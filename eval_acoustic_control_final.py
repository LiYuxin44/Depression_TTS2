#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate Acoustic Control for Depression-TTS
============================================

目标：
  - 验证我们的 depression condition 是否真的在“同一说话人 + 同一文本”下
    系统性地改变声学特征（而不是复现任何特定语料或他人结论）。

指标：
  1. 全局单调性： severity_score ↑ 时，各声学特征是否有整体单调趋势（Spearman 相关）。
  2. 组内单调性：对同一 speaker + 同一 text 的多种 severity 合成，
     看每组内部的 Spearman 相关（per-group ρ）。
  3. 配对差值：对 normal vs severe 成对句子，直接看特征差值 Δ(severe - normal)
     是否大部分方向一致（表示 TTS 控制是可靠的）。
"""

import logging
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import parselmouth
from parselmouth.praat import call
import librosa
from scipy.stats import spearmanr

# ─────────────────────────── Config ───────────────────────────
SYN_BASE_DIR = "/data/depression_tts/quality_eval_v9_099"
#SYN_BASE_DIR = "/data/depression_tts/quality_eval_test_full_1119_amplify"
EVAL_LOG_PATH = os.path.join(SYN_BASE_DIR, "eval_generation_log.csv")
LOG_DIR = "/home/i-liyuxin/Depression_TTS"
LOG_PATH = os.path.join(LOG_DIR, "eval_acoustic_control2.log")

SEVERITY_MAP = {
    "normal": 0,
    "mild": 1,
    "moderate": 2,
    "moderately_severe": 3,
    "severe": 4,
}

# ─────────────────────────── Logging ───────────────────────────


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("acoustic_control_eval")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logger()


# ─────────────────────────── Helpers ───────────────────────────


def infer_ids_from_path(wav_path):
    """
    根据文件名解析 speaker_id 和 text_id。
    你的格式示例：
        421_123_normal.wav
        453_263_severe.wav
    我们假设格式为：
        <speaker_id>_<text_id>_<severity>.wav
    则：
        speaker_id = '421'
        text_id    = '123'
    """
    base = os.path.basename(wav_path)
    name, _ = os.path.splitext(base)
    parts = name.split("_")

    # 例如 ['421', '123', 'normal']
    speaker_id = None
    text_id = None

    if len(parts) >= 2:
        speaker_id = parts[0]
        text_id = parts[1]

    return speaker_id, text_id


# ─────────────────────────── Extraction Logic ───────────────────────────


def extract_praat_features(wav_path):
    """
    用 Parselmouth(Praat) 提取:
      - F0: mean/std/range
      - Jitter/Shimmer
      - HNR
      - Formant F1/F2 及 F2/F1 中心化指标
    """
    try:
        sound = parselmouth.Sound(wav_path)

        # 1. Pitch
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values != 0]  # Remove unvoiced

        if len(f0_values) == 0:
            return None

        f0_mean = float(np.mean(f0_values))
        f0_std = float(np.std(f0_values))
        f0_range = float(np.max(f0_values) - np.min(f0_values))

        # 2. Pulses for jitter/shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        jitter = float(
            call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        )
        shimmer = float(
            call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
        )

        # 3. Harmonicity (HNR)
        harmonicity = sound.to_harmonicity()
        hnr = float(call(harmonicity, "Get mean", 0, 0))

        # 4. Formants
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        duration = sound.get_total_duration()
        if duration <= 0:
            times = np.array([])
        else:
            # 最多取 200 个时间点
            times = np.linspace(0.0, duration, num=min(int(duration * 100), 200))

        f1_values, f2_values = [], []
        for t in times:
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            if f1 and not np.isnan(f1):
                f1_values.append(f1)
            if f2 and not np.isnan(f2):
                f2_values.append(f2)

        formant_f1_mean = float(np.mean(f1_values)) if len(f1_values) > 0 else np.nan
        formant_f2_mean = float(np.mean(f2_values)) if len(f2_values) > 0 else np.nan
        formant_centralization = (
            float(formant_f2_mean / formant_f1_mean)
            if formant_f1_mean and formant_f1_mean != 0
            else np.nan
        )

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_range": f0_range,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            "formant_f1_mean": formant_f1_mean,
            "formant_f2_mean": formant_f2_mean,
            "formant_centralization": formant_centralization,
        }
    except Exception as e:
        LOGGER.warning(f"Praat Error {wav_path}: {e}")
        return None


def extract_temporal_features(wav_path):
    """
    用 Librosa 提取:
      - 能量
      - 暂停/语音时间 & 比例
      - MFCC + Δ + ΔΔ（全局均值）
    """
    try:
        y, sr = librosa.load(wav_path, sr=None)

        # 1. Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms))

        # 2. Pause / speech segmentation
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        total_duration = float(librosa.get_duration(y=y, sr=sr))

        speech_duration = (
            sum(end - start for start, end in non_silent_intervals) / sr
            if len(non_silent_intervals) > 0
            else 0.0
        )
        pause_duration = total_duration - speech_duration
        pause_duration = max(pause_duration, 0.0)

        pause_ratio = (
            pause_duration / total_duration if total_duration > 0 else np.nan
        )
        silence_speech_ratio = (
            pause_duration / speech_duration if speech_duration > 0 else np.nan
        )
        speech_ratio = (
            speech_duration / total_duration if total_duration > 0 else np.nan
        )

        # 3. MFCCs
        n_mfcc = 13
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_dd = librosa.feature.delta(mfcc, order=2)

        def _mean_features(mat, prefix):
            if mat.shape[1] == 0:
                return {f"{prefix}_{i+1:02d}": np.nan for i in range(mat.shape[0])}
            return {
                f"{prefix}_{i+1:02d}": float(np.mean(mat[i]))
                for i in range(mat.shape[0])
            }

        mfcc_means = _mean_features(mfcc, "mfcc_mean")
        mfcc_delta_means = _mean_features(mfcc_delta, "mfcc_delta_mean")
        mfcc_dd_means = _mean_features(mfcc_dd, "mfcc_dd_mean")

        return {
            "energy_mean": energy_mean,
            "pause_duration": pause_duration,
            "total_duration": total_duration,
            "pause_ratio": pause_ratio,
            "silence_speech_ratio": silence_speech_ratio,
            "speech_ratio": speech_ratio,
            **mfcc_means,
            **mfcc_delta_means,
            **mfcc_dd_means,
        }

    except Exception as e:
        LOGGER.warning(f"Librosa Error {wav_path}: {e}")
        return None


# ─────────────────────────── I/O ───────────────────────────


def load_eval_log(path):
    df = pd.read_csv(path)
    df = df[df["severity"].isin(SEVERITY_MAP.keys())].copy()
    return df


# ─────────────────────────── Main ───────────────────────────


def main():
    LOGGER.info(f"Loading log from: {EVAL_LOG_PATH}")
    if not os.path.exists(EVAL_LOG_PATH):
        LOGGER.error("Log file not found.")
        return

    df_log = load_eval_log(EVAL_LOG_PATH)
    results = []

    LOGGER.info("Extracting features using Praat (Parselmouth) and Librosa...")
    for _, row in tqdm(df_log.iterrows(), total=len(df_log)):
        wav_path = row["audio_path"]
        if not os.path.exists(wav_path):
            continue

        praat_feats = extract_praat_features(wav_path)
        if not praat_feats:
            continue

        temp_feats = extract_temporal_features(wav_path)
        if not temp_feats:
            continue

        combined = {**praat_feats, **temp_feats}
        combined["severity_score"] = SEVERITY_MAP[row["severity"]]
        combined["severity_label"] = row["severity"]

        # ── speaker_id / text_id：优先用 log 里已有的列，否则从文件名解析 ──
        if "speaker_id" in df_log.columns:
            combined["speaker_id"] = row["speaker_id"]
        if "text_id" in df_log.columns:
            combined["text_id"] = row["text_id"]

        if "speaker_id" not in combined or pd.isna(combined.get("speaker_id", None)):
            spk, utt = infer_ids_from_path(wav_path)
            if spk is not None:
                combined["speaker_id"] = spk
            if utt is not None:
                combined["text_id"] = utt

        results.append(combined)

    res_df = pd.DataFrame(results)

    if len(res_df) == 0:
        LOGGER.warning("No results.")
        return

    # 保存原始特征表，方便后续做图/其他分析
    out_path = os.path.join(SYN_BASE_DIR, "acoustic_validation_results.csv")
    res_df.to_csv(out_path, index=False)
    LOGGER.info(f"Detailed feature-level results saved to: {out_path}")

    # ─────────────────── Statistical Analysis ───────────────────

    LOGGER.info("=" * 80)
    LOGGER.info(" ACOUSTIC CONTROLLABILITY VALIDATION (Depression-TTS) ")
    LOGGER.info("=" * 80)

    # 核心关注的特征
    core_metrics = [
        "f0_mean",
        "f0_std",
        "f0_range",
        "energy_mean",
        "total_duration",
        "pause_duration",
        "pause_ratio",
        "silence_speech_ratio",
        "speech_ratio",
        "jitter",
        "shimmer",
        "hnr",
        "formant_f1_mean",
        "formant_f2_mean",
        "formant_centralization",
    ]

    mfcc_cols = [
        c
        for c in res_df.columns
        if c.startswith(("mfcc_mean_", "mfcc_delta_mean_", "mfcc_dd_mean_"))
    ]

    # ───────────────── Step 1: 全局 Spearman 单调性 ─────────────────

    LOGGER.info("[Step 1] Global correlation over all samples")
    LOGGER.info(f"{'Feature':<24} {'rho(Spearman)':<14} {'p-value':<10} {'N':<6}")
    LOGGER.info("-" * 60)

    for metric in core_metrics + mfcc_cols:
        if metric not in res_df.columns:
            continue

        subset = res_df[["severity_score", metric]].dropna()
        if len(subset) < 3:
            continue
        if subset[metric].nunique() < 2:
            continue

        rho, p_val = spearmanr(subset["severity_score"], subset[metric])
        if np.isnan(rho):
            continue

        LOGGER.info(
            f"{metric:<24} {rho:>8.3f}       {p_val:.1e}    {len(subset):<6}"
        )

    # ───────────────── Step 2: 同 speaker + 同 text 内部的单调性 ─────────────────

    LOGGER.info("=" * 80)
    LOGGER.info(" [Step 2] Within-speaker & same-text controllability ")
    LOGGER.info("=" * 80)

    if "speaker_id" not in res_df.columns or "text_id" not in res_df.columns:
        LOGGER.warning(
            "'speaker_id' and/or 'text_id' not found in res_df; "
            "skip within-utterance analysis."
        )
    else:
        group_cols = ["speaker_id", "text_id"]

        LOGGER.info(f"Unique speaker_id count: {res_df['speaker_id'].nunique()}")
        LOGGER.info(f"Unique text_id count: {res_df['text_id'].nunique()}")

        for metric in core_metrics:
            if metric not in res_df.columns:
                continue

            per_group_r = []
            for _, g in res_df.groupby(group_cols):
                g_sub = g[["severity_score", metric]].dropna()
                # 至少需要两个不同的 severity
                if g_sub["severity_score"].nunique() < 2:
                    continue
                if len(g_sub) < 3:
                    # 只有两个点时 Spearman 相关不稳定，这里先跳过
                    continue
                if g_sub[metric].nunique() < 2:
                    # 特征恒定会触发 Spearman 的 ConstantInputWarning
                    continue

                rho, _ = spearmanr(g_sub["severity_score"], g_sub[metric])
                if not np.isnan(rho):
                    per_group_r.append(rho)

            if len(per_group_r) == 0:
                continue

            per_group_r = np.array(per_group_r)
            n_groups = len(per_group_r)

            # 定一个“期望方向”（只是作为我们设计控制时的直觉）
            expected_negative = metric in [
                "f0_mean",
                "f0_std",
                "f0_range",
                "energy_mean",
                "speech_ratio",
                "hnr",
                "formant_f2_mean",
                "formant_centralization",
            ]
            expected_positive = metric in [
                "total_duration",
                "pause_duration",
                "pause_ratio",
                "silence_speech_ratio",
                "jitter",
                "shimmer",
                "formant_f1_mean",
            ]

            if expected_negative:
                prop_dir = float(np.mean(per_group_r < 0))
                direction_str = "severity↑ → feature↓"
            elif expected_positive:
                prop_dir = float(np.mean(per_group_r > 0))
                direction_str = "severity↑ → feature↑"
            else:
                prop_dir = np.nan
                direction_str = "no predefined direction"

            LOGGER.info(
                f"{metric:<24} groups={n_groups:<4d}  "
                f"mean_r={np.mean(per_group_r):>6.3f}  "
                f"median_r={np.median(per_group_r):>6.3f}  "
                f"dir_ok={prop_dir:>5.2f}  ({direction_str})"
            )

        # ───────── Step 2b: normal vs severe 的配对差值 ─────────

        LOGGER.info("[Step 2b] Pairwise difference: severe vs normal")
        for metric in core_metrics:
            if metric not in res_df.columns:
                continue

            deltas = []
            for _, g in res_df.groupby(group_cols):
                g_n = g[g["severity_score"] == 0]  # normal
                g_s = g[g["severity_score"] == 4]  # severe
                if len(g_n) == 0 or len(g_s) == 0:
                    continue

                v_n = g_n[metric].mean()
                v_s = g_s[metric].mean()
                if np.isnan(v_n) or np.isnan(v_s):
                    continue

                deltas.append(v_s - v_n)

            if len(deltas) == 0:
                continue

            deltas = np.array(deltas)
            n_pairs = len(deltas)
            mean_delta = float(np.mean(deltas))

            expected_negative = metric in [
                "f0_mean",
                "f0_std",
                "f0_range",
                "energy_mean",
                "speech_ratio",
                "hnr",
                "formant_f2_mean",
                "formant_centralization",
            ]
            expected_positive = metric in [
                "total_duration",
                "pause_duration",
                "pause_ratio",
                "silence_speech_ratio",
                "jitter",
                "shimmer",
                "formant_f1_mean",
            ]

            if expected_negative:
                prop_dir = float(np.mean(deltas < 0))
                dir_str = "severity↑ → feature↓"
            elif expected_positive:
                prop_dir = float(np.mean(deltas > 0))
                dir_str = "severity↑ → feature↑"
            else:
                prop_dir = np.nan
                dir_str = "no predefined direction"

            LOGGER.info(
                f"{metric:<24} pairs={n_pairs:<4d}  "
                f"meanΔ={mean_delta:>7.3f}  dir_ok={prop_dir:>5.2f}  ({dir_str})"
            )


if __name__ == "__main__":
    main()
