#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure X(a): Acoustic Trajectories Across Severity for One Utterance
=====================================================================

本脚本专门用于绘制论文 Case Study 图 Figure X(a)。

功能：
  - 输入：同一 speaker + 同一句文本的 5 个 severity（normal→severe）音频文件
  - 提取：Praat + Librosa 声学特征
  - 输出：一张折线图，展示 severity 对关键声学特征的影响
"""

import os
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import matplotlib.pyplot as plt

# ------------------------------ 配置 ------------------------------

SEVERITY_ORDER = ["normal", "mild", "moderate", "moderately_severe", "severe"]

CASE_STUDY_ROOT = "/home/i-liyuxin/Depression_TTS/case_study_plots"

# 特征归一化方式: "zscore" / "minmax" / None
NORMALIZE_METHOD = "zscore"

# 依据 eval_quality.py 的 subject/label 选择的两位 case study subject
# healthy_383 -> actual_subject_id=383 (PHQ8_Binary=0, PHQ8_Score=7, text: "i'm from los angeles")
# depressed_348 -> actual_subject_id=348 (PHQ8_Binary=1, PHQ8_Score=20, text: "atlanta georgia")
CASE_STUDY_SUBJECTS = {
    "healthy_383": {
        "subject_id": 383,
        "phq8_binary": 0,
        "phq8_score": 7,
        "utt_id": "301_10",
        "text": "i'm from los angeles",
        "base_dir": os.path.join(CASE_STUDY_ROOT, "subject_383_utt_301_10"),
    },
    "depressed_348": {
        "subject_id": 348,
        "phq8_binary": 1,
        "phq8_score": 20,
        "utt_id": "300_10",
        "text": "atlanta georgia",
        "base_dir": os.path.join(CASE_STUDY_ROOT, "subject_348_utt_300_10"),
    },
}

# 默认绘制哪位 subject，可在命令行或直接修改此常量进行切换
#SELECTED_CASE_KEY = "healthy_383"
SELECTED_CASE_KEY = "depressed_348"

# ------------------------------ 特征提取 ------------------------------

def extract_praat_features(wav_path):
    sound = parselmouth.Sound(wav_path)

    # Pitch
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]
    f0_mean = float(np.mean(f0_values)) if len(f0_values) else np.nan

    # Jitter / Shimmer
    pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    shimmer = float(call([sound, pp], "Get shimmer (local)",
                         0, 0, 0.0001, 0.02, 1.3, 1.6))

    # HNR
    harmonicity = sound.to_harmonicity()
    hnr = float(call(harmonicity, "Get mean", 0, 0))

    # Formants
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    duration = sound.get_total_duration()
    t_list = np.linspace(0, duration, 200)

    f1_values = []
    f2_values = []
    for t in t_list:
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        if f1 and not np.isnan(f1): f1_values.append(f1)
        if f2 and not np.isnan(f2): f2_values.append(f2)

    f1_mean = np.mean(f1_values) if f1_values else np.nan
    f2_mean = np.mean(f2_values) if f2_values else np.nan

    return dict(
        f0_mean=f0_mean,
        jitter=jitter,
        shimmer=shimmer,
        hnr=hnr,
        f1_mean=f1_mean,
        f2_mean=f2_mean,
    )


def extract_temporal_features(wav_path):
    y, sr = librosa.load(wav_path, sr=None)

    total_dur = librosa.get_duration(y=y, sr=sr)
    nonsil = librosa.effects.split(y, top_db=20)

    speech_dur = sum((end - start) for start, end in nonsil) / sr
    pause_dur = total_dur - speech_dur

    pause_ratio = pause_dur / total_dur if total_dur > 0 else np.nan
    silence_speech_ratio = pause_dur / speech_dur if speech_dur > 0 else np.nan

    return dict(
        pause_ratio=pause_ratio,
        silence_speech_ratio=silence_speech_ratio,
    )


# ------------------------------ 主逻辑 ------------------------------

def build_audio_paths(case_key: str) -> tuple[dict[str, str], dict]:
    if case_key not in CASE_STUDY_SUBJECTS:
        raise KeyError(f"未找到 case key: {case_key}; 可选项: {list(CASE_STUDY_SUBJECTS.keys())}")

    meta = CASE_STUDY_SUBJECTS[case_key]
    base_dir = meta["base_dir"]
    utt_id = meta["utt_id"]

    audio_paths = {
        sev: os.path.join(base_dir, f"{utt_id}_{sev}.wav")
        for sev in SEVERITY_ORDER
    }

    missing = [p for p in audio_paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "以下音频文件缺失，请确认已经从 /data/depression_tts/quality_eval_v9_099 拷贝到 "
            f"{base_dir}：\n" + "\n".join(missing)
        )

    return audio_paths, meta


def collect_features(audio_paths: dict[str, str]):
    features = {k: {} for k in SEVERITY_ORDER}

    for sev in SEVERITY_ORDER:
        wav = audio_paths[sev]

        praat = extract_praat_features(wav)
        temp = extract_temporal_features(wav)

        features[sev].update(praat)
        features[sev].update(temp)

    return features


# ------------------------------ 特征归一化 ------------------------------

def normalize_features(features: dict[str, dict[str, float]], method: str | None):
    """
    对每个特征在 5 个 severity 维度上做归一化：
      - zscore: (x - mean) / std
      - minmax: (x - min) / (max - min)
    """
    if method is None:
        return features

    norm = {sev: {} for sev in SEVERITY_ORDER}

    # 所有特征名（假定每个 severity 下的 key 一致）
    keys = list(next(iter(features.values())).keys())

    for k in keys:
        vals = np.array([features[sev][k] for sev in SEVERITY_ORDER], dtype=float)
        mask = ~np.isnan(vals)

        if not mask.any():
            # 全是 NaN，直接原样拷贝
            for sev, v in zip(SEVERITY_ORDER, vals):
                norm[sev][k] = v
            continue

        valid = vals[mask]

        if method == "zscore":
            mu = valid.mean()
            sigma = valid.std()
            if sigma < 1e-8:
                norm_vals = np.zeros_like(vals)
                norm_vals[~mask] = np.nan
            else:
                norm_vals = (vals - mu) / sigma
                norm_vals[~mask] = np.nan
        elif method == "minmax":
            vmin = valid.min()
            vmax = valid.max()
            if abs(vmax - vmin) < 1e-8:
                norm_vals = np.zeros_like(vals)
                norm_vals[~mask] = np.nan
            else:
                norm_vals = (vals - vmin) / (vmax - vmin)
                norm_vals[~mask] = np.nan
        else:
            raise ValueError(f"未知归一化方式: {method}")

        for sev, v in zip(SEVERITY_ORDER, norm_vals):
            norm[sev][k] = float(v)

    return norm


# ------------------------------ 绘制图（Figure X(a)） ------------------------------

def plot_figure_xa(features, case_meta):
    sev_idx = np.arange(len(SEVERITY_ORDER))

    # 只保留三个与论文最相关的指标：
    # 1) Pause ratio
    # 2) Silence–speech ratio
    # 3) HNR（为更直观展示“随严重度上升而恶化”的趋势，这里取反向：-HNR）
    selected_metrics = [
        ("pause_ratio", "Pause Ratio"),
        #("silence_speech_ratio", "Silence–Speech Ratio"),
        ("hnr", "HNR (reversed)"),
    ]

    plt.figure(figsize=(9, 6))

    for key, label in selected_metrics:
        vals = [features[sev][key] for sev in SEVERITY_ORDER]
        if key == "hnr":
            # HNR 越低代表语音质量越差，这里取反向便于与“严重度升高→数值升高”对齐
            vals = [-v if not np.isnan(v) else np.nan for v in vals]
        plt.plot(sev_idx, vals, marker="o", linewidth=2.2, label=label)

    plt.xticks(sev_idx, ["Normal", "Mild", "Moderate", "Mod-Severe", "Severe"],
               fontsize=11)
    plt.ylabel("Normalized Feature Value", fontsize=12)
    plt.xlabel("Depression Severity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=10, loc="best")
    label_str = "Depressed" if case_meta["phq8_binary"] else "Healthy"
    title = (
        "Figure X(a). Acoustic Trajectories Across Depression Severity\n"
        f"Subject {case_meta['subject_id']} (PHQ8={case_meta['phq8_score']}, label={label_str})"
    )
    plt.title(title, fontsize=14)

    plt.tight_layout()
    out_name = f"figure_Xa_case_study_{case_meta['subject_id']}.png"
    plt.savefig(out_name, dpi=300)
    plt.show()


# ------------------------------ Run ------------------------------

if __name__ == "__main__":
    audio_paths, case_meta = build_audio_paths(SELECTED_CASE_KEY)
    feats = collect_features(audio_paths)
    feats_norm = normalize_features(feats, NORMALIZE_METHOD)
    plot_figure_xa(feats_norm, case_meta)
