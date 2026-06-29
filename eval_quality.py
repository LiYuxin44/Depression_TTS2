#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate Depression-TTS objective metrics on DAIC test set:

- WER: Whisper-large-v3 ASR on synthesized audios, vs ground-truth text
- SIM-o: cosine similarity between speaker embeddings of
         synthesized audio and averaged embedding from all reference utterances
         of the same subject (from training/validation set)

Also dumps a CSV of CMOS trial candidates for subjective test:
  cmos_trials.csv: utt_id, subject_id, severity, ref_wav (representative), 
                   ref_wav_count, syn_wav, text
"""

import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

from transformers import (
    pipeline,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)

# ─────────────────────────── Config ───────────────────────────

# 合成输出目录（你的合成脚本写的地方）
SYN_BASE_DIR = "/data/depression_tts/quality_eval_v9_099"
EVAL_LOG_PATH = os.path.join(SYN_BASE_DIR, "eval_generation_log.csv")

# 原始 test filelist
TEST_FILELIST = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_test_22k.txt"

# 训练/验证 subject list（与训练脚本一致），用于复现 subject→spk 映射
TRAIN_SUBJECT_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train_subjects.txt"
VAL_SUBJECT_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_val_subjects.txt"

# 训练阶段可用的真实语音（作为 SIM-o 对照）
REFERENCE_FILELISTS = [
    "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train_22k.txt",
    "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_val_22k.txt",
]

# 模型名称（可按需要替换成你本地 / finetune 的 checkpoint）
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WAVLM_MODEL_NAME = "microsoft/wavlm-large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE_WAVLM = 16000  # WavLM 预期采样率

SEVERITIES = ["normal", "mild", "moderate", "moderately_severe", "severe"]

# 是否导出 CMOS 试验列表
DUMP_CMOS_TRIALS = True
CMOS_TRIALS_PATH = os.path.join(SYN_BASE_DIR, "cmos_trials.csv")


# ─────────────────────────── Utils ───────────────────────────

def normalize_text(text: str) -> str:
    """简单文本归一化：小写、去标点、多空格合并。"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (编辑距离 / ref_len)."""
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()

    # 动态规划编辑距离
    R, H = len(r), len(h)
    dp = np.zeros((R + 1, H + 1), dtype=int)
    for i in range(R + 1):
        dp[i, 0] = i
    for j in range(H + 1):
        dp[0, j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )
    if R == 0:
        return 0.0
    # WER = 编辑距离 / 参考文本长度，理论上可以超过 1.0（大量插入错误）
    # 但通常限制在 1.0 以内更合理
    wer_value = dp[R, H] / float(R)
    return min(wer_value, 1.0)  # 限制最大值为 1.0


# ─────────────────────────── Load data ───────────────────────────

def load_eval_log(path: str) -> pd.DataFrame:
    """
    读取合成日志 eval_generation_log.csv

    Columns:
        utt_id,subject_id,severity,alpha,audio_path,text
    """
    df = pd.read_csv(path)
    # 保守处理：确保这些列存在
    expected = ["utt_id", "subject_id", "severity", "alpha", "audio_path", "text"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in eval log: {missing}")
    for optional_col in ["actual_subject_id", "spk_idx"]:
        if optional_col not in df.columns:
            df[optional_col] = np.nan
    return df


def load_test_filelist(path: str) -> dict:
    """
    读取 test filelist:
        wav_path|text
    返回:
        utt_id -> { "wav": wav_path, "text": text }
    其中 utt_id 为 wav 文件名的 stem，如 300_10.
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            wav_path, text = line.split("|", 1)
            utt_id = Path(wav_path).stem
            mapping[utt_id] = {
                "wav": wav_path,
                "text": text.strip(),
            }
    return mapping


def _read_subject_ids(path: str) -> list[int]:
    ids = []
    seen = set()
    if not path or not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or not s.isdigit():
                continue
            sid = int(s)
            if sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
    return ids


def build_subject_mapping(train_subject_file: str, val_subject_file: str) -> tuple[list[int], dict[int, int]]:
    ordered = _read_subject_ids(train_subject_file)
    seen = set(ordered)
    for sid in _read_subject_ids(val_subject_file):
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    subject_to_spk_idx = {sid: idx for idx, sid in enumerate(ordered)}
    return ordered, subject_to_spk_idx


def build_subject_audio_library(filelists: list[str]) -> dict[int, list[str]]:
    """
    汇总训练/验证阶段可用的真实语音，返回 subject_id -> [wav paths]
    """
    audio_map: dict[int, list[str]] = defaultdict(list)
    for fl in filelists:
        if not fl or not os.path.exists(fl):
            continue
        with open(fl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                wav_path, _ = line.split("|", 1)
                utt_id = Path(wav_path).stem
                try:
                    subject_id = int(utt_id.split("_")[0])
                except Exception:
                    continue
                audio_map[subject_id].append(wav_path)
    return audio_map


def resolve_actual_subject_id(raw_subject_id, ordered_subjects: list[int], subject_to_spk_idx: dict[int, int]) -> int | None:
    """
    复现合成脚本中的 subject→spk_idx 逻辑，找出真实被使用的 training subject。
    """
    if raw_subject_id is None or (isinstance(raw_subject_id, float) and np.isnan(raw_subject_id)):
        return None
    try:
        sid = int(raw_subject_id)
    except Exception:
        return None

    if sid in subject_to_spk_idx:
        return sid

    if not ordered_subjects:
        return None

    n_spk = len(ordered_subjects)
    fallback_idx = sid % n_spk
    return ordered_subjects[fallback_idx]


# ─────────────────────────── ASR (Whisper) ───────────────────────────

def build_asr_pipeline():
    """
    Whisper-large-v3 ASR pipeline (HF transformers).
    如果你有本地 checkpoint，可以把 model 参数改成本地路径。
    """
    print(f"Loading ASR model: {WHISPER_MODEL_NAME}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        device=0 if DEVICE == "cuda" else -1,
    )
    return asr


def transcribe(asr_pipeline, wav_path: str) -> str:
    """对单个 wav 做 ASR，返回文本字符串。"""
    out = asr_pipeline(wav_path)
    # HF pipeline 可能返回 dict 或 list[dict]
    if isinstance(out, list):
        out = out[0]
    text = out.get("text", "")
    return text


# ─────────────────────────── Speaker Embeddings (WavLM) ───────────────────────────

class WavLMSpeakerEmbedder:
    """简易 WavLM-large speaker embedding：mean-pool last hidden state."""

    def __init__(self, model_name=WAVLM_MODEL_NAME, device=DEVICE):
        print(f"Loading WavLM model: {model_name}")
        self.device = device
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        wav: [T] (单通道)
        sr: sample rate
        返回: [D] 的 L2-normalized speaker embedding
        """
        if sr != SAMPLE_RATE_WAVLM:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=SAMPLE_RATE_WAVLM
            )
            sr = SAMPLE_RATE_WAVLM

        wav_np = wav.numpy()
        inputs = self.extractor(
            wav_np,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        hidden = out.last_hidden_state  # [1, T', D]
        emb = hidden.mean(dim=1).squeeze(0)  # [D]
        emb = emb / (emb.norm(p=2) + 1e-6)
        return emb.cpu()


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    """Load wav as mono [T], sr."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=False)
    else:
        wav = wav.squeeze(0)
    return wav, sr


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm(p=2) * b.norm(p=2) + 1e-8))


def print_table_from_csv(csv_path: str):
    """
    从已有的 CSV 文件读取结果并输出格式化的表格。
    如果 CSV 没有标准差列，只输出平均值。
    """
    if not os.path.exists(csv_path):
        return False
    
    df = pd.read_csv(csv_path)
    
    print("\n=== Objective metrics per severity (from existing CSV) ===")
    
    # 检查是否有标准差列
    has_wer_std = "WER_std" in df.columns
    has_sim_std = "SIM_o_std" in df.columns
    
    for _, row in df.iterrows():
        sev = row["severity"]
        wer_val = row["WER"]
        sim_val = row["SIM_o"]
        
        if has_wer_std and not pd.isna(row["WER_std"]):
            wer_std = row["WER_std"]
            wer_str = f"{wer_val:.4f}±{wer_std:.4f}"
        else:
            wer_str = f"{wer_val:.4f}"
        
        if has_sim_std and not pd.isna(row["SIM_o_std"]):
            sim_std = row["SIM_o_std"]
            sim_str = f"{sim_val:.4f}±{sim_std:.4f}"
        else:
            sim_str = f"{sim_val:.4f}"
        
        print(f"{sev:18s}  WER={wer_str}   SIM-o={sim_str}")
    
    # 输出表格格式
    print("\n=== Formatted Table ===")
    print(f"{'Severity':<20} {'WER (mean±std)':<20} {'SIM-o (mean±std)':<20}")
    print("-" * 60)
    for _, row in df.iterrows():
        sev = row["severity"]
        wer_val = row["WER"]
        sim_val = row["SIM_o"]
        
        if has_wer_std and not pd.isna(row["WER_std"]):
            wer_std = row["WER_std"]
            wer_str = f"{wer_val:.4f}±{wer_std:.4f}"
        else:
            wer_str = f"{wer_val:.4f}"
        
        if has_sim_std and not pd.isna(row["SIM_o_std"]):
            sim_std = row["SIM_o_std"]
            sim_str = f"{sim_val:.4f}±{sim_std:.4f}"
        else:
            sim_str = f"{sim_val:.4f}"
        
        print(f"{sev:<20} {wer_str:<20} {sim_str:<20}")
    
    # LaTeX 表格
    print("\n=== LaTeX Table ===")
    if has_wer_std and has_sim_std:
        print(r"""
\begin{table}[t]
\caption{Objective TTS metrics on the cross-sentence synthesis task.}
\centering
\small
\begin{tabular}{|c|c|c|}
\hline
System & WER $\downarrow$ & SIM-o $\uparrow$ \\
\hline""")
        for _, row in df.iterrows():
            sev = row["severity"]
            wer_val = row["WER"]
            wer_std = row["WER_std"] if not pd.isna(row["WER_std"]) else 0.0
            sim_val = row["SIM_o"]
            sim_std = row["SIM_o_std"] if not pd.isna(row["SIM_o_std"]) else 0.0
            sev_label = sev.replace("_", " ").title()
            print(f"Ours ({sev_label.lower()}) & {wer_val:.3f}$\\pm${wer_std:.3f} & {sim_val:.3f}$\\pm${sim_std:.3f} \\\\")
        print(r"""\hline
\end{tabular}
\label{tab:tts_obj}
\end{table}
""")
    else:
        print("Note: CSV does not contain standard deviation columns.")
        print("Re-run evaluation to get SD values.")
    
    return True


# ─────────────────────────── Main eval ───────────────────────────

def main():
    syn_base = Path(SYN_BASE_DIR)
    assert syn_base.exists(), f"SYN_BASE_DIR not found: {syn_base}"

    # 检查是否已有结果 CSV，如果有就直接输出表格
    summary_path = os.path.join(SYN_BASE_DIR, "objective_summary_by_severity.csv")
    if os.path.exists(summary_path):
        print(f"Found existing results CSV: {summary_path}")
        print("Outputting table from existing results...")
        print_table_from_csv(summary_path)
        return

    print("Loading eval log...")
    df_syn = load_eval_log(EVAL_LOG_PATH)

    print("Loading test filelist...")
    test_map = load_test_filelist(TEST_FILELIST)

    print("Loading subject mapping & reference audio library...")
    ordered_subjects, subject_to_spk_idx = build_subject_mapping(
        TRAIN_SUBJECT_FILE,
        VAL_SUBJECT_FILE,
    )
    subject_audio_library = build_subject_audio_library(REFERENCE_FILELISTS)
    if not ordered_subjects:
        print("[WARN] ordered_subjects empty, 无法复现训练时的 spk 映射；SIM-o 将继续使用 test audio 作为对照。")
    if not subject_audio_library:
        print("[WARN] subject_audio_library 为空，找不到训练音频；SIM-o 将回退到 test audio。")

    # 小 sanity check：只保留 test 中真实存在的 utt_id
    df_syn = df_syn[df_syn["utt_id"].isin(test_map.keys())].reset_index(drop=True)
    print(f"#Synth rows after matching test set: {len(df_syn)}")

    # 准备 ASR 和 speaker embedder
    asr = build_asr_pipeline()
    spk_embedder = WavLMSpeakerEmbedder()

    # 用于统计的缓存
    per_sev_stats = {
        sev: {"wer_num": 0.0, "wer_den": 0, "wer_list": [], "sims": []}
        for sev in SEVERITIES
    }

    # 可选：为 CMOS 导出 trial 列表
    cmos_rows = []

    # 遍历每条合成
    for idx, row in tqdm(df_syn.iterrows(), total=len(df_syn), desc="Evaluating"):
        utt_id = str(row["utt_id"])
        subject_id = row["subject_id"]
        severity = str(row["severity"])
        if severity not in SEVERITIES:
            continue

        syn_path = row["audio_path"]
        text_gt = test_map[utt_id]["text"]

        # 优先使用合成日志里记录的实际 subject；缺失时再推断
        actual_subject_id = row.get("actual_subject_id", np.nan)
        if pd.isna(actual_subject_id):
            actual_subject_id = resolve_actual_subject_id(
                subject_id,
                ordered_subjects,
                subject_to_spk_idx,
            )
        # 获取该 subject 的所有参考音频路径
        ref_candidates = []
        if actual_subject_id is not None and not pd.isna(actual_subject_id):
            ref_candidates = subject_audio_library.get(int(actual_subject_id), [])
        if not ref_candidates:
            raise ValueError(
                f"Cannot find reference audio for utt_id={utt_id}, "
                f"subject_id={subject_id}, actual_subject_id={actual_subject_id}. "
                f"Please ensure the subject exists in training/validation filelists."
            )

        # 1) ASR → WER
        try:
            hyp_text = transcribe(asr, syn_path)
            w = wer(text_gt, hyp_text)
            # 这里用 ref 长度作为分母
            ref_len = len(normalize_text(text_gt).split())
            per_sev_stats[severity]["wer_num"] += w * ref_len
            per_sev_stats[severity]["wer_den"] += ref_len
            # 记录每个样本的 WER 值用于计算标准差
            per_sev_stats[severity]["wer_list"].append(w)
        except Exception as e:
            print(f"[ASR ERR] {utt_id} {severity}: {e}")

        # 2) Speaker SIM-o: 对所有参考 utterance 的 embedding 取平均
        try:
            wav_syn, sr_syn = load_wav(syn_path)
            emb_syn = spk_embedder.embed(wav_syn, sr_syn)
            
            # 提取所有参考 utterance 的 embedding 并取平均
            emb_refs = []
            for ref_wav_path in ref_candidates:
                try:
                    wav_ref, sr_ref = load_wav(ref_wav_path)
                    emb_ref = spk_embedder.embed(wav_ref, sr_ref)
                    emb_refs.append(emb_ref)
                except Exception as e:
                    print(f"[WARN] Failed to load/embed {ref_wav_path}: {e}")
                    continue
            
            if not emb_refs:
                raise ValueError(
                    f"Failed to extract embeddings from any reference audio "
                    f"for subject {actual_subject_id}"
                )
            
            # 对所有 embedding 取平均并 L2 归一化
            emb_ref_mean = torch.stack(emb_refs).mean(dim=0)
            emb_ref_mean = emb_ref_mean / (emb_ref_mean.norm(p=2) + 1e-6)
            
            s = cosine_sim(emb_syn, emb_ref_mean)
            per_sev_stats[severity]["sims"].append(s)
        except Exception as e:
            print(f"[SIM ERR] {utt_id} {severity}: {e}")

        # 3) CMOS trial row
        if DUMP_CMOS_TRIALS:
            # 使用第一个参考音频作为代表（用于记录，实际计算使用所有 utterance 的平均）
            ref_wav_representative = ref_candidates[0] if ref_candidates else None
            cmos_rows.append({
                "utt_id": utt_id,
                "subject_id": subject_id,
                "severity": severity,
                "ref_wav": ref_wav_representative,
                "ref_wav_count": len(ref_candidates),  # 记录实际使用的参考音频数量
                "syn_wav": syn_path,
                "text": text_gt,
                "actual_subject_id": actual_subject_id,
            })

    # 计算最终结果
    print("\n=== Objective metrics per severity ===")
    print("Note: WER is a ratio (0-1), where 1.0 = 100% error rate")
    print("      SIM-o is cosine similarity (0-1), higher is better")
    rows = []
    for sev in SEVERITIES:
        wer_den = per_sev_stats[sev]["wer_den"]
        if wer_den > 0:
            wer_val = per_sev_stats[sev]["wer_num"] / wer_den
        else:
            wer_val = np.nan
        
        # 计算 WER 标准差
        wer_list = per_sev_stats[sev]["wer_list"]
        wer_std = float(np.std(wer_list)) if len(wer_list) > 0 else np.nan
        
        sims = per_sev_stats[sev]["sims"]
        sim_mean = float(np.mean(sims)) if len(sims) > 0 else np.nan
        sim_std = float(np.std(sims)) if len(sims) > 0 else np.nan
        
        rows.append({
            "severity": sev,
            "WER": wer_val,
            "WER_std": wer_std,
            "SIM_o": sim_mean,
            "SIM_o_std": sim_std
        })
        print(f"{sev:18s}  WER={wer_val:.4f}±{wer_std:.4f}   SIM-o={sim_mean:.4f}±{sim_std:.4f}")

    # 存成一个 summary CSV
    summary_path = os.path.join(SYN_BASE_DIR, "objective_summary_by_severity.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSaved per-severity summary to: {summary_path}")

    # 输出格式化的表格
    print("\n=== Formatted Table ===")
    print(f"{'Severity':<20} {'WER (mean±std)':<20} {'SIM-o (mean±std)':<20}")
    print("-" * 60)
    for row in rows:
        sev = row["severity"]
        wer_val = row["WER"]
        wer_std = row["WER_std"]
        sim_val = row["SIM_o"]
        sim_std = row["SIM_o_std"]
        wer_str = f"{wer_val:.4f}±{wer_std:.4f}" if not pd.isna(wer_std) else f"{wer_val:.4f}"
        sim_str = f"{sim_val:.4f}±{sim_std:.4f}" if not pd.isna(sim_std) else f"{sim_val:.4f}"
        print(f"{sev:<20} {wer_str:<20} {sim_str:<20}")

    # LaTeX table snippet
    print("\n=== LaTeX Table ===")
    print(r"""
\begin{table}[t]
\caption{Objective TTS metrics on the cross-sentence synthesis task.}
\centering
\small
\begin{tabular}{|c|c|c|}
\hline
System & WER $\downarrow$ & SIM-o $\uparrow$ \\
\hline""")
    for row in rows:
        sev = row["severity"]
        wer_val = row["WER"]
        wer_std = row["WER_std"] if not pd.isna(row["WER_std"]) else 0.0
        sim_val = row["SIM_o"]
        sim_std = row["SIM_o_std"] if not pd.isna(row["SIM_o_std"]) else 0.0
        sev_label = sev.replace("_", " ").title()
        print(f"Ours ({sev_label.lower()}) & {wer_val:.3f}$\\pm${wer_std:.3f} & {sim_val:.3f}$\\pm${sim_std:.3f} \\\\")
    print(r"""\hline
\end{tabular}
\label{tab:tts_obj}
\end{table}
""")

    # CMOS trial 列表
    if DUMP_CMOS_TRIALS and len(cmos_rows) > 0:
        df_cmos = pd.DataFrame(cmos_rows)
        df_cmos.to_csv(CMOS_TRIALS_PATH, index=False)
        print(f"\nCMOS trial list saved to: {CMOS_TRIALS_PATH}")
        print("Columns: utt_id, subject_id, severity, ref_wav (representative), ref_wav_count, syn_wav, text")
        print("Note: ref_wav is a representative path; SIM-o calculation uses average embedding from all utterances")


if __name__ == "__main__":
    main()
