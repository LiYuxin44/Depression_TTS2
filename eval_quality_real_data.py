#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate DAIC test set real recordings for objective metrics:

- WER: Whisper-large-v3 ASR on real audios vs ground-truth text
- SIM-o: cosine similarity between each utterance and the subject anchor utterance

Outputs per-severity summary CSV + formatted tables for reporting.
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

# 真实数据 filelist（wav|text）
REAL_AUDIO_FILELIST = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_test_22k.txt"

# 受试者标签（含 PHQ8 分数 -> severity）
SUBJECT_LABEL_CSV = "/home/i-liyuxin/test/data/Label/full_test_split.csv"

# 结果输出目录
OUTPUT_BASE_DIR = "/data/depression_tts/real_audio_eval"
SUMMARY_FILENAME = "objective_summary_by_severity.csv"

# 模型名称（可按需要替换成你本地 / finetune 的 checkpoint）
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WAVLM_MODEL_NAME = "microsoft/wavlm-large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE_WAVLM = 16000  # WavLM 预期采样率

SEVERITIES = ["normal", "mild", "moderate", "moderately_severe", "severe"]


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
    return dp[R, H] / float(R)


# ─────────────────────────── Load data ───────────────────────────

def phq_score_to_severity(score: float) -> str:
    """根据 PHQ8 得分映射到 severity 标签。"""
    if pd.isna(score):
        raise ValueError("PHQ8 score is NaN.")
    score = float(score)
    if score <= 4:
        return "normal"
    if score <= 9:
        return "mild"
    if score <= 14:
        return "moderate"
    if score <= 19:
        return "moderately_severe"
    return "severe"


def load_subject_severity_map(csv_path: str) -> dict:
    """
    读取 full_test_split.csv，输出:
        subject_id (str) -> {"score": phq8, "severity": severity_label}
    """
    df = pd.read_csv(csv_path)
    required_cols = {"Participant_ID", "PHQ8_Score"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    subject_map = {}
    for _, row in df.iterrows():
        subject_id = str(row["Participant_ID"])
        score = row["PHQ8_Score"]
        severity = phq_score_to_severity(score)
        subject_map[subject_id] = {
            "score": score,
            "severity": severity,
        }
    return subject_map


def load_test_filelist(path: str) -> list[dict]:
    """
    读取 test filelist:
        wav_path|text
    返回:
        [{"utt_id": ..., "subject_id": ..., "wav": ..., "text": ...}, ...]
    其中 utt_id 为 wav 文件名的 stem，如 300_10，subject_id 为 stem 的前缀。
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            wav_path, text = line.split("|", 1)
            wav_path = wav_path.strip()
            text = text.strip()
            utt_id = Path(wav_path).stem
            parts = utt_id.split("_", 1)
            subject_id = parts[0]
            rows.append({
                "utt_id": utt_id,
                "subject_id": subject_id,
                "wav": wav_path,
                "text": text,
            })
    return rows


def build_real_eval_dataframe(filelist_path: str, subject_map: dict) -> pd.DataFrame:
    """
    将 filelist 与 subject severity 对齐，返回 DataFrame:
        utt_id, subject_id, severity, phq8_score, audio_path, text
    """
    rows = []
    missing_subjects = set()
    for item in load_test_filelist(filelist_path):
        subject_id = item["subject_id"]
        if subject_id not in subject_map:
            missing_subjects.add(subject_id)
            continue
        label_info = subject_map[subject_id]
        rows.append({
            "utt_id": item["utt_id"],
            "subject_id": subject_id,
            "severity": label_info["severity"],
            "phq8_score": label_info["score"],
            "audio_path": item["wav"],
            "text": item["text"],
        })

    if missing_subjects:
        print(f"[WARN] Missing severity labels for subjects: {sorted(missing_subjects)}")

    if not rows:
        raise ValueError("No entries found after aligning filelist with subject labels.")

    return pd.DataFrame(rows)


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
    output_dir = Path(OUTPUT_BASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果已有结果 CSV，优先输出
    summary_path = output_dir / SUMMARY_FILENAME
    if summary_path.exists():
        print(f"Found existing results CSV: {summary_path}")
        print("Outputting table from existing results...")
        print_table_from_csv(str(summary_path))
        return

    print("Loading subject severity map...")
    subject_map = load_subject_severity_map(SUBJECT_LABEL_CSV)

    print("Loading real audio entries...")
    df_eval = build_real_eval_dataframe(REAL_AUDIO_FILELIST, subject_map)
    print(f"#Real utterances found: {len(df_eval)}")

    # 准备 ASR 和 speaker embedder
    asr = build_asr_pipeline()
    spk_embedder = WavLMSpeakerEmbedder()

    # 用于统计的缓存
    per_sev_stats = {
        sev: {"wer_num": 0.0, "wer_den": 0, "wer_list": [], "sims": []}
        for sev in SEVERITIES
    }

    subject_anchor_embs: dict[str, torch.Tensor] = {}

    # 遍历每条合成
    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Evaluating real audio"):
        utt_id = str(row["utt_id"])
        subject_id = row["subject_id"]
        severity = str(row["severity"])
        if severity not in SEVERITIES:
            continue

        wav_path = row["audio_path"]
        text_gt = row["text"]

        # 1) ASR → WER
        try:
            hyp_text = transcribe(asr, wav_path)
            w = wer(text_gt, hyp_text)
            # 这里用 ref 长度作为分母
            ref_len = len(normalize_text(text_gt).split())
            per_sev_stats[severity]["wer_num"] += w * ref_len
            per_sev_stats[severity]["wer_den"] += ref_len
            # 记录每个样本的 WER 值用于计算标准差
            per_sev_stats[severity]["wer_list"].append(w)
        except Exception as e:
            print(f"[ASR ERR] {utt_id} {severity}: {e}")

        # 2) Speaker SIM-o：用同一 subject 的首个发音作为锚点
        try:
            wav_real, sr_real = load_wav(wav_path)
            emb_real = spk_embedder.embed(wav_real, sr_real)
            if subject_id not in subject_anchor_embs:
                subject_anchor_embs[subject_id] = emb_real
                s = 1.0
            else:
                s = cosine_sim(emb_real, subject_anchor_embs[subject_id])
            per_sev_stats[severity]["sims"].append(s)
        except Exception as e:
            print(f"[SIM ERR] {utt_id} {severity}: {e}")

    # 计算最终结果
    print("\n=== Objective metrics per severity ===")
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


if __name__ == "__main__":
    main()
