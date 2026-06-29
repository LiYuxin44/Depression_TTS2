#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure X(b): Radar Plot of Acoustic Profiles Across Severity
============================================================

本脚本：只做雷达图（Radar Plot），用于论文 Case Study Figure X(b)。

功能：
  - 输入：同一 speaker + 同一句文本的 5 个 severity（normal→severe）音频
  - 提取：Praat + Librosa 声学特征
  - 归一化：对每个特征在 5 个 severity 上做 z-score
  - 输出：一张雷达图，展示 5 个 severity 的整体声学 profile 形状差异
"""

import os
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import matplotlib.pyplot as plt

# ------------------------------ 配置 ------------------------------

# TODO: 修改为你自己的 5 个音频路径（同一说话人 + 同一句话）
AUDIO_PATHS = {
    "normal": "case_study/utt_normal.wav",
    "mild": "case_study/utt_mild.wav",
    "moderate": "case_study/utt_moderate.wav",
    "moderately_severe": "case_study/utt_mod_severe.wav",
    "severe": "case_study/utt_severe.wav",
}

SEVERITY_ORDER = ["normal", "mild", "moderate", "moderately_severe", "severe"]
SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Mod-Severe", "Severe"]


# ------------------------------ 特征提取 ------------------------------

def extract_praat_features(wav_path):
    """
    使用 Parselmouth (Praat) 提取:
      - f0_mean
      - jitter
      - shimmer
      - HNR
      - Formant F1/F2 mean
    """
    sound = parselmouth.Sound(wav_path)

    # Pitch
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]
    f0_mean = float(np.mean(f0_values)) if len(f0_values) else np.nan

    # Pulses for jitter/shimmer
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
    t_list = np.linspace(0, duration, 200) if duration > 0 else []

    f1_vals, f2_vals = [], []
    for t in t_list:
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        if f1 and not np.isnan(f1): f1_vals.append(f1)
        if f2 and not np.isnan(f2): f2_vals.append(f2)

    f1_mean = float(np.mean(f1_vals)) if len(f1_vals) else np.nan
    f2_mean = float(np.mean(f2_vals)) if len(f2_vals) else np.nan

    return dict(
        f0_mean=f0_mean,
        jitter=jitter,
        shimmer=shimmer,
        hnr=hnr,
        f1_mean=f1_mean,
        f2_mean=f2_mean,
    )


def extract_temporal_features(wav_path):
    """
    使用 Librosa 提取:
      - pause_ratio
      - silence_speech_ratio
    """
    y, sr = librosa.load(wav_path, sr=None)

    total_dur = librosa.get_duration(y=y, sr=sr)
    nonsil = librosa.effects.split(y, top_db=20)

    speech_dur = sum((end - start) for start, end in nonsil) / sr if len(nonsil) > 0 else 0.0
    pause_dur = max(total_dur - speech_dur, 0.0)

    pause_ratio = pause_dur / total_dur if total_dur > 0 else np.nan
    silence_speech_ratio = pause_dur / speech_dur if speech_dur > 0 else np.nan

    return dict(
        pause_ratio=pause_ratio,
        silence_speech_ratio=silence_speech_ratio,
    )


# ------------------------------ 收集特征 ------------------------------

def collect_features():
    """
    返回:
      features: dict[severity(str)] -> dict[feature_name -> value]
    """
    features = {sev: {} for sev in SEVERITY_ORDER}

    for sev in SEVERITY_ORDER:
        wav = AUDIO_PATHS[sev]
        if not os.path.exists(wav):
            raise FileNotFoundError(f"Audio file not found for {sev}: {wav}")

        praat = extract_praat_features(wav)
        temp = extract_temporal_features(wav)

        features[sev].update(praat)
        features[sev].update(temp)

    return features


# ------------------------------ 雷达图绘制 ------------------------------

def normalize_for_radar(features, metric_keys):
    """
    对每个特征在 5 个 severity 上做 z-score 归一化：
      z = (x - mean) / std
    返回：
      norm_vals: dict[severity] -> list[metrics in metric_keys order]
    """
    # 先按照 [severity, metric] 做成矩阵
    data = []
    for sev in SEVERITY_ORDER:
        row = [features[sev][k] for k in metric_keys]
        data.append(row)
    data = np.array(data, dtype=float)  # shape: (5, num_metrics)

    # 对列做 z-score
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    stds[stds == 0] = 1.0  # 避免除零

    data_z = (data - means) / stds

    # 回填 NaN 为 0（表示“接近均值”）
    data_z = np.nan_to_num(data_z, nan=0.0)

    norm_vals = {}
    for i, sev in enumerate(SEVERITY_ORDER):
        norm_vals[sev] = data_z[i, :].tolist()
    return norm_vals


def plot_radar(norm_vals, metric_labels, save_path="figure_Xb_radar.png"):
    """
    绘制一张雷达图：
      - 每个轴 = 一个声学特征
      - 每条多边形 = 一个 severity
    """
    num_metrics = len(metric_labels)
    # 每个轴的角度
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    # 为了闭合多边形，再加回第一个角度
    angles = np.concatenate([angles, angles[:1]])

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    # 逐个 severity 画
    for i, sev in enumerate(SEVERITY_ORDER):
        vals = norm_vals[sev]
        vals = np.array(vals)
        vals = np.concatenate([vals, vals[:1]])  # 闭合

        ax.plot(angles, vals, linewidth=2, marker="o", label=SEVERITY_LABELS[i])
        ax.fill(angles, vals, alpha=0.08)  # 轻微填充，增加可视化区分度

    # 设置特征标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)

    # 可以适当限制径向范围，比如 [-2.5, 2.5]
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticklabels([])  # 不显示 radius 数字，避免过于拥挤

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Figure X(b). Global Acoustic Profiles Across Depression Severity",
                 fontsize=14, pad=20)

    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------ Run ------------------------------

if __name__ == "__main__":
    # 1) 提取特征
    feats = collect_features()

    # 2) 选择希望在雷达图上展示的声学指标（6~8 维比较合适）
    metric_keys = [
        "pause_ratio",
        "silence_speech_ratio",
        "shimmer",
        "hnr",
        "f1_mean",
        "f2_mean",
        # 也可以加上 "f0_mean" 或 "jitter" 等
    ]
    metric_labels = [
        "Pause Ratio",
        "Silence–Speech",
        "Shimmer",
        "HNR",
        "Formant F1",
        "Formant F2",
    ]

    # 3) 归一化到同一量纲（z-score）
    norm_vals = normalize_for_radar(feats, metric_keys)

    # 4) 画雷达图
    plot_radar(norm_vals, metric_labels, save_path="figure_Xb_radar.png")
