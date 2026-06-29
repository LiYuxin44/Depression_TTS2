#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Temporal Case Study: 三行可视化图
==================================

为一个 subject 生成10张图（2个文本 × 5个严重度），每张图包含三行：
1. 波形 + 静音掩膜（仅第一行有静音掩膜）
2. Mel 频谱
3. HNR 时间曲线

功能：
  - 从 eval_generation_log.csv 读取数据
  - 使用 DeepSeek 对文本做情感分类
  - 为一个 subject 选择一个 positive 和一个 negative 文本
  - 为每个文本的5个严重度各生成一张图（共10张）
"""

import os
import json
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
import librosa.display
import matplotlib.pyplot as plt
import asyncio
from pathlib import Path
from tqdm.auto import tqdm

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai library not installed. Please install it first.")

# ------------------------------ 配置 ------------------------------

EVAL_LOG_PATH = "/data/depression_tts/quality_eval_v9_099/eval_generation_log.csv"
LABEL_CSV_PATH = "/home/i-liyuxin/test/data/Label/train_cleaned.csv"
OUTPUT_DIR = "/home/i-liyuxin/Depression_TTS/case_study_plots/temporal"
SENTIMENT_CACHE_PATH = os.path.join(OUTPUT_DIR, "sentiment_cache.json")

# DeepSeek 配置
OPENAI_API_KEY = "ak-57d1efgh23i9jkl64mno32pqrs18tuv4k6"
OPENAI_BASE_URL = "https://models-proxy.stepfun-inc.com/v1"
OPENAI_TIMEOUT = 600
SENTIMENT_MODEL = "deepseek-r1-0528-volce-basemodel"

# 选择的 subject（只选择一个）
SELECTED_SUBJECT_ID = 348  # PHQ8_Score=20 (高)

# 五个严重度列表
ALL_SEVERITIES = ['normal', 'mild', 'moderate', 'moderately_severe', 'severe']

# HNR 阈值线
HNR_THRESHOLD_DB = 5.0

# ------------------------------ 数据加载 ------------------------------

def load_labels() -> dict:
    """加载 PHQ8 标签"""
    labels = {}
    if not os.path.exists(LABEL_CSV_PATH):
        print(f"Warning: Label CSV not found: {LABEL_CSV_PATH}")
        return labels
    
    df = pd.read_csv(LABEL_CSV_PATH)
    for _, row in df.iterrows():
        pid = int(row['Participant_ID'])
        if pd.notna(row['PHQ8_Score']):
            labels[pid] = {
                'phq8_score': float(row['PHQ8_Score']),
                'phq8_binary': int(row['PHQ8_Binary']) if pd.notna(row['PHQ8_Binary']) else None,
            }
    return labels


def load_eval_log() -> pd.DataFrame:
    """加载合成日志"""
    df = pd.read_csv(EVAL_LOG_PATH)
    df = df.dropna(subset=['actual_subject_id'])
    df['actual_subject_id'] = df['actual_subject_id'].astype(int)
    return df


# ------------------------------ 情感分类 ------------------------------

async def classify_sentiment_async(text: str, client: AsyncOpenAI, max_retries: int = 3) -> str:
    """使用 DeepSeek 对文本做情感分类"""
    prompt = f"""你是一名专业的文本情绪分析专家。请分析以下文本的情感倾向。

文本内容：
{text}

请判断这段文本的情感倾向，只回答以下三种之一（不要添加任何其他文字）：
1. positive - 文本表达积极、乐观、正面的情绪
2. neutral - 文本表达中性、客观、无明显情绪倾向
3. negative - 文本表达消极、悲观、负面的情绪

请只回答：positive、neutral 或 negative（三个词之一）"""
    
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=SENTIMENT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            result = completion.choices[0].message.content or ""
            result = result.strip().lower()
            if "positive" in result:
                return "positive"
            elif "negative" in result:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error classifying sentiment for text '{text[:50]}...': {e}")
                return "neutral"
            await asyncio.sleep(1)


async def classify_texts_batch(texts: list[str], client: AsyncOpenAI) -> dict[str, str]:
    """批量分类文本情感"""
    tasks = [classify_sentiment_async(text, client) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    sentiment_map = {}
    for text, result in zip(texts, results):
        if isinstance(result, Exception):
            sentiment_map[text] = "neutral"
        else:
            sentiment_map[text] = result
    
    return sentiment_map


def load_sentiment_cache() -> dict:
    """加载情感分类缓存"""
    if os.path.exists(SENTIMENT_CACHE_PATH):
        with open(SENTIMENT_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_sentiment_cache(cache: dict):
    """保存情感分类缓存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SENTIMENT_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


async def select_utterances_for_subjects(df: pd.DataFrame, labels: dict) -> dict:
    """为一个 subject 选择 positive 和 negative 文本，返回所有严重度的数据"""
    print("=== 选择 utterance 用于可视化 ===\n")
    
    subject_id = SELECTED_SUBJECT_ID
    print(f"Processing subject {subject_id}...")
    
    # 加载缓存
    sentiment_cache = load_sentiment_cache()
    
    # 初始化 OpenAI 客户端
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI library not installed")
    
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=OPENAI_TIMEOUT
    )
    
    # 筛选该 subject 的数据
    subj_df = df[df['actual_subject_id'] == subject_id].copy()
    if len(subj_df) == 0:
        raise ValueError(f"No data found for subject {subject_id}")
    
    # 获取唯一的文本
    unique_texts = subj_df['text'].unique().tolist()
    print(f"  Found {len(unique_texts)} unique texts")
    
    # 分类情感（使用缓存）
    texts_to_classify = [t for t in unique_texts if t not in sentiment_cache]
    if texts_to_classify:
        print(f"  Classifying {len(texts_to_classify)} new texts...")
        new_sentiments = await classify_texts_batch(texts_to_classify, client)
        sentiment_cache.update(new_sentiments)
        save_sentiment_cache(sentiment_cache)
    
    # 获取所有文本的情感
    text_sentiments = {t: sentiment_cache.get(t, "neutral") for t in unique_texts}
    
    # 选择 positive 和 negative 句子（至少5个词）
    MIN_WORDS = 5
    positive_texts = [
        t for t, s in text_sentiments.items() 
        if s == "positive" and len(t.split()) >= MIN_WORDS
    ]
    negative_texts = [
        t for t, s in text_sentiments.items() 
        if s == "negative" and len(t.split()) >= MIN_WORDS
    ]
    
    print(f"  Positive texts (>= {MIN_WORDS} words): {len(positive_texts)}")
    print(f"  Negative texts (>= {MIN_WORDS} words): {len(negative_texts)}")
    
    if not positive_texts:
        raise ValueError(f"No positive texts with at least {MIN_WORDS} words found for subject {subject_id}")
    if not negative_texts:
        raise ValueError(f"No negative texts with at least {MIN_WORDS} words found for subject {subject_id}")
    
    # 选择第一个 positive 和第一个 negative 文本
    selected_positive_text = positive_texts[0]
    selected_negative_text = negative_texts[0]
    
    print(f"  Selected positive text: '{selected_positive_text[:50]}...'")
    print(f"  Selected negative text: '{selected_negative_text[:50]}...'")
    
    # 构建返回数据结构：包含两个文本的所有严重度数据
    selected = {}
    phq8_score = labels.get(subject_id, {}).get('phq8_score', 'N/A')
    
    for sentiment, selected_text in [("positive", selected_positive_text), ("negative", selected_negative_text)]:
        # 找到该文本对应的所有 utterance
        text_df = subj_df[subj_df['text'] == selected_text]
        if len(text_df) == 0:
            print(f"  Warning: No utterance found for {sentiment} text '{selected_text[:50]}...'")
            continue
        
        # 为每个严重度收集数据
        selected[sentiment] = {
            'text': selected_text,
            'subject_id': subject_id,
            'phq8_score': phq8_score,
            'severities': {}
        }
        
        for severity in ALL_SEVERITIES:
            sev_df = text_df[text_df['severity'] == severity]
            if len(sev_df) > 0:
                row = sev_df.iloc[0]
                selected[sentiment]['severities'][severity] = {
                    'utt_id': row['utt_id'],
                    'audio_path': row['audio_path'],
                }
                print(f"  Found {sentiment} / {severity}: utt_id={row['utt_id']}")
            else:
                print(f"  Warning: No {sentiment} / {severity} data found")
    
    print()
    return selected


# ------------------------------ 可视化函数 ------------------------------

def detect_silence_regions(y: np.ndarray, sr: int, top_db: int = 20) -> list[tuple[float, float]]:
    """检测静音区域
    
    返回静音区域的时间段列表 [(start_time, end_time), ...]
    """
    try:
        nonsil = librosa.effects.split(y, top_db=top_db)
    except:
        # 如果检测失败，返回空列表（没有静音区域）
        return []
    
    if len(nonsil) == 0:
        # 如果整个音频都是静音，返回整个时间段
        duration = librosa.get_duration(y=y, sr=sr)
        return [(0.0, duration)]
    
    duration = librosa.get_duration(y=y, sr=sr)
    silence_regions = []
    last_end = 0.0
    
    for start, end in nonsil:
        start_time = start / sr
        end_time = end / sr
        
        # 如果非静音区域之前有间隙，那就是静音区域
        if start_time > last_end:
            silence_regions.append((last_end, start_time))
        
        last_end = end_time
    
    # 检查最后是否有静音区域
    if last_end < duration:
        silence_regions.append((last_end, duration))
    
    return silence_regions


def plot_temporal_analysis(wav_path: str, title_info: dict, output_path: str):
    """绘制三行可视化图"""
    # 加载音频
    y, sr = librosa.load(wav_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    t_axis = np.linspace(0, duration, len(y))
    
    # 检测静音区域
    silence_regions = detect_silence_regions(y, sr, top_db=20)
    
    # 创建三行子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # ========== 第一行：波形 + 静音掩膜 ==========
    ax1 = axes[0]
    ax1.plot(t_axis, y, linewidth=0.5, color='black', alpha=0.7)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Waveform with Silence Regions', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 标记静音区域
    for start, end in silence_regions:
        ax1.axvspan(start, end, alpha=0.3, color='lightgray', label='Silence' if start == silence_regions[0][0] else '')
    
    # ========== 第二行：Mel 频谱 ==========
    ax2 = axes[1]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(
        mel_db, x_axis='time', y_axis='mel', sr=sr,
        fmax=8000, ax=ax2, cmap='viridis'
    )
    ax2.set_ylabel('Mel Frequency (Hz)', fontsize=11)
    ax2.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    # ========== 第三行：HNR 时间曲线 ==========
    ax3 = axes[2]
    
    # 使用 parselmouth 计算 HNR
    hnr_times = []
    hnr_values = []
    
    try:
        sound = parselmouth.Sound(wav_path)
        harmonicity = sound.to_harmonicity()
        
        # 按时间采样 HNR
        time_step = 0.01  # 10ms
        t = 0.0
        while t < duration:
            try:
                hnr_val = call(harmonicity, "Get value at time", t, "Linear")
                # harmonicity 返回的是线性值（0-1之间），需要转换为dB
                # 如果值无效（静音区域），parselmouth 可能返回 undefined 或 0
                if hnr_val is not None and not np.isnan(hnr_val) and hnr_val > 0:
                    # 转换为dB：HNR(dB) = 10 * log10(linear_value)
                    hnr_db = 10 * np.log10(hnr_val)
                    if not np.isnan(hnr_db) and not np.isinf(hnr_db):
                        hnr_times.append(t)
                        hnr_values.append(hnr_db)
            except:
                # 静音或无效区域，跳过
                pass
            t += time_step
        
        if len(hnr_times) > 0:
            # 将数据转换为numpy数组，便于处理
            hnr_times_arr = np.array(hnr_times)
            hnr_values_arr = np.array(hnr_values)
            
            # 找出连续的有效区域（避免在静音区域之间画直线）
            # 如果两个点之间的时间间隔大于0.05秒（5个采样点），则认为是断开的
            max_gap = 0.05
            segments = []
            current_segment_times = [hnr_times_arr[0]]
            current_segment_values = [hnr_values_arr[0]]
            
            for i in range(1, len(hnr_times_arr)):
                if hnr_times_arr[i] - hnr_times_arr[i-1] <= max_gap:
                    # 连续的点
                    current_segment_times.append(hnr_times_arr[i])
                    current_segment_values.append(hnr_values_arr[i])
                else:
                    # 发现间隔，保存当前段，开始新段
                    if len(current_segment_times) > 0:
                        segments.append((current_segment_times, current_segment_values))
                    current_segment_times = [hnr_times_arr[i]]
                    current_segment_values = [hnr_values_arr[i]]
            
            # 添加最后一段
            if len(current_segment_times) > 0:
                segments.append((current_segment_times, current_segment_values))
            
            # 分别绘制每个连续段
            for seg_times, seg_values in segments:
                ax3.plot(seg_times, seg_values, linewidth=1.5, color='blue', alpha=0.7)
            
            # 只在第一个段添加标签
            if len(segments) > 0:
                ax3.plot([], [], linewidth=1.5, color='blue', label='HNR')
            
            ax3.axhline(y=HNR_THRESHOLD_DB, color='red', linestyle='--', linewidth=1.5, 
                       label=f'Threshold ({HNR_THRESHOLD_DB} dB)')
            ax3.legend(fontsize=9)
            
            # 设置y轴范围，避免显示异常值
            valid_values = [v for v in hnr_values if not np.isnan(v) and not np.isinf(v)]
            if len(valid_values) > 0:
                y_min = max(min(valid_values) - 5, -20)
                y_max = min(max(valid_values) + 5, 30)
                ax3.set_ylim(y_min, y_max)
        else:
            # 如果没有有效的 HNR 值，显示提示
            ax3.text(0.5, 0.5, 'No valid HNR data', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    except Exception as e:
        print(f"  Warning: Failed to compute HNR: {e}")
        ax3.text(0.5, 0.5, f'HNR computation failed: {str(e)[:50]}', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=10)
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('HNR (dB)', fontsize=11)
    ax3.set_title('Harmonic-to-Noise Ratio over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # 设置整体标题
    subject_id = title_info['subject_id']
    phq8_score = title_info['phq8_score']
    sentiment = title_info['sentiment']
    severity = title_info['severity']
    text = title_info['text']
    
    fig.suptitle(
        f"Subject {subject_id} (PHQ8={phq8_score}) | {sentiment.upper()} | {severity.upper()}\n"
        f"Text: \"{text}\"",
        fontsize=13, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


# ------------------------------ 主函数 ------------------------------

async def main_async():
    """主函数（异步版本）"""
    print("=== Temporal Case Study 可视化生成 ===\n")
    
    # 加载数据
    print("Loading data...")
    labels = load_labels()
    df = load_eval_log()
    print(f"Loaded {len(df)} rows from eval log\n")
    
    # 选择 utterance
    selected = await select_utterances_for_subjects(df, labels)
    
    # 生成图像
    print("\n=== 生成可视化图 ===\n")
    
    for sentiment, info in selected.items():
        subject_id = info['subject_id']
        phq8_score = info['phq8_score']
        text = info['text']
        severities = info['severities']
        
        # 遍历每个严重度
        for severity in ALL_SEVERITIES:
            if severity not in severities:
                print(f"Warning: No data for {sentiment} / {severity}, skipping...")
                continue
            
            sev_info = severities[severity]
            wav_path = sev_info['audio_path']
            utt_id = sev_info['utt_id']
            
            if not os.path.exists(wav_path):
                print(f"Warning: Audio file not found: {wav_path}")
                continue
            
            # 构建输出路径
            output_filename = f"temporal_{sentiment}_subj{subject_id}_utt{utt_id}_{severity}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # 构建标题信息
            title_info = {
                'subject_id': subject_id,
                'phq8_score': phq8_score,
                'sentiment': sentiment,
                'severity': severity,
                'text': text,
            }
            
            # 绘制图像
            print(f"Generating: {sentiment} / {severity}...")
            plot_temporal_analysis(wav_path, title_info, output_path)
    
    print("\n=== 完成 ===")


def main():
    """同步包装函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

