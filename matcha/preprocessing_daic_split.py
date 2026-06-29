import pandas as pd
import os
from pathlib import Path
import torchaudio
import random
import re

# ====== 配置 ======
metadata_path = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata_with_phq.csv"
transcript_dir = "/home/i-liyuxin/test/data/Text_all"
audio_dir = "/home/i-liyuxin/test/data/audio/wav_files"
output_wav_dir = "/home/i-liyuxin/Depression_TTS/matcha/data/processed_audio"
output_filelist = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist.txt"
sample_rate = 16000

# val_subjects 方式一：直接读取 val_subjects.csv
val_subjects_path = "/home/i-liyuxin/Depression_TTS/matcha/data/DAIC_val_fold2.csv"
if os.path.exists(val_subjects_path):
    val_subjects = pd.read_csv(val_subjects_path)["Participant_ID"].astype(str).tolist()

os.makedirs(output_wav_dir, exist_ok=True)

# ====== 1. 读取所有参与者的 id ======
meta = pd.read_csv(metadata_path)
all_ids = meta["Participant_ID"].astype(str).tolist()

train_lines = []
val_lines = []

# 定义文本清理函数
def clean_text(text):
    """
    清理文本，删除 <> 标记和其他不需要的内容
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 删除 <> 标记，例如 <ke>, <um>, <breath> 等
    text = re.sub(r'<[^>]*>', '', text)
    
    # 删除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

for pid in all_ids:
    transcript_path = os.path.join(transcript_dir, f"{pid}_TRANSCRIPT.csv")
    audio_path = os.path.join(audio_dir, f"{pid}_AUDIO.wav")
    if not os.path.exists(transcript_path) or not os.path.exists(audio_path):
        continue

    # 2. 读取 transcript，只保留 Participant 并清洗文本
    df = pd.read_csv(transcript_path, header=None, names=["start", "end", "speaker", "text"])
    audio, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = audio.squeeze(0)  # 单通道

    for idx, row in df.iterrows():
        if row["speaker"] != "Participant":
            continue
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        
        # 清理文本，删除 <> 标记
        text = clean_text(text)
        
        # 检查清理后的文本是否仍然有效
        if not text or len(text) < 3 or len(text.split()) < 2:
            continue
        
        # 跳过方括号内容（如 [laughter], [sigh] 等）
        if text.startswith("[") and text.endswith("]"):
            continue
        
        start_sec, end_sec = float(row["start"]), float(row["end"])
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        clip = audio[start_sample:end_sample]
        
        # 少于1秒不要
        if clip.numel() < sample_rate * 1.0:
            continue
        
        wav_out = os.path.join(output_wav_dir, f"{pid}_{idx}.wav")
        torchaudio.save(wav_out, clip.unsqueeze(0), sample_rate)
        line = f"{wav_out}|{text}"
        
        # ======= 按 subject id 分组到 val/train =======
        if pid in val_subjects:
            val_lines.append(line)
        else:
            train_lines.append(line)

# 可选：随机打乱一下，保持与之前格式一致（如果你不想打乱，可以去掉）
# random.seed(42)
# random.shuffle(train_lines)
# random.shuffle(val_lines)

# 写入文件
train_file = output_filelist.replace(".txt", "_train.txt")
val_file = output_filelist.replace(".txt", "_val.txt")

with open(train_file, "w") as f:
    for l in train_lines:
        f.write(l + "\n")
with open(val_file, "w") as f:
    for l in val_lines:
        f.write(l + "\n")

print(f"总样本: {len(train_lines) + len(val_lines)}, 训练: {len(train_lines)}, 验证: {len(val_lines)}")
print(f"Train filelist: {train_file}")
print(f"Val filelist: {val_file}")

# 可选：输出 subject id 到各自 txt，便于检查
with open(output_filelist.replace(".txt", "_train_subjects.txt"), "w") as f:
    for pid in set([l.split("/")[-1].split("_")[0] for l in train_lines]):
        f.write(pid + "\n")
with open(output_filelist.replace(".txt", "_val_subjects.txt"), "w") as f:
    for pid in set([l.split("/")[-1].split("_")[0] for l in val_lines]):
        f.write(pid + "\n")