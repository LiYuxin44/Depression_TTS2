#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Eval 合成脚本：对 DAIC test split 里的每条句子（每人最多 20 句），
生成 5 种抑郁严重度的语音：
  normal, mild, moderate, moderately_severe, severe

- 文本：来自 test filelist（全部句子，按 speaker 截断为每人 ≤20 条）
- 说话人：从 wav 文件名解析 subject_id，并用训练时的 subject→spk_idx 映射
- 抑郁条件：使用 Clinical-5 原型库，在 [-1,1] 上取 5 个固定点
"""

import datetime as dt
import json
import re
from pathlib import Path
import random
import os
import argparse
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn

from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

# ───────────────────────────  基本配置  ────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# 模型 / 声码器 / 数据路径
#MATCHA_CKPT = "/data/depression_tts/logs/train_daic_utter/decouple_unfreeze_all/runs/2025-11-19_16-45-53/checkpoints/checkpoint_epoch=399.ckpt"
MATCHA_CKPT = "/data/depression_tts/logs/train_daic_utter_train_full_film_only/decouple_unfreeze_all/runs/2025-11-27_21-37-47/checkpoints/checkpoint_epoch=099.ckpt"
HIFIGAN_WEIGHT = "/home/i-liyuxin/Depression_TTS/ckpts/VCTK_V1/generator_v1.pth"
HIFIGAN_CONFIG = "/home/i-liyuxin/Depression_TTS/ckpts/VCTK_V1/config.json"

# test filelist：每行形如 path/to/wav.wav|transcript
TEST_FILELIST = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_test_22k.txt"
# test subject 列表（每行一个 subject id），用于 sanity check / 过滤（可选）
TEST_SUBJECT_LIST = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_test_subjects.txt"

# 每个 speaker 最多使用多少条句子
MAX_UTTS_PER_SPEAKER = 20

# 输出目录：所有 severity 混在一个文件夹，文件名带 severity 后缀
OUTPUT_FOLDER_BASE = "/data/depression_tts/quality_eval_v9_099"

N_TIMESTEPS = 30
LENGTH_SCALE = 1.0
TEMPERATURE = 0.0  # evaluation 建议 0

USE_DEPRESSION_COND = True
USE_SPEAKER_COND = True  # 若想禁用说话人条件设为 False

AUDIO_GAIN = 1.0
NORMALIZE_AUDIO = False
TARGET_RMS = 0.0

# ────────── 抑郁嵌入 & 元数据路径（与你原来脚本一致） ──────────
DEPRESSION_EMBEDDINGS_FILE = "/home/i-liyuxin/Contrastive_OS/GDST_embeddings_utterance-3-trf-ordinal-asr/train_embeddings.npz"
METADATA_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata.csv"
PHQ_SCORE_METADATA_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata_with_phq.csv"
TRAIN_SUBJECT_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train_subjects.txt"
VAL_SUBJECT_FILE   = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_val_subjects.txt"

VOCODER_INPUT_DIMS = None
GLOBAL_SUBJECT_TO_SPK_IDX = None
GLOBAL_ORDERED_SUBJECTS = None

SEVERITY_CLASSES = ["normal", "mild", "moderate", "moderately_severe", "severe"]

# ────────────────────── utils: metadata / PHQ  ────────────────────────

def load_metadata():
    """加载 metadata.csv，用于 subject -> PHQ8_Binary（主要用于 dep_bank 构建）"""
    import pandas as pd
    subject_to_phq8 = {}
    try:
        df = pd.read_csv(METADATA_FILE)
        for _, row in df.iterrows():
            sid = int(row["Participant_ID"])
            subject_to_phq8[sid] = int(row["PHQ8_Binary"])
        print(f"✓ metadata.csv loaded, #subjects={len(subject_to_phq8)}")
    except Exception as e:
        print(f"[WARN] load_metadata failed: {e}")
    return subject_to_phq8


def phq_score_to_severity(score):
    """PHQ[0,24] -> 五级严重度标签"""
    try:
        if score is None:
            return None
        s = int(score)
    except Exception:
        return None

    if 0 <= s <= 4:
        return "normal"
    if 5 <= s <= 9:
        return "mild"
    if 10 <= s <= 14:
        return "moderate"
    if 15 <= s <= 19:
        return "moderately_severe"
    if 20 <= s <= 24:
        return "severe"
    return None


def load_phq_scores():
    """加载 metadata_with_phq，用于构建 Clinical-5 原型库"""
    import pandas as pd
    subject_to_score = {}
    subject_to_severity = {}
    try:
        df = pd.read_csv(PHQ_SCORE_METADATA_FILE)
        if "Participant_ID" not in df.columns:
            raise RuntimeError("metadata_with_phq.csv 缺少 Participant_ID 列")
        score_col = None
        for cand in ["PHQ_Score", "PHQ_Score_Total", "PHQ_Score_0_24",
                     "PHQ_Total", "PHQ9_Total", "PHQ9_Score"]:
            if cand in df.columns:
                score_col = cand
                break
        if score_col is None:
            raise RuntimeError("metadata_with_phq.csv 未找到 PHQ 分数列")

        for _, row in df.iterrows():
            try:
                sid = int(row["Participant_ID"])
                score = row[score_col]
                if isinstance(score, float) and np.isnan(score):
                    continue
                score = float(score)
                subject_to_score[sid] = score
                sev = phq_score_to_severity(score)
                if sev is not None:
                    subject_to_severity[sid] = sev
            except Exception:
                continue

        print(f"✓ PHQ scores loaded, #subjects={len(subject_to_score)}")
    except Exception as e:
        print(f"[WARN] load_phq_scores failed: {e}")
    return subject_to_score, subject_to_severity

# ────────────────────── utils: embedding / slerp  ─────────────────────

def _l2norm(t: torch.Tensor) -> torch.Tensor:
    return t / (t.norm(p=2) + 1e-6)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    a = _l2norm(a - a.mean())
    b = _l2norm(b - b.mean())
    dot = torch.clamp(torch.sum(a * b, dim=-1, keepdim=True), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    out = (torch.sin((1.0 - t) * omega) / (so + 1e-6)) * a + (torch.sin(t * omega) / (so + 1e-6)) * b
    mask = (so.abs() < 1e-6).view(-1)
    if mask.any():
        out[mask] = (1.0 - t) * a[mask] + t * b[mask]
    return _l2norm(out)


def build_dep_bank_clinical5(dep_embeddings, dep_subject_ids, subj_score, subj_bin):
    """
    基于 subject 级抑郁嵌入构建五级严重度原型库：
    normal / mild / moderate / moderately_severe / severe
    """
    Z = torch.tensor(dep_embeddings, dtype=torch.float32, device=device)
    ids = dep_subject_ids.astype(int).tolist() if dep_subject_ids is not None else list(range(Z.shape[0]))

    def bucket_indices(lo, hi):
        idx = []
        if subj_score is not None:
            for k, sid in enumerate(ids):
                sc = subj_score.get(int(sid), None)
                if sc is not None and lo <= sc <= hi:
                    idx.append(k)
        return idx

    bins = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)]
    names = ["normal", "mild", "moderate", "moderately_severe", "severe"]
    protos = []

    for lo, hi in bins:
        idx = bucket_indices(lo, hi)
        if len(idx) >= 3:
            v = _l2norm(Z[idx].mean(0))
        else:
            v = None
        protos.append(v)

    # 若两端缺 proto，用二分类的 mean 兜底
    if (protos[0] is None) or (protos[-1] is None):
        cls = torch.tensor([int(subj_bin.get(int(i), 1)) for i in ids],
                           dtype=torch.long, device=device)
        z_min = _l2norm(Z[cls == 0].mean(0)) if (cls == 0).any() else _l2norm(Z.mean(0))
        z_max = _l2norm(Z[cls == 1].mean(0)) if (cls == 1).any() else _l2norm(Z.mean(0))
        if protos[0] is None:
            protos[0] = z_min
        if protos[-1] is None:
            protos[-1] = z_max

    # 中间缺失，用左右做 slerp
    for i in range(5):
        if protos[i] is None:
            L = i - 1
            while L >= 0 and protos[L] is None:
                L -= 1
            R = i + 1
            while R < 5 and protos[R] is None:
                R += 1
            if L >= 0 and R < 5:
                t = (i - L) / (R - L + 1e-9)
                protos[i] = slerp(protos[L], protos[R], t)
            elif L >= 0:
                protos[i] = protos[L]
            elif R < 5:
                protos[i] = protos[R]

    return {"names": names, "protos": protos, "mid_index": 2}


def dep_from_alpha_over_bank(bank, alpha: float) -> torch.Tensor:
    """给定 α∈[-1,1]，在原型库上插值出一个抑郁条件向量。"""
    protos = bank["protos"]
    K = len(protos)
    k_mid = bank.get("mid_index", 2)
    span = max(k_mid, K - 1 - k_mid)
    a = float(np.clip(alpha, -1.0, 1.0))
    pos = k_mid + span * a
    pos = float(np.clip(pos, 0.0, K - 1.0))
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, K - 1)
    t = pos - i0
    if i0 == i1:
        result = protos[i0]
    else:
        result = slerp(protos[i0], protos[i1], t)
    if isinstance(result, torch.Tensor):
        result = result.squeeze()
        if result.dim() == 0:
            result = result.unsqueeze(0)
    else:
        result = torch.tensor(result, dtype=torch.float32)
        result = result.squeeze()
        if result.dim() == 0:
            result = result.unsqueeze(0)
    return result


def severity_to_alpha(sev: str) -> float:
    """5 级严重度 → 固定 α（eval 用固定五点）"""
    if sev == "normal":
        return -1.0
    if sev == "mild":
        return -0.5
    if sev == "moderate":
        return 0.0
    if sev == "moderately_severe":
        return 0.5
    if sev == "severe":
        return 1.0
    return 0.0

# ────────────────────── 加载嵌入 & subject→spk  ──────────────────────

def load_embeddings():
    """加载抑郁嵌入，并构建 subject→spk_idx 映射（和训练保持一致）"""
    print("Loading depression embeddings...")
    depression_data = np.load(DEPRESSION_EMBEDDINGS_FILE)
    depression_embeddings = (depression_data["embeddings"]
                             if "embeddings" in depression_data
                             else depression_data["arr_0"])
    depression_subject_ids = depression_data["subject_ids"] if "subject_ids" in depression_data else None

    # 若只有 utterance_ids，就聚合成 subject 级
    if depression_subject_ids is None and "utterance_ids" in depression_data:
        utt_ids = depression_data["utterance_ids"]
        subj_from_utt = np.asarray([int(str(u).split("_", 1)[0]) for u in utt_ids],
                                   dtype=int)
        agg_sum, agg_cnt = {}, {}
        for i, sid in enumerate(subj_from_utt):
            v = depression_embeddings[i]
            if sid in agg_sum:
                agg_sum[sid] += v
                agg_cnt[sid] += 1
            else:
                agg_sum[sid] = v.copy()
                agg_cnt[sid] = 1
        unique_vals, first_idx = np.unique(subj_from_utt, return_index=True)
        order = np.argsort(first_idx)
        unique_subj = unique_vals[order].astype(int)
        aggregated = np.stack([agg_sum[sid] / agg_cnt[sid] for sid in unique_subj],
                              axis=0).astype(depression_embeddings.dtype, copy=False)
        depression_embeddings = aggregated
        depression_subject_ids = unique_subj
        print(f"  Aggregated to subject-level: {aggregated.shape[0]} subjects")

    print(f"✓ Depression embeddings shape: {depression_embeddings.shape}")

    # 构建 subject 列表：train_subjects + val_subjects
    def _load_subject_ids_from_file(path):
        ids, seen = [], set()
        try:
            if path and os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        s = str(line).strip()
                        if s and s.isdigit():
                            sid = int(s)
                            if sid not in seen:
                                seen.add(sid)
                                ids.append(sid)
        except Exception:
            pass
        return ids

    ordered_subjects = []
    ordered_subjects += _load_subject_ids_from_file(TRAIN_SUBJECT_FILE)
    seen = set(ordered_subjects)
    for sid in _load_subject_ids_from_file(VAL_SUBJECT_FILE):
        if sid not in seen:
            seen.add(sid)
            ordered_subjects.append(sid)

    if not ordered_subjects and depression_subject_ids is not None:
        ordered_subjects = [int(x) for x in depression_subject_ids.astype(int).tolist()]

    depression_subject_ids = depression_subject_ids.astype(int)
    depression_subject_to_idx = {int(sid): idx
                                 for idx, sid in enumerate(depression_subject_ids)}
    print(f"✓ Depression subject mapping: {len(depression_subject_to_idx)} subjects")

    # 训练时的 subject→spk 映射
    subject_to_spk_idx = ({sid: idx for idx, sid in enumerate(ordered_subjects)}
                          if ordered_subjects else depression_subject_to_idx)
    print(f"✓ Subject→spk_idx mapping size: {len(subject_to_spk_idx)}")

    return (depression_embeddings, depression_subject_ids, depression_subject_to_idx,
            subject_to_spk_idx, ordered_subjects)
def resolve_spk_info(speaker_id, n_spks_model: int | None):
    """
    根据 global subject→spk 映射推断真正使用的 spk_idx 以及对应的训练 subject id。
    """
    mapping = GLOBAL_SUBJECT_TO_SPK_IDX or {}
    ordered = GLOBAL_ORDERED_SUBJECTS or []
    actual_subject_id = None
    spk_idx = None

    if speaker_id is None:
        return None, None

    try:
        sid = int(speaker_id)
    except Exception:
        return None, None

    if sid in mapping:
        spk_idx = mapping[sid]
        actual_subject_id = sid
        return spk_idx, actual_subject_id

    # fallback: align to ordered subjects（与训练时一致）
    if ordered:
        fallback_idx = sid % len(ordered)
        spk_idx = fallback_idx
        actual_subject_id = ordered[fallback_idx]
        return spk_idx, actual_subject_id

    # final fallback：使用 model n_spks 做 modulo
    if n_spks_model is not None and int(n_spks_model) > 0:
        spk_idx = sid % int(n_spks_model)
    else:
        spk_idx = 0
    actual_subject_id = None
    return spk_idx, actual_subject_id

# ────────────────────── 模型 / 声码器  ────────────────────────────────

def load_matcha_model(ckpt_path: str) -> MatchaTTS:
    print(f"Loading Matcha-TTS from: {ckpt_path}")
    model = MatchaTTS.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    print(f"✓ Matcha-TTS loaded (n_feats={model.n_feats}, n_spks={model.n_spks})")
    return model


def load_hifigan(weight_path: str, cfg_path: str) -> HiFiGAN:
    global VOCODER_INPUT_DIMS
    print(f"Loading HiFi-GAN from: {weight_path}")
    with open(cfg_path, "r") as f:
        h_json = json.load(f)
        h = AttrDict(h_json)

    g = HiFiGAN(h).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if "generator" in ckpt and isinstance(ckpt["generator"], dict):
        state_dict = ckpt["generator"]
    else:
        state_dict = ckpt
    g.load_state_dict(state_dict, strict=True)
    g.eval()
    g.remove_weight_norm()

    for k in ["num_mels", "n_mels", "mel_channels"]:
        if k in h_json:
            VOCODER_INPUT_DIMS = int(h_json[k])
            break
    print(f"✓ HiFi-GAN loaded, expected mel dims: {VOCODER_INPUT_DIMS}")
    return g


def adapt_mel_dimensions(mel, target_dims):
    if mel.dtype != torch.float32:
        mel = mel.to(torch.float32)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # [1, n_mels, T]
    current_dims = mel.shape[1]
    if target_dims is None or current_dims == target_dims:
        return mel
    if current_dims > target_dims:
        mel = mel[:, :target_dims, :]
    else:
        # pad by repeating last bands
        pad = target_dims - current_dims
        last = mel[:, -1:, :].expand(-1, pad, -1)
        mel = torch.cat([mel, last], dim=1)
    return mel


@torch.inference_mode()
def mel_to_waveform(mel, vocoder: HiFiGAN, denoiser: Denoiser):
    if VOCODER_INPUT_DIMS is not None:
        mel = adapt_mel_dimensions(mel, VOCODER_INPUT_DIMS)
    else:
        if mel.dtype != torch.float32:
            mel = mel.to(torch.float32)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

    if mel.dim() != 3:
        raise RuntimeError(f"Mel shape must be [B, n_mels, T], got {tuple(mel.shape)}")

    audio = vocoder(mel).clamp(-1, 1)  # [B,1,T]
    audio_2d = audio.squeeze(1)        # [B,T]
    denoised = denoiser(audio_2d, strength=0.00025)
    if isinstance(denoised, tuple):
        denoised = denoised[0]
    if denoised.dim() == 3 and denoised.shape[1] == 1:
        denoised = denoised.squeeze(1)
    elif denoised.dim() == 1:
        denoised = denoised.unsqueeze(0)
    audio = denoised.squeeze(0).cpu()

    audio = audio * AUDIO_GAIN
    if NORMALIZE_AUDIO and TARGET_RMS > 0:
        cur_rms = torch.sqrt(torch.mean(audio**2))
        if cur_rms > 0:
            audio = audio * (TARGET_RMS / cur_rms)
    audio = torch.clamp(audio, -1.0, 1.0)
    return audio

# ────────────────────── 文本 & 合成包装  ───────────────────────────────

@torch.inference_mode()
def process_text(text: str):
    seq, cleaned_text = text_to_sequence(text, ["english_cleaners2"])
    seq = intersperse(seq, 0)  # 与训练时 add_blank=True 对齐
    x = torch.IntTensor(seq).to(device)[None]
    l = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {
        "x_orig": text,
        "x": x,
        "x_lengths": l,
        "x_phones": sequence_to_text(x.squeeze(0).tolist()),
        "cleaned_text": cleaned_text,
    }


@torch.inference_mode()
def synthesise_with_conditions(
    model: MatchaTTS,
    text: str,
    depression_embedding,
    speaker_id: int | None,
):
    tex = process_text(text)

    # depression cond
    depression_cond = None
    if depression_embedding is not None:
        if isinstance(depression_embedding, np.ndarray):
            depression_embedding = depression_embedding.squeeze()
        depression_cond = torch.tensor(depression_embedding, dtype=torch.float32, device=device)
        depression_cond = _l2norm(depression_cond - depression_cond.mean())
        depression_cond = depression_cond.unsqueeze(0)  # [1, D]

    if not USE_DEPRESSION_COND:
        depression_cond = None

    # speaker cond: 使用训练时 subject→spk_idx
    spk_tensor = None
    actual_subject_id = None
    spk_idx = None
    if USE_SPEAKER_COND and hasattr(model, "n_spks") and int(model.n_spks) > 1 and speaker_id is not None:
        spk_idx, actual_subject_id = resolve_spk_info(speaker_id, int(model.n_spks))
        if spk_idx is not None:
            spk_tensor = torch.tensor([spk_idx], dtype=torch.long, device=device)

    out = model.synthesise(
        tex["x"], tex["x_lengths"],
        n_timesteps=N_TIMESTEPS,
        temperature=TEMPERATURE,
        length_scale=LENGTH_SCALE,
        depression_cond=depression_cond,
        speaker_cond=None,
        spks=spk_tensor if spk_tensor is not None else None,
    )
    out.update(tex)
    out["spk_idx"] = spk_idx
    out["actual_subject_id"] = actual_subject_id
    return out


def save_audio(utt_id: str, severity: str, waveform, base_folder: str | Path):
    base_folder = Path(base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)
    # 文件名：uttid_severity.wav 例如 303_145_moderate.wav
    audio_filename = f"{utt_id}_{severity}.wav"
    out_path = base_folder / audio_filename
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    sf.write(out_path, waveform, 22050, "PCM_16")
    return str(out_path)

# ────────────────────── 读取 test filelist  ───────────────────────────

def load_test_subject_list(path: str):
    """可选：读取 test subject 列表，返回 set[int]。用于 sanity check / 过滤。"""
    if not path or not os.path.exists(path):
        print(f"[INFO] Test subject list not found: {path} (skip filtering)")
        return None
    subjects = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.isdigit():
                subjects.add(int(s))
    print(f"✓ Loaded {len(subjects)} test subjects from {path}")
    return subjects


def load_test_sentences(filelist_path: str, allowed_subjects=None):
    """
    读取 test filelist，行格式：
      /path/to/303_145.wav|This is the transcript...
    返回 list[dict]: {utt_id, subject_id, wav_path, text}
    """
    items = []
    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            wav_path, text = line.split("|", 1)
            utt_id = Path(wav_path).stem            # e.g. "303_145"
            try:
                subject_id = int(utt_id.split("_")[0])
            except Exception:
                subject_id = None

            if allowed_subjects is not None and subject_id is not None:
                if subject_id not in allowed_subjects:
                    continue

            items.append({
                "utt_id": utt_id,
                "subject_id": subject_id,
                "wav_path": wav_path,
                "text": text.strip(),
            })
    print(f"✓ Loaded {len(items)} test utterances from {filelist_path}")
    return items


def subsample_per_speaker(items, max_utts: int):
    """
    按 speaker 分组，每个 speaker 最多保留 max_utts 条 utterance。
    采样受全局随机种子控制（可复现）。
    """
    if max_utts is None or max_utts <= 0:
        return items

    by_spk = defaultdict(list)
    for it in items:
        sid = it.get("subject_id", None)
        if sid is None:
            continue
        by_spk[sid].append(it)

    subsampled = []
    for sid, utts in by_spk.items():
        if len(utts) > max_utts:
            chosen = random.sample(utts, max_utts)
        else:
            chosen = utts
        subsampled.extend(chosen)

    # 按 speaker / utt_id 排一下，方便 debug
    subsampled.sort(key=lambda x: (x["subject_id"], x["utt_id"]))

    num_spk = len(by_spk)
    print(f"✓ Subsampled per speaker: speakers={num_spk}, "
          f"max_utts_per_spk={max_utts}, final_utts={len(subsampled)}")
    return subsampled

# ────────────────────── main: 合成 5 个严重度  ────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filelist", type=str, default=TEST_FILELIST,
                        help="DAIC test filelist (path|text)")
    parser.add_argument("--test_subject_list", type=str, default=TEST_SUBJECT_LIST,
                        help="DAIC test subject id list (optional)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER_BASE,
                        help="Output base folder")
    parser.add_argument("--max_utts_per_spk", type=int, default=MAX_UTTS_PER_SPEAKER,
                        help="Max #utterances per speaker for eval")
    return parser.parse_args()


def main():
    print("=== Eval: synthesize DAIC test with 5 depression severities ===")

    args = parse_args()
    out_base = Path(args.output_dir)

    # metadata / PHQ（用于构建 dep_bank）
    subject_to_phq8 = load_metadata()
    subject_to_score, _subject_to_severity_phq = load_phq_scores()

    # embeddings & subject→spk 映射
    (depression_embeddings,
     depression_subject_ids,
     depression_subject_to_idx,
     subject_to_spk_idx,
     ordered_subjects) = load_embeddings()

    global GLOBAL_SUBJECT_TO_SPK_IDX
    GLOBAL_SUBJECT_TO_SPK_IDX = subject_to_spk_idx
    global GLOBAL_ORDERED_SUBJECTS
    GLOBAL_ORDERED_SUBJECTS = ordered_subjects

    # 构建 Clinical-5 原型库（使用 PHQ 分数 + 二分类）
    dep_bank = build_dep_bank_clinical5(
        depression_embeddings,
        depression_subject_ids,
        subject_to_score,
        subject_to_phq8,
    )
    print(f"✓ Clinical-5 bank ready: {dep_bank['names']}")

    # 模型 & vocoder
    model = load_matcha_model(MATCHA_CKPT)
    vocoder = load_hifigan(HIFIGAN_WEIGHT, HIFIGAN_CONFIG)
    denoiser = Denoiser(vocoder, mode="zeros")

    if VOCODER_INPUT_DIMS is not None and hasattr(model, "n_feats"):
        n_feats_val = int(model.n_feats)
        if n_feats_val != VOCODER_INPUT_DIMS:
            print(f"[WARN] model.n_feats={n_feats_val} != vocoder_mels={VOCODER_INPUT_DIMS}")
        else:
            print(f"✓ model.n_feats and vocoder mels match: {n_feats_val}")

    # 读取 test subject 列表（可选）
    allowed_subjects = load_test_subject_list(args.test_subject_list)

    # 读取 test 句子
    test_items = load_test_sentences(args.test_filelist, allowed_subjects=allowed_subjects)

    # 每个 speaker 至多 20 句（或 args.max_utts_per_spk）
    test_items = subsample_per_speaker(test_items, max_utts=args.max_utts_per_spk)

    total = 0
    succ = 0

    # 日志文件
    out_base.mkdir(parents=True, exist_ok=True)
    log_file = out_base / "eval_generation_log.csv"
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("utt_id,subject_id,actual_subject_id,spk_idx,severity,alpha,audio_path,text\n")

    for item in tqdm(test_items, desc="Synthesizing"):
        utt_id = item["utt_id"]
        subject_id = item["subject_id"]
        text = item["text"]

        for sev in SEVERITY_CLASSES:
            total += 1
            try:
                alpha = severity_to_alpha(sev)
                dep_vec = dep_from_alpha_over_bank(dep_bank, alpha)

                out = synthesise_with_conditions(
                    model, text,
                    depression_embedding=dep_vec,
                    speaker_id=subject_id,
                )
                mel = out["mel"]  # [1, n_mels, T]
                wav = mel_to_waveform(mel, vocoder, denoiser)
                audio_path = save_audio(utt_id, sev, wav, out_base)

                with open(log_file, "a", encoding="utf-8") as f:
                    safe_text = text.replace('"', "'")
                    actual_subj = out.get("actual_subject_id", "")
                    spk_idx = out.get("spk_idx", "")
                    f.write(f"{utt_id},{subject_id},{actual_subj},{spk_idx},{sev},{alpha:.3f},{audio_path},\"{safe_text}\"\n")

                succ += 1
            except Exception as e:
                print(f"[ERR] {utt_id} sev={sev}: {e}")
                continue

    print("\n=== Done ===")
    print(f"Total trials:   {total}")
    print(f"Successful:     {succ}")
    print(f"Success rate:   {succ / total * 100:.1f}%")
    print(f"Output folder:  {out_base}")
    print(f"Log file:       {log_file}")


if __name__ == "__main__":
    main()
