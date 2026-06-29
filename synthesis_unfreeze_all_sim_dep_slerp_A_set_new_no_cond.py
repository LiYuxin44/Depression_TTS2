import datetime as dt
import json
import re
from pathlib import Path
import random
import glob
import os
import argparse
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
import torchaudio
import torch.nn as nn
import importlib

 

# 尝试导入datasets，如果失败则使用本地文件
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("Warning: datasets library not installed. Will only use local files.")
    HAS_DATASETS = False

from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

# ───────────────────────────  Paths / Const  ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定固定随机种子与确定性推理，减少不稳定性
SEED = 1234
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

# 新模型路径配置
#MATCHA_CKPT = "/scratch/users/ntu/yuxin.li/matcha-tts-new/logs/train_daic/daic_adapter_finetune/runs/2025-08-11_00-47-21/checkpoints/last.ckpt"
MATCHA_CKPT = "/home/i-liyuxin/Depression_TTS/logs/train_daic_utter/decouple_unfreeze_all/runs/2025-10-30_17-46-01/checkpoints/checkpoint_epoch=099.ckpt"
HIFIGAN_WEIGHT = "/home/i-liyuxin/Depression_TTS/ckpts/VCTK_V1/generator_v1.pth"
HIFIGAN_CONFIG = "/home/i-liyuxin/Depression_TTS/ckpts/VCTK_V1/config.json"
# 输出文件夹将根据subject ID动态生成
OUTPUT_FOLDER_BASE = "/home/i-liyuxin/data/v5_dep_slerp_A_1104"

N_TIMESTEPS = 30
LENGTH_SCALE = 1.0
TEMPERATURE = 0.2 # 降低随机性以减少失真（0.1~0.2 推荐）
TEMPERATURE_MIN = 0.3  # 多样本采样的下限温度（含直出温度）
TEMPERATURE_MAX = 0.8  # 多样本采样的上限温度（含直出温度）

# 🔥 生成配置 - 每个subject生成的句子数量
SENTENCES_PER_SUBJECT = 50  

# 单调验证：每个严重度原型固定生成的句子数
MONO_SENTENCES_PER_SEVERITY = 20

# 音频处理参数
AUDIO_GAIN = 1        # 音量增益倍数，增大音量（建议范围：2.0-5.0）
NORMALIZE_AUDIO = True   # 是否进行音频归一化
TARGET_RMS = 0.1        # 目标RMS值，用于归一化（建议范围：0.05-0.2）


MAX_GENERATION_ATTEMPTS = 1    # 每个句子只生成一次
QUALITY_LOG_ENABLED = False    # deprecated: 不再进行质量筛选

# 文本筛选参数（用于去掉太短的文本）
TEXT_MIN_CHARS = 20     # 最小字符数（清洗后计算）
TEXT_MIN_WORDS = 5      # 最少词数
TEXT_MAX_CHARS = 2000    # 最大字符数上限

# 采样与稳定性控制
RESET_SEED_EACH_SENTENCE = False  # 每句前重置随机种子，进一步消除随机性
USE_DEPRESSION_COND = True
USE_SPEAKER_COND = False

# 采样式生成配置
SAMPLING_ENABLED = False          # 是否启用采样式生成
NUM_SAMPLES = 20                 # 每个句子生成的候选数量（增加采样数量）
MAX_REF_AUDIO_PER_SUBJECT = 70    # 每个subject用于构建参考嵌入的最大音频数量

# 仅使用抑郁嵌入相似度进行候选选择（禁用说话人相似度）
SELECT_BY_DEPRESSION_ONLY = True

# 抑郁音频嵌入提取模型（与 Contrastive_OS/extract_embeddings-utterance.py 一致）
# 设置为实际的checkpoint路径；若为 None 则回退为占位实现
DEPRESSION_AUDIO_MODEL_CKPT = "/home/i-liyuxin/Contrastive_OS/runs_speaker_identification_20/full_20250928_231855/best_cls_full.pth"
DEPRESSION_EXTRACTOR_LAYER = 20     # WavLM 特征层
DEPRESSION_WAVLM_MODEL_NAME = "microsoft/wavlm-large"

# 采样策略优化
USE_ADAPTIVE_SAMPLING = True     # 启用自适应采样：如果最佳相似度低于阈值，增加采样
SIMILARITY_THRESHOLD_RETRY = 0.3  # 相似度重试阈值
MAX_RETRY_SAMPLES = 20            # 重试时的额外采样数量

# 嵌入处理
EMBED_NORM = "l2"                # 选项: "l2" | "zscore" | "none"
EMBED_EPS = 1e-6

DEPRESSION_MEAN = None
DEPRESSION_STD = None

# 加载subject和embeddings信息（改为 utterance 级 npz）
DEPRESSION_EMBEDDINGS_FILE = "/home/i-liyuxin/Contrastive_OS/GDST_embeddings_utterance-3-trf-ordinal/train_embeddings.npz"

# 添加metadata文件路径
METADATA_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata.csv"
# 训练期 subject 列表（用于构建与训练一致的 subject→spk 映射）
TRAIN_SUBJECT_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train_subjects.txt"
VAL_SUBJECT_FILE   = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_val_subjects.txt"
# 用于严重度与PHQ_Score（0-24）的元数据文件
PHQ_SCORE_METADATA_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata_with_phq.csv"

# 目标 subject 数量（可通过命令行 --num_subjects 覆盖）。
# None 表示使用全部可用 subject（与原有逻辑一致）。
TARGET_TOTAL_SUBJECTS = None

# HiFi-GAN 期望的 mel 维度（从其 config.json 读取），用于在声码器前进行维度/形状自适配
VOCODER_INPUT_DIMS = None
GLOBAL_SUBJECT_TO_SPK_IDX = None
REQUIRE_AUDIO_SPACE_BANK = False
# ───────────────────────────  CFG (Classifier-Free Guidance) 配置  ───────────────────────────
CFG_ENABLED = True            # 打开/关闭 CFG
CFG_UNCOND_MODE = "avg_proto"      # "none"→无条件分支用 depression_cond=None;
                              # "zero"→无条件分支用全零向量（维度与有条件一致）；
                              # "avg_proto"→无条件分支用五级原型的平均向量（需要 dep_bank 可用）

# 按严重度分配 guidance scale（可按听感再微调）
GUIDANCE_SCALES = {
    "normal": 1.0,
    "mild": 1.6,
    "moderate": 1.9,
    "moderately_severe": 2.4,
    "severe": 2.8,
}

def severity_to_w(sev: str) -> float:
    return float(GUIDANCE_SCALES.get(str(sev), 1.4))

# ───────────────────────────  Dense grid for alpha / w  ─────────────────────────
# 选项: "none" | "alpha" | "w"
DENSE_GRID_MODE = "none"
GRID_POINTS = 9  # 7 / 9 / 11 等
ALPHA_GRID = [float(x) for x in np.linspace(0.0, 1.0, GRID_POINTS)] if DENSE_GRID_MODE == "alpha" else []
W_GRID = []  # 若启用 w 网格, 例如: [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

@torch.inference_mode()
def _make_uncond_depression_cond(dep_cond: torch.Tensor | None,
                                 dep_bank: dict | None,
                                 mode: str = "none") -> torch.Tensor | None:
    """
    生成无条件分支的 depression_cond：
    - "none": 返回 None（即彻底无条件）
    - "zero": 返回与 dep_cond 维度一致的 0 向量
    - "avg_proto": 返回五级原型的平均向量（需 dep_bank 提供 protos）
    """
    if mode == "none":
        return None
    if dep_cond is None and mode in ("zero", "avg_proto"):
        # 若有条件分支本身就是 None，则无条件也返回 None
        return None

    if mode == "zero":
        z = torch.zeros_like(dep_cond, dtype=torch.float32, device=dep_cond.device)
        return z

    if mode == "avg_proto" and dep_bank is not None:
        try:
            protos = dep_bank.get("protos", None)
            vecs = [p for p in protos if isinstance(p, torch.Tensor)]
            if vecs:
                avg = torch.stack(vecs, dim=0).mean(0)
                avg = _l2norm(avg)
                if avg.dim() == 1:
                    avg = avg.unsqueeze(0)  # [1, D]
                return avg.to(dep_cond.device, torch.float32)
        except Exception:
            pass
    # 兜底
    return None


@torch.inference_mode()
def apply_cfg_on_mel(mel_cond: torch.Tensor, mel_uncond: torch.Tensor | None, w: float) -> torch.Tensor:
    """
    在 mel 级实现近似的 CFG 混合：
      mel_guided = mel_uncond + w * (mel_cond - mel_uncond)
    若 mel_uncond 为 None 或形状不匹配，退化为 mel_cond。
    """
    try:
        if mel_uncond is None:
            return mel_cond
        # 形状对齐（[B, n_mels, T]），必要时裁剪到相同 T
        if mel_cond.dim() == 2:
            mel_cond = mel_cond.unsqueeze(0)
        if mel_uncond.dim() == 2:
            mel_uncond = mel_uncond.unsqueeze(0)

        T = min(mel_cond.shape[-1], mel_uncond.shape[-1])
        mc = mel_cond[..., :T]
        mu = mel_uncond[..., :T]
        out = mu + float(w) * (mc - mu)
        return out
    except Exception:
        return mel_cond



def load_metadata():
    """加载metadata.csv文件，获取每个subject的PHQ8_Binary值"""
    print("Loading metadata...")
    
    import pandas as pd
    
    try:
        metadata_df = pd.read_csv(METADATA_FILE)
        print(f"✓ Metadata loaded successfully")
        print(f"  Total subjects in metadata: {len(metadata_df)}")
        
        # 创建subject ID到PHQ8_Binary的映射
        subject_to_phq8 = {}
        for _, row in metadata_df.iterrows():
            subject_id = int(row['Participant_ID'])
            phq8_binary = int(row['PHQ8_Binary'])
            subject_to_phq8[subject_id] = phq8_binary
        
        print(f"✓ Created subject to PHQ8_Binary mapping: {len(subject_to_phq8)} subjects")
        
        # 统计PHQ8_Binary分布
        phq8_counts = metadata_df['PHQ8_Binary'].value_counts()
        print(f"  PHQ8_Binary distribution:")
        print(f"    0 (non-depressed): {phq8_counts.get(0, 0)} subjects")
        print(f"    1 (depressed): {phq8_counts.get(1, 0)} subjects")
        
        return subject_to_phq8
        
    except Exception as e:
        print(f"✗ Failed to load metadata: {e}")
        print("Will use default label value of 1")
        return {}
        
def _l2norm(t: torch.Tensor) -> torch.Tensor:
    """L2归一化函数，与synthesize_gen_A_same_sentence.py保持一致"""
    return t / (t.norm(p=2) + 1e-6)

def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """在单位超球面上进行球面线性插值（SLERP）。
    - a, b: 向量（任意可广播形状，最后一维为特征维）
    - t:   插值权重 ∈ [0,1]
    返回与 a/b 维度一致的单位向量。
    """
    a = _l2norm(a)
    b = _l2norm(b)
    dot = torch.clamp(torch.sum(a * b, dim=-1, keepdim=True), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    out = (torch.sin((1.0 - t) * omega) / (so + 1e-6)) * a + (torch.sin(t * omega) / (so + 1e-6)) * b
    mask = (so.abs() < 1e-6).view(-1)
    if mask.any():
        out[mask] = (1.0 - t) * a[mask] + t * b[mask]
    return _l2norm(out)

def phq_score_to_severity(score: int | float | None) -> str | None:
    """将PHQ_Score映射到严重度标签。
    返回值 ∈ {"normal","mild","moderate","moderately_severe","severe"} 或 None（无效分数）。
    """
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
    """加载metadata_with_phq.csv，返回两个映射：subject->PHQ_Score, subject->severity。"""
    print("Loading PHQ scores (metadata_with_phq.csv)...")

    import pandas as pd

    subject_to_score = {}
    subject_to_severity = {}

    try:
        df = pd.read_csv(PHQ_SCORE_METADATA_FILE)
        if 'Participant_ID' not in df.columns:
            raise RuntimeError("metadata_with_phq.csv 缺少 Participant_ID 列")
        # 兼容列名：PHQ_Score 或 PHQ_Score_Total 等
        score_col = None
        for cand in ["PHQ_Score", "PHQ_Score_Total", "PHQ_Score_0_24", "PHQ_Total", "PHQ9_Total", "PHQ9_Score"]:
            if cand in df.columns:
                score_col = cand
                break
        if score_col is None:
            raise RuntimeError("metadata_with_phq.csv 未找到PHQ分数字段（尝试了多个候选名）")

        for _, row in df.iterrows():
            try:
                sid = int(row["Participant_ID"])
                score = row[score_col]
                subject_to_score[sid] = None if (isinstance(score, float) and np.isnan(score)) else float(score)
                sev = phq_score_to_severity(subject_to_score[sid])
                if sev is not None:
                    subject_to_severity[sid] = sev
            except Exception:
                continue

        # 打印分布
        counts = {}
        for sev in subject_to_severity.values():
            counts[sev] = counts.get(sev, 0) + 1
        print(f"✓ PHQ scores loaded. Severity distribution: {counts}")
    except Exception as e:
        print(f"✗ Failed to load metadata_with_phq.csv: {e}")

    return subject_to_score, subject_to_severity


# ───────────────────────────  Clinical-5 原型构建与插值  ─────────────────────────
def build_dep_bank_clinical5(dep_embeddings, dep_subject_ids, subj_score):
    """基于subject级抑郁嵌入构建五级严重度原型库（无二分类兜底）。
    - 仅使用 PHQ_Score 分桶均值；缺失则依赖相邻原型 SLERP 补全，或复制邻近原型。
    返回 dict: {names, protos, mid_index}
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

    bins = [(0,4), (5,9), (10,14), (15,19), (20,24)]
    names = ["normal","mild","moderate","mod_severe","severe"]
    protos = []
    for lo, hi in bins:
        idx = bucket_indices(lo, hi)
        if len(idx) >= 3:
            v = _l2norm(Z[idx].mean(0))
        else:
            v = None
        protos.append(v)

    # 去除二分类兜底：不再使用 healthy/depressed 均值填充两端

    # 中间缺失用左右原型做 SLERP 补全
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
    """给定 α∈[-1,1]，在原型库上定位并在相邻原型间做 SLERP，返回抑郁条件向量。"""
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


def _proto_cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    try:
        uu = _l2norm(u.view(-1))
        vv = _l2norm(v.view(-1))
        return float(F.cosine_similarity(uu, vv, dim=0).item())
    except Exception:
        return float('nan')


def print_bank_diagnostics(bank, name: str = "model-space", sample_alphas: list[float] | None = None):
    try:
        protos = bank.get("protos", [])
        print(f"\n[DIAG] {name} Clinical-5 prototypes cosine matrix:")
        header = ["    "] + [f"p{i}" for i in range(len(protos))]
        try:
            print(" ".join(header))
        except Exception:
            pass
        for i, pi in enumerate(protos):
            row = [f"p{i}"]
            for j, pj in enumerate(protos):
                if isinstance(pi, torch.Tensor) and isinstance(pj, torch.Tensor):
                    row.append(f"{_proto_cosine(pi, pj):.4f}")
                else:
                    row.append("nan")
            try:
                print(" ".join([f"{c:>8}" for c in row]))
            except Exception:
                print(" ".join(row))

        if sample_alphas is None:
            sample_alphas = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]
        print(f"[DIAG] {name} alpha→vector cosine to endpoints:")
        p0 = protos[0] if len(protos) > 0 else None
        p4 = protos[-1] if len(protos) > 0 else None
        for a in sample_alphas:
            v = dep_from_alpha_over_bank(bank, a)
            c0 = _proto_cosine(v, p0) if isinstance(p0, torch.Tensor) else float('nan')
            c4 = _proto_cosine(v, p4) if isinstance(p4, torch.Tensor) else float('nan')
            print(f"  alpha={a:+.2f}: cos(v,p0)={c0:.4f}, cos(v,p4)={c4:.4f}")
    except Exception as _e:
        print(f"[DIAG] diagnostics failed: {_e}")

def severity_to_alpha(sev: str) -> float:
    """将严重度标签映射为 α（拉大中间档距）。"""
    if sev == "normal":
        return -1.0
    if sev == "mild":
        return -0.7
    if sev == "moderate":
        return 0.0
    if sev == "moderately_severe":
        return 0.7
    if sev == "severe":
        return 1.0
    return 0.0

def severity_to_dirname(sev: str) -> str:
    """严重度到子目录名映射。"""
    if sev == "normal":
        return "norm"
    if sev == "mild":
        return "mild"
    if sev == "moderate":
        return "mod"
    if sev == "moderately_severe":
        return "mod_sev"
    if sev == "severe":
        return "sev"
    return str(sev)

# 严重度到每subject目标句子数
# 原始训练集目录（用于统计原始分布）
ORIG_TRAIN_DIR = "/home/i-liyuxin/test/daic_preprocessed/train"
# 采样参考音频目录（用于ECAPA参考嵌入和声学特征提取）
SAMPLING_REF_DIR = "/home/i-liyuxin/Depression_TTS/matcha/data/processed_audio_22050"
# 抑郁参考音频目录（若不设，则复用 SAMPLING_REF_DIR）
DEPRESSION_REF_AUDIO_DIR = None

# 是否启用"合成+原始均衡（二分类与五分类同时尽量均衡）"
BALANCE_WITH_ORIGINAL = True

# 全局存储：subject_id -> PHQ_Score 映射（从CSV文件加载）
SUBJECT_PHQ_SCORES = {}

# 五分类枚举顺序（二分类映射：healthy={normal,mild}；depressed={moderate,moderately_severe,severe}）
SEVERITY_CLASSES = ["normal", "mild", "moderate", "moderately_severe", "severe"]

# 旧的固定表保留但默认不用
SEVERITY_TO_SENTENCES = {
    "normal": 0,
    "mild": 18,
    "moderate": 7,
    "moderately_severe": 32,
    "severe": 163,
}

def severity_to_binary(sev: str) -> int:
    return 0 if sev in ("normal", "mild") else 1




def load_embeddings():
    """加载 depression embeddings（utterance 级 npz，必要时聚合为 subject 级）；构建训练期 subject→spk 映射。"""
    print("Loading embeddings...")

    # 加载 depression embeddings（支持 subject 或 utterance 级）
    depression_data = np.load(DEPRESSION_EMBEDDINGS_FILE)
    depression_embeddings = depression_data['embeddings'] if 'embeddings' in depression_data else depression_data['arr_0']
    depression_subject_ids = depression_data['subject_ids'] if 'subject_ids' in depression_data else None

    # 回退：当不存在 subject_ids 但存在 utterance_ids（形如 "303_140"）时，按 subject 聚合
    if depression_subject_ids is None and 'utterance_ids' in depression_data:
        try:
            utt_ids = depression_data['utterance_ids']
            subj_from_utt = np.asarray([int(str(u).split('_', 1)[0]) for u in utt_ids], dtype=int)

            agg_sum = {}
            agg_cnt = {}
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

            aggregated = np.stack([agg_sum[sid] / agg_cnt[sid] for sid in unique_subj], axis=0).astype(depression_embeddings.dtype, copy=False)
            depression_embeddings = aggregated
            depression_subject_ids = unique_subj
            print(f"✓ Aggregated utterance embeddings to subject-level: {aggregated.shape[0]} subjects from {len(subj_from_utt)} utterances")
        except Exception:
            pass

    print(f"✓ Depression embeddings shape: {depression_embeddings.shape}")

    # speaker embeddings 禁用
    speaker_embeddings = None
    speaker_subject_ids = None

    # 构建 subject→spk 映射（与训练一致）
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
                                seen.add(sid); ids.append(sid)
        except Exception:
            pass
        return ids

    ordered_subjects = []
    ordered_subjects += _load_subject_ids_from_file(TRAIN_SUBJECT_FILE)
    seen = set(ordered_subjects)
    for sid in _load_subject_ids_from_file(VAL_SUBJECT_FILE):
        if sid not in seen:
            seen.add(sid); ordered_subjects.append(sid)
    if not ordered_subjects and depression_subject_ids is not None:
        try:
            ordered_subjects = [int(x) for x in depression_subject_ids.astype(int).tolist()]
        except Exception:
            ordered_subjects = []

    depression_subject_to_idx = {}
    if depression_subject_ids is not None:
        depression_subject_ids = depression_subject_ids.astype(int)
        depression_subject_to_idx = {int(sid): idx for idx, sid in enumerate(depression_subject_ids)}
        print(f"✓ Depression subject mapping: {len(depression_subject_to_idx)} subjects")

    subject_to_spk_idx = {sid: idx for idx, sid in enumerate(ordered_subjects)} if ordered_subjects else depression_subject_to_idx
    print(f"✓ Subject→spk_idx mapping size: {len(subject_to_spk_idx)}")

    global DEPRESSION_MEAN, DEPRESSION_STD
    try:
        DEPRESSION_MEAN = depression_embeddings.mean(axis=0)
        DEPRESSION_STD = depression_embeddings.std(axis=0) + 1e-6
        print("✓ Computed depression embedding statistics (for zscore normalization)")
    except Exception:
        pass

    return (
        depression_embeddings,
        speaker_embeddings,
        depression_subject_ids,
        speaker_subject_ids,
        depression_subject_to_idx,
        subject_to_spk_idx,
    )




def load_personachat_dataset():
    """加载PersonaChat数据集"""
    print("Loading PersonaChat dataset (AlekseyKorshuk/persona-chat)...")
    ds = load_dataset("AlekseyKorshuk/persona-chat")
    train_data = ds['train'] if 'train' in ds else list(ds.values())[0]
    print("✓ AlekseyKorshuk/persona-chat dataset loaded successfully")
    print(f"  Total samples: {len(train_data)}")
    if len(train_data) > 0:
        sample = train_data[0]
        print(f"  Dataset fields: {list(sample.keys())}")
    return train_data
# ───────────────────────────  ECAPA参考嵌入构建（移除）  ───────────────────────────────────


# 文本清洗与校验辅助函数

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', '', text)
    return text


def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', str(text)))


def is_text_valid(text: str) -> bool:
    s = clean_text(text)
    lower = s.lower()
    if lower.startswith('http') or lower.startswith('www.'):
        return False
    num_chars = len(s)
    num_words = count_words(s)
    return (num_chars >= TEXT_MIN_CHARS and num_chars <= TEXT_MAX_CHARS and num_words >= TEXT_MIN_WORDS)


def extract_conversation_responses(dataset, max_needed=None):
    """从PersonaChat数据集中提取对话回复（被采访者说话内容）
    max_needed: 需要的最大句子数（用于构建全局池）；None时等同于SENTENCES_PER_SUBJECT
    """
    print("Extracting conversation responses from PersonaChat...")
    
    gather_target = max_needed if max_needed is not None else SENTENCES_PER_SUBJECT
    responses = []
    
    for idx, sample in enumerate(dataset):
        try:
            # 处理不同的数据集结构
            if hasattr(sample, 'sentences') and hasattr(sample, '__getitem__'):
                # 这是我们的fallback dataset
                response = sample['history'][0] if 'history' in sample else sample['candidates'][0]
                if is_text_valid(response):
                    responses.append(clean_text(response))
            elif 'candidates' in sample:
                # 如果有candidates字段，通常包含回复选项
                candidates = sample['candidates']
                if isinstance(candidates, list) and len(candidates) > 0:
                    # 取第一个候选回复（通常是正确答案）
                    response = candidates[0] if isinstance(candidates[0], str) else str(candidates[0])
                    if is_text_valid(response):
                        responses.append(clean_text(response))
            
            elif 'history' in sample:
                # ConvAI2格式：history字段包含对话历史
                history = sample['history']
                if isinstance(history, list) and len(history) > 0:
                    # 取最后一句作为回复（通常是被采访者的回答）
                    last_response = history[-1] if isinstance(history[-1], str) else str(history[-1])
                    if is_text_valid(last_response):
                        responses.append(clean_text(last_response))
            
            elif 'utterances' in sample:
                # 如果有utterances字段
                utterances = sample['utterances']
                if isinstance(utterances, list):
                    for utterance in utterances:
                        if isinstance(utterance, dict) and 'candidates' in utterance:
                            candidates = utterance['candidates']
                            if isinstance(candidates, list) and len(candidates) > 0:
                                response = candidates[0] if isinstance(candidates[0], str) else str(candidates[0])
                                if is_text_valid(response):
                                    responses.append(clean_text(response))
                        elif isinstance(utterance, str) and is_text_valid(utterance):
                            responses.append(clean_text(utterance))
            
            elif 'dialogue' in sample:
                # 如果有dialogue字段（一些新的PersonaChat版本）
                dialogue = sample['dialogue']
                if isinstance(dialogue, list):
                    for turn in dialogue:
                        if isinstance(turn, str) and is_text_valid(turn):
                            responses.append(clean_text(turn))
                        elif isinstance(turn, dict) and 'text' in turn:
                            text = turn['text']
                            if isinstance(text, str) and is_text_valid(text):
                                responses.append(clean_text(text))
            
            elif 'conversation' in sample:
                # Google Synthetic-Persona-Chat格式
                conversation = sample['conversation']
                if isinstance(conversation, list):
                    for turn in conversation:
                        if isinstance(turn, str) and is_text_valid(turn):
                            responses.append(clean_text(turn))
            
            # 通用处理：如果找不到特定字段，尝试所有字符串字段
            else:
                for key, value in sample.items():
                    if isinstance(value, str) and is_text_valid(value):
                        responses.append(clean_text(value))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and is_text_valid(item):
                                responses.append(clean_text(item))
            
            # 如果达到需要的数量就停止（多收集一些以防后续去重）
            if len(responses) >= gather_target * 2:
                break
                
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue
    
    print(f"✓ Extracted {len(responses)} conversation responses")
    
    # 如果没有提取到足够的句子，使用备用句子
    if len(responses) < gather_target:
        print(f"⚠️  句子池不足：仅有 {len(responses)}/{gather_target} 条，将尽力分配并继续")
    else:
        print(f"✓ 句子池就绪：{len(responses)} 条")
    
    # 去重和进一步过滤
    unique_responses = []
    seen = set()
    
    for response in responses:
        clean_response = clean_text(response)
        if (is_text_valid(clean_response) and 
            clean_response.lower() not in seen):
            unique_responses.append(clean_response)
            seen.add(clean_response.lower())
    
    print(f"✓ After filtering and deduplication: {len(unique_responses)} unique responses")
    
    # 如果还是不够，尝试分割长句子
    if len(unique_responses) < gather_target:
        print("Splitting longer responses to get more sentences...")
        additional_sentences = []
        
        for response in unique_responses[:]:
            sentences = split_sentences(response)
            for sentence in sentences:
                s = clean_text(sentence)
                if (is_text_valid(s) and 
                    s.lower() not in seen):
                    additional_sentences.append(s)
                    seen.add(s.lower())
        
        unique_responses.extend(additional_sentences)
        print(f"✓ After sentence splitting: {len(unique_responses)} sentences")
    
    final_responses = unique_responses[:gather_target]  # 返回所需数量
    print(f"✓ Final sentences pool: {len(final_responses)} sentences")
    return final_responses


def split_sentences(text):
    """按句号分割文本为句子列表"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences


def load_matcha_model(ckpt_path: str) -> MatchaTTS:
    """加载 Matcha-TTS 模型"""
    print(f"Loading Matcha-TTS from: {ckpt_path}")
    
    try:
        model = MatchaTTS.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval()
        model.to(device)
        
        print(f"✓ Matcha-TTS loaded successfully")
        print(f"  Model n_feats: {model.n_feats}")
        print(f"  Model n_spks: {model.n_spks}")
        print(f"  Model use_daic_conditions: {model.use_daic_conditions}")
        print(f"  Model depression_cond_dim: {model.depression_cond_dim}")
        print(f"  Model speaker_cond_dim: {model.speaker_cond_dim}")
        print(f"  Model use_adapter: {getattr(model, 'use_adapter', 'N/A')}")
        if hasattr(model, 'use_adapter') and model.use_adapter:
            print(f"  Model adapter_dim: {getattr(model, 'adapter_dim', 'N/A')}")
        return model
        
    except Exception as e:
        print(f"✗ Failed to load Matcha-TTS: {e}")
        raise


def get_vocoder_input_dims(config_path: str) -> int:
    """获取 HiFi-GAN 期望的输入维度"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        input_dims = None
        if "num_mels" in config:
            input_dims = config["num_mels"]
        elif "n_mels" in config:
            input_dims = config["n_mels"]
        elif "mel_channels" in config:
            input_dims = config["mel_channels"]
        
        print(f"HiFi-GAN config indicates input dims: {input_dims}")
        return input_dims
        
    except Exception as e:
        print(f"Warning: Could not determine vocoder input dims: {e}")
        return None


def load_hifigan(weight_path: str, cfg_path: str) -> HiFiGAN:
    """加载 HiFi-GAN vocoder"""
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
    
    # 记录 vocoder 期望维度，供后续适配
    global VOCODER_INPUT_DIMS
    VOCODER_INPUT_DIMS = None
    for k in ["num_mels", "n_mels", "mel_channels"]:
        if k in h_json:
            VOCODER_INPUT_DIMS = int(h_json[k])
            break
    print(f"✓ HiFi-GAN generator loaded ({len(state_dict)} params), expected mel dims: {VOCODER_INPUT_DIMS}")
    return g


def adapt_mel_dimensions(mel, target_dims):
    """调整 mel 谱维度以匹配 vocoder 期望"""
    # 确保类型为 float32，避免精度不匹配导致的听感发糊
    if mel.dtype != torch.float32:
        mel = mel.to(torch.float32)
    # 标准形状应为 [B, n_mels, T]
    if mel.dim() == 2:
        # 可能是 [n_mels, T]
        mel = mel.unsqueeze(0)
    elif mel.dim() == 3 and mel.shape[1] > 256 and mel.shape[2] < 256:
        # 罕见错误形状推断
        pass
    current_dims = mel.shape[1]
    
    if current_dims == target_dims or target_dims is None:
        return mel
    
    if current_dims > target_dims:
        adapted_mel = mel[:, :target_dims, :]
    else:
        adapted_mel = torch.zeros(mel.shape[0], target_dims, mel.shape[2], device=mel.device, dtype=mel.dtype)
        adapted_mel[:, :current_dims, :] = mel
        
        if current_dims < target_dims:
            remaining = target_dims - current_dims
            repeat_source = mel[:, -min(remaining, current_dims):, :]
            adapted_mel[:, current_dims:, :] = repeat_source
    
    return adapted_mel


@torch.inference_mode()
def process_text(text: str):
    """文本预处理 - 确保与训练时完全一致"""
    # 使用与训练时相同的文本清理器
    seq, cleaned_text = text_to_sequence(text, ["english_cleaners2"])
    
    # 确保与训练时相同的数据类型和blank token处理
    if True:  # 对应训练时的 add_blank=True
        seq = intersperse(seq, 0)
    
    # 使用与训练时相同的数据类型：torch.IntTensor
    x = torch.IntTensor(seq).to(device)[None]
    l = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    
    return {
        "x_orig": text,
        "x": x,
        "x_lengths": l,
        "x_phones": sequence_to_text(x.squeeze(0).tolist()),
        "cleaned_text": cleaned_text,  # 添加清理后的文本用于调试
    }


def inspect_ckpt_keys(ckpt_path: str, save_dir: Path | None = None):
    """打印并保存 ckpt 的键信息。
    - 打印顶层 keys
    - 若存在 state_dict，则打印其键数量与若干样例前缀
    - 将完整 keys 列表保存到 save_dir/ckpt_keys.txt（若提供）
    """
    print(f"\n=== 检查 ckpt keys: {ckpt_path} ===")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"✗ 无法加载 ckpt: {e}")
        return

    if isinstance(ckpt, dict):
        top_keys = sorted(list(ckpt.keys()))
        print(f"顶层 keys ({len(top_keys)}): {top_keys}")
        sd = ckpt.get("state_dict", None)
        if isinstance(sd, dict):
            sd_keys = sorted(list(sd.keys()))
            print(f"state_dict keys 数量: {len(sd_keys)}")
            print("state_dict 示例前20个键:")
            for k in sd_keys[:20]:
                print(f"  - {k}")
        else:
            print("未找到 'state_dict' 字段或其类型非字典")

        if save_dir is not None:
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                out_file = save_dir / "ckpt_keys.txt"
                with open(out_file, "w") as f:
                    f.write("Top-level keys:\n")
                    for k in top_keys:
                        f.write(k + "\n")
                    if isinstance(sd, dict):
                        f.write("\nstate_dict keys:\n")
                        for k in sd_keys:
                            f.write(k + "\n")
                print(f"✓ 已保存完整 keys 到: {out_file}")
            except Exception as e:
                print(f"⚠️ 保存 keys 文件失败: {e}")
    else:
        print(f"ckpt 类型: {type(ckpt)}，非字典，直接打印字符串化：")
        print(str(ckpt)[:2000])


@torch.inference_mode()
def synthesise_with_conditions(model: MatchaTTS, text: str, depression_embedding,
                               speaker_id: int | None,
                               temperature_override=None,
                               guidance_w: float | None = None,
                               dep_bank_for_uncond: dict | None = None,
                               uncond_mode: str = None):
    """
    使用 depression 条件 + spk_id 进行合成；若启用 CFG，则进行 cond / uncond 两路并在 mel 级混合。
    - guidance_w: None 或 float；当 CFG_ENABLED 且 guidance_w>0 时生效
    - dep_bank_for_uncond: 仅当 uncond_mode='avg_proto' 时需要
    - uncond_mode: 覆盖全局 CFG_UNCOND_MODE（可传 'none'|'zero'|'avg_proto'）
    """
    tex = process_text(text)

    if RESET_SEED_EACH_SENTENCE:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    # 准备条件输入（有条件分支）
    dep_cond = None
    if depression_embedding is not None:
        if isinstance(depression_embedding, np.ndarray):
            de = torch.tensor(depression_embedding.squeeze(), dtype=torch.float32, device=device)
        elif isinstance(depression_embedding, torch.Tensor):
            de = depression_embedding.to(device, torch.float32).squeeze()
        else:
            de = torch.tensor(depression_embedding, dtype=torch.float32, device=device).squeeze()
        de = _l2norm(de - de.mean())
        if de.dim() == 0:
            de = de.unsqueeze(0)
        dep_cond = de.unsqueeze(0)  # [1, D]

    if not USE_DEPRESSION_COND:
        dep_cond = None

    # spk 索引
    spk_tensor = None
    if hasattr(model, "n_spks") and int(model.n_spks) > 1 and speaker_id is not None:
        try:
            mapping = GLOBAL_SUBJECT_TO_SPK_IDX
            spk_idx = mapping[int(speaker_id)]
        except Exception:
            nspk = int(model.n_spks) if hasattr(model, 'n_spks') else 1
            spk_idx = int(int(speaker_id) % nspk) if nspk > 0 else 0
        spk_tensor = torch.tensor([spk_idx], dtype=torch.long, device=device)

    current_temperature = temperature_override if temperature_override is not None else TEMPERATURE
    synth_kwargs_base = dict(
        n_timesteps=N_TIMESTEPS,
        temperature=current_temperature,
        length_scale=LENGTH_SCALE,
        speaker_cond=None,
    )
    if spk_tensor is not None:
        synth_kwargs_base["spks"] = spk_tensor

    # 是否启用 CFG
    use_cfg = bool(CFG_ENABLED and (guidance_w is not None) and (float(guidance_w) > 0.0))

    if not use_cfg:
        # 旧路径：纯有条件分支
        out_c = model.synthesise(tex["x"], tex["x_lengths"], depression_cond=dep_cond, **synth_kwargs_base)
        out_c.update({**tex})
        return out_c

    # CFG 路径：cond / uncond 两路 mel
    # 有条件分支
    out_cond = model.synthesise(tex["x"], tex["x_lengths"], depression_cond=dep_cond, **synth_kwargs_base)
    mel_cond = out_cond["mel"]

    # 无条件分支的 depression_cond
    chosen_uncond_mode = (uncond_mode or CFG_UNCOND_MODE or "none").lower()
    dep_uncond = _make_uncond_depression_cond(dep_cond, dep_bank_for_uncond, chosen_uncond_mode)

    out_uncond = model.synthesise(tex["x"], tex["x_lengths"], depression_cond=dep_uncond, **synth_kwargs_base) \
                 if dep_uncond is not dep_cond else out_cond
    mel_uncond = out_uncond["mel"] if (dep_uncond is not None or chosen_uncond_mode == "none") else None

    # 在 mel 级做 CFG 混合
    mel_guided = apply_cfg_on_mel(mel_cond, mel_uncond, float(guidance_w))

    # 返回与原结构一致的字典
    out = dict(out_cond)
    out["mel"] = mel_guided
    out["cfg"] = {
        "enabled": True,
        "w": float(guidance_w),
        "uncond_mode": chosen_uncond_mode,
        "has_uncond": mel_uncond is not None
    }
    out.update({**tex})
    return out

@torch.inference_mode()
def generate_multiple_samples(model: MatchaTTS, text: str, depression_embedding, speaker_id: int | None,
                              vocoder: HiFiGAN, denoiser: Denoiser,
                              num_samples: int = 5,
                              guidance_w: float | None = None,
                              dep_bank_for_uncond: dict | None = None,
                              uncond_mode: str | None = None):
    """按多个温度采样；若提供 guidance_w 则在 mel 级应用 CFG。"""
    samples = []
    if num_samples <= 0:
        num_samples = 1
    temps = np.linspace(TEMPERATURE_MIN, TEMPERATURE_MAX, num_samples).tolist()
    temps.append(TEMPERATURE)
    temps = sorted(set(round(float(t), 4) for t in temps))
    print(f"    Sampling temperatures: {temps}")

    for i, sample_temperature in enumerate(temps):
        try:
            out = synthesise_with_conditions(
                model, text, depression_embedding, speaker_id,
                temperature_override=float(sample_temperature),
                guidance_w=guidance_w,
                dep_bank_for_uncond=dep_bank_for_uncond,
                uncond_mode=uncond_mode
            )
            waveform = mel_to_waveform(out["mel"], vocoder, denoiser)
            samples.append({
                "waveform": waveform,
                "mel": out["mel"],
                "temperature": float(sample_temperature),
                "sample_id": i,
                "cfg": out.get("cfg", None),
            })
        except Exception as e:
            print(f"    ⚠️ 生成样本 {i} 失败: {e}")
            continue
    return samples

# 说话人相似度选择（移除）




@torch.inference_mode()
def mel_to_waveform(mel, vocoder: HiFiGAN, denoiser: Denoiser):
    """mel 谱转换为波形 - 保证张量形状与 dtype，避免去噪器/声码器形状不匹配。"""
    try:
        # 形状与维度自检与适配
        expected_mels = VOCODER_INPUT_DIMS
        if expected_mels is not None:
            mel = adapt_mel_dimensions(mel, expected_mels)
        else:
            if mel.dtype != torch.float32:
                mel = mel.to(torch.float32)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # [1, n_mels, T]

        if mel.dim() != 3:
            raise RuntimeError(f"Mel shape must be [B, n_mels, T], got {tuple(mel.shape)}")

        print(f"    Vocoder mel shape: {tuple(mel.shape)}, dtype: {mel.dtype}")

        # HiFi-GAN 前向输出通常为 [B, 1, T]
        audio = vocoder(mel).clamp(-1, 1)            # [B, 1, T]
        # Denoiser 的 STFT 期望 1D/2D 张量，这里传入 [B, T]
        audio_2d = audio.squeeze(1)                  # [B, T]
        denoised = denoiser(audio_2d, strength=0.00025)
        # 兼容不同实现的返回形状，统一到 [B, T]
        if isinstance(denoised, tuple):
            denoised = denoised[0]
        if isinstance(denoised, torch.Tensor):
            if denoised.dim() == 3 and denoised.shape[1] == 1:
                denoised = denoised.squeeze(1)
            elif denoised.dim() == 1:
                denoised = denoised.unsqueeze(0)
        audio = denoised.squeeze(0).cpu()            # [T]

        # 应用音量增益
        audio = audio * AUDIO_GAIN

        # 音频归一化（可选）
        if NORMALIZE_AUDIO:
            current_rms = torch.sqrt(torch.mean(audio**2))
            if current_rms > 0 and TARGET_RMS > 0:
                audio = audio * (TARGET_RMS / current_rms)

        audio = torch.clamp(audio, -1.0, 1.0)
        return audio
    except Exception as e:
        print(f"Error in vocoder conversion: {e}")
        raise



def save_audio_and_label(subject_id: str, sentence_id: int, waveform, folder: str | Path, subject_to_phq8: dict, label_subject_id: int | None = None):
    """保存音频文件和对应的label文件
    使用格式: subject_id_sentence_id.wav
    例如: 300300_1.wav 表示组合ID 300300的第1个句子
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    # 文件命名用组合ID（字符串）
    audio_filename = f"{subject_id}_{sentence_id}.wav"
    label_filename = f"{subject_id}_{sentence_id}.label"
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    # 额外的音频处理：确保音频不为零
    if np.abs(waveform).max() < 1e-6:
        print(f"Warning: Audio waveform is too quiet, max amplitude: {np.abs(waveform).max()}")
    
    # 保存为22.05kHz（vocoder原生采样率）
    sample_rate = 22050
    sf.write(folder / audio_filename, waveform, sample_rate, "PCM_16")
    
    # label 使用 depression_id（若未传则尝试从 subject_id 解析）
    label_sid = label_subject_id
    if label_sid is None:
        try:
            label_sid = int(subject_id)
        except Exception:
            label_sid = None
    subject_phq8 = subject_to_phq8.get(label_sid, 1) if label_sid is not None else 1
    with open(folder / label_filename, "w") as f:
        f.write(f"{subject_phq8}\n")
    
    return audio_filename


def main():
    print("=== Matcha-TTS 简单测试：抑郁条件置空（None）能否合成语音 ===")

    # 0) 先检查并输出 ckpt 的 keys，便于确认可用字段
    out_dir = Path(OUTPUT_FOLDER_BASE) / "test_no_dep"
    inspect_ckpt_keys(MATCHA_CKPT, save_dir=out_dir)

    # 1) 加载模型与声码器
    model = load_matcha_model(MATCHA_CKPT)
    vocoder = load_hifigan(HIFIGAN_WEIGHT, HIFIGAN_CONFIG)
    denoiser = Denoiser(vocoder, mode="zeros")

    # 2) 测试文本（可自行修改）
    test_text = "This is a simple test to check speech synthesis without depression condition."
    print(f"测试文本: {test_text}")

    # 3) 置空抑郁条件，speaker 取 0（若模型是多说话人，将按 n_spks 取模）
    depression_embedding = None
    speaker_id = 0

    # 关闭引导混合（或保持 None），确保是最简单路径
    guidance_w = None

    # 4) 合成 mel，并使用 HiFi-GAN 生成波形
    out = synthesise_with_conditions(
        model,
        test_text,
        depression_embedding=depression_embedding,
        speaker_id=speaker_id,
        guidance_w=guidance_w,
        dep_bank_for_uncond=None,
        uncond_mode="none",
    )
    wav = mel_to_waveform(out["mel"], vocoder, denoiser)

    # 5) 保存到输出目录
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_no_dep.wav"
    sf.write(str(out_path), wav.cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav), 22050, "PCM_16")

    print(f"✅ 合成完成，音频已保存: {out_path}")


if __name__ == "__main__":
    main()