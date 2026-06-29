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
OUTPUT_FOLDER_BASE = "/data/depression_tts/synthese_data/v6_dep_slerp_A_1105"

N_TIMESTEPS = 45
LENGTH_SCALE = 1.0
TEMPERATURE = 0.0  # 设为0以获得完全确定性输出（不进行采样）

MONO_SENTENCES_PER_SEVERITY = 20

MAX_GENERATION_ATTEMPTS = 1    # 每个句子只生成一次

# 文本筛选参数（用于去掉太短的文本）
TEXT_MIN_CHARS = 20     # 最小字符数（清洗后计算）
TEXT_MIN_WORDS = 5      # 最少词数
TEXT_MAX_CHARS = 2000    # 最大字符数上限

# 采样与稳定性控制
RESET_SEED_EACH_SENTENCE = False  # 每句前重置随机种子，进一步消除随机性
USE_DEPRESSION_COND = True

NUM_SAMPLES = 1                  # 每个句子只生成一次
MAX_REF_AUDIO_PER_SUBJECT = 70   # 已不再使用

# 仅使用抑郁嵌入相似度进行候选选择（已禁用）
SELECT_BY_DEPRESSION_ONLY = False

# 抑郁音频嵌入提取模型（已不再使用）
DEPRESSION_AUDIO_MODEL_CKPT = None
DEPRESSION_EXTRACTOR_LAYER = 20
DEPRESSION_WAVLM_MODEL_NAME = "microsoft/wavlm-large"

# 采样策略优化（已不再使用）
USE_ADAPTIVE_SAMPLING = False
SIMILARITY_THRESHOLD_RETRY = 0.3
MAX_RETRY_SAMPLES = 20

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
REQUIRE_AUDIO_SPACE_BANK = False  # 禁用音频空间bank要求，不再进行相似度匹配

# ───────────────────────────  Speaker Similarity 代码移除  ───────────────────────────────────


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

def summarize_bank_alignment(model_bank, audio_bank):
    """计算模型空间与音频空间原型之间的余弦对齐情况。"""
    try:
        if model_bank is None or audio_bank is None:
            return None
        model_protos = model_bank.get("protos", [])
        audio_protos = audio_bank.get("protos", [])
        K = min(len(model_protos), len(audio_protos))
        if K == 0:
            return None
        cos_values = []
        for idx in range(K):
            mp = model_protos[idx]
            ap = audio_protos[idx]
            if isinstance(mp, torch.Tensor) and isinstance(ap, torch.Tensor):
                cos_values.append(_proto_cosine(mp, ap))
        if not cos_values:
            return None
        cos_array = np.array(cos_values, dtype=np.float32)
        mean_cos = float(np.mean(cos_array))
        min_cos = float(np.min(cos_array))
        max_cos = float(np.max(cos_array))
        negative_count = int(np.sum(cos_array < 0))
        print(f"[ALIGN] model↔audio prototype cosine mean={mean_cos:.4f}, range=({min_cos:.4f}, {max_cos:.4f}), negatives={negative_count}")
        if negative_count > 0:
            print("[ALIGN] ⚠️ Detected negative cosine alignments; consider multi-objective scoring or feature alignment.")
        return {
            "mean": mean_cos,
            "min": min_cos,
            "max": max_cos,
            "negative_count": negative_count,
            "values": [float(v) for v in cos_values]
        }
    except Exception as e:
        print(f"[ALIGN] alignment summary failed: {e}")
        return None

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
DEPRESSION_REF_AUDIO_DIR = None

# 是否启用"合成+原始均衡（二分类与五分类同时尽量均衡）"
BALANCE_WITH_ORIGINAL = True

# 全局存储：subject_id -> PHQ_Score 映射（从CSV文件加载）
SUBJECT_PHQ_SCORES = {}

# 五分类枚举顺序（二分类映射：healthy={normal,mild}；depressed={moderate,moderately_severe,severe}）
SEVERITY_CLASSES = ["normal", "mild", "moderate", "moderately_severe", "severe"]

def severity_to_binary(sev: str) -> int:
    return 0 if sev in ("normal", "mild") else 1

def load_phq_scores_from_csvs():
    """从CSV文件加载所有subject的PHQ分数到全局存储中"""
    global SUBJECT_PHQ_SCORES
    
    print("Loading PHQ scores from CSV files...")
    
    # CSV文件路径列表（使用本地metadata文件）
    csv_files = [
        PHQ_SCORE_METADATA_FILE,  # 使用已有的metadata_with_phq.csv
        METADATA_FILE  # 使用已有的metadata.csv作为备选
    ]
    
    import pandas as pd
    
    for csv_file in csv_files:
        try:
            if os.path.exists(csv_file):
                print(f"Loading from: {csv_file}")
                df = pd.read_csv(csv_file)
                
                # 检查必要的列
                if 'Participant_ID' not in df.columns:
                    print(f"Warning: {csv_file} missing Participant_ID column")
                    continue
                    
                # 查找PHQ分数列
                phq_col = None
                for col in ['PHQ8_Score', 'PHQ_Score', 'PHQ_Score_Total', 'PHQ_Score_0_24', 'PHQ_Total', 'PHQ9_Total', 'PHQ9_Score']:
                    if col in df.columns:
                        phq_col = col
                        break
                        
                if phq_col is None:
                    print(f"Warning: {csv_file} missing PHQ score column")
                    continue
                
                # 读取数据
                before_count = len(SUBJECT_PHQ_SCORES)
                for _, row in df.iterrows():
                    try:
                        subject_id = int(row['Participant_ID'])
                        phq_score = row[phq_col]
                        
                        # 处理NaN值
                        if pd.isna(phq_score):
                            continue
                            
                        phq_score = float(phq_score)
                        SUBJECT_PHQ_SCORES[subject_id] = phq_score
                        
                    except Exception as e:
                        continue
                        
                loaded_this_file = len(SUBJECT_PHQ_SCORES) - before_count
                print(f"  Loaded {loaded_this_file} subjects from {csv_file}")
            else:
                print(f"Warning: CSV file not found: {csv_file}")
                
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    print(f"✓ Total PHQ scores loaded: {len(SUBJECT_PHQ_SCORES)} subjects")
    if len(SUBJECT_PHQ_SCORES) > 0:
        sample_scores = list(SUBJECT_PHQ_SCORES.items())[:5]
        print(f"  Sample scores: {sample_scores}")
    
    return SUBJECT_PHQ_SCORES

def count_original_distribution(train_dir: str) -> tuple[dict, dict]:
    """扫描原始训练集，统计五分类与二分类的条数（按utterance/clip计）。
    - 五分类来自全局存储的PHQ分数（从CSV文件加载）
    - 二分类来自 .label（0/1）
    """
    sev_counts = {s: 0 for s in SEVERITY_CLASSES}
    bin_counts = {0: 0, 1: 0}
    
    try:
        # 统计每个subject的utterance数量
        subject_utterance_counts = {}
        for fn in os.listdir(train_dir):
            if not fn.endswith('.wav'):
                continue
            base = os.path.join(train_dir, fn[:-4])
            lbl_file = base + '.label'
            
            # 二分类统计
            if os.path.exists(lbl_file):
                try:
                    with open(lbl_file) as f:
                        b = int(str(f.read()).strip())
                        if b in bin_counts:
                            bin_counts[b] += 1
                except Exception:
                    pass
            
            # 统计每个subject的utterance数量
            try:
                subject_id = int(fn.split('_')[0])
                subject_utterance_counts[subject_id] = subject_utterance_counts.get(subject_id, 0) + 1
            except Exception:
                pass
        
        # 使用全局存储的PHQ分数计算五分类分布
        print(f"使用全局PHQ分数计算五分类分布，共{len(SUBJECT_PHQ_SCORES)}个subject的分数")
        for subject_id, utterance_count in subject_utterance_counts.items():
            if subject_id in SUBJECT_PHQ_SCORES:
                phq_score = SUBJECT_PHQ_SCORES[subject_id]
                sev = phq_score_to_severity(phq_score)
                if sev in sev_counts:
                    sev_counts[sev] += utterance_count
            else:
                print(f"Warning: Subject {subject_id} 没有PHQ分数")
        
        print(f"从CSV文件推断的五分类分布: {sev_counts}")
                
    except Exception as e:
        print(f"Warning: 统计原始训练集分布失败: {e}")
    return sev_counts, bin_counts

def compute_balanced_synthetic_targets(sev_orig: dict, bin_orig: dict) -> dict:
    """计算每个严重度需要生成的合成条数，使合成+原始满足：
    - 二分类总量严格平衡（bin0 == bin1）
    - 严重度在各自二分类组内相等（bin0内 normal==mild；bin1内 moderate==moderately_severe==severe）
    注：因为二分类组数量不同（2 vs 3），无法与"全五类完全相等"同时成立。本策略优先保证二分类平衡，并在各二分类组内五类均衡。
    """
    healthy_sevs = ["normal", "mild"]
    depressed_sevs = ["moderate", "moderately_severe", "severe"]

    # 各组内当前最大值
    max_healthy = max(sev_orig.get(s, 0) for s in healthy_sevs)
    max_depressed = max(sev_orig.get(s, 0) for s in depressed_sevs)

    # 选择最小的 k，使得 T0=3k >= max_healthy 且 T1=2k >= max_depressed，
    # 并满足二分类平衡：2*T0 == 3*T1（⇒ T0=3k, T1=2k）
    k = max(int(np.ceil(max_healthy / 3.0)), int(np.ceil(max_depressed / 2.0)))
    T0 = 3 * k  # bin0 内每个严重度的目标
    T1 = 2 * k  # bin1 内每个严重度的目标

    need = {s: 0 for s in SEVERITY_CLASSES}
    for s in healthy_sevs:
        need[s] = max(0, T0 - sev_orig.get(s, 0))
    for s in depressed_sevs:
        need[s] = max(0, T1 - sev_orig.get(s, 0))
    return need

def distribute_targets_to_plan(plan: list, needed_per_sev: dict) -> list:
    """将每个严重度需要的合成条数分配到对应的 subjects 上，返回更新后的 plan。
    策略：等分 + 余数前若干 subject 加 1；若某严重度 need 为 0，则该严重度下 target_sentences 置 0。
    """
    # 按严重度聚合 subject 索引
    sev_to_indices = {}
    for idx, it in enumerate(plan):
        sev = it.get("severity")
        if sev is None:
            continue
        sev_to_indices.setdefault(sev, []).append(idx)

    for sev in SEVERITY_CLASSES:
        idxs = sev_to_indices.get(sev, [])
        need = int(max(0, needed_per_sev.get(sev, 0)))
        if not idxs:
            continue
        base = need // len(idxs)
        rem = need % len(idxs)
        for j, i in enumerate(idxs):
            tgt = base + (1 if j < rem else 0)
            plan[i]["target_sentences"] = int(tgt)
    return plan



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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Batch TTS for multiple subjects with controllable dep/spk pairing")
    parser.add_argument("--num_subjects", type=int, default=400,
                        help="目标生成的subject总数（必须为偶数，将保证1/2抑郁+1/2健康）。不指定则采用全部可用subject（仅matched模式）")
    parser.add_argument("--dep_spk_pairs", type=str, default=None,
                        help="显式指定(depression_id:speaker_id)配对列表，逗号分隔，例如: 300:300,301:320,410:498。将覆盖 --num_subjects 选择逻辑。会强制类内一致与抑郁/健康数量平衡")
    return parser.parse_args()


def build_generation_plan(subject_to_phq8: dict,
                          depression_subject_to_idx: dict,
                          speaker_subject_to_idx: dict,
                          num_subjects: int | None,
                          dep_spk_pairs: str | None):
    """构建待生成条目计划。
    返回列表，元素为dict: {logical_subject_id, depression_id, speaker_id, class_label, pair_type}
    pair_type ∈ {"matched","mismatched"}
    logical_subject_id 用 depression_id（用于标签与输出文件夹命名保持兼容）。
    """
    available_ids = sorted(list(set(depression_subject_to_idx.keys()) & set(speaker_subject_to_idx.keys())))
    if not available_ids:
        raise RuntimeError("没有找到可用的subject交集")

    # 基于metadata获得class
    def get_class(sid: int) -> int:
        return int(subject_to_phq8.get(sid, 1))

    available_ids = [sid for sid in available_ids if sid in subject_to_phq8]
    if not available_ids:
        raise RuntimeError("metadata中没有这些subject的标签，请检查metadata.csv")

    depressed_ids = [sid for sid in available_ids if get_class(sid) == 1]
    healthy_ids = [sid for sid in available_ids if get_class(sid) == 0]

    plan = []

    # 模式A：显式配对
    if dep_spk_pairs:
        raw_pairs = [p.strip() for p in dep_spk_pairs.split(",") if p.strip()]
        pairs = []
        for p in raw_pairs:
            if ":" not in p:
                raise ValueError(f"非法pair格式: {p}. 期望形如 dep:spk")
            dep_str, spk_str = p.split(":", 1)
            dep_id = int(dep_str)
            spk_id = int(spk_str)
            if dep_id not in depression_subject_to_idx or spk_id not in speaker_subject_to_idx:
                raise ValueError(f"pair包含不可用id: {dep_id}:{spk_id}")
            if dep_id not in subject_to_phq8 or spk_id not in subject_to_phq8:
                raise ValueError(f"pair包含无标签id(请检查metadata): {dep_id}:{spk_id}")
            if get_class(dep_id) != get_class(spk_id):
                raise ValueError(f"pair类不一致(必须同为抑郁或健康): {dep_id}:{spk_id}")
            pairs.append((dep_id, spk_id))

        # 平衡性校验
        num_dep_class1 = sum(1 for dep_id, _ in pairs if get_class(dep_id) == 1)
        num_dep_class0 = sum(1 for dep_id, _ in pairs if get_class(dep_id) == 0)
        if num_dep_class1 != num_dep_class0:
            raise ValueError(f"显式pairs未达到类内平衡: depressed={num_dep_class1}, healthy={num_dep_class0}")

        # 检查combo唯一性
        combo_set = set()
        for dep_id, spk_id in pairs:
            combo = f"{spk_id}{dep_id}"
            if combo in combo_set:
                raise ValueError(f"显式pairs中存在重复combo: {combo} (dep={dep_id}, spk={spk_id})")
            combo_set.add(combo)

        # 仅保留 matched 配对；unmatched 将被忽略
        matched_pairs = [(d, s) for d, s in pairs if d == s]
        mismatched_pairs = [(d, s) for d, s in pairs if d != s]
        if len(mismatched_pairs) > 0:
            print(f"⚠️ 显式配对中包含 {len(mismatched_pairs)} 个 unmatched，已忽略")
        ordered_pairs = matched_pairs

        for dep_id, spk_id in ordered_pairs:
            plan.append({
                "logical_subject_id": dep_id,
                "depression_id": dep_id,
                "speaker_id": spk_id,
                "class_label": get_class(dep_id),
                "pair_type": "matched" if dep_id == spk_id else "mismatched",
            })

        return plan

    # 模式B：自动平衡选择
    if num_subjects is None:
        # 原逻辑：使用全部交集id，全部 matched
        for sid in available_ids:
            plan.append({
                "logical_subject_id": sid,
                "depression_id": sid,
                "speaker_id": sid,
                "class_label": get_class(sid),
                "pair_type": "matched",
            })
        return plan

    # 平衡与过采样
    if num_subjects % 2 != 0:
        raise ValueError("--num_subjects 必须为偶数，用于严格类内平衡")

    per_class = num_subjects // 2
    if len(depressed_ids) == 0 or len(healthy_ids) == 0:
        raise ValueError("某一类可用ID为0，无法构建平衡计划")

    if len(depressed_ids) < per_class:
        print(f"⚠️ 抑郁类不足，启用类内过采样: 需要 {per_class}，实际 {len(depressed_ids)}")
        selected_dep_class1 = [depressed_ids[i % len(depressed_ids)] for i in range(per_class)]
    else:
        selected_dep_class1 = depressed_ids[:per_class]

    if len(healthy_ids) < per_class:
        print(f"⚠️ 健康类不足，启用类内过采样: 需要 {per_class}，实际 {len(healthy_ids)}")
        selected_dep_class0 = [healthy_ids[i % len(healthy_ids)] for i in range(per_class)]
    else:
        selected_dep_class0 = healthy_ids[:per_class]

    def make_pairs_for_class_return_lists(selected_dep_ids: list[int], pool_ids_same_class: list[int], used_combos: set):
        matched_list = []
        mismatched_list = []

        # 前半: matched，后半: mismatched
        matched_count = (len(selected_dep_ids) + 1) // 2
        mismatched_count = len(selected_dep_ids) - matched_count

        matched_dep_ids = selected_dep_ids[:matched_count]
        mismatched_dep_ids = selected_dep_ids[matched_count:]

        # matched
        for dep_id in matched_dep_ids:
            combo = f"{dep_id}{dep_id}"
            if combo in used_combos:
                print(f"⚠️ 跳过重复combo: {combo} (dep={dep_id}, spk={dep_id})")
                continue
            used_combos.add(combo)
            matched_list.append({
                "logical_subject_id": dep_id,
                "depression_id": dep_id,
                "speaker_id": dep_id,
                "class_label": get_class(dep_id),
                "pair_type": "matched",
            })

        if mismatched_count > 0:
            candidates = [sid for sid in pool_ids_same_class if sid in speaker_subject_to_idx]
            candidates = list(dict.fromkeys(candidates))  # 去重并保持顺序
            if len(candidates) < 2:
                print("⚠️ 同类候选过少，无法构建mismatched，将退化为matched")
                for dep_id in mismatched_dep_ids:
                    combo = f"{dep_id}{dep_id}"
                    if combo in used_combos:
                        print(f"⚠️ 跳过重复combo: {combo} (dep={dep_id}, spk={dep_id})")
                        continue
                    used_combos.add(combo)
                    mismatched_list.append({
                        "logical_subject_id": dep_id,
                        "depression_id": dep_id,
                        "speaker_id": dep_id,
                        "class_label": get_class(dep_id),
                        "pair_type": "matched",
                    })
            else:
                # 为每个depression_id寻找唯一的speaker_id组合
                for dep_id in mismatched_dep_ids:
                    combo_found = False
                    # 尝试所有可能的speaker_id组合
                    for spk_id in candidates:
                        if spk_id == dep_id:
                            continue
                        combo = f"{spk_id}{dep_id}"
                        if combo not in used_combos:
                            used_combos.add(combo)
                            mismatched_list.append({
                                "logical_subject_id": dep_id,
                                "depression_id": dep_id,
                                "speaker_id": spk_id,
                                "class_label": get_class(dep_id),
                                "pair_type": "mismatched",
                            })
                            combo_found = True
                            break
                    
                    # 如果找不到唯一的mismatched组合，退化为matched
                    if not combo_found:
                        combo = f"{dep_id}{dep_id}"
                        if combo not in used_combos:
                            used_combos.add(combo)
                            mismatched_list.append({
                                "logical_subject_id": dep_id,
                                "depression_id": dep_id,
                                "speaker_id": dep_id,
                                "class_label": get_class(dep_id),
                                "pair_type": "matched",
                            })
                        else:
                            print(f"⚠️ 跳过重复combo: {combo} (dep={dep_id}, spk={dep_id})")

        return matched_list, mismatched_list

    # 使用全局combo集合来跟踪已使用的组合
    used_combos = set()
    
    matched_list_c1, mismatched_list_c1 = make_pairs_for_class_return_lists(selected_dep_class1, depressed_ids, used_combos)
    matched_list_c0, mismatched_list_c0 = make_pairs_for_class_return_lists(selected_dep_class0, healthy_ids, used_combos)

    # 仅生成 matched 组合（禁用 mismatched）
    plan.extend(matched_list_c1 + matched_list_c0)
    if mismatched_list_c1 or mismatched_list_c0:
        print("⚠️ 自动平衡模式中生成的 mismatched 组合已被禁用，未加入计划")

    # 最终验证combo唯一性
    final_combos = set()
    for item in plan:
        combo = f"{item['speaker_id']}{item['depression_id']}"
        if combo in final_combos:
            print(f"⚠️ 警告：最终计划中存在重复combo: {combo}")
        final_combos.add(combo)
    
    print(f"✓ 生成计划验证完成：{len(plan)}个条目，{len(final_combos)}个唯一combo")

    return plan


def build_generation_plan_by_severity(subject_to_phq8: dict,
                                      subject_to_severity: dict,
                                      depression_subject_to_idx: dict,
                                      speaker_subject_to_idx: dict):
    """基于严重度构建生成计划：使用所有可用且有严重度的subjects（不跳过 normal）；dep/spk均使用matched。
    返回列表元素包含: {logical_subject_id, depression_id, speaker_id, class_label, pair_type, severity, target_sentences}
    初始 target_sentences 置 0，后续由分配器覆盖。
    """
    available_ids = sorted(list(set(depression_subject_to_idx.keys()) & set(speaker_subject_to_idx.keys()) & set(subject_to_severity.keys())))
    if not available_ids:
        raise RuntimeError("没有找到同时具有embeddings与严重度信息的subject")

    plan = []

    for sid in available_ids:
        sev = subject_to_severity.get(sid)
        if sev is None:
            continue
        plan.append({
            "logical_subject_id": sid,
            "depression_id": sid,
            "speaker_id": sid,
            "class_label": int(subject_to_phq8.get(sid, 1)),
            "pair_type": "matched",
            "severity": sev,
            "target_sentences": 0,
        })

    print(f"✓ 严重度计划构建完成：总 {len(plan)} 个subject（包含 normal）")
    return plan


def validate_generation_plan(plan: list) -> bool:
    """验证生成计划的唯一性（按 combo+severity 组合）。"""
    print("验证生成计划...")
    key_set = set()
    duplicates = []
    for i, item in enumerate(plan):
        sev = str(item.get("severity", ""))
        key = f"{item['speaker_id']}{item['depression_id']}:{sev}"
        if key in key_set:
            duplicates.append((i, key, item))
        key_set.add(key)
    if duplicates:
        print("❌ 发现重复的 combo+severity 组合:")
        for idx, key, item in duplicates:
            print(f"  条目 {idx}: key={key} (dep={item['depression_id']}, spk={item['speaker_id']}, sev={item.get('severity')})")
        return False
    print(f"✓ 验证通过：{len(plan)} 个条目，{len(key_set)} 个唯一 combo+severity")
    return True


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
    max_needed: 需要的最大句子数（用于构建全局池）；None时使用MONO_SENTENCES_PER_SEVERITY
    """
    print("Extracting conversation responses from PersonaChat...")
    
    gather_target = max_needed if max_needed is not None else MONO_SENTENCES_PER_SEVERITY
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


@torch.inference_mode()
def synthesise_with_conditions(model: MatchaTTS, text: str, depression_embedding, speaker_id: int | None, temperature_override=None):
    """使用 depression 条件 + spk_id 进行合成（固定音色）。"""
    tex = process_text(text)

    if RESET_SEED_EACH_SENTENCE:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    # 准备条件输入 - 确保维度正确
    if isinstance(depression_embedding, np.ndarray):
        depression_embedding = depression_embedding.squeeze()

    depression_cond = torch.tensor(depression_embedding, dtype=torch.float32, device=device) if depression_embedding is not None else None
    if depression_cond is not None:
        depression_cond = _l2norm(depression_cond)

    if not USE_DEPRESSION_COND:
        depression_cond = None

    if depression_cond is not None:
        depression_cond = depression_cond.squeeze()
        if depression_cond.dim() == 0:
            depression_cond = depression_cond.unsqueeze(0)
        depression_cond = depression_cond.unsqueeze(0)

    # 构造 spk 索引（与训练一致）
    spk_tensor = None
    if hasattr(model, "n_spks") and int(model.n_spks) > 1 and speaker_id is not None:
        try:
            mapping = GLOBAL_SUBJECT_TO_SPK_IDX
            if mapping is None:
                raise KeyError("GLOBAL_SUBJECT_TO_SPK_IDX is None")
            spk_idx = mapping[int(speaker_id)]
            spk_tensor = torch.tensor([spk_idx], dtype=torch.long, device=device)
            try:
                map_size = len(mapping) if isinstance(mapping, dict) else "N/A"
                back_sid = None
                try:
                    for _sid, _idx in mapping.items():
                        if _idx == spk_idx:
                            back_sid = _sid
                            break
                except Exception:
                    back_sid = None
                print(f"    [SPK-DEBUG] speaker_id={int(speaker_id)} -> spk_idx={int(spk_idx)} (n_spks={int(model.n_spks)}, map_size={map_size}, reverse_check={back_sid})")
            except Exception:
                pass
        except Exception:
            try:
                nspk = int(model.n_spks)
                raw_sid = int(speaker_id)
                fallback_idx = int(raw_sid % nspk)
            except Exception:
                nspk = int(model.n_spks) if hasattr(model, 'n_spks') else 1
                fallback_idx = 0
            spk_tensor = torch.tensor([fallback_idx], dtype=torch.long, device=device)
            try:
                print(f"    [SPK-DEBUG] speaker_id={speaker_id} 未在映射表中，使用兜底索引 spk_idx={fallback_idx}（n_spks={nspk}）")
            except Exception:
                pass
    else:
        try:
            print(f"    [SPK-DEBUG] 未启用多说话人路径（n_spks={getattr(model,'n_spks','N/A')}, speaker_id={speaker_id}）")
        except Exception:
            pass

    current_temperature = temperature_override if temperature_override is not None else TEMPERATURE

    synth_kwargs = dict(
        n_timesteps=N_TIMESTEPS,
        temperature=current_temperature,
        length_scale=LENGTH_SCALE,
        depression_cond=depression_cond,
        speaker_cond=None,
    )
    if spk_tensor is not None:
        synth_kwargs["spks"] = spk_tensor

    out = model.synthesise(
        tex["x"], tex["x_lengths"],
        **synth_kwargs,
    )

    out.update({**tex})
    return out


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
    print("=== 新Matcha-TTS模型批量所有Subject语音合成 (PersonaChat数据集) ===")
    
    # 解析命令行参数
    args = parse_args()
    target_total = args.num_subjects if args.num_subjects is not None else TARGET_TOTAL_SUBJECTS
    dep_spk_pairs = args.dep_spk_pairs
    
    # 加载PHQ分数
    load_phq_scores_from_csvs()
    subject_to_phq8 = load_metadata()
    
    # 加载embeddings
    (depression_embeddings, _speaker_embeddings, depression_subject_ids, 
     _speaker_subject_ids, depression_subject_to_idx, speaker_subject_to_idx) = load_embeddings()

    # 将训练期 subject→spk 映射设置为全局变量，供合成阶段使用
    try:
        global GLOBAL_SUBJECT_TO_SPK_IDX
        GLOBAL_SUBJECT_TO_SPK_IDX = speaker_subject_to_idx
        print(f"✓ GLOBAL_SUBJECT_TO_SPK_IDX set: size={len(GLOBAL_SUBJECT_TO_SPK_IDX) if isinstance(GLOBAL_SUBJECT_TO_SPK_IDX, dict) else 'N/A'}")
    except Exception:
        pass

    # 构建"单调验证"生成计划（每 subject × 五严重度 × 20 句）
    try:
        if dep_spk_pairs is not None and len(str(dep_spk_pairs).strip()) > 0:
            print("使用显式 dep:spk 配对作为基础计划")
            base_plan = build_generation_plan(
                subject_to_phq8, depression_subject_to_idx,
                speaker_subject_to_idx, target_total, dep_spk_pairs
            )
        else:
            print("使用所有可用交集 subject（matched）作为基础计划")
            base_plan = build_generation_plan(
                subject_to_phq8, depression_subject_to_idx,
                speaker_subject_to_idx, None, None
            )

        generation_plan = []
        for it in base_plan:
            for sev in SEVERITY_CLASSES:
                expanded = dict(it)
                expanded["severity"] = sev
                expanded["target_sentences"] = MONO_SENTENCES_PER_SEVERITY
                generation_plan.append(expanded)

        print(f"✓ 单调验证计划构建完成：基础 subjects={len(base_plan)}，扩展条目={len(generation_plan)}")
    except Exception as e:
        print(f"❌ 构建单调验证计划失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 验证计划
    if not validate_generation_plan(generation_plan):
        print("❌ 生成计划验证失败")
        return
    
    # 精简计划日志（不再计算均衡/需求）
    print(f"\n{'='*80}")
    print(f"📊 单调验证计划")
    print(f"{'='*80}")
    plan_log_file = Path(OUTPUT_FOLDER_BASE) / "generation_plan_monotonic.json"
    plan_log_file.parent.mkdir(parents=True, exist_ok=True)
    sev_subjects = {sev: len(set(it["logical_subject_id"] for it in generation_plan if it.get("severity") == sev)) for sev in SEVERITY_CLASSES}
    sev_target_utterances = {sev: sum(int(it.get("target_sentences", 0)) for it in generation_plan if it.get("severity") == sev) for sev in SEVERITY_CLASSES}
    plan_log_data = {
        "timestamp": dt.datetime.now().isoformat(),
        "mode": "monotonic_severity",
        "severities": SEVERITY_CLASSES,
        "per_severity_sentences": MONO_SENTENCES_PER_SEVERITY,
        "base_subjects": len(set(it["logical_subject_id"] for it in generation_plan)),
        "expanded_items": len(generation_plan),
        "five_class_subjects": sev_subjects,
        "five_class_target_utterances": sev_target_utterances,
        "generation_plan_sample": generation_plan[:5],
            "sampling_config": {
                "sampling_enabled": False,
                "num_samples": 1,
                "similarity_selection": None,
                "temperature": TEMPERATURE
            }
    }
    with open(plan_log_file, "w", encoding="utf-8") as f:
        json.dump(plan_log_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 单调验证计划已保存到: {plan_log_file}")
    print(f"{'='*80}")
    
    print(f"✓ 将按计划使用对应的 embeddings（仅 matched，dep==spk）")
    print(f"  Depression embeddings shape: {depression_embeddings.shape}")
    
    # 加载PersonaChat数据集
    dataset = load_personachat_dataset()
    print(f"✓ PersonaChat dataset loaded with {len(dataset)} samples")
    
    # 准备全局句子池：所有 subject 与严重度共享同一组 20 句
    dataset_needed = int(MONO_SENTENCES_PER_SEVERITY)
    print(f"\n📝 准备共享句子池，共需 {dataset_needed} 句（所有 subject/severity 复用）")
    global_sentences_pool = extract_conversation_responses(dataset, max_needed=dataset_needed)
    
    if len(global_sentences_pool) < dataset_needed:
        print(f"⚠️  句子池不足：仅有 {len(global_sentences_pool)}/{dataset_needed} 条，将尽力继续")
    else:
        print(f"✓ 句子池就绪：{len(global_sentences_pool)} 条")
    
    # 打印前几个句子作为示例
    print("  示例句子:")
    for i, sentence in enumerate(global_sentences_pool[:5]):
        print(f"    {i+1}. {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
    
    # 构建 Clinical-5 原型库（基于 PHQ_Score 和二分类信息）
    try:
        # 需要 subject_to_score 与二分类映射 subject_to_phq8
        subject_to_score, subject_to_severity = load_phq_scores()
    except Exception:
        subject_to_score, subject_to_severity = ({}, {})
    dep_bank = build_dep_bank_clinical5(
        depression_embeddings,
        depression_subject_ids,
        subject_to_score
    )
    print(f"✓ Clinical-5 bank ready: {['normal','mild','moderate','mod_severe','severe']}")
    # 诊断打印：模型空间原型相似度与alpha曲线
    try:
        print_bank_diagnostics(dep_bank, name="model-space")
    except Exception:
        pass
    # 若后续存在抑郁音频提取器，将再构建一个"音频空间"的原型库以对齐相似度空间
    dep_bank_audio = None
    bank_alignment_metrics = None

    # 加载模型
    model = load_matcha_model(MATCHA_CKPT)
    vocoder = load_hifigan(HIFIGAN_WEIGHT, HIFIGAN_CONFIG)
    denoiser = Denoiser(vocoder, mode="zeros")
    
    # 采样已禁用：不再加载抑郁音频嵌入提取器和音频空间bank
    print("✓ 采样已禁用：每个样本只生成一次，不进行相似度匹配")
    depression_extractor = None
    dep_bank_audio = None
    bank_alignment_metrics = None

    # 为每个subject生成语音
    total_successful = 0
    total_attempted = 0
    sentence_ptr = 0  # 在全局句子池中的指针
    
    for subject_idx, item in enumerate(generation_plan):
        logical_subject_id = int(item["logical_subject_id"])  # depression_id（用于统计）
        depression_id = int(item["depression_id"])
        speaker_id = int(item["speaker_id"])
        class_label = int(item["class_label"])                # 0/1
        pair_type = str(item["pair_type"])                    # matched/mismatched

        # 组合ID = speaker_id + depression_id（无分隔符）
        combo_id = f"{speaker_id}{depression_id}"

        print(f"\n{'='*60}")
        print(f"处理subject {subject_idx + 1}/{len(generation_plan)}: combo={combo_id} (dep={depression_id}, spk={speaker_id}, class={class_label}, {pair_type})")
        print(f"{'='*60}")
        
        # 确定当前条目的严重度与输出目录（组合/严重度子目录）
        sev_log = item.get("severity", None)
        if sev_log is None:
            sev_log = subject_to_severity.get(depression_id, "moderate") if 'subject_to_severity' in locals() else "moderate"
        sev_dir = severity_to_dirname(sev_log)
        # 输出目录：基础组合目录 + 严重度子目录
        subject_folder = Path(OUTPUT_FOLDER_BASE) / combo_id
        output_folder = subject_folder / sev_dir
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"输出文件夹: {output_folder}")
        
        # 日志文件（若不存在则写表头）
        log_file = output_folder / "processing_log.txt"
        if not log_file.exists():
            with open(log_file, "w") as log:
                log.write("Combo_ID,Depression_ID,Speaker_ID,Pair_Type,PHQ8_Binary,Sentence_ID,Text,Audio_File,Status,Similarity_Score,Sample_ID,Is_Sampled\n")
        
        # 为这个subject生成语音
        successful_generations = 0

        # 句子编号续接：在严重度子目录下扫描已存在的 combo_id_*.wav
        existing = list(output_folder.glob(f"{combo_id}_*.wav"))
        max_idx = 0
        for p in existing:
            try:
                stem = p.stem  # e.g., "300467_19"
                parts = stem.split("_")
                if len(parts) >= 2 and parts[0] == combo_id:
                    idx = int(parts[1])
                    if idx > max_idx:
                        max_idx = idx
            except Exception:
                pass
        sentence_counter = max_idx + 1

        # 使用 spk_id 控制音色（不再依赖 speaker embedding 向量）
        current_speaker_emb = None

        alpha_log = severity_to_alpha(sev_log)
        dep_vec_log = dep_from_alpha_over_bank(dep_bank, alpha_log)
        
        print(f"  Subject combo={combo_id} embeddings:")
        print(f"    Depression embedding shape: {dep_vec_log.shape}")
        
        # 使用共享的固定 20 句
        subject_sentences = list(global_sentences_pool)

        with tqdm(total=len(subject_sentences), desc=f"Generating subject {subject_idx+1}/{len(generation_plan)} combo={combo_id}") as pbar:
            for _, sentence_text in enumerate(subject_sentences):
                try:
                    total_attempted += 1
                    
                    # 计算严重度 alpha，并从原型库得到抑郁条件向量
                    sev = item.get("severity", None)
                    if sev is None:
                        # 若计划未携带严重度，则尝试用 subject_to_severity；否则默认 moderate
                        sev = subject_to_severity.get(depression_id, "moderate") if 'subject_to_severity' in locals() else "moderate"
                    alpha = severity_to_alpha(sev)
                    # 条件向量在"模型空间"
                    dep_vec = dep_from_alpha_over_bank(dep_bank, alpha)

                    # 单次生成（不进行采样和相似度匹配）
                    out = synthesise_with_conditions(
                        model, sentence_text, 
                        dep_vec, speaker_id
                    )
                    waveform = mel_to_waveform(out["mel"], vocoder, denoiser)
                    similarity_score = 1.0
                    sample_id = 0
                    is_sampled = False
                    
                    # 保存（命名用 combo_id；label 查表用 depression_id）
                    audio_filename = save_audio_and_label(combo_id, sentence_counter, waveform, output_folder, subject_to_phq8, label_subject_id=depression_id)
                    
                    successful_generations += 1
                    total_successful += 1
                    pbar.update(1)
                    
                    # PHQ8 基于 depression_id
                    phq8_binary = subject_to_phq8.get(depression_id, 1)
                    with open(log_file, "a") as log:
                        log.write(f"{combo_id},{depression_id},{speaker_id},{pair_type},{phq8_binary},{sentence_counter},\"{sentence_text}\",{audio_filename},Success,{similarity_score:.4f},{sample_id},{is_sampled}\n")
                    
                    sentence_counter += 1
                    
                    if successful_generations % 10 == 0:
                        print(f"    📈 Generated {successful_generations}/{len(subject_sentences)} sentences")
                        
                except Exception as e:
                    print(f"Error generating sentence: {e}")
                    phq8_binary = subject_to_phq8.get(depression_id, 1)
                    with open(log_file, "a") as log:
                        log.write(f"{combo_id},{depression_id},{speaker_id},{pair_type},{phq8_binary},{sentence_counter},\"{sentence_text}\",N/A,Failed: {str(e)},0.0,-1,False\n")
                    sentence_counter += 1
                    continue
                        
        # 统计文件按组合+严重度命名，避免覆盖
        sev_name = str(item.get("severity", "n_a")).replace("/", "-")
        stats_file = output_folder / f"generation_stats_dep{depression_id}_spk{speaker_id}_{pair_type}_sev-{sev_name}.json"
        phq8_binary = subject_to_phq8.get(depression_id, 1)
        with open(stats_file, "w") as f:
            json.dump({
                "combo_id": combo_id,
                "logical_subject_id": logical_subject_id,
                "depression_id": depression_id,
                "speaker_id": speaker_id,
                "phq8_binary": phq8_binary,
                "pair_type": pair_type,
                "embedding_usage": "matched" if pair_type == "matched" else "mismatched_same_class",
                "severity": item.get("severity", "n/a"),
                "target_sentences": int(item.get("target_sentences", MONO_SENTENCES_PER_SEVERITY)),
                "successful_generations": successful_generations,
                "success_rate": successful_generations/len(subject_sentences) if len(subject_sentences) > 0 else 0,
                "data_source": "PersonaChat_conversations",
                "quality_filter": None,
                "model_config": {
                    "n_timesteps": N_TIMESTEPS,
                    "temperature": TEMPERATURE,
                    "length_scale": LENGTH_SCALE
                },

                "embeddings_info": {
                    "current_depression_embedding_shape": list(dep_vec_log.shape)
                },
            "sampling_config": {
                "sampling_enabled": False,
                "num_samples": 1,
                "similarity_selection": None,
                "temperature": TEMPERATURE
            },
                "prototype_alignment": None
            }, f, indent=2)

        print(f"✅ Subject combo={combo_id} 完成: {successful_generations}/{len(subject_sentences)} 成功 ({pair_type})")    
    # 保存总体统计信息
    overall_stats_file = Path(OUTPUT_FOLDER_BASE) / "overall_generation_stats.json"
    
    # 统计PHQ8_Binary分布（按 logical_subject_id）
    phq8_distribution = {}
    for item in generation_plan:
        phq8_binary = subject_to_phq8.get(int(item["logical_subject_id"]), 1)
        phq8_distribution[phq8_binary] = phq8_distribution.get(phq8_binary, 0) + 1
    
    with open(overall_stats_file, "w") as f:
        json.dump({
            "total_subjects": len(generation_plan),
            "total_target_sentences": len(generation_plan) * MONO_SENTENCES_PER_SEVERITY,
            "total_attempted": total_attempted,
            "total_successful": total_successful,
            "overall_success_rate": total_successful / total_attempted if total_attempted > 0 else 0,
            "data_source": "PersonaChat_conversations",
            "plan_summary": {
                "matched": sum(1 for it in generation_plan if it["pair_type"] == "matched"),
                "mismatched": sum(1 for it in generation_plan if it["pair_type"] == "mismatched"),
            },
            "severity_distribution": {
                "mild": sum(1 for it in generation_plan if it.get("severity") == "mild"),
                "moderate": sum(1 for it in generation_plan if it.get("severity") == "moderate"),
                "moderately_severe": sum(1 for it in generation_plan if it.get("severity") == "moderately_severe"),
                "severe": sum(1 for it in generation_plan if it.get("severity") == "severe"),
            },
            "quality_filter": None,
            "generation_plan": generation_plan,
            "phq8_binary_distribution": phq8_distribution,
            "sample_sentences": global_sentences_pool[:10],  # 保存前10个句子作为示例
            "model_config": {
                "n_timesteps": N_TIMESTEPS,
                "temperature": TEMPERATURE,
                "length_scale": LENGTH_SCALE
            },
            
            "sampling_config": {
                "sampling_enabled": False,
                "num_samples": 1,
                "similarity_selection": None,
                "temperature": TEMPERATURE
            },
            "prototype_alignment": None
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"=== 批量合成完成 (PersonaChat数据集) ===")
    print(f"总subject数: {len(generation_plan)}")
    print(f"总目标句子数: {len(generation_plan) * MONO_SENTENCES_PER_SEVERITY}")
    print(f"总尝试数: {total_attempted}")
    print(f"总成功数: {total_successful}")
    print(f"总体成功率: {total_successful/total_attempted*100:.1f}%" if total_attempted > 0 else "0%")
    print(f"数据来源: PersonaChat对话回复")
    print(f"Embedding使用方式: 仅 matched（dep==spk）")
    print(f"采样模式: 禁用（每个样本只生成一次）")
    if bank_alignment_metrics:
        mean_cos = bank_alignment_metrics.get("mean")
        min_cos = bank_alignment_metrics.get("min")
        max_cos = bank_alignment_metrics.get("max")
        neg_cnt = bank_alignment_metrics.get("negative_count")
        print(f"原型对齐: mean={mean_cos:.4f}, range=({min_cos:.4f}, {max_cos:.4f}), negatives={neg_cnt}")
        if neg_cnt and neg_cnt > 0:
            print("  建议：引入多目标打分（如F0/能量/ASR）或重新对齐音频空间嵌入。")
    else:
        print("原型对齐: 未构建音频空间原型，跳过对齐分析")
    # 音频质量筛选：已完全移除
    print(f"PHQ8_Binary分布: {phq8_distribution}")
    print(f"输出基础文件夹: {OUTPUT_FOLDER_BASE}")
    print(f"总体统计信息: {overall_stats_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()