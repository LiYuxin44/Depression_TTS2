import datetime as dt
import json
import re
from pathlib import Path
import random
import glob
import os
import argparse

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn

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
MATCHA_CKPT = "/home/i-liyuxin/Depression_TTS/logs/train_daic_utter/decouple_no_adapter/runs/2025-09-09_19-57-22/checkpoints/checkpoint_epoch=199.ckpt"
HIFIGAN_WEIGHT = "/home/i-liyuxin/Depression_TTS/ckpts/VCTK_V1/generator_v1.pth"
HIFIGAN_CONFIG = "/home/i-liyuxin/Depression_TTS/ckpts/VCTK_V1/config.json"
# 输出文件夹将根据subject ID动态生成
OUTPUT_FOLDER_BASE = "/home/i-liyuxin/Depression_TTS/synthesis_no_adapter"

# 合成参数
N_TIMESTEPS = 20
LENGTH_SCALE = 1.0
TEMPERATURE = 0  # 设为0以获得确定性输出（如需少量随机性可调到0.1~0.2）

# 🔥 生成配置 - 每个subject生成的句子数量
SENTENCES_PER_SUBJECT = 20  

# 音频处理参数
AUDIO_GAIN = 1.5        # 音量增益倍数，增大音量（建议范围：2.0-5.0）
NORMALIZE_AUDIO = False   # 是否进行音频归一化
TARGET_RMS = 0        # 目标RMS值，用于归一化（建议范围：0.05-0.2）


MAX_GENERATION_ATTEMPTS = 1    # 每个句子只生成一次
QUALITY_LOG_ENABLED = False    # deprecated: 不再进行质量筛选

# 文本筛选参数（用于去掉太短的文本）
TEXT_MIN_CHARS = 20     # 最小字符数（清洗后计算）
TEXT_MIN_WORDS = 5      # 最少词数
TEXT_MAX_CHARS = 2000    # 最大字符数上限

# 采样与稳定性控制
RESET_SEED_EACH_SENTENCE = False  # 每句前重置随机种子，进一步消除随机性
USE_DEPRESSION_COND = True
USE_SPEAKER_COND = True

# 嵌入处理
EMBED_NORM = "l2"                # 选项: "l2" | "zscore" | "none"
EMBED_EPS = 1e-6

DEPRESSION_MEAN = None
DEPRESSION_STD = None
SPEAKER_MEAN = None
SPEAKER_STD = None

# 加载subject和embeddings信息
DEPRESSION_EMBEDDINGS_FILE = "/home/i-liyuxin/Depression_TTS/embeddings/GDST_embeddings/train_embeddings.npz"
SPEAKER_EMBEDDINGS_FILE = "/home/i-liyuxin/Depression_TTS/embeddings/speaker_embeddings/train_speaker_embeddings.npz"

# 添加metadata文件路径
METADATA_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata.csv"
# 用于严重度与PHQ_Score（0-24）的元数据文件
PHQ_SCORE_METADATA_FILE = "/home/i-liyuxin/Depression_TTS/matcha/data/metadata_with_phq.csv"

# 目标 subject 数量（可通过命令行 --num_subjects 覆盖）。
# None 表示使用全部可用 subject（与原有逻辑一致）。
TARGET_TOTAL_SUBJECTS = None

# HiFi-GAN 期望的 mel 维度（从其 config.json 读取），用于在声码器前进行维度/形状自适配
VOCODER_INPUT_DIMS = None


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


# 严重度到每subject目标句子数
# 原始训练集目录（用于统计原始分布）
ORIG_TRAIN_DIR = "/home/i-liyuxin/test/daic_5cv_preprocessed_20_46/train_mode"

# 是否启用"合成+原始均衡（二分类与五分类同时尽量均衡）"
BALANCE_WITH_ORIGINAL = True

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

def count_original_distribution(train_dir: str) -> tuple[dict, dict]:
    """扫描原始训练集，统计五分类与二分类的条数（按utterance/clip计）。
    - 五分类来自 .phq_label（PHQ_Score）转换
    - 二分类来自 .label（0/1）
    """
    sev_counts = {s: 0 for s in SEVERITY_CLASSES}
    bin_counts = {0: 0, 1: 0}
    try:
        for fn in os.listdir(train_dir):
            if not fn.endswith('.wav'):
                continue
            base = os.path.join(train_dir, fn[:-4])
            lbl_file = base + '.label'
            phq_file = base + '.phq_label'
            # 二分类
            if os.path.exists(lbl_file):
                try:
                    with open(lbl_file) as f:
                        b = int(str(f.read()).strip())
                        if b in bin_counts:
                            bin_counts[b] += 1
                except Exception:
                    pass
            # 五分类（通过PHQ_Score）
            if os.path.exists(phq_file):
                try:
                    with open(phq_file) as f:
                        sc = float(str(f.read()).strip())
                    sev = phq_score_to_severity(sc)
                    if sev in sev_counts:
                        sev_counts[sev] += 1
                except Exception:
                    pass
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
    """加载depression和speaker embeddings"""
    print("Loading embeddings...")
    
    # 加载depression embeddings - 使用与训练时相同的映射方式
    depression_data = np.load(DEPRESSION_EMBEDDINGS_FILE)
    depression_embeddings = depression_data['embeddings'] if 'embeddings' in depression_data else depression_data['arr_0']
    depression_subject_ids = depression_data['subject_ids'] if 'subject_ids' in depression_data else None
    
    # 加载speaker embeddings - 使用与训练时相同的映射方式
    speaker_data = np.load(SPEAKER_EMBEDDINGS_FILE)
    speaker_embeddings = speaker_data['embeddings'] if 'embeddings' in speaker_data else speaker_data['arr_0']
    speaker_subject_ids = speaker_data['subject_ids'] if 'subject_ids' in speaker_data else None
    
    print(f"✓ Depression embeddings shape: {depression_embeddings.shape}")
    print(f"✓ Speaker embeddings shape: {speaker_embeddings.shape}")
    
    # 创建subject ID到索引的映射（与训练时一致）
    depression_subject_to_idx = {}
    speaker_subject_to_idx = {}
    
    if depression_subject_ids is not None:
        depression_subject_ids = depression_subject_ids.astype(int)
        depression_subject_to_idx = {int(sid): idx for idx, sid in enumerate(depression_subject_ids)}
        print(f"✓ Depression subject mapping: {len(depression_subject_to_idx)} subjects")
        print(f"  Available subjects: {sorted(list(depression_subject_to_idx.keys()))[:10]}...")
    
    if speaker_subject_ids is not None:
        speaker_subject_ids = speaker_subject_ids.astype(int)
        speaker_subject_to_idx = {int(sid): idx for idx, sid in enumerate(speaker_subject_ids)}
        print(f"✓ Speaker subject mapping: {len(speaker_subject_to_idx)} subjects")
        print(f"  Available subjects: {sorted(list(speaker_subject_to_idx.keys()))[:10]}...")
    
    # 计算全局统计量（用于zscore归一化）
    global DEPRESSION_MEAN, DEPRESSION_STD, SPEAKER_MEAN, SPEAKER_STD
    try:
        DEPRESSION_MEAN = depression_embeddings.mean(axis=0)
        DEPRESSION_STD = depression_embeddings.std(axis=0) + 1e-6
        SPEAKER_MEAN = speaker_embeddings.mean(axis=0)
        SPEAKER_STD = speaker_embeddings.std(axis=0) + 1e-6
        print("✓ Computed global embedding statistics (for zscore normalization)")
    except Exception:
        pass
    
    return (
        depression_embeddings,
        speaker_embeddings,
        depression_subject_ids,
        speaker_subject_ids,
        depression_subject_to_idx,
        speaker_subject_to_idx,
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

        # 顺序：matched 在前，mismatched 在后
        matched_pairs = [(d, s) for d, s in pairs if d == s]
        mismatched_pairs = [(d, s) for d, s in pairs if d != s]
        ordered_pairs = matched_pairs + mismatched_pairs

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

    # 全局顺序：所有 matched 在前，然后所有 mismatched
    plan.extend(matched_list_c1 + matched_list_c0)
    plan.extend(mismatched_list_c1 + mismatched_list_c0)

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
    """验证生成计划的combo唯一性"""
    print("验证生成计划...")
    
    combo_set = set()
    duplicate_combos = []
    
    for i, item in enumerate(plan):
        combo = f"{item['speaker_id']}{item['depression_id']}"
        if combo in combo_set:
            duplicate_combos.append((i, combo, item))
        combo_set.add(combo)
    
    if duplicate_combos:
        print("❌ 发现重复的combo:")
        for idx, combo, item in duplicate_combos:
            print(f"  条目 {idx}: combo={combo} (dep={item['depression_id']}, spk={item['speaker_id']})")
        return False
    
    print(f"✓ 验证通过：{len(plan)}个条目，{len(combo_set)}个唯一combo")
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


@torch.inference_mode()
def synthesise_with_conditions(model: MatchaTTS, text: str, depression_embedding, speaker_embedding):
    """使用depression和speaker条件进行合成"""
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
    if isinstance(speaker_embedding, np.ndarray):
        speaker_embedding = speaker_embedding.squeeze()
    
    # 转换为tensor
    depression_cond = torch.tensor(depression_embedding, dtype=torch.float32, device=device) if depression_embedding is not None else None
    speaker_cond = torch.tensor(speaker_embedding, dtype=torch.float32, device=device) if speaker_embedding is not None else None
    
    # 嵌入归一化 - 使用统一的_l2norm函数
    if EMBED_NORM == "l2":
        if depression_cond is not None:
            depression_cond = _l2norm(depression_cond - depression_cond.mean())
        if speaker_cond is not None:
            speaker_cond = _l2norm(speaker_cond - speaker_cond.mean())
    
    # 关闭某些条件（可选）
    if not USE_DEPRESSION_COND:
        depression_cond = None
    if not USE_SPEAKER_COND:
        speaker_cond = None
    
    # 添加batch维度 - 确保维度正确 [batch_size, feature_dim]
    if depression_cond is not None:
        depression_cond = depression_cond.squeeze()
        if depression_cond.dim() == 0:
            depression_cond = depression_cond.unsqueeze(0)
        depression_cond = depression_cond.unsqueeze(0)
        
    if speaker_cond is not None:
        speaker_cond = speaker_cond.squeeze()
        if speaker_cond.dim() == 0:
            speaker_cond = speaker_cond.unsqueeze(0)
        speaker_cond = speaker_cond.unsqueeze(0)

    out = model.synthesise(
        tex["x"], tex["x_lengths"],
        n_timesteps=N_TIMESTEPS,
        temperature=TEMPERATURE,
        length_scale=LENGTH_SCALE,
        depression_cond=depression_cond,
        speaker_cond=speaker_cond,
    )

    out.update({**tex})
    return out


@torch.inference_mode()
def mel_to_waveform(mel, vocoder: HiFiGAN, denoiser: Denoiser):
    """mel 谱转换为波形 - 增加音量增益"""
    try:
        # 形状与维度自检与适配
        expected_mels = VOCODER_INPUT_DIMS
        if expected_mels is not None:
            mel = adapt_mel_dimensions(mel, expected_mels)
        else:
            # 没有读到配置时，也保证标准形状 [B, n_mels, T] 且为 float32
            if mel.dtype != torch.float32:
                mel = mel.to(torch.float32)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

        # 调试输出，帮助定位"糊"的根因（维度/顺序）
        print(f"    Vocoder mel shape: {tuple(mel.shape)}, dtype: {mel.dtype}")

        # 与 notebook 保持一致：直接送入 vocoder
        audio = vocoder(mel).clamp(-1, 1)
        audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
        
        # 应用音量增益
        audio = audio * AUDIO_GAIN
        
        # 音频归一化（可选）
        if NORMALIZE_AUDIO:
            # 计算当前RMS
            current_rms = torch.sqrt(torch.mean(audio**2))
            if current_rms > 0:
                # 归一化到目标RMS
                normalize_factor = TARGET_RMS / current_rms
                audio = audio * normalize_factor
        
        # 重新限制在[-1, 1]范围内，避免削波
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
    print("按PHQ严重度动态分配每subject句子数：normal=0, mild=18, moderate=7, moderately_severe=32, severe=163")
    
    # 解析命令行参数
    args = parse_args()
    target_total = args.num_subjects if args.num_subjects is not None else TARGET_TOTAL_SUBJECTS
    dep_spk_pairs = args.dep_spk_pairs
    
    # 加载metadata
    subject_to_phq8 = load_metadata()
    
    # 加载subject和embeddings
    depression_embeddings, speaker_embeddings, depression_subject_ids, speaker_subject_ids, depression_subject_to_idx, speaker_subject_to_idx = load_embeddings()

    # 基于严重度构建生成计划（若显式指定dep_spk_pairs，则沿用旧逻辑）
    try:
        if dep_spk_pairs is not None and len(str(dep_spk_pairs).strip()) > 0:
            print("使用显式dep:spk配对列表构建计划（忽略严重度策略）")
            generation_plan = build_generation_plan(
                subject_to_phq8,
                depression_subject_to_idx,
                speaker_subject_to_idx,
                target_total,
                dep_spk_pairs,
            )
            # 显式模式下，默认每subject仍按固定值，用字段对齐
            for it in generation_plan:
                it.setdefault("severity", "explicit")
                it.setdefault("target_sentences", SENTENCES_PER_SUBJECT)
        else:
            # 加载PHQ_Score并映射严重度
            subject_to_score, subject_to_severity = load_phq_scores()
            generation_plan = build_generation_plan_by_severity(
                subject_to_phq8,
                subject_to_severity,
                depression_subject_to_idx,
                speaker_subject_to_idx,
            )
            if BALANCE_WITH_ORIGINAL:
                # 统计原始训练集分布（避免重复计算）
                sev_orig, bin_orig = None, None
                if BALANCE_WITH_ORIGINAL:
                    sev_orig, bin_orig = count_original_distribution(ORIG_TRAIN_DIR)
                    print(f"\n📝 原始训练集分布:")
                    print(f"  五分类分布: {sev_orig}")
                    print(f"  二分类分布: {bin_orig}")
                    print(f"  原始总utterance数: {sum(sev_orig.values())}")
                # 计算每严重度合成所需数
                need_per_sev = compute_balanced_synthetic_targets(sev_orig, bin_orig)
                print(f"为均衡目标计算的每严重度需要的合成条数: {need_per_sev}")
                # 将目标条数分配到各subject
                generation_plan = distribute_targets_to_plan(generation_plan, need_per_sev)
    except Exception as e:
        print(f"❌ 构建生成计划失败: {e}")
        return

    print(f"✓ 生成计划条目数: {len(generation_plan)}")
    for i, item in enumerate(generation_plan[:5]):
        sev = item.get('severity', 'n/a')
        tgt = item.get('target_sentences', SENTENCES_PER_SUBJECT)
        print(f"  [{i+1}] logical_id={item['logical_subject_id']}, dep={item['depression_id']}, spk={item['speaker_id']}, class={item['class_label']}, type={item['pair_type']}, severity={sev}, target={tgt}")
    
    # 验证生成计划的combo唯一性
    if not validate_generation_plan(generation_plan):
        print("❌ 生成计划验证失败，程序退出")
        return
    
    # 详细计划分析和日志记录
    print(f"\n{'='*80}")
    print(f"📊 生成计划详细分析")
    print(f"{'='*80}")
    
    # 统计原始数据分布
    if BALANCE_WITH_ORIGINAL:
        sev_orig, bin_orig = count_original_distribution(ORIG_TRAIN_DIR)
        print(f"\n📝 原始训练集分布:")
        print(f"  五分类分布: {sev_orig}")
        print(f"  二分类分布: {bin_orig}")
        print(f"  原始总utterance数: {sum(sev_orig.values())}")
    
    # 统计计划中的subject分布
    sev_subjects = {}
    sev_target_utterances = {}
    bin_subjects = {0: 0, 1: 0}
    bin_target_utterances = {0: 0, 1: 0}
    
    for item in generation_plan:
        sev = item.get("severity", "unknown")
        target = int(item.get("target_sentences", 0))
        class_label = int(item.get("class_label", 1))
        
        # 五分类统计
        sev_subjects[sev] = sev_subjects.get(sev, 0) + 1
        sev_target_utterances[sev] = sev_target_utterances.get(sev, 0) + target
        
        # 二分类统计
        bin_subjects[class_label] += 1
        bin_target_utterances[class_label] += target
    
    print(f"\n📋 合成计划分布:")
    print(f"  五分类subject数: {sev_subjects}")
    print(f"  五分类目标utterance数: {sev_target_utterances}")
    print(f"  二分类subject数: {bin_subjects}")
    print(f"  二分类目标utterance数: {bin_target_utterances}")
    print(f"  计划总subject数: {len(generation_plan)}")
    print(f"  计划总utterance数: {sum(sev_target_utterances.values())}")
    
    # 计算合成后的预期分布
    if BALANCE_WITH_ORIGINAL:
        print(f"\n📝 合成后预期分布:")
        sev_final = {}
        bin_final = {0: 0, 1: 0}
        
        for sev in SEVERITY_CLASSES:
            orig_count = sev_orig.get(sev, 0)
            synth_count = sev_target_utterances.get(sev, 0)
            sev_final[sev] = orig_count + synth_count
            
            # 二分类统计
            if sev in ("normal", "mild"):
                bin_final[0] += sev_final[sev]
            else:
                bin_final[1] += sev_final[sev]
        
        print(f"  五分类最终utterance数: {sev_final}")
        print(f"  二分类最终utterance数: {bin_final}")
        print(f"  最终总utterance数: {sum(sev_final.values())}")
        
        # 计算平衡度
        bin_diff = abs(bin_final[0] - bin_final[1])
        print(f"  二分类平衡度: 差异={bin_diff} (健康:{bin_final[0]}, 抑郁:{bin_final[1]})")
    
    # 详细subject计划
    print(f"\n📝 详细subject计划:")
    sev_plans = {}
    for sev in SEVERITY_CLASSES:
        sev_plans[sev] = []
    
    for item in generation_plan:
        sev = item.get("severity", "unknown")
        if sev in sev_plans:
            sev_plans[sev].append({
                "subject_id": item["logical_subject_id"],
                "target_utterances": item.get("target_sentences", 0),
                "class_label": item["class_label"]
            })
    
    for sev in SEVERITY_CLASSES:
        if sev_plans[sev]:
            print(f"  {sev.upper()}:")
            total_utts = sum(p["target_utterances"] for p in sev_plans[sev])
            print(f"    总subject数: {len(sev_plans[sev])}, 总utterance数: {total_utts}")
            for p in sev_plans[sev][:5]:  # 只显示前5个
                print(f"      Subject {p['subject_id']}: {p['target_utterances']} utterances (class={p['class_label']})")
            if len(sev_plans[sev]) > 5:
                print(f"      ... 还有 {len(sev_plans[sev]) - 5} 个subjects")
    
    # 保存详细计划到日志文件
    plan_log_file = Path(OUTPUT_FOLDER_BASE) / "generation_plan_detailed.json"
    plan_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    plan_log_data = {
        "timestamp": dt.datetime.now().isoformat(),
        "original_distribution": {
            "five_class": sev_orig if BALANCE_WITH_ORIGINAL else None,
            "binary_class": bin_orig if BALANCE_WITH_ORIGINAL else None,
            "total_utterances": sum(sev_orig.values()) if BALANCE_WITH_ORIGINAL else None
        },
        "synthesis_plan": {
            "five_class_subjects": sev_subjects,
            "five_class_target_utterances": sev_target_utterances,
            "binary_class_subjects": bin_subjects,
            "binary_class_target_utterances": bin_target_utterances,
            "total_subjects": len(generation_plan),
            "total_target_utterances": sum(sev_target_utterances.values())
        },
        "expected_final_distribution": {
            "five_class": sev_final if BALANCE_WITH_ORIGINAL else None,
            "binary_class": bin_final if BALANCE_WITH_ORIGINAL else None,
            "total_utterances": sum(sev_final.values()) if BALANCE_WITH_ORIGINAL else None,
            "binary_balance_diff": bin_diff if BALANCE_WITH_ORIGINAL else None
        },
        "detailed_subject_plans": sev_plans,
        "generation_plan": generation_plan
    }
    
    with open(plan_log_file, "w", encoding="utf-8") as f:
        json.dump(plan_log_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细计划已保存到: {plan_log_file}")
    print(f"{'='*80}")
    
    print(f"✓ 将按计划使用对应的 embeddings（支持 dep/spk 匹配与不匹配，同类约束）")
    print(f"  Depression embeddings shape: {depression_embeddings.shape}")
    print(f"  Speaker embeddings shape: {speaker_embeddings.shape}")
    
    # 加载PersonaChat数据集
    dataset = load_personachat_dataset()
    print(f"✓ PersonaChat dataset loaded with {len(dataset)} samples")
    
    # 准备全局句子池：总需求为各subject target_sentences 之和
    total_needed = int(sum(int(it.get("target_sentences", SENTENCES_PER_SUBJECT)) for it in generation_plan))
    print(f"\n📝 为所有subjects准备全局句子池，共需 {total_needed} 句")
    global_sentences_pool = extract_conversation_responses(dataset, max_needed=total_needed)
    
    if len(global_sentences_pool) < total_needed:
        print(f"⚠️  句子池不足：仅有 {len(global_sentences_pool)}/{total_needed} 条，将尽力分配并继续")
    else:
        print(f"✓ 句子池就绪：{len(global_sentences_pool)} 条")
    
    # 打印前几个句子作为示例
    print("  示例句子:")
    for i, sentence in enumerate(global_sentences_pool[:5]):
        print(f"    {i+1}. {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
    
    # 加载模型
    model = load_matcha_model(MATCHA_CKPT)
    vocoder = load_hifigan(HIFIGAN_WEIGHT, HIFIGAN_CONFIG)
    denoiser = Denoiser(vocoder, mode="zeros")

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
        
        # 输出目录使用组合ID
        output_folder = Path(OUTPUT_FOLDER_BASE) / combo_id
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"输出文件夹: {output_folder}")
        
        # 日志文件（若不存在则写表头）
        log_file = output_folder / "processing_log.txt"
        if not log_file.exists():
            with open(log_file, "w") as log:
                log.write("Combo_ID,Depression_ID,Speaker_ID,Pair_Type,PHQ8_Binary,Sentence_ID,Text,Audio_File,Status\n")
        
        # 为这个subject生成语音
        successful_generations = 0

        # 句子编号续接：扫描已存在的 combo_id_*.wav
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

        # 获取当前条目的embeddings（dep/spk可不同）
        current_depression_emb = depression_embeddings[depression_subject_to_idx[depression_id]].squeeze()
        current_speaker_emb = speaker_embeddings[speaker_subject_to_idx[speaker_id]].squeeze()
        
        print(f"  Subject combo={combo_id} embeddings:")
        print(f"    Depression embedding shape: {current_depression_emb.shape}")
        print(f"    Speaker embedding shape: {current_speaker_emb.shape}")
        
        # 为该subject分配不重复的句子切片（按target）
        target_sentences = int(item.get("target_sentences", SENTENCES_PER_SUBJECT))
        subject_sentences = global_sentences_pool[sentence_ptr:sentence_ptr + target_sentences]
        if len(subject_sentences) < target_sentences:
            print(f"  ⚠️ 可用句子不足：目标 {target_sentences}，仅分配 {len(subject_sentences)} 条")
        sentence_ptr += len(subject_sentences)

        with tqdm(total=len(subject_sentences), desc=f"Generating subject {subject_idx+1}/{len(generation_plan)} combo={combo_id}") as pbar:
            for _, sentence_text in enumerate(subject_sentences):
                try:
                    total_attempted += 1
                    # 合成
                    out = synthesise_with_conditions(
                        model, sentence_text, 
                        current_depression_emb, current_speaker_emb
                    )
                    waveform = mel_to_waveform(out["mel"], vocoder, denoiser)
                    
                    # 保存（命名用 combo_id；label 查表用 depression_id）
                    audio_filename = save_audio_and_label(combo_id, sentence_counter, waveform, output_folder, subject_to_phq8, label_subject_id=depression_id)
                    
                    successful_generations += 1
                    total_successful += 1
                    pbar.update(1)
                    
                    # PHQ8 基于 depression_id
                    phq8_binary = subject_to_phq8.get(depression_id, 1)
                    with open(log_file, "a") as log:
                        log.write(f"{combo_id},{depression_id},{speaker_id},{pair_type},{phq8_binary},{sentence_counter},\"{sentence_text}\",{audio_filename},Success\n")
                    
                    sentence_counter += 1
                    
                    if successful_generations % 10 == 0:
                        print(f"    📈 Generated {successful_generations}/{SENTENCES_PER_SUBJECT} sentences")
                        
                except Exception as e:
                    print(f"Error generating sentence: {e}")
                    phq8_binary = subject_to_phq8.get(depression_id, 1)
                    with open(log_file, "a") as log:
                        log.write(f"{combo_id},{depression_id},{speaker_id},{pair_type},{phq8_binary},{sentence_counter},\"{sentence_text}\",N/A,Failed: {str(e)}\n")
                    sentence_counter += 1
                    continue
                        
        # 统计文件按组合命名，避免覆盖
        stats_file = output_folder / f"generation_stats_dep{depression_id}_spk{speaker_id}_{pair_type}.json"
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
                "target_sentences": int(item.get("target_sentences", SENTENCES_PER_SUBJECT)),
                "successful_generations": successful_generations,
                "success_rate": successful_generations/len(subject_sentences) if len(subject_sentences) > 0 else 0,
                "data_source": "PersonaChat_conversations",
                "quality_filter": None,
                "model_config": {
                    "n_timesteps": N_TIMESTEPS,
                    "temperature": TEMPERATURE,
                    "length_scale": LENGTH_SCALE
                },
                "audio_config": {
                    "audio_gain": AUDIO_GAIN,
                    "normalize_audio": NORMALIZE_AUDIO,
                    "target_rms": TARGET_RMS
                },
                "embeddings_info": {
                    "current_depression_embedding_shape": list(current_depression_emb.shape),
                    "current_speaker_embedding_shape": list(current_speaker_emb.shape)
                }
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
            "total_target_sentences": total_needed,
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
            "audio_config": {
                "audio_gain": AUDIO_GAIN,
                "normalize_audio": NORMALIZE_AUDIO,
                "target_rms": TARGET_RMS
            }
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"=== 批量合成完成 (PersonaChat数据集) ===")
    print(f"总subject数: {len(generation_plan)}")
    print(f"总目标句子数: {total_needed}")
    print(f"总尝试数: {total_attempted}")
    print(f"总成功数: {total_successful}")
    print(f"总体成功率: {total_successful/total_attempted*100:.1f}%" if total_attempted > 0 else "0%")
    print(f"数据来源: PersonaChat对话回复")
    print(f"Embedding使用方式: 根据生成计划使用 dep/spk（含 matched 与 mismatched 同类）")
    # 音频质量筛选：已完全移除
    print(f"PHQ8_Binary分布: {phq8_distribution}")
    print(f"输出基础文件夹: {OUTPUT_FOLDER_BASE}")
    print(f"总体统计信息: {overall_stats_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()