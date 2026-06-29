import csv
import random
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
 

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def extract_subject_id_from_path_or_line(s: str) -> Optional[int]:
    """从一行文本中提取 subject id。支持三种情况：
    1) 纯 subject id（如 "325"）
    2) filelist 行（如 "/path/300_1.wav|text..."）
    3) 直接是文件路径（如 "/path/300_1.wav"）
    """
    s = s.strip()
    if not s:
        return None
    # 情况1：纯数字
    if s.isdigit():
        return int(s)
    # 情况2：filelist，取第一列当作路径
    if "|" in s:
        s = s.split("|")[0].strip()
    # 情况3：从文件名/路径中提取
    try:
        filename = Path(s).stem
        part0 = filename.split("_")[0]
        if part0.isdigit():
            return int(part0)
    except Exception:
        pass
    # 回退：从路径各段中反向找纯数字目录
    try:
        for part in reversed(Path(s).parts):
            if part.isdigit():
                return int(part)
    except Exception:
        pass
    return None


def load_subject_mapping(subject_file_path):
    """从 subject 列表文件或 utterance 级 filelist 中生成去重后的 subject 列表（按出现顺序）。"""
    ordered_unique: list[int] = []
    seen = set()
    with open(subject_file_path, 'r') as f:
        for line in f:
            sid = extract_subject_id_from_path_or_line(line)
            if sid is None:
                continue
            if sid not in seen:
                seen.add(sid)
                ordered_unique.append(sid)
    return ordered_unique


def load_embeddings(embedding_path):
    """Load embeddings from .npz file.

    支持两种索引：
    - subject_ids: 按 subject 级别（整型ID）
    - utterance_ids: 按 utterance 级别（文件名去扩展名的字符串，如 "300_1"）

    返回三元组: (embeddings, id_to_index, ids_kind)
    其中 ids_kind ∈ {"subject", "utterance"}
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    data = np.load(embedding_path, allow_pickle=True)
    embeddings = data['embeddings']

    if 'subject_ids' in data:
        subject_ids = data['subject_ids']
        try:
            subject_ids = subject_ids.astype(int)
        except Exception:
            subject_ids = np.array([int(s) for s in subject_ids])
        id_to_index = {int(sid): idx for idx, sid in enumerate(subject_ids)}
        return embeddings, id_to_index, "subject"

    if 'utterance_ids' in data:
        utt_ids = data['utterance_ids']
        # 统一转为纯字符串
        try:
            utt_ids = utt_ids.astype(str)
        except Exception:
            utt_ids = np.array([str(s) for s in utt_ids])
        id_to_index = {str(uid): idx for idx, uid in enumerate(utt_ids)}
        return embeddings, id_to_index, "utterance"

    raise KeyError(f"Unknown id field in {embedding_path}: expected 'subject_ids' or 'utterance_ids'")


DEFAULT_PHQ_LEVEL_THRESHOLDS = [0, 5, 10, 15, 20, 28]


def load_phq_scores(
    csv_path: str,
    id_column: str = "Participant_ID",
    score_columns: Optional[Sequence[str]] = None,
) -> Dict[int, float]:
    """读取 subject→PHQ 分数的映射。"""
    if not csv_path:
        raise ValueError("phq_score_metadata_path 不能为空")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PHQ metadata csv not found: {csv_path}")

    if score_columns is None:
        score_columns = ("PHQ8_Score", "PHQ_Score")

    scores: Dict[int, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = (row.get(id_column) or "").strip()
            if not raw_id or not raw_id.isdigit():
                continue
            subject_id = int(raw_id)

            score_val = None
            for col in score_columns:
                if col is None:
                    continue
                raw_score = row.get(col)
                if raw_score is None or raw_score == "":
                    continue
                try:
                    score_val = float(raw_score)
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    raise ValueError(f"无法解析 {col} 列的 PHQ 分数: {raw_score}") from exc
            if score_val is None:
                continue
            scores[subject_id] = score_val

    if not scores:
        raise RuntimeError(f"未能从 {csv_path} 读取任何 subject 的 PHQ 分数")
    return scores


def phq_score_to_level(score: float, thresholds: Sequence[float] | None = None) -> int:
    """根据阈值列表将 PHQ-8 分数映射到 0-based level。"""
    if thresholds is None or len(thresholds) < 2:
        thresholds = DEFAULT_PHQ_LEVEL_THRESHOLDS
    value = float(score)
    for idx in range(len(thresholds) - 1):
        low = thresholds[idx]
        high = thresholds[idx + 1]
        if (value >= low and value < high) or idx == len(thresholds) - 2:
            return idx
    return len(thresholds) - 2


def normalize_score(score: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        raise ValueError("phq_scalar_max 必须大于 phq_scalar_min")
    norm = (float(score) - float(min_val)) / (float(max_val) - float(min_val))
    return float(np.clip(norm, 0.0, 1.0))



class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        # New parameters for DAIC dataset
        train_subject_file_path=None,
        valid_subject_file_path=None,
        train_depression_embeddings_path=None,
        valid_depression_embeddings_path=None,
        train_speaker_embeddings_path=None,
        valid_speaker_embeddings_path=None,
        precomputed_subject_to_spk_idx: Optional[Dict[int, int]] = None,
        depression_condition_type: str = "precomputed_embedding",
        phq_score_metadata_path: Optional[str] = None,
        phq_score_column: Optional[str] = None,
        phq_level_thresholds: Optional[Sequence[float]] = None,
        phq_scalar_min: float = 0.0,
        phq_scalar_max: float = 24.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self._precomputed_subject_to_spk_idx = precomputed_subject_to_spk_idx
        self.global_subject_to_spk_idx = None
        self.depression_condition_type = depression_condition_type or "precomputed_embedding"
        self._phq_score_lookup: Optional[Dict[int, float]] = None
        self._phq_level_thresholds = (
            list(phq_level_thresholds) if phq_level_thresholds is not None else list(DEFAULT_PHQ_LEVEL_THRESHOLDS)
        )
        self._phq_scalar_min = float(phq_scalar_min)
        self._phq_scalar_max = float(phq_scalar_max)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        # 先构建“全局一致”的 subject→spk 映射（合并 train/val 源），以避免两边顺序/子集不同导致不一致
        global_subject_to_spk_idx = self._precomputed_subject_to_spk_idx

        # 规范化 n_spks：支持 auto/None/-1；字符串数字转 int
        raw_n_spks = self.hparams.n_spks
        auto_flag = False
        if isinstance(raw_n_spks, str):
            if raw_n_spks.lower() == "auto":
                auto_flag = True
            else:
                try:
                    raw_n_spks = int(raw_n_spks)
                except Exception:
                    pass
        if raw_n_spks in (None, -1):
            auto_flag = True

        need_multi_spk = auto_flag or (isinstance(raw_n_spks, int) and raw_n_spks > 1)

        if need_multi_spk:
            if global_subject_to_spk_idx is None:
                combined_subjects: list[int] = []
                seen = set()
                # 优先从 subject 文件构建（可接受 utterance 级 filelist 并去重）
                for path in [self.hparams.train_subject_file_path, self.hparams.valid_subject_file_path]:
                    if path is None:
                        continue
                    try:
                        ids = load_subject_mapping(path)
                        for sid in ids:
                            if sid not in seen:
                                seen.add(sid)
                                combined_subjects.append(sid)
                    except Exception:
                        pass
                # 若 subject 文件缺失或为空，回退到 embeddings（合并 train/val 的 subject_ids）
                if not combined_subjects:
                    for emb_path in [self.hparams.train_depression_embeddings_path, self.hparams.valid_depression_embeddings_path]:
                        if emb_path is None:
                            continue
                        try:
                            _, subj2idx = load_embeddings(emb_path)
                            # 根据 idx 顺序重建 ids 列表
                            ids_sorted = [sid for sid, _ in sorted(subj2idx.items(), key=lambda kv: kv[1])]
                            for sid in ids_sorted:
                                if sid not in seen:
                                    seen.add(sid)
                                    combined_subjects.append(sid)
                        except Exception:
                            pass
                if combined_subjects:
                    global_subject_to_spk_idx = {sid: idx for idx, sid in enumerate(combined_subjects)}

            # 根据模式调整/校验 n_spks
            if auto_flag:
                if global_subject_to_spk_idx is None:
                    raise ValueError("n_spks=auto 但无法构建全局 subject→spk 映射，请提供 subject 列表或 embeddings")
                # 回填 n_spks
                self.hparams.n_spks = len(global_subject_to_spk_idx)
            else:
                if global_subject_to_spk_idx is not None and len(global_subject_to_spk_idx) != int(raw_n_spks):
                    raise ValueError(
                        f"n_spks={raw_n_spks} 与全局映射大小 {len(global_subject_to_spk_idx)} 不一致。"
                        f" 请确保全局唯一 subject 数量与 n_spks 对齐，或设置 n_spks=auto。"
                    )

        self.global_subject_to_spk_idx = global_subject_to_spk_idx

        condition_type = self.depression_condition_type
        phq_score_lookup = None
        if condition_type in {"phq_level", "phq_scalar"}:
            metadata_path = self.hparams.get("phq_score_metadata_path")
            if metadata_path is None:
                raise ValueError("phq_score_metadata_path is required for PHQ-based conditioning.")
            score_column = self.hparams.get("phq_score_column")
            score_columns = (score_column,) if score_column else None
            phq_score_lookup = load_phq_scores(metadata_path, score_columns=score_columns)
        self._phq_score_lookup = phq_score_lookup

        train_dep_emb_path = (
            self.hparams.train_depression_embeddings_path
            if condition_type == "precomputed_embedding"
            else None
        )
        valid_dep_emb_path = (
            self.hparams.valid_depression_embeddings_path
            if condition_type == "precomputed_embedding"
            else None
        )

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            # New parameters for DAIC dataset
            self.hparams.train_subject_file_path,
            train_dep_emb_path,
            self.hparams.train_speaker_embeddings_path,
            subject_to_spk_idx=self.global_subject_to_spk_idx,
            is_train=True,
            depression_condition_type=condition_type,
            phq_score_lookup=self._phq_score_lookup,
            phq_level_thresholds=self._phq_level_thresholds,
            phq_scalar_min=self._phq_scalar_min,
            phq_scalar_max=self._phq_scalar_max,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            # New parameters for DAIC dataset
            self.hparams.valid_subject_file_path,
            valid_dep_emb_path,
            self.hparams.valid_speaker_embeddings_path,
            subject_to_spk_idx=self.global_subject_to_spk_idx,
            is_train=False,
            depression_condition_type=condition_type,
            phq_score_lookup=self._phq_score_lookup,
            phq_level_thresholds=self._phq_level_thresholds,
            phq_scalar_min=self._phq_scalar_min,
            phq_scalar_max=self._phq_scalar_max,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        # New parameters for DAIC dataset
        subject_file_path=None,
        depression_embeddings_path=None,
        speaker_embeddings_path=None,
        subject_to_spk_idx: Optional[Dict[int, int]] = None,
        is_train: bool = False,
        depression_condition_type: str = "precomputed_embedding",
        phq_score_lookup: Optional[Dict[int, float]] = None,
        phq_level_thresholds: Optional[Sequence[float]] = None,
        phq_scalar_min: float = 0.0,
        phq_scalar_max: float = 24.0,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.is_train = is_train
        self.depression_condition_type = depression_condition_type or "precomputed_embedding"
        self._phq_score_lookup = phq_score_lookup
        self._phq_level_thresholds = (
            list(phq_level_thresholds) if phq_level_thresholds is not None else list(DEFAULT_PHQ_LEVEL_THRESHOLDS)
        )
        self._phq_scalar_min = float(phq_scalar_min)
        self._phq_scalar_max = float(phq_scalar_max)

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        
        # Load subject mappings and embeddings for DAIC dataset
        self.subject_mapping = None
        self.depression_embeddings = None
        self.depression_id_to_idx = None
        self.depression_ids_kind: str | None = None  # "subject" | "utterance"
        self.speaker_embeddings = None
        self.speaker_id_to_idx = None
        self.speaker_ids_kind: str | None = None
        self.subject_to_spk_idx = subject_to_spk_idx
        
        if self.subject_to_spk_idx is None and subject_file_path is not None:
            self.subject_mapping = load_subject_mapping(subject_file_path)
        
        if depression_embeddings_path is not None and self.depression_condition_type == "precomputed_embedding":
            self.depression_embeddings, self.depression_id_to_idx, self.depression_ids_kind = load_embeddings(depression_embeddings_path)
            
        # speaker embeddings disabled in training pipeline
        
        # Build a stable subject→spk mapping（仅在未提供预计算映射时）
        if self.subject_to_spk_idx is None:
            # Priority 1: use provided subject list file order (支持 utterance 级文件并去重)
            if self.subject_mapping is not None:
                ordered_ids = []
                for s in self.subject_mapping:
                    try:
                        sid = int(str(s))
                    except Exception:
                        continue
                    if sid not in ordered_ids:
                        ordered_ids.append(sid)
                self.subject_to_spk_idx = {sid: idx for idx, sid in enumerate(ordered_ids)}
            # Priority 2: fallback to depression embeddings subject_ids order
            elif self.depression_id_to_idx is not None and self.depression_ids_kind == "subject":
                self.subject_to_spk_idx = {int(sid): int(idx) for sid, idx in self.depression_id_to_idx.items()}
        
        # Validate mapping when multi-speaker
        if int(self.n_spks) > 1:
            if self.subject_to_spk_idx is None:
                raise ValueError(
                    "n_spks>1 但未提供可用的 subject→spk 映射。请提供 depression_embeddings_path 或 subject_file_path。"
                )
            if len(self.subject_to_spk_idx) != int(self.n_spks):
                raise ValueError(
                    f"n_spks={self.n_spks} 与 subject→spk 映射大小 {len(self.subject_to_spk_idx)} 不一致，请对齐配置/映射。"
                )
        
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def extract_subject_id(self, filepath):
        """
        从文件路径中提取 subject ID。
        你需要根据你的实际文件命名规则来实现这个函数。
        """
        # 示例 1: 文件名格式如 '.../300_1.wav' or '.../sub_300_1.wav'
        # 提取第一个数字序列作为 subject ID
        try:
            filename = Path(filepath).stem  # e.g., '300_1'
            subject_id_str = filename.split('_')[0]
            if subject_id_str.isdigit():
                return int(subject_id_str)
        except Exception:
            pass

        # 示例 2: 路径中包含 subject ID, 如 '.../300/mic_1.wav'
        try:
            parts = Path(filepath).parts
            # 从后往前找，找到的第一个纯数字的目录名作为ID
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
        except Exception:
            pass
        
        # 如果以上都不匹配，返回 None
        return None

    def get_datapoint(self, filepath_and_text):
        # 兼容两列或三列文件格式：
        # - 两列: filepath|text -> 当 n_spks>1 时，从 subject id 衍生 spk id
        # - 三列: filepath|spk|text
        parts = filepath_and_text
        if len(parts) == 3:
            filepath, spk, text = parts[0], int(parts[1]), parts[2]
        elif len(parts) == 2:
            filepath, text = parts[0], parts[1]
            if self.n_spks > 1:
                # 从 subject id 映射 spk id（稳定映射）
                sid = self.extract_subject_id(filepath)
                if sid is None:
                    raise ValueError(f"Cannot extract subject id for speaker mapping: {filepath}")
                if self.subject_to_spk_idx is None or sid not in self.subject_to_spk_idx:
                    raise KeyError(f"Subject {sid} 不在 subject→spk 映射中: {filepath}")
                spk = int(self.subject_to_spk_idx[sid])
            else:
                spk = None
        else:
            raise ValueError(f"Unexpected filelist format: {parts}")

        text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        # Get subject-level embeddings for DAIC dataset
        depression_cond = None
        speaker_cond = None
        subject_id = self.extract_subject_id(filepath)
        if subject_id is None:
            raise ValueError(f"Cannot extract subject id from {filepath}")
        utterance_id = Path(filepath).stem

        if self.depression_condition_type == "precomputed_embedding" and self.depression_embeddings is not None:
            if self.depression_ids_kind == "subject":
                if subject_id not in self.depression_id_to_idx:
                    raise KeyError(f"Subject {subject_id} not in depression embeddings for {filepath}")
                idx = self.depression_id_to_idx[subject_id]
            else:  # utterance
                if utterance_id not in self.depression_id_to_idx:
                    raise KeyError(f"Utterance {utterance_id} not in depression embeddings for {filepath}")
                idx = self.depression_id_to_idx[utterance_id]
            depression_cond = torch.from_numpy(self.depression_embeddings[idx]).float().squeeze()
        elif self.depression_condition_type in {"phq_level", "phq_scalar"}:
            if self._phq_score_lookup is None:
                raise RuntimeError("PHQ score lookup未初始化，无法构建条件向量。")
            if subject_id not in self._phq_score_lookup:
                raise KeyError(f"Subject {subject_id} 在 PHQ metadata 中不存在。")
            phq_score = self._phq_score_lookup[subject_id]
            if self.depression_condition_type == "phq_level":
                level = phq_score_to_level(phq_score, self._phq_level_thresholds)
                depression_cond = torch.tensor(level, dtype=torch.long)
            else:
                norm_score = normalize_score(phq_score, self._phq_scalar_min, self._phq_scalar_max)
                depression_cond = torch.tensor([norm_score], dtype=torch.float32)

        # speaker_cond disabled

        # 归一化：仅对外部嵌入做零均值 + L2 归一
        eps = 1e-6
        if depression_cond is not None and self.depression_condition_type == "precomputed_embedding":
            depression_cond = depression_cond - depression_cond.mean()
            depression_cond = depression_cond / (depression_cond.norm(p=2) + eps)
        # no speaker_cond normalization

        return {
            "x": text, 
            "y": mel, 
            "spk": spk, 
            "filepath": filepath, 
            "x_text": cleaned_text, 
            "durations": durations,
            "depression_cond": depression_cond,
        }

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        depression_conds = []
        # speaker_conds removed
        
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            depression_conds.append(item["depression_cond"])
            # no speaker_conds collection
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        # 只强制 depression_cond 存在
        if not all(c is not None for c in depression_conds):
            miss_idx = [i for i, c in enumerate(depression_conds) if c is None]
            raise ValueError(
                f"Missing depression_cond at indices {miss_idx} for files: {[filepaths[i] for i in miss_idx]}"
            )
        depression_cond = torch.stack(depression_conds)

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
            "depression_cond": depression_cond,
        }
