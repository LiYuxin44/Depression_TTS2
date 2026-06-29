import datetime as dt
import math
import random

import torch

import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)
log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
        # New parameters for DAIC dataset
        use_daic_conditions=False,
        use_speaker_id_with_daic= False,
        depression_cond_dim=None,
        speaker_cond_dim=None,
        # Adapter parameters
        use_adapter=False,
        adapter_dim=64,
        adapter_dropout=0.1,
        freeze_encoder_stages=None,
        freeze_decoder_stages=None,
        # 新增：解耦条件开关（Hydra 会将 model.use_decoupled_conditions 传入此处）
        use_decoupled_conditions: bool = False,
        pretrained_ckpt_path: str | None = None,
        train_new_condition_modules_only: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        # 解析 n_spks，兼容字符串（如 "auto"）
        resolved_n_spks = n_spks
        if isinstance(resolved_n_spks, str):
            try:
                resolved_n_spks = int(resolved_n_spks)
            except Exception:
                resolved_n_spks = 1  # 暂置为单说话人，稍后在 on_fit_start 延迟创建 embedding
        self.n_spks = int(resolved_n_spks)
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        self.use_daic_conditions = use_daic_conditions
        self.depression_cond_dim = depression_cond_dim
        self.speaker_cond_dim = speaker_cond_dim
        self.use_adapter = use_adapter
        # 新增：可切换解耦开关（默认关闭以保持旧行为）
        self.use_decoupled_conditions = bool(use_decoupled_conditions)
        self.train_new_condition_modules_only = bool(train_new_condition_modules_only)

        # Speaker embedding: 多说话人时始终启用 ID 嵌入（不再受 DAIC 条件开关影响）
        self.use_speaker_id_with_daic = bool(getattr(self.hparams, "use_speaker_id_with_daic", False))
        if self.n_spks > 1:
            self.spk_emb = torch.nn.Embedding(self.n_spks, spk_emb_dim)
        else:
            self.spk_emb = None

        # 使用与当前说话人数一致的编码器，以匹配 ckpt 中的通道数（多说话人/单说话人）
        enc_cond_dim = 0
        if self.use_daic_conditions and isinstance(self.depression_cond_dim, int):
            enc_cond_dim = max(0, self.depression_cond_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            max(1, self.n_spks),
            spk_emb_dim,
            condition_dim=enc_cond_dim,
        )

        # 解码器输入通道：2 * n_feats (+ spk_emb_dim for multi-speaker)
        decoder_in_channels = 2 * encoder.encoder_params.n_feats + (spk_emb_dim if self.n_spks > 1 else 0)

        # 恢复 decoder 的 FiLM/Adapter 能力，确保抑郁条件可以被下游模块接收
        decoder_params = dict(decoder)
        decoder_params["use_adapter"] = bool(self.use_adapter)
        if decoder_params["use_adapter"]:
            decoder_params.setdefault("adapter_dim", adapter_dim)
            decoder_params.setdefault("adapter_dropout", adapter_dropout)
            if not self.use_decoupled_conditions:
                decoder_params.setdefault(
                    "adapter_cond_dim",
                    max(0, int(self.depression_cond_dim or 0)),
                )
        decoder_params["use_decoupled_conditions"] = bool(self.use_decoupled_conditions)
        decoder_params["depression_cond_dim"] = max(0, int(self.depression_cond_dim or 0))
        decoder_params["speaker_cond_dim"] = max(0, int(self.speaker_cond_dim or 0))

        self.decoder = CFM(
            in_channels=decoder_in_channels,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder_params,
            n_spks=max(1, self.n_spks),
            spk_emb_dim=spk_emb_dim,
            use_daic_conditions=self.use_daic_conditions,
            depression_cond_dim=(self.depression_cond_dim or 0),
            speaker_cond_dim=(self.speaker_cond_dim or 0),
        )
        
        # 可选：按配置冻结 TextEncoder 的部分层（1-based 索引）
        if freeze_encoder_stages:
            try:
                enc_core = self.encoder.encoder  # Transformer 编码器主体
                blocks = list(zip(
                    enc_core.attn_layers,
                    enc_core.norm_layers_1,
                    enc_core.ffn_layers,
                    enc_core.norm_layers_2,
                ))
                if isinstance(freeze_encoder_stages, (list, tuple)):
                    target_indices = [int(s) - 1 for s in freeze_encoder_stages]
                elif isinstance(freeze_encoder_stages, str) and freeze_encoder_stages.lower() == "all":
                    target_indices = list(range(len(blocks)))
                else:
                    target_indices = []
                frozen = []
                for idx in target_indices:
                    if 0 <= idx < len(blocks):
                        for module in blocks[idx]:
                            for p in module.parameters():
                                p.requires_grad = False
                        frozen.append(idx + 1)
                if frozen:
                    log.info(f"Frozen TextEncoder stages (1-based): {frozen}")
            except Exception as e:  # pylint: disable=broad-except
                log.warning(f"freeze_encoder_stages failed to apply: {e}")

        # 保持 decoder 可训练：忽略 decoder 冻结设置

        self.update_data_statistics(data_statistics)

        # 从官方 ckpt 加载 encoder 与 decoder（失败直接抛出异常，终止训练）
        if isinstance(pretrained_ckpt_path, str) and len(pretrained_ckpt_path) > 0:
            self._load_pretrained_encoder_decoder(pretrained_ckpt_path)

        if self.train_new_condition_modules_only:
            self._freeze_for_new_condition_training()


    def on_fit_start(self) -> None:
        """在训练开始时，从 datamodule 获取最终的说话人数，必要时延迟创建 spk_emb。"""
        try:
            if (not self.use_daic_conditions or self.use_speaker_id_with_daic) and self.spk_emb is None:
                num_spks: int | None = None
                dm = getattr(self.trainer, "datamodule", None)
                if dm is not None:
                    # 优先使用 datamodule 构建的全局映射大小
                    mapping = getattr(dm, "global_subject_to_spk_idx", None)
                    if isinstance(mapping, dict) and len(mapping) > 1:
                        num_spks = int(len(mapping))
                    elif hasattr(dm.hparams, "n_spks"):
                        try:
                            n = dm.hparams.n_spks
                            if isinstance(n, str):
                                n = int(n) if n.isdigit() else None
                            if isinstance(n, int) and n > 1:
                                num_spks = n
                        except Exception:
                            pass
                if isinstance(num_spks, int) and num_spks > 1:
                    self.n_spks = num_spks
                    self.spk_emb = torch.nn.Embedding(self.n_spks, self.spk_emb_dim).to(self.device)
        except Exception:
            # 保守回退：不阻断训练
            pass

    def _prepare_depression_condition(self, depression_cond: torch.Tensor | None):
        if (
            depression_cond is None
            or not self.use_daic_conditions
            or not isinstance(self.depression_cond_dim, int)
            or self.depression_cond_dim <= 0
        ):
            return None

        if not isinstance(depression_cond, torch.Tensor):
            raise TypeError("depression_cond 必须是 torch.Tensor。")

        cond = depression_cond.to(dtype=torch.float32, device=self.device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)
        if cond.dim() == 3 and cond.shape[-1] == 1:
            cond = cond.squeeze(-1)

        return cond

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0,
                   depression_cond=None, speaker_cond=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        # For RTF computation
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        depression_cond_vec = self._prepare_depression_condition(depression_cond)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, cond=depression_cond_vec)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(
            mu_y,
            y_mask,
            n_timesteps,
            temperature,
            spks,
            cond=depression_cond_vec,
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, durations=None,
                depression_cond=None, speaker_cond=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        depression_cond_vec = self._prepare_depression_condition(depression_cond)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, cond=depression_cond_vec)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, max_length=y_cut.shape[-1]).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        decoder_cond = depression_cond_vec if depression_cond_vec is not None else cond

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(
            x1=y,
            mask=y_mask,
            mu=mu_y,
            spks=spks,
            cond=decoder_cond,
        )

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn

    # ---------------------
    # Pretrained loading
    # ---------------------
    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _freeze_for_new_condition_training(self) -> None:
        """
        Freeze所有旧权重，只保留新增的抑郁条件路径可训练：
            - TextEncoder: 仅 cond_projection + DurationPredictor
            - Decoder(UNet): 仅 FiLM 生成器
        """
        log.info("Enabling new-condition-only training: freezing legacy encoder/decoder weights.")

        # 1) Lock entire encoder & decoder
        self._set_module_requires_grad(self.encoder, False)
        self._set_module_requires_grad(self.decoder, False)
        if hasattr(self, "spk_emb"):
            self._set_module_requires_grad(self.spk_emb, False)

        # 2) Re-enable encoder cond_projection + duration predictor
        cond_proj = getattr(self.encoder, "cond_projection", None)
        if cond_proj is None:
            log.warning("cond_projection not found; nothing to fine-tune for encoder conditions.")
        self._set_module_requires_grad(cond_proj, True)
        self._set_module_requires_grad(getattr(self.encoder, "proj_w", None), True)

        # 3) Re-enable decoder FiLM generator
        estimator = getattr(self.decoder, "estimator", None)
        if estimator is None:
            log.warning("Decoder estimator missing; cannot target FiLM generator for fine-tuning.")
            return
        film_generator = getattr(estimator, "film_generator", None)
        if film_generator is None:
            log.warning("Film generator not found on decoder; skip enabling FiLM-specific training.")
        self._set_module_requires_grad(film_generator, True)

    def _safe_load_checkpoint(self, ckpt_path: str):
        """直接按信任源方式加载 ckpt（weights_only=False）。失败则抛出异常。"""
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Failed to load checkpoint with weights_only=False: {ckpt_path}"
            ) from e

    @torch.no_grad()
    def _load_pretrained_encoder_decoder(self, ckpt_path: str) -> None:
        """从 ckpt 加载 Encoder + Decoder 权重，用于全量 finetune。

        尽量鲁棒地对齐不同保存前缀（encoder.*, decoder.estimator.*、estimator.*、decoder.* 等），
        仅加载与当前模型键名、形状匹配的参数，strict=False。
        """
        state = self._safe_load_checkpoint(ckpt_path)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        current = self.state_dict()
        filtered_sd: dict[str, torch.Tensor] = {}

        for k, v in state.items():
            candidate_keys = []

            # 直匹配
            candidate_keys.append(k)

            # 兼容 lightning/封装器带前缀的 decoder.estimator
            if ".decoder.estimator." in k:
                suffix = k.split(".decoder.estimator.", 1)[1]
                candidate_keys.append(f"decoder.estimator.{suffix}")

            # 兼容仅保存 estimator.* 的情况
            if k.startswith("estimator."):
                candidate_keys.append(f"decoder.estimator.{k[len('estimator.'):]}" )

            # 兼容 decoder.* 直挂在根上的情况
            if k.startswith("decoder.") and not k.startswith("decoder.estimator."):
                candidate_keys.append(k)

            # 兼容 encoder.* 前缀或带外层前缀的情况
            if ".encoder." in k:
                suffix = k.split(".encoder.", 1)[1]
                candidate_keys.append(f"encoder.{suffix}")

            # 去重，按顺序尝试对齐
            seen = set()
            unique_candidates = []
            for ck in candidate_keys:
                if ck not in seen:
                    unique_candidates.append(ck)
                    seen.add(ck)

            matched = False
            for ck in unique_candidates:
                if ck in current and hasattr(v, "shape") and hasattr(current[ck], "shape"):
                    if tuple(v.shape) == tuple(current[ck].shape):
                        filtered_sd[ck] = v
                        matched = True
                        break
            # 忽略未匹配或形状不一致的参数（包括 spk_emb 维度不一致）

        # 执行加载
        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)
        log.info(
            f"Loaded pretrained params: matched={len(filtered_sd)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
