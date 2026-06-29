import os, sys
from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from matcha import utils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


log = utils.get_pylogger(__name__)


def _load_pretrained_weights(
    model: LightningModule,
    ckpt_path: str,
    spk_emb_mode: str = "ignore",  # "ignore" | "expand_copy"
) -> None:
    """Partially load weights from a pretrained checkpoint into current model.
    - ignore: skip loading spk_emb; initialize new embedding with pretrained stats if available
    - expand_copy: copy as many rows as possible from pretrained spk_emb, fill the rest with random init
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        log.warning(f"Pretrained ckpt not found or empty path: {ckpt_path}")
        return

    log.info(f"Loading pretrained checkpoint (partial): {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    model_sd = model.state_dict()

    # Prepare filtered state dict (exclude keys missing in current model, and optionally spk_emb)
    filtered_sd = {}
    for k, v in sd.items():
        if k not in model_sd:
            continue
        if k == "spk_emb.weight" and spk_emb_mode == "ignore":
            continue
        # Only load tensors with same shape (safer)
        if hasattr(v, "shape") and tuple(v.shape) != tuple(model_sd[k].shape):
            if k != "spk_emb.weight":
                log.warning(f"Skip due to shape mismatch: {k}: {tuple(v.shape)} != {tuple(model_sd[k].shape)}")
            continue
        filtered_sd[k] = v

    # Load non-spk_emb params first
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if missing:
        log.info(f"Missing keys when loading pretrained (expected): {len(missing)}")
    if unexpected:
        log.info(f"Unexpected keys when loading pretrained (ignored): {len(unexpected)}")

    # Handle spk_emb according to mode
    has_spk_now = "spk_emb.weight" in model.state_dict()
    has_spk_prev = "spk_emb.weight" in sd

    if not has_spk_now:
        # Current model has no speaker embedding (n_spks <= 1)
        return

    if spk_emb_mode == "ignore":
        # Initialize with pretrained statistics if available
        if has_spk_prev:
            prev_w = sd["spk_emb.weight"]
            mean = prev_w.mean().item()
            std = prev_w.std().item()
            std = std if std > 1e-6 else 0.02
            with torch.no_grad():
                model.spk_emb.weight.normal_(mean=mean, std=std)
            log.info(f"Initialized spk_emb with pretrained stats: mean={mean:.4f}, std={std:.4f}")
        else:
            # leave default init
            log.info("spk_emb: keep default initialization (no pretrained stats available)")
        return

    if spk_emb_mode == "expand_copy":
        if not has_spk_prev:
            log.warning("No spk_emb in pretrained ckpt; fallback to default init for current spk_emb")
            return
        prev_w = sd["spk_emb.weight"]
        cur_w = model.spk_emb.weight.data
        if prev_w.shape[1] != cur_w.shape[1]:
            log.warning(
                f"spk_emb dim mismatch: pretrained {tuple(prev_w.shape)} vs current {tuple(cur_w.shape)}; "
                f"fallback to default init"
            )
            return
        # Initialize current with pretrained stats, then copy overlapping rows
        mean = prev_w.mean().item(); std = prev_w.std().item(); std = std if std > 1e-6 else 0.02
        with torch.no_grad():
            cur_w.normal_(mean=mean, std=std)
            num_rows = min(prev_w.shape[0], cur_w.shape[0])
            cur_w[:num_rows] = prev_w[:num_rows]
        log.info(f"spk_emb expanded: copied {num_rows} rows from pretrained, total {cur_w.shape[0]} rows")
        return

    log.warning(f"Unknown spk_emb_mode: {spk_emb_mode}; skipped spk_emb handling")


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # pylint: disable=protected-access
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")  # pylint: disable=protected-access
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Optional: load pretrained checkpoint with custom spk_emb strategy
    pretrained_ckpt = cfg.get("pretrained_ckpt") or cfg.model.get("pretrained_ckpt") if hasattr(cfg, "model") else None
    spk_emb_mode = cfg.get("spk_emb_load_mode") or cfg.model.get("spk_emb_load_mode") if hasattr(cfg, "model") else None
    if pretrained_ckpt:
        _load_pretrained_weights(model, pretrained_ckpt, spk_emb_mode or "ignore")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # pylint: disable=protected-access
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        ckpt_path = cfg.get("ckpt_path")
        # 与旧库对齐：直接把 ckpt_path 交给 Trainer
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
