import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch


def load_checkpoint(path_to_checkpoint: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load a PyTorch checkpoint and return its state_dict and the raw checkpoint dict.

    The function supports plain state_dict files and Lightning-style checkpoints
    containing a top-level "state_dict" key.
    """
    if not os.path.exists(path_to_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {path_to_checkpoint}")

    checkpoint: Dict[str, Any] = torch.load(path_to_checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # assume it is already a state_dict

    if not isinstance(state_dict, dict):
        raise ValueError("Loaded checkpoint is not a dict-like state_dict.")

    return state_dict, checkpoint


def get_tensor_shape(state_dict: Dict[str, torch.Tensor], key: str) -> Optional[Tuple[int, ...]]:
    """Return the shape of a tensor in the state_dict if present."""
    tensor = state_dict.get(key)
    if tensor is None:
        return None
    if hasattr(tensor, "shape"):
        return tuple(int(dim) for dim in tensor.shape)
    return None


def infer_encoder_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the encoder hidden dimension from common encoder parameter shapes."""
    probe_keys = [
        "encoder.emb.weight",  # shape: (n_vocab, hidden_dim)
        "encoder.prenet.proj.weight",  # shape: (hidden_dim, hidden_dim, 1)
        "encoder.encoder.attn_layers.0.conv_q.weight",  # shape: (hidden_dim, hidden_dim, 1)
        "encoder.encoder.ffn_layers.0.conv_2.bias",  # shape: (hidden_dim,)
        "encoder.encoder.norm_layers_1.0.gamma",  # shape: (hidden_dim,)
        "encoder.proj_m.weight",  # shape: (n_mels, hidden_dim, 1)
    ]
    for key in probe_keys:
        shape = get_tensor_shape(state_dict, key)
        if shape is None:
            continue
        if key == "encoder.emb.weight" and len(shape) == 2:
            return int(shape[1])
        if key in {"encoder.encoder.ffn_layers.0.conv_2.bias", "encoder.encoder.norm_layers_1.0.gamma"} and len(shape) == 1:
            return int(shape[0])
        if key in {"encoder.prenet.proj.weight", "encoder.encoder.attn_layers.0.conv_q.weight", "encoder.proj_m.weight"} and len(shape) >= 2:
            return int(shape[1])
    return None


def infer_decoder_time_mlp_input_dim(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the decoder time-MLP input dimension if present in the checkpoint."""
    key = "decoder.estimator.time_mlp.linear_1.weight"  # shape: (out_dim, in_dim)
    shape = get_tensor_shape(state_dict, key)
    if shape is None or len(shape) != 2:
        return None
    return int(shape[1])


def summarize_known_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
    """Collect shapes for a set of commonly useful keys if they exist."""
    keys_of_interest = [
        "encoder.emb.weight",
        "encoder.prenet.conv_layers.0.weight",
        "encoder.prenet.conv_layers.0.bias",
        "encoder.prenet.proj.weight",
        "encoder.prenet.proj.bias",
        "encoder.encoder.attn_layers.0.conv_q.weight",
        "encoder.encoder.attn_layers.0.conv_k.weight",
        "encoder.encoder.attn_layers.0.conv_v.weight",
        "encoder.encoder.attn_layers.0.conv_o.weight",
        "encoder.encoder.ffn_layers.0.conv_1.weight",
        "encoder.encoder.ffn_layers.0.conv_2.weight",
        "encoder.encoder.ffn_layers.0.conv_2.bias",
        "encoder.encoder.norm_layers_1.0.gamma",
        "encoder.encoder.norm_layers_1.0.beta",
        "encoder.proj_m.weight",
        "encoder.proj_w.conv_1.weight",
        "decoder.estimator.time_mlp.linear_1.weight",
        "decoder.estimator.down_blocks.0.0.block1.block.0.weight",
        "decoder.estimator.down_blocks.0.0.res_conv.weight",
    ]
    shapes: Dict[str, Tuple[int, ...]] = {}
    for key in keys_of_interest:
        shape = get_tensor_shape(state_dict, key)
        if shape is not None:
            shapes[key] = shape
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect shapes and hyper-parameters from a Matcha-TTS checkpoint")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the checkpoint file")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save the inspection result as JSON")
    args = parser.parse_args()

    state_dict, checkpoint = load_checkpoint(args.ckpt)

    num_params = len(state_dict)
    encoder_hidden_dim = infer_encoder_hidden_dim(state_dict)
    time_mlp_input_dim = infer_decoder_time_mlp_input_dim(state_dict)
    has_hparams = isinstance(checkpoint, dict) and ("hyper_parameters" in checkpoint)
    hyper_parameters = checkpoint.get("hyper_parameters", {}) if has_hparams else {}
    model_hparams = hyper_parameters.get("model", {}) if isinstance(hyper_parameters, dict) else {}

    summary: Dict[str, Any] = {
        "ckpt_path": os.path.abspath(args.ckpt),
        "num_tensors_in_state_dict": num_params,
        "encoder_hidden_dim_inferred": encoder_hidden_dim,
        "decoder_time_mlp_input_dim_inferred": time_mlp_input_dim,
        "has_hyper_parameters": has_hparams,
        "hyper_parameters_keys": list(hyper_parameters.keys()) if isinstance(hyper_parameters, dict) else None,
        "model_hyper_parameters": model_hparams,
        "known_key_shapes": summarize_known_keys(state_dict),
    }

    print("==== Checkpoint Inspection ====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved inspection JSON to: {os.path.abspath(args.save_json)}")


if __name__ == "__main__":
    main()


