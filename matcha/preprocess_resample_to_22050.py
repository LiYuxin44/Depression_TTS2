import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple
import concurrent.futures as futures

import torchaudio as ta


DEFAULT_IN_DIR = \
    "/data/test/processed_audio_test"
DEFAULT_OUT_DIR = \
    "/home/i-liyuxin/Depression_TTS/matcha/data/processed_audio_test_22050"

# DEFAULT_TRAIN_IN_FILELIST = \
#     "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train.txt"
# DEFAULT_TRAIN_OUT_FILELIST = \
#     "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train_22k.txt"
# DEFAULT_VAL_IN_FILELIST = \
#     "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_val.txt"
# DEFAULT_VAL_OUT_FILELIST = \
#     "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_val_22k.txt"
DEFAULT_TEST_IN_FILELIST = \
    "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_test.txt"
DEFAULT_TEST_OUT_FILELIST = \
    "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_test_22k.txt"


def list_wav_files(input_dir: str) -> List[Path]:
    input_dir_path = Path(input_dir)
    if not input_dir_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    wav_paths: List[Path] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".wav"):
                wav_paths.append(Path(root) / fn)
    return wav_paths


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_mono(waveform):
    # waveform: (channels, num_samples)
    if waveform.shape[0] == 1:
        return waveform
    # average channels to mono
    return waveform.mean(dim=0, keepdim=True)


# Changed: include input_dir in args and use it for relative path computation
def resample_one(args: Tuple[Path, str, int, str, str]) -> Tuple[str, bool, str]:
    src_path, out_dir, target_sr, encoding, input_dir = args
    try:
        rel = src_path.relative_to(Path(input_dir))
    except Exception:
        # Fallback if src_path is not under input_dir
        rel = src_path.name
    try:
        dst_path = Path(out_dir) / rel
        ensure_parent_dir(dst_path)
        wav, sr = ta.load(str(src_path))
        if sr != target_sr:
            wav = ta.functional.resample(wav, sr, target_sr)
        wav = to_mono(wav)
        ta.save(str(dst_path), wav, target_sr, encoding=encoding, bits_per_sample=16)
        return (str(src_path), True, "")
    except Exception as e:  # pylint: disable=broad-except
        return (str(src_path), False, repr(e))


def resample_dir(input_dir: str, output_dir: str, target_sr: int, num_workers: int = 4) -> None:
    wav_files = list_wav_files(input_dir)
    if len(wav_files) == 0:
        print(f"No .wav files found under: {input_dir}")
        return

    print(f"Found {len(wav_files)} wav files. Starting resampling to {target_sr} Hz → {output_dir}")
    encoding = "PCM_S"

    # Changed: pass input_dir into each task
    tasks = [(p, output_dir, target_sr, encoding, input_dir) for p in wav_files]

    ok, fail = 0, 0
    errors: List[Tuple[str, str]] = []

    # Use process pool for CPU-bound resampling
    with futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        for src, success, err in ex.map(resample_one, tasks, chunksize=16):
            if success:
                ok += 1
                if ok % 500 == 0:
                    print(f"Resampled {ok}/{len(wav_files)}...")
            else:
                fail += 1
                errors.append((src, err))

    print(f"Resampling done. Success: {ok}, Failed: {fail}")
    if fail:
        print("Some files failed to process (showing up to first 20):")
        for i, (src, err) in enumerate(errors[:20]):
            print(f"  [{i+1}] {src}: {err}")


def remap_filelist(fin: str, fout: str, old_prefix: str, new_prefix: str) -> None:
    in_path = Path(fin)
    out_path = Path(fout)
    ensure_parent_dir(out_path)

    num = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for raw in f_in:
            line = raw.rstrip("\n")
            if not line:
                continue
            # split into path and the rest (text or spk|text)
            parts = line.split("|", 1)
            if len(parts) < 2:
                # keep line unchanged
                f_out.write(line + "\n")
                continue
            audio_path, tail = parts[0], parts[1]
            if audio_path.startswith(old_prefix):
                audio_path = audio_path.replace(old_prefix, new_prefix, 1)
            f_out.write(f"{audio_path}|{tail}\n")
            num += 1
    print(f"Remapped {num} entries → {fout}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline resampling DAIC audio to 22050 Hz and generating new filelists "
            "that point to the resampled directory."
        )
    )
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR, type=str,
                        help="Input directory containing original wav files")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=str,
                        help="Output directory to write resampled wav files")
    parser.add_argument("--target-sr", default=22050, type=int,
                        help="Target sampling rate")
    parser.add_argument("--num-workers", default=max(1, (os.cpu_count() or 4) - 1), type=int,
                        help="Number of worker processes for resampling")

    # parser.add_argument("--train-filelist-in", default=DEFAULT_TRAIN_IN_FILELIST, type=str)
    # parser.add_argument("--train-filelist-out", default=DEFAULT_TRAIN_OUT_FILELIST, type=str)
    # parser.add_argument("--val-filelist-in", default=DEFAULT_VAL_IN_FILELIST, type=str)
    # parser.add_argument("--val-filelist-out", default=DEFAULT_VAL_OUT_FILELIST, type=str)
    parser.add_argument("--test-filelist-in", default=DEFAULT_TEST_IN_FILELIST, type=str)
    parser.add_argument("--test-filelist-out", default=DEFAULT_TEST_OUT_FILELIST, type=str)

    parser.add_argument("--old-prefix", default=DEFAULT_IN_DIR.rstrip("/") + "/", type=str,
                        help="Prefix in filelists to replace")
    parser.add_argument("--new-prefix", default=DEFAULT_OUT_DIR.rstrip("/") + "/", type=str,
                        help="New prefix in filelists")

    parser.add_argument("--copy-if-same-sr", action="store_true",
                        help="If input already has target_sr, copy file bytes instead of re-encoding")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Resample directory
    resample_dir(args.in_dir, args.out_dir, args.target_sr, num_workers=args.num_workers)

    # Remap filelists
    # remap_filelist(args.train_filelist_in, args.train_filelist_out, args.old_prefix, args.new_prefix)
    # remap_filelist(args.val_filelist_in, args.val_filelist_out, args.old_prefix, args.new_prefix)
    remap_filelist(args.test_filelist_in, args.test_filelist_out, args.old_prefix, args.new_prefix)

    print("All done. Next steps:")
    print("  1) Update configs/data/daic_depr_only.yaml:")
    print("     - sample_rate: 22050")
    print("     - train_filelist_path: " + args.train_filelist_out)
    print("     - valid_filelist_path: " + args.val_filelist_out)
    print("  2) (Optional) Re-compute data_statistics for best results.")


if __name__ == "__main__":
    main() 