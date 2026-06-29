#!/usr/bin/env bash
set -euo pipefail

cd /scratch/users/ntu/yuxin.li/matcha-tts-new

python -m matcha.train +experiment=null -cn train_daic_depr_only.yaml


