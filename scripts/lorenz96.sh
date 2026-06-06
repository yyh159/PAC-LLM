#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --system Lorenz96
  --feat_dim 8
  --model_variant pac_llm
  --feature_mode raw
  --aux_model_type mlp
  --aux_source patch
  --input_size 30
  --patch_len 8
  --stride 4
  --takens_embed_dim 8
  --d_model 32
  --d_ff 128
  --n_heads 8
  --dropout 0.1
  --llm_dim 768
  --llm_layers 2
  --lambda_main 0.9
  --train_horizon 30
  --valid_horizon 30
  --eval_horizon 300
  --rollout_horizons 30 120 300
)

python train.py \
  "${COMMON_ARGS[@]}" \
  --batch_size 32 \
  --epochs 150 \
  --learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --warmup_epochs 8 \
  --patience 20 \
  --grad_clip 0.3

python evaluate.py "${COMMON_ARGS[@]}"
