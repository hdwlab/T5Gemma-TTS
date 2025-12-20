#!/usr/bin/env bash

set -euo pipefail

# このスクリプトは、あみたろの声素材工房様のデータセットを用いてLoRAでファインチューニングを行う例です。
# This script is an example of fine-tuning with LoRA using the dataset from Amitaro's Voice Material Workshop.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

T5GEMMA_MODEL_NAME=google/t5gemma-2b-2b-ul2
XCODEC2_MODEL_NAME=NandemoGHS/Anime-XCodec2-44.1kHz-v2
PRETRAINED_MODEL_PATH="${PROJECT_ROOT}/ckpt/pretrained.pth"

# Dataset root directory
DATASET_ROOT="${PROJECT_ROOT}/datasets/amitaro_dataset"

EXP_ROOT="${PROJECT_ROOT}/runs/amitaro"

NUM_GPUS="${NUM_GPUS:-1}"
NUM_STEPS=1000
LR=0.035
WARMUP_FRAC=0.3
VAL_EVERY=10
PRINT_EVERY=5

NEIGHBOR_PROB=0.5
NEIGHBOR_FOLDER="${NEIGHBOR_FOLDER_NAME:-neighbors}"

# T55Gemmaの予約トークン
# Reserved token for T5Gemma
X_SEP_TOKEN=255999
N_SPECIAL=5
AUDIO_VOCAB_SIZE=65536
# EMPTY_TOKEN=AUDIO_VOCAB_SIZEに内部で設定される
# EOG=AUDIO_VOCAB_SIZE+1に内部で設定される
# AUDIO_PAD_TOKEN=AUDIO_VOCAB_SIZE+2に内部で設定される
# EOS=AUDIO_VOCAB_SIZE+3に内部で設定される
# Y_SEP_TOKEN=AUDIO_VOCAB_SIZE+4に内部で設定される
# EMPTY_TOKEN is set to AUDIO_VOCAB_SIZE internally
# EOG is set to AUDIO_VOCAB_SIZE+1 internally
# AUDIO_PAD_TOKEN is set to AUDIO_VOCAB_SIZE+2 internally
# EOS is set to AUDIO_VOCAB_SIZE+3 internally
# Y_SEP_TOKEN is set to AUDIO_VOCAB_SIZE+4 internally

# If OOM, try reducing these values or increasing gradient_accumulation_steps
# or reduce MAX_NUM_TOKENS
MAX_NUM_TOKENS=5000
VAL_MAX_NUM_TOKENS=1500
GRADIENT_ACCUMULATION_STEPS=1

mkdir -p "${EXP_ROOT}"

DATASET_DIRS="['${DATASET_ROOT}']"
MANIFEST_NAMES="['manifest_final']"

echo "[Info] launching torchrun with ${NUM_GPUS} GPU(s)"
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_ROOT}/main.py" \
  --model_arch t5gemma \
  --t5gemma_model_name "${T5GEMMA_MODEL_NAME}" \
  --text_input_type text \
  --text_tokenizer_name "${T5GEMMA_MODEL_NAME}" \
  --audio_tokenizer xcodec2 \
  --xcodec2_model_name "${XCODEC2_MODEL_NAME}" \
  --audio_vocab_size "${AUDIO_VOCAB_SIZE}" \
  --progress_scale 2000 \
  --neighbor_prompt_prob "${NEIGHBOR_PROB}" \
  --neighbor_folder_name "${NEIGHBOR_FOLDER}" \
  --n_special "${N_SPECIAL}" \
  --x_sep_token "${X_SEP_TOKEN}" \
  --no_loss_on_prefix 1 \
  --min_prompt_len 0.5 \
  --audio_max_length 30 \
  --audio_min_length 0.2 \
  --text_max_length 500 \
  --encodec_sr 50 \
  --dataset_dir "${DATASET_DIRS}" \
  --manifest_name "${MANIFEST_NAMES}" \
  --encodec_folder_name xcodec2_1cb \
  --audio_folder_name audio \
  --target_time_stretch_prob 0 \
  --target_time_stretch_bound 0 \
  --time_stretch_prob 0 \
  --time_stretch_bound 0 \
  --num_workers 2 \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --val_max_num_tokens "${VAL_MAX_NUM_TOKENS}" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LR}" \
  --warmup_fraction "${WARMUP_FRAC}" \
  --precision bfloat16 \
  --print_every_n_steps "${PRINT_EVERY}" \
  --val_every_n_steps "${VAL_EVERY}" \
  --inference_every_n_steps 100000000 \
  --save_every_n_steps 10 \
  --tb_write_every_n_steps 1 \
  --seed 1 \
  --exp_dir "${EXP_ROOT}" \
  --drop_long 1 \
  --pad_x 0 \
  --text_pad_token 0 \
  --num_buckets 20 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --optimizer_name "ScaledAdam" \
  --pseudo_epoch_size 100 \
  --reduce_lr_start_step 100 \
  --reduce_lr_start_epoch 6 \
  --clipping_update_period 50 \
  --validation_sample_cap 30000 \
  --t5_gradient_checkpointing 1 \
  --prune_text_modules 2 \
  --compile 0 \
  --attn_implementation sdpa \
  --ddp_find_unused_parameters 0 \
  --load_model_from "${PRETRAINED_MODEL_PATH}" \
  --use_lora 1 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --wandb_entity your_wandb_entity
