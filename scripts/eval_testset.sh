#!/usr/bin/env bash
# Evaluate FdAudio on the full AudioCaps test set: generate all clips, then compute metrics
# (FD / FAD / KL / IS / CLAP). No paths are hard-coded; supply the ground-truth audio with --gt_audio.
#
# Usage:
#   bash scripts/eval_testset.sh --gt_audio /path/to/audiocaps_test_audio \
#        [--nfe 1] [--seed 2024] [--output ./eval_out] [--model_path your_ckpt.pth]
#
# The captions TSV (sets/test-audiocaps.tsv) is bundled. The FdAudio checkpoint and the
# VAE / vocoder / CLAP weights are downloaded from kph68/FdAudio on first run.
set -euo pipefail

NFE=1; SEED=2024; OUTPUT=./eval_out; GT_AUDIO=""; MODEL_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt_audio)   GT_AUDIO="$2"; shift 2;;
    --nfe)        NFE="$2"; shift 2;;
    --seed)       SEED="$2"; shift 2;;
    --output)     OUTPUT="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done
if [ -z "$GT_AUDIO" ]; then
  echo "error: --gt_audio is required (directory of AudioCaps test ground-truth audio)" >&2
  exit 1
fi

# Ensure VAE + vocoder + CLAP are present (needed for generation and CLAP metric).
CLAP=music_speech_audioset_epoch_15_esc_89.98.pt
if [ ! -f ./weights/v1-16.pth ] || [ ! -f ./weights/best_netG.pt ] || [ ! -f "./weights/${CLAP}" ]; then
  echo "[eval] downloading VAE / vocoder / CLAP from kph68/FdAudio ..."
  huggingface-cli download kph68/FdAudio v1-16.pth best_netG.pt "${CLAP}" --local-dir weights
fi

GEN_DIR="${OUTPUT}/gen_nfe${NFE}"
echo "[eval] 1/2 generating the AudioCaps test set (NFE=${NFE}, seed=${SEED}) -> ${GEN_DIR}"
python eval.py \
  ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
  --use_meanflow --use_rope --encoder_name t5_clap --text_c_dim 512 \
  --num_steps "$NFE" --cfg_strength 0.9 --seed "$SEED" --full_precision \
  --output "$GEN_DIR"

echo "[eval] 2/2 computing metrics against ground-truth"
python training/eval_full_metrics.py \
  --gt_audio "$GT_AUDIO" \
  --gt_captions_tsv sets/test-audiocaps.tsv \
  --pred_audio "$GEN_DIR" \
  --out_json "${OUTPUT}/metrics_nfe${NFE}.json"

echo "[eval] done -> ${OUTPUT}/metrics_nfe${NFE}.json"
