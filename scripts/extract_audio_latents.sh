#!/usr/bin/env bash
# Extract VAE audio latents + text features (for FD post-training).
#
# extract_audio_latents.py runs under torchrun (uses torch.distributed). Pass your
# dataset paths through to it; everything after the script name is forwarded to the
# Python script (see `python training/extract_audio_latents.py -h` for all options).
#
# Example:
#   NPROC=1 bash scripts/extract_audio_latents.sh \
#     --data_dir     /path/to/wavs \
#     --captions_tsv /path/to/captions.tsv \
#     --clips_tsv    /path/to/clips.tsv \
#     --latent_dir   ./data/audio-latents \
#     --output_dir   ./data/memmap/audiocaps
#
# Set NPROC for multi-GPU extraction (default: 1).
set -euo pipefail

# VAE + vocoder + CLAP are required to encode audio and text into features.
CLAP=music_speech_audioset_epoch_15_esc_89.98.pt
if [ ! -f ./weights/v1-16.pth ] || [ ! -f ./weights/best_netG.pt ] || [ ! -f "./weights/${CLAP}" ]; then
  echo "[extract] required weights not found in ./weights — downloading from kph68/FdAudio ..."
  huggingface-cli download kph68/FdAudio v1-16.pth best_netG.pt "${CLAP}" --local-dir weights
fi

NPROC="${NPROC:-1}"
torchrun --standalone --nproc_per_node="${NPROC}" \
  training/extract_audio_latents.py \
  --text_encoder t5_clap \
  "$@"
