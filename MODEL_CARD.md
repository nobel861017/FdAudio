---
license: cc-by-nc-sa-4.0
language: en
pipeline_tag: text-to-audio
tags:
  - text-to-audio
  - one-step-generation
  - meanflow
  - flow-matching
---

# FdAudio

**FdAudio** is a one-step text-to-audio (T2A) generator produced by **MeanFlow-anchored
Fréchet-distance (FD) post-training**, initialized from **MeanAudio-S-Full**. It optimizes the
one-step output distribution against real audio in several pretrained encoder spaces
(PANNs, PaSST, BEATs, AudioMAE) while a MeanFlow consistency objective preserves multi-step sampling.
With only 120M parameters it achieves state-of-the-art one-step generation on AudioCaps.

- 📄 Paper: https://arxiv.org/abs/xxxxxx
- 💻 Code: https://github.com/nobel861017/FdAudio
- 🔊 Demo: https://fdoneaudio.github.io/

## Files
- `fdaudio.pth` — the FdAudio one-step generator weights.

## Usage
See the [code repository](https://github.com/nobel861017/FdAudio) for installation and inference:

```bash
huggingface-cli download kph68/FdAudio fdaudio.pth --local-dir weights
python infer.py --variant meanaudio_s --model_path weights/fdaudio.pth \
  --use_meanflow --use_rope --encoder_name t5_clap --text_c_dim 512 \
  --num_steps 1 --cfg_strength 0.9 --full_precision \
  --prompt "A dog barking in the distance" --output ./output
```

## Results (AudioCaps, one-step / NFE=1)
| FD (PANNs) ↓ | FAD ↓ | KL ↓ | IS ↑ | CLAP ↑ |
|---|---|---|---|---|
| 12.71 | 1.26 | 1.30 | 11.14 | 0.274 |

## Acknowledgement
Built on [MeanAudio](https://github.com/xiquan-li/MeanAudio).
