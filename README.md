<div align="center">
<h1>FdAudio: MeanFlow-Anchored Fréchet-Distance Post-Training for One-Step Text-to-Audio Generation</h1>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.10421)
[![Hugging Face Model](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/kph68/FdAudio)
[![Webpage](https://img.shields.io/badge/Website-Visit-orange?logo=googlechrome&logoColor=white)](https://fdoneaudio.github.io/)

</div>

## Overview

**FdAudio** is a one-step text-to-audio (T2A) generator. Starting from the MeanFlow-based
**MeanAudio-S-Full** model, we apply **MeanFlow-anchored Fréchet-distance (FD) post-training**:
the one-step output distribution is optimized directly against real audio in the embedding
spaces of several pretrained audio encoders (PANNs, PaSST, BEATs, AudioMAE), while a MeanFlow
consistency objective anchors the velocity field so that multi-step sampling is preserved.
With only **120M** parameters, FdAudio reaches state-of-the-art **one-step** generation quality
on AudioCaps and stays competitive under 25-step sampling.

> This codebase is modified from [MeanAudio](https://github.com/xiquan-li/MeanAudio).

## Installation

```bash
git clone https://github.com/nobel861017/FdAudio.git
cd FdAudio
conda create -n fdaudio python=3.10 -y && conda activate fdaudio
pip install -e .
```

## FD Post-Training

FdAudio is produced by MeanFlow-anchored FD post-training, **initialized from MeanAudio-S-Full**.

### 1. Data preparation

Training uses **AudioCaps** + **WavCaps**.

- **AudioCaps** — download following [AudioLDM-training-finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning#download-checkpoints-and-dataset).
- **WavCaps** — download from [`cvssp/WavCaps`](https://huggingface.co/datasets/cvssp/WavCaps).

> **⚠️ Note on AudioSet_SL.** Some `.wav` files are missing from the WavCaps AudioSet_SL split
> (see [this discussion](https://huggingface.co/datasets/cvssp/WavCaps/discussions/5)). The complete
> set can be obtained from [this mirror](https://drive.google.com/file/d/1rVBRJbPPu63jzEojpdZ35Xumb6pRNCIj/view?usp=sharing).

You need (i) a directory of audio files and (ii) a captions TSV with columns `id` (file name
without extension) and `caption`. First partition the audio into 10-second clips, then extract
the VAE latents + text features (writes the memmap dataset used for training):

```bash
# 1) partition audio -> clips.tsv (columns: id, name, start_sample, end_sample)
python training/partition_clips.py \
  --data_dir /path/to/wavs \
  --output_dir ./data/clips.tsv

# 2) extract VAE latents + text features into a memmap dataset
NPROC=1 bash scripts/extract_audio_latents.sh \
  --data_dir     /path/to/wavs \
  --captions_tsv /path/to/captions.tsv \
  --clips_tsv    ./data/clips.tsv \
  --latent_dir   ./data/audio-latents \
  --output_dir   ./data/memmap/audiocaps
```

Then configure the resulting dataset paths in `config/data/t5_clap.yaml`. The FD-loss reference
statistics are precomputed with `training/extract_fd_ref_stats_multi.py`.

### 2. Base model

Download the MeanAudio-S-Full initialization checkpoint into `./weights/`:

```bash
huggingface-cli download AndreasXi/MeanAudio meanaudio_s_full.pth --local-dir weights
```

### 3. Train

```bash
python train.py exp_id=fdaudio_posttrain \
  weights=./weights/meanaudio_s_full.pth \
  use_meanflow=True use_fd=True \
  fd.enable=true fd.encoders=[panns,passt,beats,audiomae] \
  fd.mf_weight=0.25 \
  learning_rate=1e-5
```

The MeanFlow anchor (`fd.mf_weight`) regularizes the velocity field during FD optimization,
preventing the multi-step collapse of naive FD post-training while improving one-step fidelity.

> **Note.** The `panns`, `passt`, and `audiomae` FD encoders are installed with `pip install -e .`
> (via `av-bench`, `hear21passt`, and `timm`). The `beats` encoder additionally requires the
> [BEATs](https://github.com/microsoft/unilm/tree/master/beats) code on your `PYTHONPATH` and its
> pretrained checkpoint; omit `beats` from `fd.encoders` if you don't need it.

## Inference

By default, inference uses our released **FdAudio** checkpoint, which (together with the VAE,
BigVGAN vocoder, and LAION-CLAP checkpoint) is **downloaded automatically from
[`kph68/FdAudio`](https://huggingface.co/kph68/FdAudio)** on first run (FLAN-T5 is fetched from the HF hub):

```bash
python infer.py \
  --use_meanflow --use_rope --encoder_name t5_clap --text_c_dim 512 \
  --num_steps 1 --cfg_strength 0.9 --full_precision \
  --prompt "A dog barking in the distance" --output ./output
```

To use **your own** FD-post-trained checkpoint instead, pass `--model_path`:

```bash
python infer.py --model_path ./exps/fdaudio_posttrain/your_checkpoint.pth \
  --use_meanflow --use_rope --encoder_name t5_clap --text_c_dim 512 \
  --num_steps 1 --cfg_strength 0.9 --full_precision \
  --prompt "A dog barking in the distance" --output ./output
```

For multi-step sampling, set `--num_steps 25` (FdAudio preserves high-fidelity multi-step generation).

## Evaluation

The evaluation dependencies (`av-bench`, `hear21passt`) are installed by `pip install -e .`.
Computing FD / FAD / KL / IS / CLAP requires the **AudioCaps test ground-truth audio**, which is
**not bundled** — obtain the AudioCaps test set (e.g. via
[AudioLDM-training-finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning#download-checkpoints-and-dataset))
and point `--gt_audio` at it. The captions TSV (`sets/test-audiocaps.tsv`, 957 clips) is included.

`scripts/eval_testset.sh` runs the whole pipeline — generate all 957 test clips, then compute metrics:

```bash
# one-step (NFE=1)
bash scripts/eval_testset.sh --gt_audio /path/to/audiocaps_test_audio

# 25-step
bash scripts/eval_testset.sh --gt_audio /path/to/audiocaps_test_audio --nfe 25
```

Results are written to `./eval_out/metrics_nfe<N>.json`. Use `--model_path` to evaluate your own
FD-post-trained checkpoint, or `--output` to change the output directory.

<details><summary>Equivalent manual steps</summary>

```bash
python eval.py --use_meanflow --use_rope --encoder_name t5_clap --text_c_dim 512 \
  --num_steps 1 --cfg_strength 0.9 --full_precision --output ./pred_audio
python training/eval_full_metrics.py \
  --gt_audio /path/to/audiocaps_test_audio \
  --gt_captions_tsv sets/test-audiocaps.tsv --pred_audio ./pred_audio
```
</details>

## License

- **Code:** released under the terms in [`LICENSE`](LICENSE).
- **Model weights** ([`kph68/FdAudio`](https://huggingface.co/kph68/FdAudio)): **CC BY-NC-SA 4.0** (non-commercial).

## Acknowledgements

This codebase is modified from [MeanAudio](https://github.com/xiquan-li/MeanAudio). We also gratefully acknowledge:

- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) — VAE and BigVGAN vocoder.
- FD-loss audio encoders: [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn),
  [PaSST](https://github.com/kkoutini/PaSST),
  [BEATs](https://github.com/microsoft/unilm/tree/master/beats),
  [AudioMAE](https://github.com/facebookresearch/AudioMAE).
- [LAION-CLAP](https://github.com/LAION-AI/CLAP) — text/audio alignment encoder.
- The [AudioCaps](https://github.com/cdjkim/audiocaps) and [WavCaps](https://github.com/XinhaoMei/WavCaps) datasets.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{fdaudio2026,
  title   = {FdAudio: MeanFlow-Anchored Fr\'echet-Distance Post-Training for One-Step Text-to-Audio Generation},
  author  = {Huang, Kuan-Po and Lu, Bo-Ru and Chung, Ho-Lam and Wang, Shih-Hsin and Lee, Hung-yi},
  journal = {arXiv preprint arXiv:2607.10421},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.10421},
}
```
