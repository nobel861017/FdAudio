"""Compute FD-VGG/PANN/PASST + KL + IS for a pred audio dir vs GT.

Reuses existing feature caches from eval_fd_metrics.py (pann/passt/vggish).
KL and IS are computed from PANN logits which are already in those caches.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

_orig_load = torchaudio.load
def _patched_load(*a, **kw):
    kw.setdefault('backend', 'soundfile')
    wav, sr = _orig_load(*a, **kw)
    return wav.clone(), sr
torchaudio.load = _patched_load

from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import subprocess
import sys

from av_bench.data.audio_dataset import AudioDataset
from av_bench.metrics import compute_fd, compute_isc, compute_kl
from av_bench.panns import Cnn14
from av_bench.utils import unroll_dict, unroll_dict_all_keys, unroll_paired_dict
from av_bench.vggish.vggish import VGGish
from hear21passt.base import get_basic_model

_CLAP_CKPT = Path('./weights/music_speech_audioset_epoch_15_esc_89.98.pt')


def _pad_collate(batch):
    max_len = max(item[0].shape[-1] for item in batch)
    wavs, names = [], []
    for wav, fn in batch:
        if wav.shape[-1] < max_len:
            wav = F.pad(wav, (0, max_len - wav.shape[-1]))
        wavs.append(wav)
        names.append(fn)
    return torch.stack(wavs, dim=0), names


def cache_for(audio_dir: Path) -> Path:
    return audio_dir.parent / (audio_dir.name + '_cache')


def maybe_extract(audio_dir: Path, cache_dir: Path, audio_length: float = 10.0, batch_size: int = 16):
    needed = ['pann_features.pth', 'vggish_features.pth',
              'passt_features_embed.pth', 'passt_logits.pth']
    if all((cache_dir / n).exists() for n in needed):
        print(f'[skip] {cache_dir} already complete')
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    import gc, torch as _torch
    _torch.cuda.empty_cache(); gc.collect()
    free, total = _torch.cuda.mem_get_info()
    print(f'[extract] GPU free: {free/1e9:.2f} GB / {total/1e9:.2f} GB before loading encoders')
    device = 'cuda'
    audios = sorted(list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.flac')),
                    key=lambda x: x.stem)

    import gc

    # --- PANN + VGGish (16 kHz) ---
    panns = Cnn14(features_list=['2048', 'logits'], sample_rate=16000, window_size=512,
                  hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527).to(device).eval()
    vggish = VGGish(postprocess=False).to(device).eval()

    ds16k = AudioDataset(audios, audio_length=audio_length, sr=16000)
    dl16k = DataLoader(ds16k, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=_pad_collate)
    pann_out, vgg_out = {}, {}
    for wav, fns in tqdm(dl16k, desc='PANN+VGGish'):
        wav = wav.squeeze(1).float().to(device)
        with torch.no_grad():
            pf = {k: v.cpu() for k, v in panns(wav).items()}
            vf = vggish(wav).cpu()
        for i, fn in enumerate(fns):
            pann_out[fn] = {k: v[i] for k, v in pf.items()}
            vgg_out[fn] = vf[i]
    torch.save(pann_out, cache_dir / 'pann_features.pth')
    torch.save(vgg_out, cache_dir / 'vggish_features.pth')

    # Free PANN + VGGish before loading PaSST to avoid OOM
    del panns, vggish
    gc.collect()
    torch.cuda.empty_cache()

    # --- PaSST (32 kHz) ---
    passt = get_basic_model(mode='all').to(device).eval()
    ds32k = AudioDataset(audios, audio_length=audio_length, sr=32000)
    dl32k = DataLoader(ds32k, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=_pad_collate)
    feats, logits = {}, {}
    for wav, fns in tqdm(dl32k, desc='PaSST'):
        wav = wav.squeeze(1).float().to(device)
        if wav.size(-1) >= 320000:
            wav = wav[..., :320000]
        else:
            wav = F.pad(wav, (0, 320000 - wav.size(-1)))
        with torch.no_grad():
            out = passt(wav).cpu()
        for i, fn in enumerate(fns):
            feats[fn] = out[i, 527:]
            logits[fn] = out[i, :527]
    torch.save(feats, cache_dir / 'passt_features_embed.pth')
    torch.save(logits, cache_dir / 'passt_logits.pth')
    del passt
    gc.collect()
    torch.cuda.empty_cache()


def extract_clap(audio_dir: Path, audio_cache: Path, captions_tsv: Path, gt_cache: Path,
                 audio_length: float = 10.0):
    # Run as a subprocess with CUDA_VISIBLE_DEVICES='' so laion_clap stays on CPU
    script = Path(__file__).parent / 'extract_clap_features.py'
    cmd = [
        sys.executable, str(script),
        '--audio_dir',    str(audio_dir),
        '--audio_cache',  str(audio_cache),
        '--captions_tsv', str(captions_tsv),
        '--gt_cache',     str(gt_cache),
        '--ckpt',         str(_CLAP_CKPT),
        '--audio_length', str(audio_length),
    ]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    print('[clap] launching CPU subprocess ...')
    subprocess.run(cmd, env=env, check=True)


def clap_score(gt_cache: Path, pred_cache: Path) -> float:
    text_feats  = torch.load(gt_cache   / 'clap_laion_text.pth',  weights_only=True)
    audio_feats = torch.load(pred_cache / 'clap_laion_audio.pth', weights_only=True)
    scores = []
    for vid in audio_feats:
        if vid not in text_feats:
            continue
        t = F.normalize(text_feats[vid].float(),  dim=-1)
        a = F.normalize(audio_feats[vid].float(), dim=-1)
        scores.append((t * a).sum().item())
    return float(sum(scores) / len(scores)) if scores else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_audio', type=Path, required=True)
    ap.add_argument('--gt_captions_tsv', type=Path, required=True)
    ap.add_argument('--pred_audio', type=Path, required=True)
    ap.add_argument('--label', type=str, default='pred')
    ap.add_argument('--audio_length', type=float, default=10.0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--out_json', type=Path, default=None)
    args = ap.parse_args()

    gt_cache   = cache_for(args.gt_audio)
    pred_cache = cache_for(args.pred_audio)

    maybe_extract(args.gt_audio,   gt_cache,   args.audio_length, args.batch_size)
    maybe_extract(args.pred_audio, pred_cache, args.audio_length, args.batch_size)

    extract_clap(args.pred_audio, pred_cache, args.gt_captions_tsv, gt_cache, args.audio_length)

    gt_pann    = torch.load(gt_cache   / 'pann_features.pth', weights_only=True)
    pred_pann  = torch.load(pred_cache / 'pann_features.pth', weights_only=True)
    gt_passt   = torch.load(gt_cache   / 'passt_features_embed.pth', weights_only=True)
    pred_passt = torch.load(pred_cache / 'passt_features_embed.pth', weights_only=True)
    gt_vgg     = torch.load(gt_cache   / 'vggish_features.pth', weights_only=True)
    pred_vgg   = torch.load(pred_cache / 'vggish_features.pth', weights_only=True)

    # FD
    gt_vgg_u      = unroll_dict(gt_vgg,   cat=True)
    pred_vgg_u    = unroll_dict(pred_vgg, cat=True)
    gt_pann_all   = unroll_dict_all_keys(gt_pann)
    pred_pann_all = unroll_dict_all_keys(pred_pann)
    gt_passt_u, pred_passt_u, _ = unroll_paired_dict(gt_passt, pred_passt)

    fd_vgg   = float(compute_fd(pred_vgg_u.numpy(),            gt_vgg_u.numpy()))
    fd_pann  = float(compute_fd(pred_pann_all['2048'].numpy(), gt_pann_all['2048'].numpy()))
    fd_passt = float(compute_fd(pred_passt_u.numpy(),          gt_passt_u.numpy()))

    # KL (paired PANN logits — match by filename)
    common = sorted(set(gt_pann.keys()) & set(pred_pann.keys()))
    gt_logits_t   = torch.stack([gt_pann[k]['logits']   for k in common])
    pred_logits_t = torch.stack([pred_pann[k]['logits'] for k in common])
    kl = compute_kl([pred_logits_t], gt_logits_t)

    # IS (pred PANN logits)
    pred_logits_all = torch.stack([v['logits'] for v in pred_pann.values()])
    isc = compute_isc({'logits': pred_logits_all}, 'logits',
                      rng_seed=2020, samples_shuffle=True, splits=10)

    clap = clap_score(gt_cache, pred_cache)

    metrics = {
        args.label: {
            'FD-VGG':      fd_vgg,
            'FD-PANN':     fd_pann,
            'FD-PASST':    fd_passt,
            'KL-sigmoid':  kl['kl_sigmoid'],
            'KL-softmax':  kl['kl_softmax'],
            'IS-mean':     isc['inception_score_mean'],
            'IS-std':      isc['inception_score_std'],
            'CLAP':        clap,
            'n_pred':      int(pred_passt_u.shape[0]),
        }
    }

    for k, v in metrics[args.label].items():
        print(f'  {k}: {v}')

    out_json = args.out_json or (args.pred_audio.parent / 'full_metrics.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f'\nSaved to {out_json}')


if __name__ == '__main__':
    main()
