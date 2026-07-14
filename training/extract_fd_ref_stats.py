"""Extract reference (μ, Σ) statistics for FD-PANN and FD-PaSST from real AudioCaps audio.

Path (i): real waveform → resample per-encoder → encoder → features. No VAE involved.

Output: sets/fd_ref_stats_panns.pt and sets/fd_ref_stats_passt.pt with keys
{mu, sigma, n, warm_features}.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from meanaudio.data.data_setup import error_avoidance_collate
from meanaudio.data.extraction.wav_dataset import WavTextClipsDataset
from meanaudio.model.fd_loss import build_encoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
NUM_SAMPLES = SAMPLE_RATE * 10  # 10 seconds


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Directory with AudioCaps wav/flac files')
    parser.add_argument('--captions_tsv', type=Path, required=True)
    parser.add_argument('--clips_tsv', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, default=Path('sets'))
    parser.add_argument('--encoders', nargs='+', default=['panns', 'passt'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--warm_size', type=int, default=5000)
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Cap total audios processed (-1 = all)')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info('Loading WavTextClipsDataset')
    dataset = WavTextClipsDataset(
        args.data_dir,
        captions_tsv=args.captions_tsv,
        clips_tsv=args.clips_tsv,
        sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        normalize_audio=True,
        reject_silent=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=error_avoidance_collate,
    )

    encoders = {}
    resamplers = {}
    # Online accumulators: store sum_x (D,) and sum_xx (D,D) in float64.
    # This is O(D²) memory regardless of dataset size — critical for frame-level
    # encoders that return B*T vectors per batch (e.g. passt_patch: T≈1188).
    acc = {}  # name → {'n': int, 'sum_x': (D,), 'sum_xx': (D,D), 'warm': list}
    for name in args.encoders:
        log.info(f'Building encoder: {name}')
        enc = build_encoder(name).to(device).eval()
        encoders[name] = enc
        if enc.target_sr != SAMPLE_RATE:
            resamplers[name] = torchaudio.transforms.Resample(SAMPLE_RATE, enc.target_sr).to(device)
        else:
            resamplers[name] = None
        acc[name] = {'n': 0, 'sum_x': None, 'sum_xx': None, 'warm': []}

    total = 0
    for batch in tqdm(loader, desc='Extracting features'):
        wav = batch['waveform'].to(device).float()  # (B, T) at 16k
        for name, enc in encoders.items():
            x = wav
            if resamplers[name] is not None:
                x = resamplers[name](x)
            feats = enc(x).detach().float()  # (N, D) — N may be B*T for frame encoders
            feats64 = feats.cpu().to(torch.float64)  # keep on CPU for accumulation
            N, D = feats64.shape
            a = acc[name]
            if a['sum_x'] is None:
                a['sum_x']  = torch.zeros(D,    dtype=torch.float64)
                a['sum_xx'] = torch.zeros(D, D, dtype=torch.float64)
            a['n']      += N
            a['sum_x']  += feats64.sum(dim=0)
            a['sum_xx'] += feats64.T @ feats64
            if len(a['warm']) < args.warm_size:
                a['warm'].append(feats.cpu())
        total += wav.shape[0]
        if 0 < args.max_samples <= total:
            break

    for name in args.encoders:
        a = acc[name]
        n = a['n']
        mu = a['sum_x'] / n
        # Σ = (Σ xᵢxᵢᵀ)/n − μμᵀ, with n−1 Bessel correction
        sigma = (a['sum_xx'] - n * mu.outer(mu)) / max(n - 1, 1)
        log.info(f'[{name}] N={n} D={mu.shape[0]}')

        warm = torch.cat(a['warm'], dim=0)[: args.warm_size].clone() if a['warm'] else torch.zeros(0)
        out = {
            'mu': mu.float(),
            'sigma': sigma.float(),
            'n': n,
            'warm_features': warm,
        }
        out_path = args.out_dir / f'fd_ref_stats_{name}.pt'
        torch.save(out, out_path)
        log.info(f'[{name}] saved to {out_path}')


if __name__ == '__main__':
    main()
