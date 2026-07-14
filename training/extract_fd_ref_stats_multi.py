"""Extract reference (μ, Σ) FD statistics across multiple datasets (AudioCaps + WavCaps).

Builds a ConcatDataset over (root, captions_tsv, clips_tsv) triples, runs all
selected encoders in a single I/O pass, and writes per-encoder ref stats.

Output: <out_dir>/fd_ref_stats_<name>_<suffix>.pt with keys {mu, sigma, n, warm_features}.
"""
import argparse
import logging
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from meanaudio.data.data_setup import error_avoidance_collate
from meanaudio.data.extraction.wav_dataset import WavTextClipsDataset
from meanaudio.model.fd_loss import build_encoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
NUM_SAMPLES = SAMPLE_RATE * 10  # 10 seconds


def parse_dataset_spec(spec: str):
    """Parse 'root|captions_tsv|clips_tsv' into a 3-tuple of Paths."""
    parts = spec.split('|')
    if len(parts) != 3:
        raise ValueError(f"Expected 'root|captions|clips', got: {spec}")
    return Path(parts[0]), Path(parts[1]), Path(parts[2])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True,
                        help="One or more 'root|captions_tsv|clips_tsv' specs")
    parser.add_argument('--out_dir', type=Path, default=Path('sets'))
    parser.add_argument('--suffix', type=str, default='acwc',
                        help='Filename suffix: fd_ref_stats_<name>_<suffix>.pt')
    parser.add_argument('--encoders', nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--warm_size', type=int, default=5000)
    parser.add_argument('--max_samples', type=int, default=-1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'Building ConcatDataset over {len(args.datasets)} datasets')
    sub_datasets = []
    for spec in args.datasets:
        root, captions_tsv, clips_tsv = parse_dataset_spec(spec)
        log.info(f'  root={root}')
        sub_datasets.append(WavTextClipsDataset(
            root,
            captions_tsv=captions_tsv,
            clips_tsv=clips_tsv,
            sample_rate=SAMPLE_RATE,
            num_samples=NUM_SAMPLES,
            normalize_audio=True,
            reject_silent=True,
        ))
    dataset = ConcatDataset(sub_datasets)
    log.info(f'Total clips: {len(dataset)}')

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
    acc = {}
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
        wav = batch['waveform'].to(device).float()
        for name, enc in encoders.items():
            x = wav
            if resamplers[name] is not None:
                x = resamplers[name](x)
            feats = enc(x).detach().float()
            feats64 = feats.cpu().to(torch.float64)
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
        sigma = (a['sum_xx'] - n * mu.outer(mu)) / max(n - 1, 1)
        log.info(f'[{name}] N={n} D={mu.shape[0]}')

        warm = torch.cat(a['warm'], dim=0)[: args.warm_size].clone() if a['warm'] else torch.zeros(0)
        out = {
            'mu': mu.float(),
            'sigma': sigma.float(),
            'n': n,
            'warm_features': warm,
        }
        out_path = args.out_dir / f'fd_ref_stats_{name}_{args.suffix}.pt'
        torch.save(out, out_path)
        log.info(f'[{name}] saved to {out_path}')


if __name__ == '__main__':
    main()
