"""Extract LAION-CLAP audio and text features on CPU (avoids GPU OOM from HTSAT attention).
Run as a subprocess with CUDA_VISIBLE_DEVICES='' set before any torch import.

Usage:
    python training/extract_clap_features.py \
        --audio_dir PATH --audio_cache PATH \
        --captions_tsv PATH --gt_cache PATH \
        --ckpt PATH [--audio_length 10.0]
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # must be before any torch import

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import laion_clap
from torch.utils.data import DataLoader
from tqdm import tqdm

_orig_load = torchaudio.load
def _patched_load(*a, **kw):
    kw.setdefault('backend', 'soundfile')
    wav, sr = _orig_load(*a, **kw)
    return wav.clone(), sr
torchaudio.load = _patched_load

from av_bench.data.audio_dataset import AudioDataset


def _pad_collate(batch):
    max_len = max(item[0].shape[-1] for item in batch)
    wavs, names = [], []
    for wav, fn in batch:
        if wav.shape[-1] < max_len:
            wav = F.pad(wav, (0, max_len - wav.shape[-1]))
        wavs.append(wav)
        names.append(fn)
    return torch.stack(wavs, dim=0), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio_dir',    type=Path, required=True)
    ap.add_argument('--audio_cache',  type=Path, required=True)
    ap.add_argument('--captions_tsv', type=Path, required=True)
    ap.add_argument('--gt_cache',     type=Path, required=True)
    ap.add_argument('--ckpt',         type=str,  required=True)
    ap.add_argument('--audio_length', type=float, default=10.0)
    args = ap.parse_args()

    need_audio = not (args.audio_cache / 'clap_laion_audio.pth').exists()
    need_text  = not (args.gt_cache    / 'clap_laion_text.pth').exists()
    if not need_audio and not need_text:
        print('[skip-clap] both caches exist')
        return

    print('[clap] loading model on CPU ...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(args.ckpt)
    model = model.cpu().eval()

    if need_text:
        rows = {}
        with open(args.captions_tsv) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                rows[row['id']] = row['caption']
        out, ids = {}, list(rows.keys())
        for i in tqdm(range(0, len(ids), 64), desc='CLAP text'):
            batch_ids = ids[i:i+64]
            with torch.inference_mode():
                emb = model.get_text_embedding([rows[k] for k in batch_ids], use_tensor=True)
            for j, vid in enumerate(batch_ids):
                out[vid] = emb[j].cpu()
        torch.save(out, args.gt_cache / 'clap_laion_text.pth')
        print(f'[clap] text features saved: {args.gt_cache}/clap_laion_text.pth')

    if need_audio:
        args.audio_cache.mkdir(parents=True, exist_ok=True)
        audios = sorted(list(args.audio_dir.glob('*.wav')) + list(args.audio_dir.glob('*.flac')),
                        key=lambda x: x.stem)
        ds = AudioDataset(audios, audio_length=args.audio_length, sr=48000)
        dl = DataLoader(ds, batch_size=8, num_workers=0, collate_fn=_pad_collate)
        out = {}
        for wav, fns in tqdm(dl, desc='CLAP audio'):
            wav = wav.squeeze(1).float()
            with torch.inference_mode():
                emb = model.get_audio_embedding_from_data(wav, use_tensor=True)
            for i, fn in enumerate(fns):
                out[fn] = emb[i].cpu()
        torch.save(out, args.audio_cache / 'clap_laion_audio.pth')
        print(f'[clap] audio features saved: {args.audio_cache}/clap_laion_audio.pth')


if __name__ == '__main__':
    main()
