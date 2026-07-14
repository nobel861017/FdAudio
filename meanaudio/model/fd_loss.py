import logging
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torchaudio

log = logging.getLogger()


def _gather_features(features: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return features
    world_size = dist.get_world_size()
    if world_size == 1:
        return features
    gathered = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features.contiguous())
    return torch.cat(gathered, dim=0)


def _matrix_sqrt_psd(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Symmetric PSD matrix square root via eigendecomposition. A must be symmetric.
    A = 0.5 * (A + A.T)
    w, V = torch.linalg.eigh(A)
    w = torch.clamp(w, min=eps)
    return V @ torch.diag(torch.sqrt(w)) @ V.T


def _frechet_distance(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_real: torch.Tensor,
    sigma_real: torch.Tensor,
    sigma_real_sqrt: torch.Tensor,
    eig_eps: float = 1e-6,
) -> torch.Tensor:
    # FD = ||μ_q - μ_real||² + Tr(Σ_q + Σ_real - 2 (Σ_real^{1/2} Σ_q Σ_real^{1/2})^{1/2}).
    # All inputs in float64. sigma_real_sqrt is precomputed since Σ_real is fixed.
    diff = mu_q - mu_real
    mean_term = (diff * diff).sum()

    M = sigma_real_sqrt @ sigma_q @ sigma_real_sqrt
    M = 0.5 * (M + M.T)  # numerical symmetrization
    eigvals = torch.linalg.eigvalsh(M)
    eigvals = torch.clamp(eigvals, min=eig_eps)
    trace_sqrt = torch.sqrt(eigvals).sum()

    trace_term = torch.diagonal(sigma_q).sum() + torch.diagonal(sigma_real).sum() - 2.0 * trace_sqrt
    return mean_term + trace_term


def _batch_mean_cov(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # features: (N, D), float64
    n = features.shape[0]
    mu = features.mean(dim=0)
    centered = features - mu
    sigma = (centered.T @ centered) / max(n - 1, 1)
    return mu, sigma


class _PannEncoder(nn.Module):
    # Wraps av_bench's Cnn14 with the av-benchmark 16k config and returns 2048-dim features.
    target_sr = 16000

    def __init__(self):
        super().__init__()
        from av_bench.panns import Cnn14
        self.model = Cnn14(
            features_list=['2048', 'logits'],
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self._patch_inplace_relu()

    def _patch_inplace_relu(self):
        # PANN's Cnn14 uses F.relu_ (in-place) and nn.ReLU(inplace=True), which corrupt
        # tensors saved for backward when PANN is part of a larger autograd graph.
        # 1) Replace inplace ReLU modules with non-inplace.
        # 2) Globally swap torch.nn.functional.relu_ → torch.nn.functional.relu.
        # The global swap is permanent for the process — F.relu_ is functionally
        # interchangeable with F.relu (just no inplace) so other models are unaffected.
        import torch.nn.functional as F
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                module.inplace = False
        if not getattr(F, '_meanaudio_relu_patched', False):
            F.relu_ = F.relu
            F._meanaudio_relu_patched = True
        # Also kill in-place dropout (used in eval=no-op anyway, but safer).
        if not getattr(F, '_meanaudio_dropout_patched', False):
            _orig_dropout = F.dropout
            def _safe_dropout(input, p=0.5, training=True, inplace=False):
                return _orig_dropout(input, p=p, training=training, inplace=False)
            F.dropout = _safe_dropout
            F._meanaudio_dropout_patched = True

    @property
    def feature_dim(self) -> int:
        return 2048

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) at 16 kHz
        out = self.model(wav)
        return out['2048']


class _PasstEncoder(nn.Module):
    # Wraps hear21passt and returns 768-dim features (drops the 527 logits).
    target_sr = 32000

    def __init__(self):
        super().__init__()
        from hear21passt.base import get_basic_model
        self.model = get_basic_model(mode='all')
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @property
    def feature_dim(self) -> int:
        return 768

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) at 32 kHz, hear21passt expects up to 10s = 320000 samples
        if wav.size(-1) > 320000:
            wav = wav[..., :320000]
        out = self.model(wav)
        # av-bench: first 527 dims are logits, rest are 768-dim features
        return out[..., 527:]


class _VGGishEncoder(nn.Module):
    # Differentiable VGGish: exact VGGish mel filterbank as a fixed buffer + differentiable STFT.
    # The VGG backbone (preprocess=False) is frozen; mel→feature path carries gradients.
    target_sr = 16000

    # VGGish preprocessing constants — must match av_bench/vggish/mel_features.py
    _N_FFT   = 512   # next power-of-2 above 25 ms window (400 samples)
    _WIN_LEN = 400   # 25 ms at 16 kHz
    _HOP_LEN = 160   # 10 ms at 16 kHz
    _N_MELS  = 64
    _F_MIN   = 125.0
    _F_MAX   = 7500.0
    _LOG_OFFSET = 0.01
    _N_FRAMES   = 96   # frames per 0.96-s example

    def __init__(self):
        super().__init__()
        from av_bench.vggish.vggish import VGGish
        from av_bench.vggish.mel_features import spectrogram_to_mel_matrix
        # preprocess=False: we supply our own differentiable mel frontend
        self.model = VGGish(preprocess=False, postprocess=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # Patch inplace ReLU so backward through frozen VGGish doesn't corrupt saved activations
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                module.inplace = False

        # Exact VGGish mel filterbank matrix (computed once via numpy, then frozen as a buffer).
        # Shape: (n_fft//2+1, n_mels) = (257, 64). This matches av_bench's spectrogram_to_mel_matrix.
        mel_fb = spectrogram_to_mel_matrix(
            num_mel_bins=self._N_MELS,
            num_spectrogram_bins=self._N_FFT // 2 + 1,
            audio_sample_rate=self.target_sr,
            lower_edge_hertz=self._F_MIN,
            upper_edge_hertz=self._F_MAX,
        )
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb).float())  # (257, 64)

        # Left-aligned window of length n_fft: [hann(win_len), 0, ..., 0]
        # torch.stft center-pads shorter windows, but VGGish numpy zero-pads on the right.
        # Passing a full n_fft-length window avoids center-padding and gives exact match.
        win = torch.zeros(self._N_FFT)
        win[:self._WIN_LEN] = torch.hann_window(self._WIN_LEN, periodic=True)
        self.register_buffer('window', win)

    @property
    def feature_dim(self) -> int:
        return 128

    def _wav_to_examples(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) → (B, N, 96, 64) non-overlapping 0.96-s patches, fully differentiable
        # STFT magnitude: identical formula to VGGish's stft_magnitude (periodic Hann, zero-pad to N_FFT)
        stft = torch.stft(
            wav.reshape(-1, wav.shape[-1]),  # flatten batch for torch.stft
            n_fft=self._N_FFT,
            hop_length=self._HOP_LEN,
            window=self.window,              # full n_fft-length left-aligned window
            center=False,                   # no reflect-padding at signal edges
            return_complex=True,
        )                                    # (B, 257, T_frames)
        mag = stft.abs()                          # (B, 257, T_frames)
        mel = torch.matmul(mag.permute(0, 2, 1), self.mel_fb)  # (B, T_frames, 64)
        log_mel = torch.log(mel + self._LOG_OFFSET)            # (B, T_frames, 64)
        T = log_mel.shape[1]
        n = max(T // self._N_FRAMES, 1)
        if T < self._N_FRAMES:
            log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, self._N_FRAMES - T))
        log_mel = log_mel[:, :n * self._N_FRAMES]   # (B, n*96, 64)
        return log_mel.reshape(wav.shape[0], n, self._N_FRAMES, self._N_MELS)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) at 16 kHz → (B, 128), fully differentiable
        x = self._wav_to_examples(wav).contiguous()  # (B, N, 96, 64); contiguous for VGGish's .view()
        out = self.model(x)                          # (B, N, 128)
        return out.mean(dim=1)                       # (B, 128)


class _CLAPEncoder(nn.Module):
    # LAION-CLAP HTSAT-base audio encoder returning 512-dim projected embeddings.
    # torchlibrosa STFT + mel inside HTSAT is fully differentiable (pure PyTorch).
    target_sr = 48000

    _CKPT = './weights/music_speech_audioset_epoch_15_esc_89.98.pt'

    def __init__(self):
        super().__init__()
        import laion_clap
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        clap.load_ckpt(self._CKPT)
        self.htsat      = clap.model.audio_branch      # HTSAT_Swin_Transformer
        self.projection = clap.model.audio_projection  # Linear(1024→512)→ReLU→Linear(512→512)
        for p in self.htsat.parameters():
            p.requires_grad_(False)
        for p in self.projection.parameters():
            p.requires_grad_(False)
        self.htsat.eval()
        self.projection.eval()
        # Patch inplace ReLU in both submodules
        for m in list(self.htsat.modules()) + list(self.projection.modules()):
            if isinstance(m, nn.ReLU) and m.inplace:
                m.inplace = False

    @property
    def feature_dim(self) -> int:
        return 512

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) at 48 kHz → (B, 512)
        # HTSAT expects a dict; 'waveform' is extracted internally when enable_fusion=False.
        # torchlibrosa STFT+mel inside HTSAT is fully differentiable (pure PyTorch).
        out = self.htsat({'waveform': wav}, mixup_lambda=None, device=wav.device)
        emb = out['embedding']          # (B, 1024)
        return self.projection(emb)     # (B, 512)


class _BEATsEncoder(nn.Module):
    """BEATs (iter3+AS2M finetuned) encoder — 768-dim clip-level features at 16 kHz.

    Bypasses the classification head and averages the transformer encoder output
    over time patches to produce a single clip-level embedding per audio clip.
    """
    target_sr = 16000
    _CKPT = './weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    _BEATS_DIR = './third_party/beats'

    def __init__(self):
        super().__init__()
        import sys
        if self._BEATS_DIR not in sys.path:
            sys.path.insert(0, self._BEATS_DIR)
        from BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(self._CKPT, map_location='cpu', weights_only=False)
        cfg = BEATsConfig(checkpoint['cfg'])
        self._beats = BEATs(cfg)
        self._beats.load_state_dict(checkpoint['model'])
        for p in self._beats.parameters():
            p.requires_grad_(False)
        self._beats.eval()

    @property
    def feature_dim(self) -> int:
        return 768

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Replicate extract_features up to the encoder, skipping the predictor head.
        b = self._beats
        fbank = b.preprocess(wav, fbank_mean=15.41663, fbank_std=6.55582)
        feats = b.patch_embedding(fbank.unsqueeze(1))
        feats = feats.reshape(feats.shape[0], feats.shape[1], -1).transpose(1, 2)
        feats = b.layer_norm(feats)
        if b.post_extract_proj is not None:
            feats = b.post_extract_proj(feats)
        x, _ = b.encoder(b.dropout_input(feats), padding_mask=None)  # (B, T, 768)
        return x.mean(dim=1)  # (B, 768)


class _AudioMAEEncoder(nn.Module):
    """AudioMAE (ViT-B, AS2M pretrained) encoder — 768-dim clip-level features at 16 kHz.

    Converts audio to a log-mel spectrogram (128 bins × 1024 frames), feeds into
    the AudioMAE ViT, and averages patch tokens (excluding CLS) for a clip embedding.
    AudioMAE preprocessing follows the original paper (AudioSet statistics).
    """
    target_sr = 16000
    _N_FFT    = 400
    _HOP_LEN  = 160
    _N_MELS   = 128
    _T_FRAMES = 1024   # model expects exactly 1024 time frames
    _MEAN     = -4.2677393
    _STD      =  4.5689974

    def __init__(self):
        super().__init__()
        import timm
        self._vit = timm.create_model(
            'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
            pretrained=True,
        )
        for p in self._vit.parameters():
            p.requires_grad_(False)
        self._vit.eval()
        win = torch.hann_window(self._N_FFT)
        self.register_buffer('_win', win)

    @property
    def feature_dim(self) -> int:
        return 768

    def _to_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) at 16 kHz → (B, 1, T_FRAMES, N_MELS) log-mel, normalised
        import torch.nn.functional as F
        stft = torch.stft(
            wav.reshape(-1, wav.shape[-1]),
            n_fft=self._N_FFT, hop_length=self._HOP_LEN,
            window=self._win, center=True, return_complex=True,
        )  # (B, F, T)
        power = stft.abs().pow(2)
        # Build mel filterbank on the fly (frequency bins → mel bins)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self._N_FFT // 2 + 1,
            f_min=0.0, f_max=self.target_sr / 2,
            n_mels=self._N_MELS,
            sample_rate=self.target_sr,
        ).to(wav.device)  # (F, n_mels)
        mel = torch.matmul(power.permute(0, 2, 1), mel_fb)  # (B, T, n_mels)
        log_mel = torch.log(mel + 1e-5)
        log_mel = (log_mel - self._MEAN) / (self._STD * 2)  # normalise
        # Pad / crop time axis to _T_FRAMES
        T = log_mel.shape[1]
        if T < self._T_FRAMES:
            log_mel = F.pad(log_mel, (0, 0, 0, self._T_FRAMES - T))
        else:
            log_mel = log_mel[:, :self._T_FRAMES, :]
        # AudioMAE expects (B, 1, T_FRAMES, N_MELS)
        return log_mel.unsqueeze(1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        fbank = self._to_fbank(wav)                        # (B, 1, 1024, 128)
        tokens = self._vit.forward_features(fbank)         # (B, 1+N, 768)
        return tokens[:, 1:, :].mean(dim=1)                # (B, 768), skip CLS


class _PasstPatchBase(nn.Module):
    """Base: loads PaSST once, hooks the norm layer to expose CLS + patch tokens.

    PaSST (DeiT-style) sequence after norm: [CLS, dist, patch_0, ..., patch_{N-1}]
    Patches are stored in freq-major order (freq axis varies slowest):
        patch index = freq_i * T_time + time_j
    Freq-averaging collapses the 12 frequency bins → one 768-dim vector per time step.

    _N_FREQ = floor((128 - 16) / 10) + 1 = 12  (PaSST mel-bins=128, kernel=16, stride=10)
    """
    target_sr = 32000
    _N_FREQ = 12  # frequency-axis patches (fixed by PaSST architecture)
    _N_SPECIAL = 2  # CLS + distillation tokens prepended before patches

    def __init__(self):
        super().__init__()
        from hear21passt.base import get_basic_model
        self._passt = get_basic_model(mode='all').eval()
        for p in self._passt.parameters():
            p.requires_grad_(False)
        self._tokens = None  # (B, _N_SPECIAL + N_patches, 768), set by hook
        # get_basic_model returns PasstBasicWrapper; the ViT lives in .net
        self._passt.net.norm.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._tokens = output  # (B, 2+N, 768); grad_fn intact when input has grad

    def _forward_passt(self, wav: torch.Tensor):
        """Single PaSST forward. Returns (cls, time_patches).

        cls:          (B, 768)    CLS token — global clip identity
        time_patches: (B, T, 768) freq-averaged temporal patches
        """
        if wav.size(-1) > 320000:
            wav = wav[..., :320000]
        self._passt(wav)                                      # fires hook
        tokens = self._tokens                                 # (B, 2+N, 768)
        cls = tokens[:, 0, :]                                 # (B, 768)
        patches = tokens[:, self._N_SPECIAL:, :]              # (B, N, 768)
        B, N, D = patches.shape
        T = N // self._N_FREQ
        patches = patches[:, : T * self._N_FREQ, :]           # align to full freq blocks
        # freq-major → reshape to (B, N_freq, T, D), mean over freq axis
        time_patches = patches.reshape(B, self._N_FREQ, T, D).mean(dim=1)  # (B, T, 768)
        return cls, time_patches

    @property
    def feature_dim(self) -> int:
        return 768


class _PasstHierEncoder(_PasstPatchBase):
    """Option 1 — Hierarchical: CLS clip token + time patches in one joint distribution.

    Returns (B*(1+T), 768): B CLS vectors stacked with B*T frame vectors.
    FD matches both clip-level scene identity and frame-level temporal texture jointly.
    """
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        cls, patches = self._forward_passt(wav)   # (B, 768), (B, T, 768)
        B, T, D = patches.shape
        cls_exp = cls.unsqueeze(1)                # (B, 1, 768)
        combined = torch.cat([cls_exp, patches], dim=1)  # (B, 1+T, 768)
        return combined.reshape(B * (1 + T), D)


class _PasstDevEncoder(_PasstPatchBase):
    """Option 2 — Within-clip deviation: frame vectors minus the clip mean.

    Returns (B*T, 768). Captures how frames deviate from their clip centre —
    orthogonal to clip-level FD, so no redundancy with PANN.
    Mean of deviations is zero by construction per clip; FD over deviations
    measures within-clip temporal variation across the dataset.
    """
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _, patches = self._forward_passt(wav)             # (B, T, 768)
        deviations = patches - patches.mean(dim=1, keepdim=True)  # (B, T, 768)
        B, T, D = deviations.shape
        return deviations.reshape(B * T, D)


class _PasstCLSEncoder(_PasstPatchBase):
    """Option 3a — CLS token only (clip-level identity, one vector per clip).

    Paired with _PasstPatchEncoder as two separate FD terms.
    CLS captures global scene identity; patch FD captures local texture.
    """
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        cls, _ = self._forward_passt(wav)
        return cls  # (B, 768)


class _PasstPatchEncoder(_PasstPatchBase):
    """Option 3b — Freq-averaged time patches (frame-level texture, T vectors per clip).

    Paired with _PasstCLSEncoder. Each vector = 768-dim summary of one ~100ms window.
    """
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _, patches = self._forward_passt(wav)   # (B, T, 768)
        B, T, D = patches.shape
        return patches.reshape(B * T, D)


class _PasstTransEncoder(_PasstPatchBase):
    """Option 4 — Temporal transitions: consecutive frame differences (B*(T-1), 768).

    FD over transitions captures rhythm, dynamics and onset patterns — explicitly
    encodes temporal ordering, which raw frame FD treats as i.i.d.
    """
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _, patches = self._forward_passt(wav)                    # (B, T, 768)
        transitions = patches[:, 1:, :] - patches[:, :-1, :]    # (B, T-1, 768)
        B, T1, D = transitions.shape
        return transitions.reshape(B * T1, D)


def build_encoder(name: str) -> nn.Module:
    if name == 'panns':       return _PannEncoder()
    if name == 'passt':       return _PasstEncoder()
    if name == 'vggish':      return _VGGishEncoder()
    if name == 'clap':        return _CLAPEncoder()
    if name == 'beats':       return _BEATsEncoder()
    if name == 'audiomae':    return _AudioMAEEncoder()
    if name == 'passt_hier':  return _PasstHierEncoder()
    if name == 'passt_dev':   return _PasstDevEncoder()
    if name == 'passt_cls':   return _PasstCLSEncoder()
    if name == 'passt_patch': return _PasstPatchEncoder()
    if name == 'passt_trans': return _PasstTransEncoder()
    raise ValueError(f'Unknown FD encoder: {name}')


class SingleFDLoss(nn.Module):
    """FD loss against a single frozen audio encoder, with a FIFO feature queue."""

    def __init__(
        self,
        encoder: nn.Module,
        ref_stats_path: str,
        source_sr: int = 16000,
        queue_size: int = 50000,
        ema_beta: float = 0.0,
        eig_eps: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.queue_size = queue_size
        self.ema_beta = ema_beta
        self.eig_eps = eig_eps

        self.target_sr = encoder.target_sr
        self.feature_dim = encoder.feature_dim
        if source_sr != self.target_sr:
            self.resampler = torchaudio.transforms.Resample(source_sr, self.target_sr)
        else:
            self.resampler = None

        ref = torch.load(ref_stats_path, map_location='cpu', weights_only=False)
        mu_real = ref['mu'].to(torch.float64)
        sigma_real = ref['sigma'].to(torch.float64)
        # Precompute Σ_real^{1/2} once; it's needed for every FD computation.
        sigma_real_sqrt = _matrix_sqrt_psd(sigma_real, eps=eig_eps)

        self.register_buffer('mu_real', mu_real)
        self.register_buffer('sigma_real', sigma_real)
        self.register_buffer('sigma_real_sqrt', sigma_real_sqrt)

        # Queue starts EMPTY. Filling it with real `warm_features` would bias
        # μ_q, Σ_q toward μ_real, Σ_real (artificially small FD early), and
        # then "drift outward" as generated samples replace real ones — making
        # the loss appear to grow even when the model is improving.
        # Initial iterations rely on the small batch alone; the queue fills
        # naturally over training.
        queue = torch.zeros(queue_size, self.feature_dim, dtype=torch.float32)
        queue_fill = torch.tensor(0, dtype=torch.long)
        self.register_buffer('queue', queue)
        self.register_buffer('queue_fill', queue_fill)
        self.register_buffer('queue_ptr', torch.tensor(0, dtype=torch.long))

        # EMA estimator (alternative to the FIFO queue). When ema_beta>0, the
        # generated stats are an exponential moving average of the per-batch mean
        # and second moment instead of a hard FIFO window. ema_beta is the decay
        # on history; the current batch carries weight (1-ema_beta). The FD-loss
        # paper (Jiawei-Yang/FD-loss) recommends and defaults to ema_beta=0.999.
        if ema_beta > 0:
            self.register_buffer('ema_mu', torch.zeros(self.feature_dim, dtype=torch.float64))
            self.register_buffer('ema_M2', torch.zeros(self.feature_dim, self.feature_dim, dtype=torch.float64))
            self.register_buffer('ema_init', torch.tensor(0, dtype=torch.long))

    def _extract(self, wav_16k: torch.Tensor) -> torch.Tensor:
        wav = wav_16k.float()
        if self.resampler is not None:
            wav = self.resampler(wav)
        # Clone to detach from any in-place ops that PANN/PaSST may perform internally
        # (their final ReLU outputs sometimes get modified in-place in their forwards).
        return self.encoder(wav).clone()

    @torch.no_grad()
    def _push_queue(self, features_detached: torch.Tensor):
        # features_detached is the global-batch (after all_gather) detached features
        n = features_detached.shape[0]
        ptr = int(self.queue_ptr.item())
        if n >= self.queue_size:
            self.queue.copy_(features_detached[-self.queue_size:].to(self.queue.dtype))
            self.queue_ptr.fill_(0)
            self.queue_fill.fill_(self.queue_size)
            return
        end = ptr + n
        if end <= self.queue_size:
            self.queue[ptr:end].copy_(features_detached.to(self.queue.dtype))
        else:
            first = self.queue_size - ptr
            self.queue[ptr:].copy_(features_detached[:first].to(self.queue.dtype))
            self.queue[:end - self.queue_size].copy_(features_detached[first:].to(self.queue.dtype))
        self.queue_ptr.fill_(end % self.queue_size)
        new_fill = min(int(self.queue_fill.item()) + n, self.queue_size)
        self.queue_fill.fill_(new_fill)

    def _forward_ema(self, features: torch.Tensor) -> torch.Tensor:
        # EMA estimator (matches Jiawei-Yang/FD-loss queue.py): beta is the decay
        # on the running stats; the current batch carries weight (1-beta) and the
        # gradient. mu = beta*mu_ema + (1-beta)*batch_mean; m2 likewise; Sigma = m2 - mu mu^T.
        beta = self.ema_beta
        B = features.shape[0]
        f64 = features.to(torch.float64)
        batch_mu = f64.mean(dim=0)                # (D,) grad
        batch_m2 = (f64.t() @ f64) / B            # (D,D) grad
        if int(self.ema_init.item()) == 0:
            # First step: use the batch directly and seed the EMA (avoids a long
            # cold start when beta is near 1; our generator is pretrained so the
            # first batch is already a reasonable estimate).
            mu_q, m2_q = batch_mu, batch_m2
            with torch.no_grad():
                self.ema_mu.copy_(batch_mu.detach())
                self.ema_M2.copy_(batch_m2.detach())
                self.ema_init.fill_(1)
        else:
            mu_q = beta * self.ema_mu.detach() + (1.0 - beta) * batch_mu
            m2_q = beta * self.ema_M2.detach() + (1.0 - beta) * batch_m2
            with torch.no_grad():
                g = _gather_features(features.detach().clone().float()).to(torch.float64)
                gB = g.shape[0]
                self.ema_mu.mul_(beta).add_(g.mean(dim=0), alpha=1.0 - beta)
                self.ema_M2.mul_(beta).addmm_(g.t(), g, alpha=(1.0 - beta) / gB)
        sigma_q = m2_q - torch.outer(mu_q, mu_q)
        return _frechet_distance(mu_q, sigma_q, self.mu_real, self.sigma_real,
                                 self.sigma_real_sqrt, eig_eps=self.eig_eps)

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        # wav_16k: (B, T) waveform at 16 kHz with grad
        features = self._extract(wav_16k)  # (B, D)

        if self.ema_beta > 0:
            return self._forward_ema(features)

        # Combine current batch (with grad) with queue (detached) for empirical stats
        fill = int(self.queue_fill.item())
        if fill > 0:
            queue_feats = self.queue[:fill].detach()
            all_feats = torch.cat([features, queue_feats.to(features.dtype)], dim=0)
        else:
            all_feats = features

        all_feats_64 = all_feats.to(torch.float64)
        mu_q, sigma_q = _batch_mean_cov(all_feats_64)

        fd = _frechet_distance(mu_q, sigma_q, self.mu_real, self.sigma_real,
                               self.sigma_real_sqrt, eig_eps=self.eig_eps)

        # Update queue with this step's features (detached, gathered across ranks).
        # Clone to ensure storage independence from `features` (which is still in the
        # autograd graph and must not be aliased to the queue buffer).
        gathered = _gather_features(features.detach().clone().float())
        self._push_queue(gathered)

        return fd


class MultiFDLoss(nn.Module):
    """Sum of magnitude-normalized SingleFDLoss terms across multiple encoders."""

    def __init__(
        self,
        encoders: list[str],
        ref_stats_paths: dict[str, str],
        source_sr: int = 16000,
        queue_size: int = 50000,
        weights: Optional[list[float]] = None,
        norm_eps: float = 0.01,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.encoder_names = list(encoders)
        if weights is None:
            weights = [1.0] * len(self.encoder_names)
        assert len(weights) == len(self.encoder_names)
        self.weights = weights
        self.norm_eps = norm_eps

        modules = {}
        for name in self.encoder_names:
            enc = build_encoder(name)
            modules[name] = SingleFDLoss(
                encoder=enc,
                ref_stats_path=ref_stats_paths[name],
                source_sr=source_sr,
                queue_size=queue_size,
                ema_beta=ema_beta,
            )
        self.fd_modules = nn.ModuleDict(modules)

    def forward(self, wav_16k: torch.Tensor) -> tuple[torch.Tensor, dict]:
        total = wav_16k.new_zeros(())
        info = {}
        for name, w in zip(self.encoder_names, self.weights):
            fd = self.fd_modules[name](wav_16k)
            fd_norm = fd / (fd.detach() + self.norm_eps)
            total = total + w * fd_norm
            info[name] = fd.detach().float()
        return total, info
