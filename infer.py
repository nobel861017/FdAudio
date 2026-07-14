import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
from meanaudio.eval_utils import (ModelConfig, all_model_cfg, generate_mf, generate_fm, setup_eval_logging)
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import MeanAudio, get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=9.975)  # for 312 latents (used in training), seq_config should has a duration of 9.975s 
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=2024)
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--model_path', type=str, help='Ckpt path of trained model')
    parser.add_argument('--encoder_name', choices=['clip', 't5', 't5_clap'], type=str, help='text encoder name')
    parser.add_argument('--use_rope', action='store_true', help='Whether or not use position embedding for model')
    parser.add_argument('--text_c_dim', type=int, default=512, 
                        help='Dim of the text_features_c, 1024 for pooled T5 and 512 for CLAP')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_meanflow', action='store_true', help='Whether or not use mean flow for inference')
    args = parser.parse_args()

    if args.debug: 
        import debugpy
        debugpy.listen(6666) 
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()  
    
    model: ModelConfig = all_model_cfg['fdaudio']  # FdAudio is the only released model
    seq_cfg = model.seq_cfg  

    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)
    # resolve checkpoints: the kph68/FdAudio repo bundles every component needed for inference
    # (fdaudio.pth, the VAE, the BigVGAN vocoder, and the LAION-CLAP checkpoint). Download the
    # whole repo into ./weights on first run; FLAN-T5 is fetched separately from the HF hub.
    model_path = args.model_path
    if model_path is None or not Path(model_path).exists():
        from huggingface_hub import snapshot_download
        log.info('FdAudio components not found locally, downloading from Hugging Face (kph68/FdAudio)...')
        snapshot_download(repo_id='kph68/FdAudio', local_dir='./weights')
        model_path = './weights/fdaudio.pth'
    # load a pretrained model
    net: MeanAudio = get_mean_audio(model.model_name,
                                    use_rope=args.use_rope,
                                    text_c_dim=args.text_c_dim).to(device, dtype).eval()
    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    if args.use_meanflow:
        mf = MeanFlow(steps=num_steps)
    else:
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  enable_conditions=True,
                                  encoder_name=args.encoder_name, 
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len)
    prompts: str = [args.prompt]

   
    if args.use_meanflow:
        for prompt in tqdm(prompts): 
            log.info(f'Prompt: {prompt}')
            log.info(f'Negative prompt: {negative_prompt}')
            audios = generate_mf([prompt],
                                  negative_text=[negative_prompt],
                                  feature_utils=feature_utils,
                                  net=net,
                                  mf=mf,
                                  rng=rng,
                                  cfg_strength=cfg_strength)
            audio = audios.float().cpu()[0]
            safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = output_dir / f'{safe_filename}--numsteps{num_steps}.wav'
            torchaudio.save( save_path, audio, seq_cfg.sampling_rate)
            log.info(f'Audio saved to {save_path}')
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    else:
        for prompt in tqdm(prompts): 
            log.info(f'Prompt: {prompt}')
            log.info(f'Negative prompt: {negative_prompt}')
            audios = generate_fm([prompt],
                                  negative_text=[negative_prompt],
                                  feature_utils=feature_utils,
                                  net=net,
                                  fm=fm,
                                  rng=rng,
                                  cfg_strength=cfg_strength)
            audio = audios.float().cpu()[0]
            safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = output_dir / f'{safe_filename}--numsteps{num_steps}.wav'
            torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

            log.info(f'Audio saved to {save_path}')
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
