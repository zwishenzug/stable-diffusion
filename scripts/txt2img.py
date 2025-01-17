import argparse, os, sys, glob
import cv2
import torch
from transformers import logging
logging.set_verbosity_error()
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from sd_utils import read_prompts_from_file, PromptInfo, build_cond, clean_prompt
from sd_utils import hrfix_process

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        nargs="?",
        default="",
        help="the negative prompt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--save_grid",
        action='store_true',
        help="save a grid of the generated images at the end of the process",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--half",
        action='store_true',
        help="use half precision (fp16)",
    )
    parser.add_argument(
        "--benchmark",
        action='store_true',
        help="benchmark operations and attempt to optimize on first run",
    )
    parser.add_argument(
        "--jpeg",
        action='store_true',
        help="output as jpeg instead of png",
    )
    parser.add_argument(
        "--hrfix",
        action='store_true',
        help="hires fix, generates a smaller version of the image first then upscales using img2img",
    )
    parser.add_argument(
        "--hrstrength",
        type=float,
        default=0.60,
        help="strength for noising/unnoising during hires fix. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--interactive",
        action='store_true',
        help="instead of using the --prompt param, will sit in an input() loop taking prompts that way",
    )
    opt = parser.parse_args()

    output_format = "png"
    if opt.jpeg:
        output_format = "jpeg"

    if opt.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    if opt.half:
        print("Using half precision (fp16) where possible")
        torch.set_default_tensor_type(torch.HalfTensor)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    if opt.half:
        model = model.half()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms and opt.hrfix:
        print("WARNING: plms sampling and hrfix are incompatible.  Using ddim sampling instead")
        opt.plms = False
        
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        assert opt.prompt is not None
        prompts_data = [PromptInfo(prompt=clean_prompt(opt.prompt), neg_prompt=clean_prompt(opt.neg_prompt))]
    else:
        print(f"reading prompts from {opt.from_file}")
        prompts_data = read_prompts_from_file(opt.from_file)

    # figure out the buffs/nerfs and remake the prompt
    for pi in prompts_data:
        pi.parse()

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    running = True

    while running:
        if opt.interactive:
            opt.n_iter = 1
            prompt_text = input('>')
            if prompt_text.lower().startswith('--seed'):
                try:
                    seed_val = int(prompt_text[6:])
                except:
                    print("Invalid seed value")
                    continue
                opt.seed = seed_val
                continue
            pi = PromptInfo(prompt = clean_prompt(prompt_text))
            neg_pos = prompt_text.find('###')
            if neg_pos >= 0:
                pi.prompt = clean_prompt(prompt_text[0:neg_pos])
                pi.neg_prompt = clean_prompt(prompt_text[neg_pos+3:])
            pi.parse()
            prompts_data = [pi]

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        seed = opt.seed + n
                        for pi in tqdm(prompts_data, desc="data"):
                            seed_everything(seed)
                        
                            print(f"prompt    : '{pi.prompt}'")
                            print(f"neg prompt: '{pi.neg_prompt}'")
                        
                            uc = None
                            if opt.scale != 1.0:
                                uc = build_cond(model, device, batch_size, pi.neg_prompt, pi.neg_buffs, pi.neg_nerfs)

                            c = build_cond(model, device, batch_size, pi.prompt, pi.buffs, pi.nerfs)

                            if opt.hrfix:
                                x_samples_ddim = hrfix_process(model,
                                                               device,
                                                               opt.W,
                                                               opt.H,
                                                               c,
                                                               uc,
                                                               batch_size,
                                                               opt,
                                                               seed,
                                                               sampler)
                            else:
                                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                 conditioning=c,
                                                                 batch_size=opt.n_samples,
                                                                 shape=shape,
                                                                 verbose=False,
                                                                 unconditional_guidance_scale=opt.scale,
                                                                 unconditional_conditioning=uc,
                                                                 eta=opt.ddim_eta,
                                                                 x_T=start_code)

                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for i, x_sample in enumerate(x_samples_ddim):
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img.save(os.path.join(sample_path, f"{base_count:05}_{seed}_{i}.{output_format}"))
                                    base_count += 1

                            if opt.save_grid:
                                all_samples.append(x_samples_ddim)

                    if opt.save_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img.save(os.path.join(outpath, f'grid-{grid_count:04}.{output_format}'))
                        grid_count += 1

                    toc = time.time()
        if not opt.interactive:
            running = False

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
