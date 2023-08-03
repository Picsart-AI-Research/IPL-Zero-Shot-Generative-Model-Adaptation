import os
import torch
import numpy as np
from model.ZSSGAN import ZSSGAN, SG2Generator
from options.train_options import TrainOptions
from utils.file_utils import save_images
import warnings
warnings.filterwarnings("ignore")

dataset_sizes = {
    "ffhq":   1024,
    "dog":    512,
}

def eval(args):
    if args.auto_compute:
        args.size = dataset_sizes[args.source_model_type]

    sample_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    generator_ema = SG2Generator(args.adapted_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier, device=device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier, device=device)
    generator_frozen.freeze_layers()
    generator_frozen.eval()

    torch.manual_seed(args.seed2)
    torch.cuda.manual_seed_all(args.seed2)
    np.random.seed(args.seed2)

    fixed_z = torch.randn(args.n_sample, 512, device=device)

    with torch.no_grad():
        sample_w = generator_ema.style([fixed_z])
        sample = generator_ema(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]
        sample_src = generator_frozen(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]

    grid_rows = int(args.n_sample ** 0.5) 
    save_images(sample, sample_dir, "iter", grid_rows, 300)
    save_images(sample_src, sample_dir, "src", grid_rows, 0)



if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()
    eval(args)