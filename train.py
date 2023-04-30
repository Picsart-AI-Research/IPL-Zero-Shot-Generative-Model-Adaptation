'''
Example commands:
    CUDA_VISIBLE_DEVICES=0 python train.py  --frozen_gen_ckpt ./pre_stylegan/stylegan2-ffhq-config-f.pt \
                                            --source_model_type "ffhq" \
                                            --output_interval 300 \
                                            --save_interval 300 \
                                            --auto_compute \
                                            --source_class "photo" \
                                            --target_class "disney" \
                                            --run_stage1 \
                                            --batch_mapper 32 \
                                            --lr_mapper 0.05 \
                                            --iter_mapper 300 \
                                            --ctx_init "a photo of a" \
                                            --n_ctx 4 \
                                            --lambda_l 1 \
                                            --run_stage2 \
                                            --batch 2 \
                                            --lr 0.002 \
                                            --iter 300 \
                                            --output_dir ./output/disney
'''


import os
import numpy as np
import torch
from tqdm import tqdm
from model.ZSSGAN import ZSSGAN, SG2Generator
from utils.file_utils import save_images
from utils.training_utils import mixing_noise
from mapper import latent_mappers
from options.train_options import TrainOptions
import clip
import warnings
warnings.filterwarnings("ignore")


dataset_sizes = {
    "ffhq":   1024,
    "dog":    512,
}

def text_encoder(source_prompts, source_tokenized_prompts, clip_model):
    x = source_prompts + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(0, 2, 1, 3)  # NLD -> LND
    for j in range(len(x)):
        x[j] = clip_model.transformer(x[j])
    x = x.permute(0, 2, 1, 3)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    text_features = x[:, torch.arange(x.shape[1]), source_tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection

    return text_features

def compute_text_features(prompts, source_prefix, source_suffix, source_tokenized_prompts, clip_model, batch):
    source_ctx = prompts.unsqueeze(1)
    source_prefix = source_prefix.expand(batch, -1, -1, -1)
    source_suffix = source_suffix.expand(batch, -1, -1, -1)
    source_prompts = torch.cat(
        [
            source_prefix,  # (batch, n_cls, 1, dim)
            source_ctx,  # (batch, n_cls, n_ctx, dim)
            source_suffix,  # (batch, n_cls, *, dim)
        ],
        dim=2,
    )
    text_features = text_encoder(source_prompts, source_tokenized_prompts, clip_model)
    return text_features

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))

def train(args):
    if args.auto_compute:
        args.auto_layer_k = int(2 * (2 * np.log2(dataset_sizes[args.source_model_type]) - 2) / 3)
        args.size = dataset_sizes[args.source_model_type]

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)
    with torch.no_grad():
        clip_loss_models, clip_model_weights = net.get_clip_loss_models()  

    g_reg_ratio = args.g_reg_every / (
            args.g_reg_every + 1)  # using original SG2 params. Not currently using r1 regularization, may need to change.

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")  
    ckpt_dir = os.path.join(args.output_dir, "checkpoint")  
    ckpt_dir_m = os.path.join(ckpt_dir, "mapper")
    ckpt_dir_g = os.path.join(ckpt_dir, "generator")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ckpt_dir_m, exist_ok=True)
    os.makedirs(ckpt_dir_g, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(args.seed1)
    torch.cuda.manual_seed_all(args.seed1)
    np.random.seed(args.seed1)

    # pre
    clip_model = clip_loss_models[args.clip_models[0]].model
    n_dim = clip_model.ln_final.weight.shape[0]
    if args.ctx_init != "":
        ctx_init = args.ctx_init.replace("_", " ")
        args.n_ctx = len(ctx_init.split(" "))
        prompt_prefix = ctx_init
    else:
        prompt_prefix = " ".join(["X"] * args.n_ctx)
    source_prompts = [prompt_prefix + " " + args.source_class]
    target_prompts = [prompt_prefix + " " + args.target_class]
    print("source prompts", source_prompts)
    print("target prompts", target_prompts)
    source_tokenized_prompts = torch.cat(  
        [clip.tokenize(p) for p in source_prompts]).to(device)
    target_tokenized_prompts = torch.cat(  
        [clip.tokenize(p) for p in target_prompts]).to(device)
    source_embedding = clip_model.token_embedding(source_tokenized_prompts).type(clip_model.dtype)
    target_embedding = clip_model.token_embedding(target_tokenized_prompts).type(clip_model.dtype)
    source_prefix = source_embedding[:, :1, :].detach()
    source_suffix = source_embedding[:, 1 + args.n_ctx:, :].detach()
    target_prefix = target_embedding[:, :1, :].detach()
    target_suffix = target_embedding[:, 1 + args.n_ctx:, :].detach()

    if args.run_stage1:
        # stage 1
        print("stage 1: training mapper")
        mapper = latent_mappers.SingleMapper(args, n_dim)  
        m_optim = torch.optim.Adam(
            mapper.mapping.parameters(),
            lr=args.lr_mapper,
        )
        
        for i in tqdm(range(args.iter_mapper)):
            
            mapper.train()
            sample_z = mixing_noise(args.batch_mapper, 512, args.mixing, device)
            sample_w = net.generator_frozen.style(sample_z)
            prompts = torch.reshape(mapper(sample_w[0]), (args.batch_mapper, args.n_ctx, n_dim)).type(clip_model.dtype)
            source_text_features = compute_text_features(prompts, source_prefix, source_suffix, source_tokenized_prompts,
                                                            clip_model, args.batch_mapper)
            target_text_features = compute_text_features(prompts, target_prefix, target_suffix, target_tokenized_prompts,
                                                            clip_model, args.batch_mapper)

            with torch.no_grad():
                imgs = net.generator_frozen(sample_z, input_is_latent=False, truncation=1, randomize_noise=True)[0].detach()
            loss = clip_loss_models[args.clip_models[0]].global_clip_loss(imgs, args.source_class, source_text_features,
                                                                            is_contrastive=1, logit_scale=clip_model.logit_scale,
                                                                            prompt_prefix=prompt_prefix, target_text=args.target_class,
                                                                            target_delta_features=target_text_features,
                                                                            lambda_l=args.lambda_l, lambda_src=args.lambda_src)

            m_optim.zero_grad()
            clip_model.zero_grad()
            net.zero_grad()
            loss.backward()
            m_optim.step()
          
        torch.save({
            "m": mapper.state_dict(),
            "m_optim": m_optim.state_dict(),
            },
            f"{ckpt_dir_m}/mapper.pt",
        )

    generator_ema = SG2Generator(args.frozen_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier).to(device)
    generator_ema.freeze_layers()
    generator_ema.eval()


    # reset seed
    torch.manual_seed(args.seed2)
    torch.cuda.manual_seed_all(args.seed2)
    np.random.seed(args.seed2)

    if args.run_stage2:
        # stage 2
        print("stage 2: training generator")
        if not args.run_stage1:
            print("loading mapper...")
            checkpoint_path = os.path.join(ckpt_dir_m, "mapper.pt")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            mapper = latent_mappers.SingleMapper(args, n_dim)
            mapper.load_state_dict(checkpoint["m"], strict=True)
        mapper.eval()
        g_optim = torch.optim.Adam(
            net.generator_trainable.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )

        # Training loop
        fixed_z = torch.randn(args.n_sample, 512, device=device)  # random vectors

        for i in tqdm(range(1, args.iter+1)):
            
            net.train()
            sample_z = mixing_noise(args.batch, 512, args.mixing, device)

            with torch.no_grad():
                sample_w = net.generator_frozen.style(sample_z)
                prompts = torch.reshape(mapper(sample_w[0]), (args.batch, args.n_ctx, n_dim)).type(clip_model.dtype)
                source_text_features = compute_text_features(prompts, source_prefix, source_suffix, source_tokenized_prompts, clip_model, args.batch)
                target_text_features = compute_text_features(prompts, target_prefix, target_suffix, target_tokenized_prompts,clip_model, args.batch)

            [sampled_src, sampled_dst], loss = net(sample_w, input_is_latent=True,
                                                    source_text_features=source_text_features,
                                                    target_text_features=target_text_features,
                                                    templates=prompt_prefix)  
            
            net.zero_grad()
            loss.backward()
            g_optim.step()

            ema(net.generator_trainable.generator, generator_ema.generator, args.ema_decay)

            if i % args.output_interval == 0:  
                net.eval()

                with torch.no_grad():
                    sample_w = generator_ema.style([fixed_z])
                    sample = generator_ema(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]
                    sample_src = net.generator_frozen(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]

                grid_rows = int(args.n_sample ** 0.5) 
                save_images(sample, sample_dir, "iter", grid_rows, i)
                save_images(sample_src, sample_dir, "src", grid_rows, 0)


            if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):  
                torch.save(
                    {
                        "g_ema": generator_ema.generator.state_dict(),
                        "g_optim": g_optim.state_dict(),
                    },
                    f"{ckpt_dir_g}/{str(i).zfill(6)}.pt",
                )
    

if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()
    train(args)