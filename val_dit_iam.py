import argparse

import os
import torch
import wandb
import copy
from datetime import datetime
from torch.nn.parallel import DataParallel

from model.vae import VAE
from dataset.vae_dataset import ValDataset
from torch.utils.data import DataLoader
from utils.optim import dit_build_optimizer_and_scheduler
from utils.utils import ModelConfig, save_checkpoint, load_config_from_yaml, set_seed
from trainer.iam_inference import infer_diffink

from model.dit import TextEmbedding, InputEmbedding, DiT

def strip_module_prefix(state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

def val_diffink(config_path="./configs/dit_val_iam.yaml"):
    # === 加载配置 ===
    config_dict = load_config_from_yaml(config_path)

    for field in ('vae_model_path', 'dit_resume_ckpt'):
        if not os.path.isfile(config_dict.get(field) or ''):
            raise FileNotFoundError(f"Missing {field}: {config_dict.get(field)}")

    # === 构建数据集与加载器，并更新类别数 ===
    dataset = ValDataset(config_dict['val_file'], config_dict['text_file'], None)
    if not len(dataset):
        raise ValueError('No validation samples pass the existing length filter')
    config_dict['num_text_embedding'] = len(dataset.text_cache) + 1
    val_loader = DataLoader(dataset, batch_size=config_dict.get('val_batch_size', 1),
                            shuffle=False, num_workers=config_dict.get('val_num_workers', 0),
                            collate_fn=ValDataset.collate_fn)
    config = ModelConfig(config_dict)

    # === set devices===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # === init ===
    set_seed(42)
    vae = VAE(config).to(device)
    dit = DiT(config).to(device)

    # === load vae ====
    vae_model_path = config_dict.get("vae_model_path")
    if vae_model_path and os.path.exists(vae_model_path):
        print(f"Resuming VAE from checkpoint: {vae_model_path}")
        ckpt = torch.load(vae_model_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = strip_module_prefix(state_dict)
        model_state = vae.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        required_prefixes = ('encoder.', 'conv_mu.', 'conv_logvar.', 'decoder.', 'transformer_decoder.')
        missing_required = [k for k in model_state if k.startswith(required_prefixes) and k not in filtered_state]
        if missing_required:
            raise ValueError(f'Incomplete VAE generation weights: {missing_required}')
        # 加载
        missing_keys, unexpected_keys = vae.load_state_dict(filtered_state, strict=False)
        print(f"Loaded VAE with {len(filtered_state)} parameters.")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    # Load checkpoint if provided
    dit_ckpt_path = config_dict.get("dit_resume_ckpt", None)
    if dit_ckpt_path and os.path.exists(dit_ckpt_path):
        print(f"Resuming DIT from checkpoint: {dit_ckpt_path}")
        ckpt = torch.load(dit_ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"]

        # 自动处理 DDP 保存的模型（含 module. 前缀）
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = strip_module_prefix(state_dict)

        dit.load_state_dict(state_dict)
    else:
        print("No valid dit checkpoint found. Starting from scratch.")

    # 开始测试
    output_path = config_dict.get("output_base", './')
    infer_diffink(dit, vae, val_loader, device, f'{output_path}/step_20_cfg_1',
                  max_samples=config_dict.get('max_samples', 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/dit_val_iam.yaml')
    args = parser.parse_args()
    val_diffink(args.config)