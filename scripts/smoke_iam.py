"""Run IAM through the real dataset, VAE training losses and DiT prefix masks on CPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from dataset.vae_dataset import TrainDataset, ValDataset
from model.vae import VAE
from model.gmm import get_mixture_coef_max, get_loss
from utils.mask import downsample_mask, build_prefix_mask_from_char_points
from utils.utils import ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='datas/iam_example/vae_iam.yaml')
    parser.add_argument('--checkpoint', help='Optional existing VAE; load encoder/decoder weights only')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = yaml.safe_load(Path(args.config).read_text())
    path = config.get('train_file') or config.get('val_file')
    ds = TrainDataset(path, config['text_file'], config['writer_file'])
    if not len(ds):
        raise ValueError('No lines pass the existing TrainDataset 200..2000 point filter')
    data, mask, text, ends, writer = next(iter(DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)))
    assert len(ends[0]) == (text[0] >= 0).sum().item()
    assert ends[0][-1] == mask[0].sum().item()
    assert np.all(np.diff(ends[0]) >= 0)
    assert torch.all(data[mask][:, 2:].sum(-1) == 1)
    vae = VAE(ModelConfig(config))
    loaded = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = {k.removeprefix('module.'): v for k, v in ckpt['model_state_dict'].items()}
        own = vae.state_dict()
        state = {k: v for k, v in state.items() if k.startswith(('encoder.', 'decoder.', 'conv_mu.', 'conv_logvar.', 'transformer_decoder.')) and k in own and v.shape == own[k].shape}
        if not state:
            raise ValueError('No compatible trajectory weights in checkpoint')
        vae.load_state_dict(state, strict=False)
        loaded = len(state)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5)
    latent_mask = downsample_mask(mask, 8)
    output, ctc, kl, style = vae(data.transpose(1, 2), latent_mask, text, writer)
    params = get_mixture_coef_max(output, num_mixture=20)
    density, pen = get_loss(*params, data.transpose(1, 2)[:, :1], data.transpose(1, 2)[:, 1:2], data.transpose(1, 2)[:, 2:])
    gmm = (density * mask).sum() / mask.sum()
    loss = config['gmm_weight'] * gmm + config['pen_weight'] * pen + config['kl_weight'] * kl + config['ctc_weight'] * ctc + config['style_weight'] * style
    assert torch.isfinite(loss), 'Non-finite loss'
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in vae.parameters() if p.grad is not None)
    optimizer.step()
    prefix, padding, _ = build_prefix_mask_from_char_points(ends, mask, compression_factor=8)
    assert prefix.shape == padding.shape == latent_mask.shape
    assert ((prefix * padding) > 0).any(), 'Empty DiT suffix'
    val = ValDataset(path, config['text_file'], None)
    next(iter(DataLoader(val, batch_size=1, collate_fn=val.collate_fn)))
    report = {'status': 'passed', 'points': int(mask.sum()), 'batch_shape': list(data.shape), 'characters_including_spaces': int((text >= 0).sum()), 'latent_mask_shape': list(prefix.shape), 'loaded_trajectory_weight_tensors': loaded, 'loss': float(loss.detach()), 'ctc_loss': float(ctc.detach()), 'gmm_loss': float(gmm.detach()), 'checks': ['TrainDataset + collate', 'ValDataset + collate', 'VAE forward', 'GMM/pen/CTC/KL/style losses', 'finite backward gradients', 'AdamW step', 'DiT prefix mask'], 'scope': 'One-batch integration check; no DiT forward or trained handwriting generation. OCR/style heads initialized for IAM.'}
    destination = Path(args.config).parent / 'smoke_report.json'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ds.hf.close()
    val.hf.close()


if __name__ == '__main__':
    main()
