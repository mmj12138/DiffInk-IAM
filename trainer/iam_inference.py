"""IAM-only validation loop; original DiffInk validation remains unchanged."""
import os
import torch
from tqdm import tqdm
from model.diffusion import Diffusion
from model.gmm import get_mixture_coef, sample_from_params
from utils.visual import plot_line_cv2_new
from utils.mask import build_prefix_mask_from_char_points
from .dit_trainer import save_diffink_svg


def infer_diffink(dit, vae, val_loader, device, save_path, max_samples=3):
    dit.eval()
    vae.eval()
    vae = vae.module if hasattr(vae, 'module') else vae

    diffusion = Diffusion(
        noise_steps=1000,
        schedule_type='cosine',
        device=device
    )

    os.makedirs(save_path, exist_ok=True)
    import tempfile
    save_path = tempfile.mkdtemp(prefix='svg_test_', dir=save_path)
    print('SVG test output:', save_path, flush=True)
    import json
    from pathlib import Path
    dataset = val_loader.dataset
    samples = []
    for index, source_index in enumerate(dataset.sorted_indices):
        key = dataset.keys[source_index]
        group = dataset.hf[key]
        samples.append({'index': index, 'key': key,
                        'text': group['line_text'].asstr()[()],
                        'writer_id': group['writer_id'].asstr()[()]})
    Path(save_path, 'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    saved_samples = 0

    with torch.no_grad():
        for p, batch in tqdm(enumerate(val_loader), desc="val", unit="batch"):
            data, mask, text_idx, char_points_idx, writer_id = batch
            data, mask, text_idx = data.to(device), mask.to(device), text_idx.to(device)

            batch_size = data.size(0)

            data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

            feat = vae.encode(data)[0].permute(0, 2, 1)

            latent_mask, latent_padding_mask, prefix_label_mask = build_prefix_mask_from_char_points(
                char_points_idx=char_points_idx,
                mask=mask,  # e.g., mask from dataloader
                compression_factor=8,
                prefix_ratio=0.3
            )
            final_noise_mask = latent_mask * latent_padding_mask  # [B, T_latent]

            x_pred = diffusion.ddim_sample(dit, batch_size, feat, text_idx, latent_mask, latent_padding_mask, sampling_timesteps=20, eta=0.0, cfg_scale=1.0)

            x_mix = x_pred.permute(0, 2, 1) * (final_noise_mask.unsqueeze(1)) + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)

            # decoder and visualized
            output = vae.decode(x_mix)

            # sample from gmm
            pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = get_mixture_coef(output, num_mixture=20)

            for i in range(data.size(0)):
                sample_idx = i
                seq_len = int(mask[i].sum().item())
                params = [
                    pi[sample_idx].cpu(),  # 混合系数
                    mu1[sample_idx].cpu(),  # 第一个分量的均值
                    mu2[sample_idx].cpu(),  # 第二个分量的均值
                    sigma1[sample_idx].cpu(),  # 第一个分量的标准差
                    sigma2[sample_idx].cpu(),  # 第二个分量的标准差
                    corr[sample_idx].cpu(),  # 相关系数
                    pen[sample_idx].cpu()  # 结束状态的概率
                ]

                # 生成样本
                recon = sample_from_params(params, temp=0.1, max_seq_len=seq_len, greedy=True)  # T 5
                plot_line_cv2_new(data[i], save_path=f"{save_path}/gt_{p * batch_size + i}.png", canvas_height=256,
                                  padding=20, line_thickness=2, max_dist=200)
                plot_line_cv2_new(recon, save_path=f"{save_path}/recon_{p * batch_size + i}.png", canvas_height=256,
                                  padding=20, line_thickness=2, max_dist=200)
                save_diffink_svg(recon, seq_len, f"{save_path}/recon_{p * batch_size + i}.svg")
                saved_samples += 1
                if max_samples is not None and saved_samples >= max_samples:
                    print(f'SVG test finished: {saved_samples} samples', flush=True)
                    return

            # break

