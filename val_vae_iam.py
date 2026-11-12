"""VAE-only IAM reconstruction diagnostics; no DiT or training."""
import argparse
import html
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from check.iam_prefix import prefix_info, trajectory_svg
from model.vae import VAE
from model.gmm import get_mixture_coef, sample_from_params
from utils.utils import ModelConfig


def decode_argmax(output, length):
    pi, x, y, _, _, _, pen, _ = get_mixture_coef(output, num_mixture=20)
    index = pi[0,:,:length].argmax(dim=0,keepdim=True)
    xy = torch.stack((x[0,:,:length].gather(0,index)[0],
                      y[0,:,:length].gather(0,index)[0]),dim=1)
    states = torch.nn.functional.one_hot(pen[0,:,:length].argmax(dim=0),3)
    return torch.cat((xy,states),dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/dit_val_iam.yaml'))
    parser.add_argument('--checkpoint',type=Path,help='Override VAE checkpoint (e.g. IAM best.pt)')
    parser.add_argument('--output',type=Path,default=Path('outputs/iam_vae_check'))
    parser.add_argument('--max-samples',type=int,default=None)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error('--max-samples must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run inside your GPU job')
    config=yaml.safe_load(args.config.read_text())
    vocab=json.loads(Path(config['text_file']).read_text())
    config['num_text_embedding']=len(vocab)+1
    torch.manual_seed(42);np.random.seed(42)
    model=VAE(ModelConfig(config))
    checkpoint=args.checkpoint or config.get('vae_model_path')
    if not checkpoint:parser.error('Provide --checkpoint or vae_model_path in config')
    ckpt=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    weights={k.removeprefix('module.'):v for k,v in ckpt['model_state_dict'].items()}
    required=('encoder.','conv_mu.','conv_logvar.','decoder.','transformer_decoder.')
    state=model.state_dict()
    incompatible=[k for k,v in state.items() if k.startswith(required) and (k not in weights or weights[k].shape!=v.shape)]
    if incompatible:raise ValueError(f'Missing/incompatible VAE weights: {incompatible}')
    model.load_state_dict({k:v for k,v in weights.items() if k in state and state[k].shape==v.shape},strict=False)
    model=model.to(args.device).eval()
    args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='vae_',dir=args.output))
    rows=[]
    page=['<!doctype html><meta charset="utf-8"><title>VAE reconstruction</title><style>body{font-family:system-ui;margin:24px}svg{width:100%;max-width:1140px}article{border-bottom:1px solid #aaa}</style><h1>VAE-only reconstruction</h1><p>No DiT. Blue/orange use the exact GT prefix point index. All characters shown. All panels of a sample share coordinate bounds. mu_argmax isolates VAE reconstruction; z_argmax adds posterior noise; z_current also uses existing mixture sampling.</p>']
    with h5py.File(config['val_file']) as hf, torch.inference_mode():
        # Original filter (temporarily disabled): len(hf[k]['point_seq']) >= 200
        keys=sorted((k for k in hf if len(hf[k]['line_text'].asstr()[()]) >= 2),key=lambda k:len(hf[k]['point_seq']),reverse=True)
        if args.max_samples is not None:keys=keys[:args.max_samples]
        for index,key in enumerate(keys):
            g=hf[key];seq=g['point_seq'][:];n=len(seq)
            info=prefix_info(seq,g['char_points_idx'][:],g['line_text'].asstr()[()])
            info.update(index=index,key=key)
            padded=((n+7)//8)*8
            data=np.tile(np.array([0,0,0,0,1],dtype=np.float32),(padded,1));data[:n]=seq
            x=torch.from_numpy(data).T.unsqueeze(0).to(args.device)
            z,mu,_=model.encode(x)
            mu_output=model.decode(mu);z_output=model.decode(z)
            curves={'gt':seq,'mu_argmax':decode_argmax(mu_output,n),'z_argmax':decode_argmax(z_output,n)}
            params=get_mixture_coef(z_output,num_mixture=20)[:7]
            curves['z_current']=sample_from_params([p[0].cpu() for p in params],temp=.1,max_seq_len=n,greedy=True)[:n]
            info['metrics']={}
            boundary=info['prefix_end_exclusive']
            lo=np.min([a[:,:2].min(0) for a in curves.values()],axis=0)
            hi=np.max([a[:,:2].max(0) for a in curves.values()],axis=0)
            page.append(f'<article><h2>{index}: {html.escape(key)} — {html.escape(info["text"])}</h2><p>Prefix {html.escape(repr(info["prefix_text"]))}, points [0,{boundary}); latent block boundary {info["effective_prefix_end_exclusive"]}</p>')
            for name,curve in curves.items():
                np.save(out/f'{index:03d}_{name}.npy',curve)
                svg=trajectory_svg(curve,boundary,info['effective_prefix_end_exclusive'],bounds=(lo,hi))
                (out/f'{index:03d}_{name}.svg').write_text(svg)
                page.append(f'<h3>{name}</h3>'+svg)
                if name!='gt':
                    info['metrics'][name]={'xy_mse':float(np.mean((curve[:,:2]-seq[:,:2])**2)),
                                          'prefix_xy_mse':float(np.mean((curve[:boundary,:2]-seq[:boundary,:2])**2)) if boundary else None,
                                          'state_accuracy':float(np.mean(curve[:,2:].argmax(1)==seq[:,2:].argmax(1)))}
            page.append('</article>');rows.append(info)
            print(f'{index+1}/{len(keys)} {key} {info["text"]!r}',flush=True)
    (out/'report.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'index.html').write_text('\n'.join(page))
    print('Results:',out,flush=True)


if __name__=='__main__':main()
