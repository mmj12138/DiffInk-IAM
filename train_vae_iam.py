"""Single-GPU IAM VAE training from scratch, with joint OCR/style learning."""
import argparse
import hashlib
import html
import json
import math
import random
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.iam_training_dataset import IAMTrainingDataset, collate_iam
from model.vae import VAE
from trainer.iam_vae import compute_losses
from utils.utils import ModelConfig


def dataset_signature(config):
    digest=hashlib.sha256()
    for field in ['text_file','writer_file','splits_file','report_file']:
        digest.update(Path(config[field]).read_bytes())
    return digest.hexdigest()


def load_data(config):
    train=IAMTrainingDataset(config['train_file'],config['text_file'],config['writer_file'],True)
    val=IAMTrainingDataset(config['val_file'],config['text_file'],config['writer_file'],False)
    writers=json.loads(Path(config['writer_file']).read_text())
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(writers[a])&set(writers[b]):raise ValueError('Writer split overlap')
    with h5py.File(config['train_file']) as a,h5py.File(config['val_file']) as b:
        aw={g['writer_id'].asstr()[()] for g in a.values()};bw={g['writer_id'].asstr()[()] for g in b.values()}
        if aw&bw or not aw<=set(writers['train']) or not bw<=set(writers['val']):raise ValueError('H5 writer leakage')
    config['num_text_embedding']=len(train.vocab)+1
    config['num_writer']=len(train.writers)
    return train,val


def run_epoch(model,loader,config,device,optimizer=None,scheduler=None):
    training=optimizer is not None
    model.train(training)
    totals={};samples=0
    for index,batch in enumerate(loader):
        for key in ['data','mask','labels','writers']:batch[key]=batch[key].to(device)
        if training:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss,metrics=compute_losses(model,batch,config,training)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss in {batch["keys"]}')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizer.step();scheduler.step()
        count=len(batch['keys']);samples+=count
        for key,value in metrics.items():
            value=float(value.detach()) if torch.is_tensor(value) else value
            totals[key]=totals.get(key,0)+value*(1 if key.startswith('ctc_') and key.endswith('_samples') else count)
        if training and (index+1)%config['log_every']==0:
            print(f'  step {index+1}/{len(loader)} loss={float(loss):.5f} lr={optimizer.param_groups[0]["lr"]:.3g}',flush=True)
    return {k:v if k.endswith('_samples') else v/samples for k,v in totals.items()}


def save_checkpoint(path,model,optimizer,scheduler,epoch,best,config,signature,stale):
    payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
             'scheduler_state_dict':scheduler.state_dict(),'epoch':epoch,'best_val_reconstruction':best,
             'config':config,'dataset_signature':signature,'stale_epochs':stale,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
             'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def save_preview(model,config,device,destination):
    from check.iam_prefix import prefix_info,trajectory_svg
    from val_vae_iam import decode_argmax
    destination.mkdir(parents=True,exist_ok=True)
    page=['<!doctype html><meta charset="utf-8"><title>IAM VAE validation</title><style>svg{width:100%;max-width:1140px}</style><h1>Unseen-writer validation: GT / deterministic reconstruction</h1>']
    model.eval()
    with h5py.File(config['val_file']) as hf,torch.inference_mode():
        keys=[k for k in hf if len(hf[k]['line_text'].asstr()[()])>=2][:3]
        for key in keys:
            g=hf[key];seq=g['point_seq'][:];n=len(seq)
            info=prefix_info(seq,g['char_points_idx'][:],g['line_text'].asstr()[()])
            padded=((n+7)//8)*8
            data=torch.tensor([0,0,0,0,1],dtype=torch.float32).repeat(padded,1)
            data[:n]=torch.from_numpy(seq)
            _,mu,_=model.encode(data.T.unsqueeze(0).to(device))
            recon=decode_argmax(model.decode(mu),n)
            bounds=(np.minimum(seq[:,:2].min(0),recon[:,:2].min(0)),np.maximum(seq[:,:2].max(0),recon[:,:2].max(0)))
            page.append(f'<h2>{html.escape(key)}: {html.escape(info["text"])}</h2>')
            for name,curve in [('gt',seq),('recon',recon)]:
                svg=trajectory_svg(curve,info['prefix_end_exclusive'],bounds=bounds)
                (destination/f'{key}_{name}.svg').write_text(svg)
                page.append(f'<h3>{name}</h3>'+svg)
    (destination/'index.html').write_text('\n'.join(page))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/vae_iam_scratch.yaml'))
    parser.add_argument('--resume',type=Path,help='Resume ONLY a checkpoint from this IAM training entry')
    parser.add_argument('--check-data',action='store_true',help='Check data and a batch without constructing a model')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    train,val=load_data(config)
    signature=dataset_signature(config)
    print(f'Train={len(train)} val={len(val)} writer classes={config["num_writer"]} text classes={config["num_text_embedding"]}',flush=True)
    if args.check_data:
        for name,dataset in [('train',train),('val',val)]:
            batch=collate_iam([dataset[i] for i in range(min(2,len(dataset)))])
            print(name,'data',list(batch['data'].shape),'labels',list(batch['labels'].shape),flush=True)
            dataset.close()
        return
    if not torch.cuda.is_available():raise RuntimeError('Training requires a UBELIX CUDA GPU job')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=torch.device('cuda')
    train_loader=DataLoader(train,batch_size=config['train_batch_size'],shuffle=True,num_workers=config['num_workers'],collate_fn=collate_iam)
    val_loader=DataLoader(val,batch_size=config['val_batch_size'],shuffle=False,num_workers=config['num_workers'],collate_fn=collate_iam)
    # No pretrained checkpoint is loaded on the scratch path.
    model=VAE(ModelConfig(config)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['base_lr'],betas=tuple(config['betas']),weight_decay=config['weight_decay'])
    steps=len(train_loader)*config['num_epochs'];warmup=max(1,int(steps*config['warmup_fraction']))
    def schedule(step):
        if step<warmup:return (step+1)/warmup
        progress=min(1.,(step-warmup)/max(1,steps-warmup))
        floor=config['min_lr']/config['base_lr']
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    start=0;best=float('inf');stale=0
    if args.resume:
        ckpt=torch.load(str(args.resume),map_location='cpu',weights_only=False)
        if ckpt.get('dataset_signature')!=signature or ckpt.get('config')!=config:
            raise ValueError('Resume dataset/config mismatch; use the original run config and prepared data')
        model.load_state_dict(ckpt['model_state_dict'],strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict']);scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start=ckpt['epoch'];best=ckpt['best_val_reconstruction'];stale=ckpt.get('stale_epochs',0)
        torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        np.random.set_state(ckpt['numpy_rng']);random.setstate(ckpt['python_rng'])
        output=args.resume.resolve().parent.parent
    else:
        base=Path(config['output_base']);base.mkdir(parents=True,exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix='run_',dir=base))
    (output/'ckpt').mkdir(exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print(f'Output: {output}',flush=True)
    for epoch in range(start,config['num_epochs']):
        print(f'Epoch {epoch+1}/{config["num_epochs"]}',flush=True)
        training=run_epoch(model,train_loader,config,device,optimizer,scheduler)
        validation=run_epoch(model,val_loader,config,device)
        improved=validation['reconstruction']<best
        if improved:best=validation['reconstruction'];stale=0
        else:stale+=1
        record={'epoch':epoch+1,'train':training,'val':validation,'lr':optimizer.param_groups[0]['lr']}
        with (output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        save_checkpoint(output/'ckpt/latest.pt',model,optimizer,scheduler,epoch+1,best,config,signature,stale)
        if improved:save_checkpoint(output/'ckpt/best.pt',model,optimizer,scheduler,epoch+1,best,config,signature,stale)
        if epoch==0 or (epoch+1)%config['preview_every']==0:
            # Diagnostics must not consume the RNG state saved for resuming training.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                save_preview(model,config,device,output/f'preview_epoch_{epoch+1:03d}')
        if stale>=config['patience']:
            print(f'Early stopping after {stale} epochs without validation reconstruction improvement',flush=True);break
    train.close();val.close()


if __name__=='__main__':main()
