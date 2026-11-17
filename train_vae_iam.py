"""Single-GPU IAM entry using the original trainer.vae_trainer implementation."""
import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.iam_training_dataset import IAMTrainingDataset, collate_original_vae
from model.vae import VAE
from utils.utils import ModelConfig
from utils.optim import dit_build_optimizer_and_scheduler

TRAINING_IMPL = "vae_trainer_ctc_exact_v1"


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


def save_checkpoint(path,model,optimizer,scheduler,epoch,best,config,signature,stale):
    payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
             'scheduler_state_dict':scheduler.state_dict(),'epoch':epoch,'best_val_reconstruction':best,
             'config':config,'dataset_signature':signature,'stale_epochs':stale,
             'training_impl':TRAINING_IMPL,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
             'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def report_ctc_compatibility(config):
    for split in ('train','val'):
        total=eligible=exact=0
        with h5py.File(config[f'{split}_file']) as hf:
            for group in hf.values():
                text=group['line_text'].asstr()[()];points=len(group['point_seq'])
                if len(text)<2 or (split=='train' and points>2000):continue
                length=(points+7)//8
                total+=1
                eligible+=length>=2*len(text)-1
                exact+=length>=len(text)+sum(a==b for a,b in zip(text,text[1:]))
        print(f'{split}: CTC eligible={exact}/{total}; CTC skipped={total-exact}; previous conservative check={eligible}/{total}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/vae_iam_scratch.yaml'))
    parser.add_argument('--resume',type=Path,help='Resume ONLY a checkpoint from this IAM training entry')
    parser.add_argument('--check-data',action='store_true',help='Check data and a batch without constructing a model')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    train,val=load_data(config)
    signature=dataset_signature(config)
    report_ctc_compatibility(config)
    print(f'Train={len(train)} val={len(val)} writer classes={config["num_writer"]} text classes={config["num_text_embedding"]}',flush=True)
    if args.check_data:
        for name,dataset in [('train',train),('val',val)]:
            batch=collate_original_vae([dataset[i] for i in range(min(2,len(dataset)))])
            print(name,'data',list(batch[0].shape),'labels',list(batch[2].shape),'format=original trainer tuple',flush=True)
            dataset.close()
        return
    if not torch.cuda.is_available():raise RuntimeError('Training requires a UBELIX CUDA GPU job')
    from trainer.vae_trainer import train_vae_one_epoch, val_vae_one_epoch, val_vae_one_batch

    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=torch.device('cuda')
    train_loader=DataLoader(train,batch_size=config['train_batch_size'],shuffle=True,num_workers=config['num_workers'],collate_fn=collate_original_vae)
    val_loader=DataLoader(val,batch_size=config['val_batch_size'],shuffle=False,num_workers=config['num_workers'],collate_fn=collate_original_vae)
    # No pretrained checkpoint is loaded on the scratch path.
    model=VAE(ModelConfig(config)).to(device)
    config['warmup_steps']=max(1,int(len(train_loader)*config['num_epochs']*config.get('warmup_fraction',.05)))
    optimizer,scheduler=dit_build_optimizer_and_scheduler(model,config,len(train_loader))
    model_config=ModelConfig(config)
    start=0;best=float('inf');stale=0
    if args.resume:
        ckpt=torch.load(str(args.resume),map_location='cpu',weights_only=False)
        if ckpt.get('training_impl') != TRAINING_IMPL:
            raise ValueError('This checkpoint used a different trainer; start a new baseline run instead of resuming the old custom-loss run')
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
        training=train_vae_one_epoch(model,model_config,train_loader,optimizer,scheduler,
                                     epoch,config['num_epochs'],device,ddp=False)
        if not training['updated_batches']:
            raise FloatingPointError('Original trainer skipped all updates in this epoch')
        validation=val_vae_one_epoch(model,model_config,val_loader,device)
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
                val_vae_one_batch(model,val_loader,epoch,device,str(output/f'preview_epoch_{epoch+1:03d}'))
        if stale>=config['patience']:
            print(f'Early stopping after {stale} epochs without validation reconstruction improvement',flush=True);break
    train.close();val.close()


if __name__=='__main__':main()
