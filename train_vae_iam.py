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

from dataset import build_datasets_and_loaders_ddp
from model.vae import VAE
from utils.utils import ModelConfig, load_config_from_yaml, set_seed
from utils.optim import dit_build_optimizer_and_scheduler
from trainer.vae_trainer import train_vae_one_epoch, val_vae_one_epoch, val_vae_one_batch

TRAINING_IMPL = "vae_trainer_ctc_exact_v1"


def dataset_signature(config):
    digest=hashlib.sha256()
    for field in ['text_file','writer_file','splits_file','report_file']:
        digest.update(Path(config[field]).read_bytes())
    return digest.hexdigest()


def load_data(config_dict):
    """Check author isolation and use the original dataset/loader builder."""
    writers = json.loads(Path(config_dict['writer_file']).read_text())
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(writers[left]) & set(writers[right]):
            raise ValueError('Writer split overlap')
    with h5py.File(config_dict['train_file']) as train, h5py.File(config_dict['val_file']) as val:
        train_writers = {group['writer_id'].asstr()[()] for group in train.values()}
        val_writers = {group['writer_id'].asstr()[()] for group in val.values()}
        if (train_writers & val_writers or not train_writers <= set(writers['train'])
                or not val_writers <= set(writers['val'])):
            raise ValueError('H5 writer leakage')
    # Preserve the current IAM data conditions when reusing the original builder.
    return build_datasets_and_loaders_ddp(
        config_dict, ddp=False, augment=False, append_val_sentinel=False
    )



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


def train_vae(config_path, resume=None):
    """Single-GPU entry, organized like train_vae_ddp.train_vae."""
    # === Set device (one logical GPU in the current Slurm job) ===
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU job')
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # === Load config and datasets ===
    config_dict = load_config_from_yaml(config_path)
    train_loader, val_loader, config_dict = load_data(config_dict)
    config = ModelConfig(config_dict)
    signature = dataset_signature(config_dict)
    report_ctc_compatibility(config_dict)

    # === Initialize, using the same seed helper as the DDP entry ===
    set_seed(config_dict.get('seed', 42))
    vae = VAE(config).to(device)

    # === Optimizer and scheduler ===
    config_dict['warmup_steps'] = max(1, int(
        len(train_loader) * config_dict['num_epochs'] * config_dict.get('warmup_fraction', .05)
    ))
    optimizer, scheduler = dit_build_optimizer_and_scheduler(
        vae, config_dict, len(train_loader)
    )

    # === Restore checkpoint, or create a fresh run ===
    start=0;best=float('inf');stale=0
    if resume:
        ckpt=torch.load(str(resume),map_location='cpu',weights_only=False)
        if ckpt.get('training_impl') != TRAINING_IMPL:
            raise ValueError('This checkpoint used a different trainer; start a new baseline run instead of resuming the old custom-loss run')
        if ckpt.get('dataset_signature')!=signature or ckpt.get('config')!=config_dict:
            raise ValueError('Resume dataset/config mismatch; use the original run config and prepared data')
        vae.load_state_dict(ckpt['model_state_dict'],strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict']);scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start=ckpt['epoch'];best=ckpt['best_val_reconstruction'];stale=ckpt.get('stale_epochs',0)
        torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        np.random.set_state(ckpt['numpy_rng']);random.setstate(ckpt['python_rng'])
        output=resume.resolve().parent.parent
    else:
        base=Path(config_dict['output_base']);base.mkdir(parents=True,exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix='run_',dir=base))
    (output/'ckpt').mkdir(exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config_dict,sort_keys=False))
    print(f'Output: {output}',flush=True)
    for epoch in range(start,config_dict['num_epochs']):
        print(f'Epoch {epoch+1}/{config_dict["num_epochs"]}',flush=True)
        training=train_vae_one_epoch(vae,config,train_loader,optimizer,scheduler,
                                     epoch,config_dict['num_epochs'],device,ddp=False)
        if not training['updated_batches']:
            raise FloatingPointError('Original trainer skipped all updates in this epoch')
        validation=val_vae_one_epoch(vae,config,val_loader,device)
        improved=validation['reconstruction']<best
        if improved:best=validation['reconstruction'];stale=0
        else:stale+=1
        record={'epoch':epoch+1,'train':training,'val':validation,'lr':optimizer.param_groups[0]['lr']}
        with (output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        save_checkpoint(output/'ckpt/latest.pt',vae,optimizer,scheduler,epoch+1,best,config_dict,signature,stale)
        if improved:save_checkpoint(output/'ckpt/best.pt',vae,optimizer,scheduler,epoch+1,best,config_dict,signature,stale)
        if epoch==0 or (epoch+1)%config_dict['preview_every']==0:
            # Diagnostics must not consume the RNG state saved for resuming training.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                val_vae_one_batch(vae,val_loader,epoch,device,str(output/f'preview_epoch_{epoch+1:03d}'))
        if stale>=config_dict['patience']:
            print(f'Early stopping after {stale} epochs without validation reconstruction improvement',flush=True);break
    train_loader.dataset.hf.close()
    val_loader.dataset.hf.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/vae_iam_scratch.yaml'))
    parser.add_argument('--resume', type=Path, help='Resume a checkpoint from this IAM training entry')
    parser.add_argument('--check-data', action='store_true', help='Check data without creating a model')
    args = parser.parse_args()

    if args.check_data:
        config_dict = load_config_from_yaml(args.config)
        train_loader, val_loader, config_dict = load_data(config_dict)
        report_ctc_compatibility(config_dict)
        for name, loader in [('train', train_loader), ('val', val_loader)]:
            batch = next(iter(loader))
            print(name, 'samples', len(loader.dataset), 'data', list(batch[0].shape),
                  'labels', list(batch[2].shape), flush=True)
            loader.dataset.hf.close()
        return

    train_vae(args.config, resume=args.resume)


if __name__=='__main__':main()
