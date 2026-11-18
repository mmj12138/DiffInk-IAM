"""VAE-only IAM reconstruction diagnostics; no DiT or training."""
import argparse
import html
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from check.iam_prefix import prefix_info, trajectory_svg
from model.vae import VAE
from model.gmm import get_mixture_coef, sample_from_params
from utils.utils import ModelConfig
from utils.mask import downsample_mask


def decode_argmax(output, length):
    pi, x, y, _, _, _, pen, _ = get_mixture_coef(output, num_mixture=20)
    index = pi[0,:,:length].argmax(dim=0,keepdim=True)
    xy = torch.stack((x[0,:,:length].gather(0,index)[0],
                      y[0,:,:length].gather(0,index)[0]),dim=1)
    states = torch.nn.functional.one_hot(pen[0,:,:length].argmax(dim=0),3)
    return torch.cat((xy,states),dim=1).cpu().numpy()


def edit_distance(reference, prediction):
    previous=list(range(len(prediction)+1))
    for i,left in enumerate(reference,1):
        current=[i]
        for j,right in enumerate(prediction,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]


def text_errors(reference, prediction):
    reference_words=reference.split()
    return {'character_errors':edit_distance(reference,prediction),'reference_characters':len(reference),
            'word_errors':edit_distance(reference_words,prediction.split()),'reference_words':len(reference_words)}


def dtw_distance(reference, prediction):
    """Exact Euclidean XY DTW; report minimum total cost divided by its path length."""
    reference=np.asarray(reference,dtype=np.float64)[:,:2]
    prediction=np.asarray(prediction,dtype=np.float64)[:,:2]
    if not len(reference) or not len(prediction) or not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError('DTW requires nonempty finite XY sequences')
    costs=np.full(len(prediction)+1,np.inf);costs[0]=0.
    lengths=np.zeros(len(prediction)+1,dtype=np.int64)
    for point in reference:
        new_costs=np.full(len(prediction)+1,np.inf)
        new_lengths=np.zeros(len(prediction)+1,dtype=np.int64)
        distances=np.linalg.norm(prediction-point,axis=1)
        for j,distance in enumerate(distances,1):
            cost,length=min((costs[j-1],lengths[j-1]),(costs[j],lengths[j]),(new_costs[j-1],new_lengths[j-1]))
            new_costs[j]=cost+distance;new_lengths[j]=length+1
        costs,lengths=new_costs,new_lengths
    total=float(costs[-1]);steps=int(lengths[-1])
    return {'total_cost':total,'path_length':steps,'normalized':total/steps}


def ctc_greedy(ids, characters):
    result=[];previous=None
    for token in ids:
        token=int(token)
        if token!=previous and token!=0:
            if token<1 or token>len(characters):raise ValueError('OCR token outside vocabulary')
            result.append(characters[token-1])
        previous=token
    return ''.join(result)


def recognize_latent(model, mu, mask, characters, length, resolution='points'):
    if resolution=='latent':
        # Original VAE OCR: no interpolation, same clamp and time resolution.
        logits=model.ocr_model(mu).clamp(-30,30)
        valid=int(downsample_mask(mask,8).sum().item())
        return ctc_greedy(logits[:valid,0].argmax(-1).cpu().tolist(),characters)
    # Compatibility with checkpoints trained by the previous IAM custom loss.
    feature=F.interpolate(mu,size=mask.size(1),mode='linear',align_corners=False)
    ocr=model.ocr_model
    feature=ocr.pos_encoder(ocr.input_proj(feature.permute(0,2,1)))
    feature=ocr.transformer(feature,src_key_padding_mask=~mask)
    ids=ocr.output_fc(feature)[0,:length].argmax(-1).cpu().tolist()
    return ctc_greedy(ids,characters)


def summarize(rows):
    text=[r['text_evaluation'] for r in rows if r['text_evaluation'] is not None]
    errors={key:sum(r[key] for r in text) for key in ['character_errors','reference_characters','word_errors','reference_words']}
    classified=[r['writer_evaluation'] for r in rows if r['writer_evaluation']['correct'] is not None]
    return {'samples':len(rows),'ocr_evaluated_samples':len(text),
            'dtw_mean_normalized':float(np.mean([r['metrics']['mu_argmax']['dtw']['normalized'] for r in rows])),
            'latent_ocr_cer':errors['character_errors']/errors['reference_characters'] if errors['reference_characters'] else None,
            'latent_ocr_wer':errors['word_errors']/errors['reference_words'] if errors['reference_words'] else None,
            **errors,'writer_classification_accuracy':sum(r['correct'] for r in classified)/len(classified) if classified else None,
            'writer_evaluated_samples':len(classified),
            'unknown_writer_samples':sum(r['writer_evaluation']['status']=='unseen_writer' for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/dit_val_iam.yaml'))
    parser.add_argument('--checkpoint',type=Path,help='Override VAE checkpoint')
    parser.add_argument('--input',type=Path,help='Override val_file, e.g. train.h5 for known-writer accuracy')
    parser.add_argument('--output',type=Path,default=Path('outputs/iam_vae_check'))
    parser.add_argument('--max-samples',type=int,default=None)
    parser.add_argument('--metrics-only',action='store_true',help='Only deterministic evaluation; skip SVG/NPY and stochastic reconstructions')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if args.max_samples is not None and args.max_samples<=0:parser.error('--max-samples must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run inside your GPU job')
    config=yaml.safe_load(args.config.read_text())
    vocab=json.loads(Path(config['text_file']).read_text());characters=list(vocab)
    config['num_text_embedding']=len(vocab)+1
    writer_path=config.get('writer_file')
    writer_ids=json.loads(Path(writer_path).read_text())['train'] if writer_path else []
    writer_labels={writer:index for index,writer in enumerate(writer_ids)}
    torch.manual_seed(42);np.random.seed(42)
    model=VAE(ModelConfig(config))
    checkpoint=args.checkpoint or config.get('vae_model_path')
    if not checkpoint:parser.error('Provide --checkpoint or vae_model_path in config')
    ckpt=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    weights={(k[len('module.'):] if k.startswith('module.') else k):v
             for k,v in ckpt['model_state_dict'].items()}
    state=model.state_dict()
    def available(prefix):
        keys=[key for key in state if key.startswith(prefix)]
        return bool(keys) and all(key in weights and weights[key].shape==state[key].shape for key in keys)
    required=('encoder.','conv_mu.','conv_logvar.','decoder.','transformer_decoder.')
    if not all(available(prefix) for prefix in required):raise ValueError('Missing/incompatible VAE reconstruction weights')
    implementation=ckpt.get('training_impl')
    ocr_resolution='points' if implementation is None and 'dataset_signature' in ckpt else 'latent'
    ocr_available=available('ocr_model.')
    style_available=available('style_classifier.') and len(writer_ids)==config['num_writer']
    model.load_state_dict({key:value for key,value in weights.items() if key in state and state[key].shape==value.shape},strict=False)
    model=model.to(args.device).eval()
    data_file=args.input or Path(config['val_file'])
    args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='vae_',dir=args.output))
    rows=[]
    page=['<!doctype html><meta charset="utf-8"><title>VAE evaluation</title><style>body{font-family:system-ui;margin:24px}svg{width:100%;max-width:1140px}article{border-bottom:1px solid #aaa}pre{white-space:pre-wrap}</style><h1>VAE evaluation</h1><p>DTW evaluates deterministic mu/argmax reconstruction in the GT normalized XY coordinates. CER/WER evaluate OCR on the original trajectory latent, not an independent recognizer of reconstruction. Writer accuracy is only defined for training writers. Case and punctuation are preserved; CER includes spaces and WER splits on whitespace.</p>']
    with h5py.File(data_file) as hf,torch.inference_mode():
        # Original 200-point filter remains disabled.
        keys=sorted((key for key in hf if len(hf[key]['line_text'].asstr()[()])>=2),key=lambda key:len(hf[key]['point_seq']),reverse=True)
        if args.max_samples is not None:keys=keys[:args.max_samples]
        if not keys:raise ValueError('No eligible samples')
        for index,key in enumerate(keys):
            group=hf[key];seq=group['point_seq'][:];n=len(seq)
            text=group['line_text'].asstr()[()];writer=group['writer_id'].asstr()[()]
            if set(text)-vocab.keys():raise ValueError(f'{key}: reference contains unknown characters')
            info=prefix_info(seq,group['char_points_idx'][:],text)
            info.update(index=index,key=key,writer_id=writer)
            padded=((n+7)//8)*8
            data=np.tile(np.array([0,0,0,0,1],dtype=np.float32),(padded,1));data[:n]=seq
            x=torch.from_numpy(data).T.unsqueeze(0).to(args.device)
            mask=(torch.arange(padded,device=args.device)<n).unsqueeze(0)
            z,mu,_=model.encode(x)
            mu_output=model.decode(mu)
            curves={'gt':seq,'mu_argmax':decode_argmax(mu_output,n)}
            info['text_evaluation']=None
            if ocr_available:
                prediction=recognize_latent(model,mu,mask,characters,n,resolution=ocr_resolution)
                info['text_evaluation']={'prediction':prediction,**text_errors(text,prediction)}
            info['writer_evaluation']={'status':'unseen_writer','correct':None,'prediction':None}
            if writer in writer_labels:
                if style_available:
                    predicted=int(model.style_classifier(mu,downsample_mask(mask,8)).argmax(1).item())
                    info['writer_evaluation']={'status':'known_writer','correct':predicted==writer_labels[writer],
                                               'prediction':writer_ids[predicted]}
                else:info['writer_evaluation']['status']='classifier_unavailable'
            if not args.metrics_only:
                z_output=model.decode(z)
                curves['z_argmax']=decode_argmax(z_output,n)
                params=get_mixture_coef(z_output,num_mixture=20)[:7]
                curves['z_current']=sample_from_params([p[0].cpu() for p in params],temp=.1,max_seq_len=n,greedy=True)[:n]
            info['metrics']={}
            boundary=info['prefix_end_exclusive']
            page.append(f'<article><h2>{index}: {html.escape(key)} — {html.escape(text)}</h2>')
            for name,curve in curves.items():
                if name!='gt':
                    info['metrics'][name]={'xy_mse':float(np.mean((curve[:,:2]-seq[:,:2])**2)),
                                          'prefix_xy_mse':float(np.mean((curve[:boundary,:2]-seq[:boundary,:2])**2)) if boundary else None}
                    if name=='mu_argmax':info['metrics'][name]['dtw']=dtw_distance(seq,curve)
                if not args.metrics_only:
                    np.save(out/f'{index:03d}_{name}.npy',curve)
                    lo=np.min([a[:,:2].min(0) for a in curves.values()],axis=0)
                    hi=np.max([a[:,:2].max(0) for a in curves.values()],axis=0)
                    svg=trajectory_svg(curve,boundary,info['effective_prefix_end_exclusive'],bounds=(lo,hi))
                    (out/f'{index:03d}_{name}.svg').write_text(svg)
                    page.append(f'<h3>{name}</h3>'+svg)
            details={'DTW':info['metrics']['mu_argmax']['dtw'],'OCR':info['text_evaluation'],'writer':info['writer_evaluation']}
            page.append('<pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2))+'</pre></article>')
            rows.append(info)
            print(f'{index+1}/{len(keys)} {key} DTW={info["metrics"]["mu_argmax"]["dtw"]["normalized"]:.5f}',flush=True)
    summary=summarize(rows)
    metadata={'checkpoint':str(checkpoint),'input':str(data_file),'ocr_head_loaded':ocr_available,
              'style_head_loaded':style_available,'ocr_time_resolution':ocr_resolution,'decode':'posterior mean mu; highest-weight mixture mean per point',
              'dtw':'XY Euclidean minimum-total-cost path, divided by path length; no separate curve rescaling',
              'text':'latent OCR; greedy CTC; original case/punctuation; CER includes spaces; WER whitespace tokenization',
              'writer':'accuracy over known training writers only; null if no eligible writers'}
    (out/'report.json').write_text(json.dumps({'summary':summary,'evaluation':metadata,'samples':rows},ensure_ascii=False,indent=2))
    page.insert(1,'<h2>Summary</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    (out/'index.html').write_text('\n'.join(page))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Results:',out,flush=True)


if __name__=='__main__':main()
