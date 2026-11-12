"""IAM joint VAE/OCR/style losses with valid-point masking and CTC time alignment."""
import math
import torch
import torch.nn.functional as F

from utils.mask import downsample_mask


def reconstruction_loss(output, data, mask):
    """Stable GMM NLL and class-balanced focal pen loss; padding excluded."""
    pen=output[:,:3]
    pi,mux,muy,sx,sy,rho=output[:,3:].chunk(6,dim=1)
    sx=F.softplus(sx)+1e-3;sy=F.softplus(sy)+1e-3
    rho=rho.tanh().clamp(-.999,.999)
    x=(data[:,0:1]-mux)/sx;y=(data[:,1:2]-muy)/sy
    one_minus=(1-rho.square()).clamp_min(1e-6)
    log_density=-math.log(2*math.pi)-sx.log()-sy.log()-.5*one_minus.log()-(x.square()+y.square()-2*rho*x*y)/(2*one_minus)
    nll=-torch.logsumexp(pi.log_softmax(1)+log_density,dim=1)
    target=data[:,2:].argmax(dim=1)
    ce=F.cross_entropy(pen,target,reduction='none')
    counts=(F.one_hot(target,3)*mask.unsqueeze(-1)).sum(dim=1).to(output.dtype)
    weights=mask.sum(dim=1,keepdim=True)/counts.clamp_min(1)
    alpha=weights.gather(1,target)
    focal=alpha*(1-(-ce).exp()).square()*ce
    return (nll*mask).sum()/mask.sum(),(focal*mask).sum()/mask.sum()


def ocr_loss(model, features, point_mask, labels):
    """Interpolate latent features for OCR only; true trajectory lengths unchanged."""
    features=F.interpolate(features,size=point_mask.size(1),mode='linear',align_corners=False)
    ocr=model.ocr_model
    x=ocr.pos_encoder(ocr.input_proj(features.permute(0,2,1)))
    x=ocr.transformer(x,src_key_padding_mask=~point_mask)
    log_probs=ocr.output_fc(x).log_softmax(-1).transpose(0,1)
    target_lengths=(labels>=0).sum(1)
    repeats=((labels[:,1:]==labels[:,:-1]) & (labels[:,1:]>=0)).sum(1)
    input_lengths=point_mask.sum(1)
    valid=(target_lengths>0)&(input_lengths>=target_lengths+repeats)
    if not valid.any():raise ValueError('No feasible CTC paths in batch; inspect preprocessing')
    selected=labels[valid]
    targets=selected[selected>=0]+1
    loss=F.ctc_loss(log_probs[:,valid],targets,input_lengths[valid],target_lengths[valid],blank=0,zero_infinity=False)
    return loss,int(valid.sum()),int((~valid).sum())


def compute_losses(model,batch,config,training):
    data=batch['data'].permute(0,2,1);mask=batch['mask'];labels=batch['labels']
    z,mu,logvar=model.encode(data)
    features=z if training else mu
    output=model.decode(features)
    latent_mask=downsample_mask(mask,8).bool()
    gmm,pen=reconstruction_loss(output,data,mask)
    kl=.5*(mu.square()+logvar.exp()-1-logvar)
    kl=(kl*latent_mask.unsqueeze(1)).sum()/(latent_mask.sum()*mu.size(1))
    ctc,valid,invalid=ocr_loss(model,features,mask,labels)
    style=F.cross_entropy(model.style_classifier(features,latent_mask),batch['writers']) if training else features.sum()*0
    recon=gmm*config['gmm_weight']+pen*config['pen_weight']
    total=recon+kl*config['kl_weight']+ctc*config['ctc_weight']+style*config['style_weight']
    return total,{'total':total,'reconstruction':recon,'gmm':gmm,'pen':pen,'kl':kl,'ctc':ctc,'style':style,
                  'ctc_valid_samples':valid,'ctc_invalid_samples':invalid}
