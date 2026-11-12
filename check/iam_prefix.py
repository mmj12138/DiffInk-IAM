"""Audit actual DiT prefix masks and render complete trajectories without truncation."""
import argparse
import html
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.mask import build_prefix_mask_from_char_points


def prefix_info(seq, ends, text, ratio=0.3):
    n = len(seq)
    ends = np.asarray(ends, dtype=np.int64)
    if len(ends) != len(text) or not len(ends):
        raise ValueError('Text/character endpoint count mismatch')
    if np.any(np.diff(ends) < 0) or ends[0] < 0 or ends[-1] != n:
        raise ValueError('Invalid cumulative character endpoints')
    nonempty = np.r_[ends[0] > 0, np.diff(ends) > 0]
    if not np.all(seq[ends[nonempty]-1, 4] == 1):
        raise ValueError('Character endpoints do not have end-of-character state')
    count = max(1, round(len(ends)*ratio))
    boundary = int(ends[count-1]) if count < len(ends) else n
    padded = ((n+7)//8)*8
    mask = torch.zeros(1,padded,dtype=torch.bool)
    mask[0,:n] = True
    latent, valid, point_mask = build_prefix_mask_from_char_points(
        [ends], mask, compression_factor=8, prefix_ratio=ratio)
    retained = int((latent[0] == 0).sum())
    assert torch.all(point_mask[0,:boundary] == 0)
    assert torch.all(point_mask[0,boundary:n] == 1)
    assert retained == (boundary+7)//8
    effective = min(retained*8,n)
    return {'text':text,'characters':len(text),'prefix_characters':count,
            'prefix_text':text[:count], 'prefix_end_exclusive':boundary,
            'points':n,'point_fraction':boundary/n,
            'retained_latent_tokens':retained,'latent_tokens':padded//8,
            'effective_prefix_end_exclusive':effective,
            'extra_points_in_boundary_block':effective-boundary,
            'actual_latent_generation_mask':latent[0].tolist()}


def trajectory_svg(seq, boundary, effective=None, bounds=None):
    """Blue prefix, orange suffix; no last-character clipping or distance filter."""
    seq = np.asarray(seq)
    xy = seq[:,:2]
    if not np.isfinite(xy).all(): raise ValueError('Non-finite coordinates')
    lo, hi = (xy.min(0),xy.max(0)) if bounds is None else bounds
    scale = min(1100/max(float(hi[0]-lo[0]),1e-6),200/max(float(hi[1]-lo[1]),1e-6))
    points = np.column_stack((20+(xy[:,0]-lo[0])*scale, 20+(hi[1]-xy[:,1])*scale))
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 240">']
    for i,(x,y) in enumerate(points):
        color='#1665c1' if i < boundary else '#d66a00'
        if i+1<len(seq) and seq[i,2] == 1:
            xx,yy=points[i+1]
            parts.append(f'<path d="M{x:.3f},{y:.3f} L{xx:.3f},{yy:.3f}" stroke="{color}" stroke-width="1.6" fill="none"/>')
        else:
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="1.2" fill="{color}"/>')
    if 0 < boundary <= len(points):
        x,y=points[boundary-1]
        parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="none" stroke="#c00"/>')
    if effective is not None and effective > boundary:
        for x,y in points[boundary:effective]:
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="none" stroke="#a0a"/>')
    return '\n'.join(parts+['</svg>'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('datas/iam_validation/val.h5'))
    parser.add_argument('--output',type=Path,default=Path('datas/iam_prefix_check'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    page=['<!doctype html><meta charset="utf-8"><title>IAM prefix audit</title><style>body{font-family:system-ui;margin:24px}article{border-bottom:1px solid #aaa}svg{width:100%;max-width:1140px}</style><h1>IAM prefix audit</h1><p>Blue: exact character prefix. Orange: suffix. Red circle: last prefix point. Purple rings: extra points covered by latent block rounding. No character clipping.</p>']
    with h5py.File(args.input) as hf:
        keys=sorted(hf,key=lambda k:len(hf[k]['point_seq']),reverse=True)
        for index,key in enumerate(keys):
            g=hf[key];seq=g['point_seq'][:];text=g['line_text'].asstr()[()]
            info=prefix_info(seq,g['char_points_idx'][:],text)
            info.update(index=index,key=key)
            rows.append(info)
            page.append(f'<article><h2>{index}: {html.escape(key)} — {html.escape(text)}</h2><p>Prefix: {html.escape(repr(info["prefix_text"]))}; {info["prefix_characters"]}/{len(text)} characters; {info["prefix_end_exclusive"]}/{len(seq)} points; latent boundary={info["effective_prefix_end_exclusive"]}</p>')
            page.append(trajectory_svg(seq,info['prefix_end_exclusive'],info['effective_prefix_end_exclusive'])+'</article>')
    (args.output/'report.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (args.output/'index.html').write_text('\n'.join(page))
    print(f'Checked {len(rows)} samples: {args.output}/index.html')
    for r in rows:
        if r['text']=='photos':print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
