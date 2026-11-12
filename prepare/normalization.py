"""Explicit coordinate conventions for IAM line preprocessing."""
import numpy as np


def normalize_points(points, mode='point_y'):
    xy=np.asarray([[p['x'],p['y']] for p in points],dtype=np.float64)
    if not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Empty or non-finite coordinates')
    if mode=='point_y':
        center=xy.mean(axis=0)
        scale=float(xy[:,1].std())
    elif mode=='arc_y':
        same=np.asarray([a['stroke']==b['stroke'] for a,b in zip(points[:-1],points[1:])],dtype=bool)
        a,b=xy[:-1][same],xy[1:][same]
        lengths=np.linalg.norm(b-a,axis=1)
        total=float(lengths.sum())
        if total<=0:
            raise ValueError('No nonzero within-stroke segments for arc normalization')
        center=(lengths[:,None]*(a+b)/2).sum(axis=0)/total
        u,v=a[:,1]-center[1],b[:,1]-center[1]
        variance=float((lengths*(u*u+u*v+v*v)/3).sum()/total)
        scale=float(np.sqrt(max(variance,0)))
    else:
        raise ValueError(f'Unknown normalization mode: {mode}')
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Zero vertical extent')
    normalized=(xy-center)/scale
    normalized[:,1]*=-1
    return normalized,center,scale
