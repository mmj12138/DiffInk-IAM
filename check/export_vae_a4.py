"""Lay out saved VAE GT/mu_argmax NPY trajectories; no model inference.

Requires numpy and reportlab (PDF); --preview additionally requires pdftoppm.
Example: python -m check.export_vae_a4 --input outputs/vae_w1ks0iw8 --output outputs/iam_plotter_test
"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def trajectory_strokes(seq):
    """Absolute XY; column 2 connects to next point. No character-end cropping.

    Retain the legacy SVG jump rule: integer pixels at 256 px high with
    20 px margins; reject segments >= 200 px. Isolated points remain dots.
    """
    if seq.ndim != 2 or seq.shape[1] != 5 or not len(seq) or not np.isfinite(seq).all():
        raise ValueError('Expected finite nonempty [T,5] trajectory')
    if not np.all(np.isin(seq[:,2:], [0,1])) or not np.all(seq[:,2:].sum(1)==1):
        raise ValueError('Expected one-hot pen states')
    xy=seq[:,:2].astype(np.float64)
    scale=216/(np.ptp(xy[:,1])+1e-6)
    pixels=np.column_stack(((xy[:,0]-xy[:,0].min())*scale+20,
                           (xy[:,1].max()-xy[:,1])*scale+20)).astype(np.int64)
    jumps=np.linalg.norm(np.diff(pixels,axis=0).astype(float),axis=1)>=200
    drawable=(seq[:-1,2]==1)&~jumps
    used=np.zeros(len(seq),dtype=bool);runs=[];active=[]
    for i,draw in enumerate(drawable):
        if draw:
            if not active:active=[i]
            active.append(i+1);used[i:i+2]=True
        elif active:
            runs.append(xy[active]);active=[]
    if active:runs.append(xy[active])
    runs.extend(xy[i:i+1] for i in np.flatnonzero(~used))
    return runs,int(((seq[:-1,2]==1)&jumps).sum())


def load_pairs(inputs, sample_keys, max_dtw, limit):
    pairs=[];seen=set()
    for directory in inputs:
        report_path=directory/'report.json'
        report=json.loads(report_path.read_text())
        evaluation=report['evaluation']
        if 'mu' not in evaluation.get('decode','') or not evaluation.get('checkpoint'):
            raise ValueError(f'{report_path}: expected saved VAE mu reconstruction report')
        for row in report['samples']:
            key=row['key'];dtw=row['metrics']['mu_argmax']['dtw']['normalized']
            if key in seen or (sample_keys and key not in sample_keys) or (not sample_keys and dtw>max_dtw):continue
            paths=[directory/f'{row["index"]:03d}_{name}.npy' for name in ('gt','mu_argmax')]
            if not all(p.is_file() for p in paths):
                if sample_keys:raise FileNotFoundError(f'Missing GT/recon NPY for {key}')
                continue
            curves=[np.load(p,allow_pickle=False) for p in paths]
            if any(len(a)!=row['points'] for a in curves):raise ValueError(f'{key}: truncated trajectory')
            extracted=[trajectory_strokes(a) for a in curves]
            both=np.concatenate([s for runs,_ in extracted for s in runs])
            gtxy=np.concatenate(extracted[0][0])
            pairs.append(dict(row=row,evaluation=evaluation,source_report=str(report_path),
                              npy_paths=list(map(str,paths)),strokes=[v[0] for v in extracted],
                              rejected=[v[1] for v in extracted],lo=both.min(0),hi=both.max(0),
                              gt_height=float(np.ptp(gtxy[:,1]))))
            seen.add(key)
    if sample_keys:
        missing=set(sample_keys)-seen
        if missing:raise ValueError(f'Requested samples unavailable: {sorted(missing)}')
        pairs.sort(key=lambda p:sample_keys.index(p['row']['key']))
    else:pairs.sort(key=lambda p:p['row']['metrics']['mu_argmax']['dtw']['normalized'])
    pairs=pairs[:limit]
    if not pairs:raise ValueError('No usable saved reconstruction pairs')
    return pairs


def svg_document(width,height,groups,stroke_width):
    root=ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width=f'{width}mm',height=f'{height}mm',
                    viewBox=f'0 0 {width} {height}')
    for identifier,strokes in groups:
        group=ET.SubElement(root,'g',id=identifier,fill='none',stroke='black',
                            **{'stroke-width':str(stroke_width),'stroke-linecap':'round','stroke-linejoin':'round'})
        for points in strokes:
            # A 0.02 mm round-capped segment makes an isolated dot plotter-visible.
            if len(points)==1:points=np.vstack((points,points+[.02,0]))
            commands=['M {:.8f} {:.8f}'.format(*points[0])]
            commands.extend('L {:.8f} {:.8f}'.format(*point) for point in points[1:])
            ET.SubElement(group,'path',d=' '.join(commands))
    return ET.tostring(root,encoding='unicode')


def write_pdf(path,width,height,groups,stroke_width):
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.units import mm
    canvas=Canvas(str(path),pagesize=(width*mm,height*mm),pageCompression=1)
    canvas.setLineWidth(stroke_width*mm);canvas.setLineCap(1);canvas.setLineJoin(1)
    for _,strokes in groups:
        for points in strokes:
            if len(points)==1:points=np.vstack((points,points+[.02,0]))
            shape=canvas.beginPath();shape.moveTo(points[0,0]*mm,(height-points[0,1])*mm)
            for x,y in points[1:]:shape.lineTo(x*mm,(height-y)*mm)
            canvas.drawPath(shape,stroke=1,fill=0)
    canvas.showPage();canvas.save()


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-keys',nargs='+')
    parser.add_argument('--max-dtw',type=float,default=.2,help='Automatic selection only; not a guarantee of visual quality')
    parser.add_argument('--limit',type=int,help='Default: all explicit keys, otherwise three automatic candidates')
    parser.add_argument('--page-width-mm',type=float,default=210)
    parser.add_argument('--page-height-mm',type=float,default=297)
    parser.add_argument('--margin-mm',type=float,default=15)
    parser.add_argument('--target-height-mm',type=float,default=10)
    parser.add_argument('--line-gap-mm',type=float,default=7)
    parser.add_argument('--pair-gap-mm',type=float,default=12)
    parser.add_argument('--max-pairs-per-page',type=int,default=5)
    parser.add_argument('--stroke-width-mm',type=float,default=.3)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if args.limit is None:args.limit=len(args.sample_keys) if args.sample_keys else 3
    positives=[args.page_width_mm,args.page_height_mm,args.margin_mm,args.target_height_mm,args.stroke_width_mm]
    if not all(np.isfinite(v) and v>0 for v in positives) or args.limit<1 or args.max_pairs_per_page<1:
        parser.error('Dimensions and counts must be positive')
    if not all(np.isfinite(v) and v>=0 for v in [args.line_gap_mm,args.pair_gap_mm,args.max_dtw]):
        parser.error('Gaps and max DTW must be nonnegative')
    if args.margin_mm<14:parser.error('Margin must be >=14 mm to keep handwriting clear of calibration crosses')
    width,height=args.page_width_mm,args.page_height_mm;margin=args.margin_mm
    if min(width,height)<=2*margin:parser.error('Margins leave no usable page area')
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    # Fail before creating partial output if the PDF dependency is unavailable.
    try:import reportlab
    except ImportError:parser.error('PDF dependency missing: pip install reportlab')
    if args.preview and not shutil.which('pdftoppm'):parser.error('--preview requires pdftoppm (Poppler)')
    pairs=load_pairs(args.input,args.sample_keys,args.max_dtw,args.limit)
    max_width=max(float((p['hi']-p['lo'])[0]) for p in pairs)
    max_height=max(float((p['hi']-p['lo'])[1]) for p in pairs)
    median_height=float(np.median([p['gt_height'] for p in pairs]))
    if min(max_height,max_width,median_height)<=0:raise ValueError('Degenerate trajectories')
    # One global isotropic scale, fitted to median GT height, width and a full pair.
    room_w=width-2*margin-args.stroke_width_mm-.02
    room_h=height-2*margin-args.stroke_width_mm
    scale=min(args.target_height_mm/median_height,room_w/max_width,
              (room_h-args.line_gap_mm)/(2*max_height))
    if scale<=0:raise ValueError('Page too short for a pair')
    args.output.mkdir(parents=True);(args.output/'pages').mkdir();(args.output/'individual').mkdir()
    pages=[];metadata=[];calibration=[];cursor=margin+args.stroke_width_mm/2;pair_count=0
    for pair_number,pair in enumerate(pairs,1):
        span=(pair['hi']-pair['lo'])*scale;needed=2*span[1]+args.line_gap_mm
        if not pages or pair_count>=args.max_pairs_per_page or cursor+needed>height-margin-args.stroke_width_mm/2:
            pages.append([]);cursor=margin+args.stroke_width_mm/2;pair_count=0
        page_id=f'page_{len(pages):03d}';pair_id=f'pair_{pair_number:03d}'
        x_origin=margin+args.stroke_width_mm/2
        for index,sample_type in enumerate(('GT','VAE_RECON')):
            y_origin=cursor+index*(span[1]+args.line_gap_mm)
            placed=[np.column_stack((x_origin+(s[:,0]-pair['lo'][0])*scale,
                                      y_origin+(pair['hi'][1]-s[:,1])*scale)) for s in pair['strokes'][index]]
            isolated=sum(len(s)==1 for s in placed)
            placed=[np.vstack((s,s+[.02,0])) if len(s)==1 else s for s in placed]
            sample_id=f'{pair_id}_{sample_type.lower()}'
            points=np.concatenate(placed);lo=points.min(0);hi=points.max(0)
            if lo[0]<margin or lo[1]<margin or hi[0]>width-margin or hi[1]>height-margin:
                raise ValueError('Layout overflow')
            pages[-1].append((sample_id,placed))
            # Standalone SVG keeps the exact page position and physical scale.
            individual=f'individual/{sample_id}.svg'
            (args.output/individual).write_text(svg_document(width,height,[(sample_id,placed)],args.stroke_width_mm))
            row=pair['row'];metadata.append(dict(page_id=page_id,sample_id=sample_id,pair_id=pair_id,
                sample_type=sample_type,writer_id=row.get('writer_id',''),text=row.get('text',''),
                source_line_id=row['key'],source_file=pair['evaluation'].get('input',''),
                svg_file=f'pages/{page_id}.svg',individual_svg=individual,
                x_mm=float(lo[0]),y_mm=float(lo[1]),width_mm=float(hi[0]-lo[0]),height_mm=float(hi[1]-lo[1]),
                scale_factor=scale,dtw=row['metrics']['mu_argmax']['dtw']['normalized'],
                source_report=pair['source_report'],source_npy=pair['npy_paths'][index],
                checkpoint=pair['evaluation']['checkpoint'],rejected_jump_segments=pair['rejected'][index],
                isolated_dot_paths=isolated))
        cursor+=needed+args.pair_gap_mm;pair_count+=1
    for index,groups in enumerate(pages,1):
        page_id=f'page_{index:03d}'
        for point_id,x,y in [('top_left',10,10),('top_right',width-10,10),
                             ('bottom_left',10,height-10),('bottom_right',width-10,height-10)]:
            groups.append(('calibration_'+point_id,[np.array([[x-3,y],[x+3,y]]),np.array([[x,y-3],[x,y+3]])]))
            calibration.append(dict(page_id=page_id,point_id=point_id,x_mm=x,y_mm=y))
        svg=args.output/'pages'/f'{page_id}.svg';pdf=svg.with_suffix('.pdf')
        svg.write_text(svg_document(width,height,groups,args.stroke_width_mm))
        ET.parse(svg)
        write_pdf(pdf,width,height,groups,args.stroke_width_mm)
        if args.preview:subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(pdf),str(svg.with_suffix(''))],check=True)
    write_csv(args.output/'metadata.csv',metadata);write_csv(args.output/'calibration_points.csv',calibration)
    notes={'page_width_mm':width,'page_height_mm':height,'global_mm_per_coordinate_unit':scale,
           'target_median_gt_height_mm':args.target_height_mm,'actual_median_gt_height_mm':median_height*scale,
           'coordinate_policy':'Saved original normalized XY -> fixed isotropic mm/unit scale; Y flipped downward; not calibrated physical IAM units. GT/recon share pair origin and global scale.',
           'trajectory_policy':'Complete saved arrays; no penultimate-character cropping; column 2 connects to next point; isolated dots are 0.02 mm round-capped paths.',
           'jump_policy':'Legacy SVG rule: height 256 px, padding 20 px, integer-pixel segment distance <200; applied without character truncation. Rejected counts in metadata.csv.',
           'pdf_dependency':'reportlab; PDF and SVG use the same placed path coordinates; print at 100% actual size.',
           'calibration_policy':'Crosses are scan-registration marks, not physical plotter-origin calibration.',
           'selection_policy':'Explicit keys, or available saved VAE reconstructions below max DTW; not representative evaluation. Reconstruction defects are not repaired.',
           'missing_metadata':[{'sample_id':m['sample_id'],'fields':[k for k in ('writer_id','text','source_file') if not m[k]]} for m in metadata if any(not m[k] for k in ('writer_id','text','source_file'))],
           'pairs':len(pairs),'pages':len(pages),'requested_limit':args.limit}
    (args.output/'layout.json').write_text(json.dumps(notes,indent=2))
    print(json.dumps(notes,indent=2));print('Results:',args.output)


if __name__=='__main__':main()
