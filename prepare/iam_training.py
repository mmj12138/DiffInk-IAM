"""Prepare reproducible writer-disjoint IAM datasets for training from scratch."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .iam import convert_line


NORMALIZATIONS = {
    'point_y': 'line point mean and population y std; y up; before RDP',
    'arc_y': 'within-stroke arc-length weighted line center and integral y std; y up; before RDP; explicit line-level adaptation'
}


def rdp_indices(xy, epsilon):
    """Iterative RDP, distance to the finite endpoint segment; retain endpoints."""
    xy = np.asarray(xy, dtype=np.float64)
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError('epsilon must be finite and nonnegative')
    if len(xy) <= 2:
        return np.arange(len(xy), dtype=np.int64)
    keep = {0, len(xy)-1}
    pending = [(0, len(xy)-1)]
    while pending:
        start, end = pending.pop()
        if end-start <= 1:
            continue
        v = xy[end]-xy[start]
        middle = xy[start+1:end]
        norm = float(v @ v)
        if norm == 0:
            distances = np.linalg.norm(middle-xy[start], axis=1)
        else:
            t = np.clip((middle-xy[start]) @ v / norm, 0, 1)
            distances = np.linalg.norm(middle-(xy[start]+t[:,None]*v), axis=1)
        far = int(np.argmax(distances))
        if distances[far] > epsilon:
            split = start+1+far
            keep.add(split)
            pending.extend([(start, split), (split, end)])
    return np.asarray(sorted(keep), dtype=np.int64)


def simplify_line(seq, ends, epsilon):
    """Keep stroke/range/character endpoints, then remap exclusive character ends."""
    seq = np.asarray(seq)
    kept, output, start = [], [], 0
    for stop in np.flatnonzero(seq[:,2] == 0):
        indices = rdp_indices(seq[start:stop+1,:2], epsilon) + start
        piece = seq[indices].copy()
        piece[:,2:] = [1,0,0]
        piece[-1,2:] = seq[stop,2:]
        kept.extend(indices.tolist())
        output.append(piece)
        start = stop+1
    if start != len(seq):
        raise ValueError('Input trajectory has no terminating state')
    kept = np.asarray(kept, dtype=np.int64)
    new_ends = np.searchsorted(kept, ends, side='left').astype(np.int32)
    return np.concatenate(output).astype(np.float32), new_ends, kept


def split_writers(writers, seed, test_count, val_count):
    writers = sorted(set(writers))
    if test_count < 1 or val_count < 1 or test_count+val_count >= len(writers):
        raise ValueError('Need nonempty training, validation and test writer sets')
    random.Random(seed).shuffle(writers)
    return {'train':sorted(writers[test_count+val_count:]),
            'val':sorted(writers[test_count:test_count+val_count]),
            'test':sorted(writers[:test_count])}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def check_record(seq, ends, text):
    if seq.ndim != 2 or seq.shape[1] != 5 or not np.isfinite(seq).all():
        raise ValueError('Invalid trajectory')
    if len(ends) != len(text) or ends[-1] != len(seq) or np.any(np.diff(ends)<0):
        raise ValueError('Invalid character endpoints')
    if not np.all(np.isin(seq[:,2:], [0,1])) or not np.all(seq[:,2:].sum(1)==1):
        raise ValueError('Invalid pen states')
    nonempty = np.r_[ends[0]>0, np.diff(ends)>0]
    if not np.all(seq[ends[nonempty]-1,4]==1):
        raise ValueError('Character end state mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('datas/iam'))
    parser.add_argument('--output', type=Path, default=Path('datas/iam_train_rdp005'))
    parser.add_argument('--normalization', choices=['arc_y','point_y'], default='arc_y')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-writers', type=int, default=25)
    parser.add_argument('--val-writers', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error('--epsilon must be finite and nonnegative')
    normalization_description = NORMALIZATIONS[args.normalization]
    paths = sorted(args.input.rglob('*.json'))
    if not paths:
        parser.error('No input JSON files')
    records, excluded, seen = [], [], set()
    for number,path in enumerate(paths,1):
        raw = path.read_bytes()
        data = json.loads(raw)
        segments = data['gt_segmentation']['segments']
        missing = [i for i,s in enumerate(segments) if not s.get('inkRanges') and not s['substring'].isspace()]
        reason = 'nonspace_missing_ink' if missing else ('unresolved_text_mismatch' if ''.join(s['substring'] for s in segments)!=data['text'] else None)
        relative = path.relative_to(args.input).as_posix()
        if reason:
            excluded.append({'source':relative,'reason':reason})
            continue
        try:
            seq,ends,text,writer,scale = convert_line(data,reorder_by_character=True,normalization=args.normalization)
            reduced,new_ends,kept = simplify_line(seq,ends,args.epsilon)
            check_record(reduced,new_ends,text)
        except Exception as exc:
            raise ValueError(f'{path}: {exc}') from exc
        key = data['key'].replace('/', '_')
        if key in seen:
            raise ValueError(f'Duplicate key: {key}')
        seen.add(key)
        records.append({'key':key,'source':relative,'source_sha256':hashlib.sha256(raw).hexdigest(),
                        'text':text,'writer_id':writer,'original_points':len(seq),'points':len(reduced),
                        'coordinate_scale':scale,'seq':reduced,'ends':new_ends,'kept':kept})
        if number%1000==0:
            print(f'Processed {number}/{len(paths)} JSON files',flush=True)
    writers = split_writers([r['writer_id'] for r in records],args.seed,args.test_writers,args.val_writers)
    writer_split = {writer:split for split,ids in writers.items() for writer in ids}
    for r in records:
        r['split'] = writer_split[r['writer_id']]
    # Fit the character vocabulary on training text only, not held-out text.
    frequencies = Counter(''.join(r['text'] for r in records if r['split']=='train'))
    vocabulary = {char:index for index,char in enumerate(sorted(set(frequencies)|{'、'}))}
    oov = {split:sorted(set(''.join(r['text'] for r in records if r['split']==split))-vocabulary.keys()) for split in ['val','test']}
    if any(oov.values()):
        raise ValueError(f'Held-out characters not covered by training vocabulary: {oov}; no outputs written')
    args.output.mkdir(parents=True)
    for split in writers:
        with h5py.File(args.output/f'{split}.h5','w') as hf:
            hf.attrs['normalization'] = normalization_description
            hf.attrs['rdp_epsilon'] = args.epsilon
            hf.attrs['rdp_units'] = 'normalized coordinate units'
            hf.attrs['rdp_scope'] = 'each continuous stroke piece; character/range endpoints forced'
            hf.attrs['point_order'] = 'character order; original order within ranges; source JSON split ignored'
            hf.attrs['split'] = split
            for r in records:
                if r['split'] != split: continue
                g=hf.create_group(r['key'])
                g.create_dataset('point_seq',data=r['seq'],dtype='float32')
                g.create_dataset('char_points_idx',data=r['ends'],dtype='int32')
                for field in ['line_text','writer_id']:
                    g.create_dataset(field,data=r['text'] if field=='line_text' else r[field],dtype=h5py.string_dtype('utf-8'))
                # Indices refer to character-ordered points BEFORE RDP, not JSON point IDs.
                g.create_dataset('pre_rdp_point_indices',data=r['kept'],dtype='int32')
                for field in ['source','source_sha256','original_points','coordinate_scale']:
                    g.attrs[field]=r[field]
    write_json(args.output/'text.json',vocabulary)
    write_json(args.output/'writers.json',writers)
    metadata = [{k:v for k,v in r.items() if k not in ['seq','ends','kept']} for r in records]
    write_json(args.output/'splits.json',{'seed':args.seed,'writers':writers,'samples':metadata})
    stats = {}
    for split in writers:
        subset = [r for r in records if r['split']==split]
        lengths = [r['points'] for r in subset]
        stats[split] = {'writers':len(writers[split]),'lines':len(subset),
                        'points_before_rdp':sum(r['original_points'] for r in subset),
                        'points_after_rdp':sum(lengths), 'length_min':min(lengths),'length_median':float(np.median(lengths)),
                        'length_max':max(lengths),'legacy_train_length_eligible':sum(200<=n<=2000 for n in lengths),
                        'legacy_val_length_eligible':sum(n>=200 for n in lengths),
                        'current_train_eligible':sum(len(r['text'])>=2 and r['points']<=2000 for r in subset),
                        'current_val_eligible':sum(len(r['text'])>=2 for r in subset),
                        'writers_with_train_eligible':len({r['writer_id'] for r in subset if len(r['text'])>=2 and r['points']<=2000})}
    for split, item in stats.items():
        subset = [r for r in records if r['split'] == split]
        required = lambda r: len(r['text']) + sum(a == b for a,b in zip(r['text'],r['text'][1:]))
        item['ctc_feasible_at_latent_resolution'] = sum((r['points']+7)//8 >= required(r) for r in subset)
        item['ctc_feasible_at_point_resolution'] = sum(r['points'] >= required(r) for r in subset)
    # Verify actual saved HDF5 files, not only in-memory data.
    actual_writers = {}
    for split in writers:
        with h5py.File(args.output/f'{split}.h5','r') as hf:
            if len(hf)!=stats[split]['lines']: raise ValueError('Written count mismatch')
            actual_writers[split]=set()
            for g in hf.values():
                text=g['line_text'].asstr()[()]
                check_record(g['point_seq'][:],g['char_points_idx'][:],text)
                writer=g['writer_id'].asstr()[()]
                if writer not in writers[split] or set(text)-vocabulary.keys(): raise ValueError('Invalid split/vocabulary')
                actual_writers[split].add(writer)
    if any(actual_writers[a]&actual_writers[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Writer leakage')
    summary = {'input_lines':len(paths),'accepted_lines':len(records),'excluded_lines':len(excluded),
               'excluded':excluded,'seed':args.seed,'rdp_epsilon':args.epsilon,'normalization':normalization_description,
               'normalization_mode':args.normalization,
               'paper_source':'https://arxiv.org/html/2509.23624#A1',
               'reproduction_limits':['Arc-length weighted y scaling is our explicit line-level adaptation; cited single-character formula uses x scaling.',
                                      'RDP keeps character/range endpoints in addition to stroke endpoints; epsilon 0.05 selected by visual inspection, differing from the paper value 0.5.',
                                      '20 validation writers are our additional selection; paper only specifies 25 test writers.'],
               'vocabulary_size':len(vocabulary),'vocabulary_fit':'training text only; Unicode sorted; values equal key-order IDs; includes legacy validation sentinel 、',
               'num_text_embedding':len(vocabulary)+1,'oov':oov,'splits':stats,
               'checks':'all HDF5 records reread; no shared writers; text coverage and remapped boundaries checked',
               'loader_policy':'at least 2 text characters; max 2000 points for training; 200 point minimum disabled',
               'normalization_status':'explicit_line_adaptation_recorded',
               'training_started':False}
    write_json(args.output/'report.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='excluded'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
