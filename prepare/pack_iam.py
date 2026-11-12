"""Pack checked IAM JSON into one HDF5, with an explicit text-mismatch policy."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .iam import convert_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('datas/iam'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-text-mismatches', action='store_true', required=True,
                        help='Explicitly exclude unresolved text/segmentation disagreements')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.input.rglob('*.json'))
    if not paths:
        raise ValueError('No JSON files found')
    accepted, excluded, seen = [], [], set()
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        segments = data['gt_segmentation']['segments']
        missing = [i for i,s in enumerate(segments) if not s.get('inkRanges') and not s['substring'].isspace()]
        if missing:
            excluded.append({'source':str(path), 'reason':'nonspace_missing_ink', 'indices':missing})
            continue
        if ''.join(s['substring'] for s in segments) != data['text']:
            excluded.append({'source':str(path), 'reason':'unresolved_text_mismatch'})
            continue
        result = convert_line(data, reorder_by_character=True)
        key = data['key'].replace('/', '_')
        if key in seen:
            raise ValueError(f'Duplicate key: {key}')
        seen.add(key)
        accepted.append((key, result, str(path)))
    args.output.mkdir(parents=True)
    with h5py.File(args.output/'all.h5', 'w') as hf:
        hf.attrs['coordinate_format'] = 'absolute, centered, y-up, divided by line y std'
        hf.attrs['point_order'] = 'character order; source order retained within each inkRange'
        hf.attrs['split_policy'] = 'unassigned; original JSON split ignored'
        for key,(seq,ends,text,writer,scale),source in accepted:
            group = hf.create_group(key)
            group.create_dataset('point_seq',data=seq,dtype='float32')
            group.create_dataset('char_points_idx',data=ends,dtype='int32')
            group.create_dataset('line_text',data=text,dtype=h5py.string_dtype('utf-8'))
            group.create_dataset('writer_id',data=writer,dtype=h5py.string_dtype('utf-8'))
            group.attrs['source'] = source
            group.attrs['coordinate_scale'] = scale
    # Reopen the actual file and verify every record, including character endpoints.
    with h5py.File(args.output/'all.h5','r') as hf:
        assert len(hf) == len(accepted)
        for g in hf.values():
            seq = g['point_seq'][:]; ends = g['char_points_idx'][:]
            text = g['line_text'].asstr()[()]
            assert seq.shape[1] == 5 and np.isfinite(seq).all()
            assert len(ends) == len(text) and ends[-1] == len(seq)
            assert np.all(np.diff(ends) >= 0)
            assert np.all(np.isin(seq[:,2:],[0,1])) and np.all(seq[:,2:].sum(1) == 1)
            assert np.all(seq[np.unique(ends[ends>0])-1,4] == 1)
    chars = sorted(set(''.join(r[1][2] for r in accepted)) | {'、'})
    (args.output/'preview_text.json').write_text(json.dumps(dict.fromkeys(chars,0),ensure_ascii=False,indent=2))
    report = {'input_lines':len(paths),'written_lines':len(accepted),'writers':len({r[1][3] for r in accepted}),
              'train_length_eligible':sum(200 <= len(r[1][0]) <= 2000 for r in accepted),
              'validation_length_eligible':sum(len(r[1][0]) >= 200 for r in accepted),
              'excluded':excluded,'validation':'all written HDF5 records checked',
              'vocabulary':'preview only; not compatible with pretrained text embedding without migration'}
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='excluded'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
