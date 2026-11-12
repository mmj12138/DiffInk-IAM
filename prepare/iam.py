"""Convert segmented IAM JSON lines into DiffInk's existing HDF5 schema."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .normalization import normalize_points


def convert_line(data, *, reorder_by_character=False, normalization="point_y"):
    points = data['points']
    segments = data['gt_segmentation']['segments']
    text = ''.join(s['substring'] for s in segments)
    if text != data['text']:
        raise ValueError('Segmentation text differs from line text')
    if not points or not text:
        raise ValueError('Empty line')
    lookup = {(p['stroke'], p['index']): i for i, p in enumerate(points)}
    if len(lookup) != len(points):
        raise ValueError('Duplicate stroke/point IDs')
    xy, _, scale = normalize_points(points, mode=normalization)
    rows, ends, used = [], [], set()
    last_source = -1
    for segment in segments:
        if len(segment['substring']) != 1:
            raise ValueError('Expected one Unicode character per segment')
        ranges = segment.get('inkRanges', [])
        if not ranges and not segment['substring'].isspace():
            raise ValueError('Non-space character has no inkRanges')
        for ink in ranges:
            start = lookup[(ink['startStroke'], ink['startPoint'])]
            stop = lookup[(ink['endStroke'], ink['endPoint'])]
            if stop < start or (not reorder_by_character and start <= last_source):
                raise ValueError('Reversed, overlapping, or out-of-order inkRanges')
            for i in range(start, stop + 1):
                if i in used:
                    raise ValueError('Overlapping inkRanges reuse a point')
                used.add(i)
                connected = i < stop and points[i]['stroke'] == points[i + 1]['stroke']
                rows.append([*xy[i], 1 if connected else 0, 0 if connected else 1, 0])
            last_source = stop
        if ranges:
            rows[-1][2:] = [0, 0, 1]  # Character end, as in the shipped dataset.
        # Exclusive endpoint: spaces consume no points but retain their text index.
        ends.append(len(rows))
    if len(used) != len(points):
        raise ValueError(f'{len(points) - len(used)} points are outside segmentation')
    writer = str(data['writer_id'])
    if not writer or writer.lower() in ('none', 'unknown'):
        raise ValueError('Missing writer_id')
    return np.asarray(rows, np.float32), np.asarray(ends, np.int64), text, writer, scale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One JSON or a directory (recursive)')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--text-file', type=Path, help='Reuse a training vocabulary for val/test')
    parser.add_argument('--writer-file', type=Path, help='Reuse training writer metadata')
    args = parser.parse_args(argv)
    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob('*.json'))
    if not paths:
        parser.error('No JSON files found')
    # Validate everything before writing any output; never silently drop a bad line.
    converted = []
    keys = set()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            key = data['key'].replace('/', '_')
            if key in keys:
                raise ValueError(f'Duplicate line key {key}')
            keys.add(key)
            converted.append((key, convert_line(data), str(path)))
        except Exception as exc:
            raise ValueError(f'{path}: {exc}') from exc
    vocab = json.loads(args.text_file.read_text()) if args.text_file else dict.fromkeys(
        sorted(set(''.join(item[1][2] for item in converted)) | {'、'}), 0)
    writers = json.loads(args.writer_file.read_text()) if args.writer_file else {
        'train': sorted(set(item[1][3] for item in converted)), 'val': [], 'test': []}
    for key, (_, _, text, writer, _), _ in converted:
        if set(text) - vocab.keys():
            raise ValueError(f'{key}: characters missing from vocabulary')
        if args.split == 'train' and writer not in writers['train']:
            raise ValueError(f'{key}: writer missing from training metadata')
    args.output.mkdir(parents=True, exist_ok=True)
    h5_path = args.output / f'{args.split}.h5'
    destinations = [h5_path, args.output / 'text.json', args.output / 'writers.json', args.output / 'manifest.json', args.output / 'vae_iam.yaml']
    if any(p.exists() for p in destinations):
        raise FileExistsError('Output files already exist; choose a new output directory')
    with h5py.File(h5_path, 'w') as hf:
        hf.attrs['coordinate_format'] = 'absolute, centered, y-up, divided by line y std'
        for key, (seq, ends, text, writer, scale), source in converted:
            group = hf.create_group(key)
            group.create_dataset('point_seq', data=seq)
            group.create_dataset('char_points_idx', data=ends)
            group.create_dataset('line_text', data=text.encode('utf-8'))
            group.create_dataset('writer_id', data=writer.encode('utf-8'))
            group.attrs['source'] = source
            group.attrs['coordinate_scale'] = scale
    (args.output / 'text.json').write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    (args.output / 'writers.json').write_text(json.dumps(writers, indent=2))
    manifest = {'split': args.split, 'lines': len(converted), 'train_length_eligible': sum(200 <= len(x[1][0]) <= 2000 for x in converted), 'sources': [x[2] for x in converted]}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs' / 'vae_config.yaml').read_text())
    config.update(train_file=str(h5_path.resolve()) if args.split == 'train' else None,
                  val_file=str(h5_path.resolve()) if args.split in ('val', 'test') else None,
                  text_file=str((args.output / 'text.json').resolve()),
                  writer_file=str((args.output / 'writers.json').resolve()),
                  num_text_embedding=len(vocab) + 1, num_writer=len(writers['train']),
                  train_batch_size=1, val_batch_size=1, aug_prob=0.0)
    (args.output / 'vae_iam.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(manifest, indent=2))
    print(f'HDF5: {h5_path}')


if __name__ == '__main__':
    main()
