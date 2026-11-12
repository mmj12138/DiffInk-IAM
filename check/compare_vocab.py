"""Compare preview vocabulary keys against the pretrained model vocabulary."""
import argparse
import json
from pathlib import Path


def load_characters(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or any(len(key) != 1 for key in data):
        raise ValueError(f'{path}: expected a JSON object with single-character keys')
    return data


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, default=root/'datas/iam_prepared/preview_text.json')
    parser.add_argument('--original', type=Path, default=root/'datas/meta/All_zi.json')
    args = parser.parse_args()
    preview = load_characters(args.preview)
    original = load_characters(args.original)
    missing = sorted(set(preview) - set(original))
    shared = set(preview) & set(original)
    print(f'Preview: {args.preview} ({len(preview)} keys, including validation sentinel if present)')
    print(f'Original: {args.original} ({len(original)} keys)')
    print(f'Shared: {len(shared)}; missing from original: {len(missing)}')
    for index, char in enumerate(missing, 1):
        label = 'SPACE' if char == ' ' else repr(char)
        print(f'{index:2d}. {label:10s} U+{ord(char):04X}')
    print('Missing characters (JSON):', json.dumps(missing, ensure_ascii=False))


if __name__ == '__main__':
    main()
