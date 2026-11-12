"""Validate IAM lines with the strict converter and produce an inspection report."""
import argparse
import json
from collections import Counter
from pathlib import Path
from prepare.iam import convert_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('datas/iam'))
    parser.add_argument('--output', type=Path, default=Path('datas/iam_scan_report.json'))
    args = parser.parse_args()
    counts, errors, keys = Counter(), [], set()
    writers, chars, lengths = set(), set(), []
    paths = sorted(args.input.rglob('*.json'))
    if not paths:
        parser.error('No JSON files found')
    for path in paths:
        counts[path.parent.name] += 1
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            seq, _, text, writer, _ = convert_line(data)
            key = data['key'].replace('/', '_')
            if key in keys:
                raise ValueError(f'Duplicate key: {key}')
            keys.add(key); writers.add(writer); chars.update(text); lengths.append(len(seq))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({'file':str(path), 'error':str(exc)})
    report = {'directory_counts':dict(counts), 'valid':len(lengths), 'writers':len(writers),
              'characters':len(chars), 'train_length_eligible':sum(200 <= n <= 2000 for n in lengths),
              'length_range':[min(lengths),max(lengths)] if lengths else [],
              'error_count':len(errors), 'errors':errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
