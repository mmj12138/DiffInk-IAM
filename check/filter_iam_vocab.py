"""List IAM JSON lines whose text is covered by an existing vocabulary."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .compare_vocab import load_characters
from prepare.iam import convert_line


def inspect_line(data, vocabulary):
    text = data['text']
    missing = sorted(set(text) - vocabulary.keys())
    result = {'text': text, 'writer_id': data.get('writer_id'),
              'missing_characters': missing, 'point_count': len(data['points'])}
    try:
        seq, _, _, _, _ = convert_line(data, reorder_by_character=True)
        result['conversion_error'] = None
        # Original length filter (temporarily disabled): len(seq) >= 200
        result['validation_eligible'] = not missing and len(text) >= 2
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        result['conversion_error'] = str(exc)
        result['validation_eligible'] = False
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root/'datas/iam')
    parser.add_argument('--vocab', type=Path, default=root/'datas/meta/All_zi.json')
    parser.add_argument('--output', type=Path, default=root/'datas/iam_vocab_check.json')
    args = parser.parse_args()
    vocabulary = load_characters(args.vocab)
    paths = sorted(args.input.rglob('*.json'))
    if not paths:
        parser.error('No JSON files found')
    covered, rejected, errors = [], [], []
    missing_counts = Counter()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            record = {'source': str(path.resolve()), **inspect_line(data, vocabulary)}
            if record['missing_characters']:
                rejected.append(record)
                missing_counts.update(record['missing_characters'])
            else:
                covered.append(record)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({'source': str(path.resolve()), 'error': str(exc)})
    summary = {'input_lines': len(paths), 'text_fully_covered': len(covered),
               'covered_and_convertible': sum(r['conversion_error'] is None for r in covered),
               'covered_and_validation_eligible': sum(r['validation_eligible'] for r in covered),
               'out_of_vocabulary_lines': len(rejected), 'read_or_schema_errors': len(errors)}
    report = {'summary':summary, 'vocabulary': str(args.vocab.resolve()),
              'policy': 'Check text verbatim, including spaces/case/punctuation; validation requires successful conversion and at least 2 text characters. No characters removed.',
              'missing_character_line_counts':dict(sorted(missing_counts.items())),
              'covered':covered, 'rejected':rejected, 'errors':errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for record in covered:
        print(f"{'VAL' if record['validation_eligible'] else 'TEXT ONLY'} {record['source']} {record['text']!r}")
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
