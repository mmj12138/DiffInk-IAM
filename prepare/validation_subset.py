"""Build the original-vocabulary-compatible IAM validation subset."""
import argparse
import json
import shutil
from pathlib import Path
import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('datas/iam_vocab_check.json'))
    parser.add_argument('--source',type=Path,default=Path('datas/iam_prepared/all.h5'))
    parser.add_argument('--vocab',type=Path,default=Path('datas/meta/All_zi.json'))
    parser.add_argument('--output',type=Path,default=Path('datas/iam_validation'))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    report=json.loads(args.report.read_text())
    selected=[r for r in report['covered'] if r['validation_eligible']]
    vocab=json.loads(args.vocab.read_text())
    if '、' not in vocab: raise ValueError('Validation sentinel missing')
    entries=[]
    with h5py.File(args.source,'r') as source:
        groups={Path(g.attrs['source']).resolve():key for key,g in source.items()}
        for r in selected:
            key=groups[Path(r['source']).resolve()];g=source[key]
            text=g['line_text'].asstr()[()]
            if set(text)-vocab.keys() or len(g['point_seq'])<200: raise ValueError(key)
            entries.append({'key':key,'source':r['source'],'text':text,
                            'writer_id':g['writer_id'].asstr()[()], 'points':len(g['point_seq'])})
        args.output.mkdir(parents=True)
        with h5py.File(args.output/'val.h5','w') as dest:
            for k,v in source.attrs.items(): dest.attrs[k]=v
            for r in entries: source.copy(r['key'],dest)
    shutil.copyfile(args.vocab,args.output/'text.json')
    (args.output/'manifest.json').write_text(json.dumps({'lines':len(entries),'samples':entries},ensure_ascii=False,indent=2))
    with h5py.File(args.output/'val.h5','r') as hf:
        assert len(hf)==len(selected)
        for g in hf.values():
            assert len(g['point_seq'])>=200
            assert not set(g['line_text'].asstr()[()])-vocab.keys()
    print(f'Created {len(entries)} validation samples in {args.output}')


if __name__=='__main__': main()
