"""Render rejected IAM lines without changing their original point order."""
import colorsys
import html
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    report = json.loads(Path('datas/iam_scan_report.json').read_text())
    output = Path('datas/iam_inspection')
    (output / 'lines').mkdir(parents=True, exist_ok=True)
    (output / 'sheets').mkdir(exist_ok=True)
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 16)
    small = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 12)
    records = []
    thumbnails = []
    for number, item in enumerate(report['errors'], 1):
        data = json.loads(Path(item['file']).read_text())
        points = data['points']
        segments = data['gt_segmentation']['segments']
        lookup = {(p['stroke'], p['index']): i for i, p in enumerate(points)}
        owners = [set() for _ in points]
        details = []
        last = -1
        counts = Counter()
        for ci, seg in enumerate(segments):
            ranges = seg.get('inkRanges', [])
            if not ranges and not seg['substring'].isspace():
                counts['missing_ink'] += 1
                details.append(f"char {ci} {seg['substring']!r}: no ink")
            for ink in ranges:
                a = lookup[(ink['startStroke'], ink['startPoint'])]
                b = lookup[(ink['endStroke'], ink['endPoint'])]
                if b < a:
                    counts['reversed_range'] += 1
                if a <= last:
                    counts['nonmonotonic_range'] += 1
                    details.append(f"char {ci} {seg['substring']!r}: source [{a},{b}] after {last}")
                for j in range(a, b + 1):
                    owners[j].add(ci)
                last = b
        counts['overlap_points'] = sum(len(o) > 1 for o in owners)
        counts['uncovered_points'] = sum(not o for o in owners)
        x0, x1 = min(p['x'] for p in points), max(p['x'] for p in points)
        y0, y1 = min(p['y'] for p in points), max(p['y'] for p in points)
        scale = min(1540 / max(x1-x0, 1), 160 / max(y1-y0, 1))
        coords = [(30+(p['x']-x0)*scale, (p['y']-y0)*scale) for p in points]
        image = Image.new('RGB', (1600, 620), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((20, 8), f"{number:04d}  {Path(item['file']).name}  writer={data.get('writer_id')}", font=font, fill='black')
        draw.text((20, 34), 'TEXT: '+data['text'], font=font, fill='black')
        draw.text((20, 60), 'SEG:  '+''.join(s['substring'] for s in segments), font=font, fill='black')
        draw.text((20, 86), item['error'], font=font, fill='red')
        draw.text((20, 113), 'Original strokes (no reordering)', font=small, fill='black')
        draw.text((20, 315), 'Character colors; red=overlap, gray=uncovered; dots show isolated points', font=small, fill='black')
        for offset, colored in [(140, False), (340, True)]:
            for j, (x,y) in enumerate(coords):
                color = 'black'
                if colored:
                    color = ('red' if len(owners[j]) > 1 else '#aaaaaa' if not owners[j] else tuple(int(v*210) for v in colorsys.hsv_to_rgb((min(owners[j])*.618)%1,.8,.85)))
                draw.ellipse((x-1,y+offset-1,x+1,y+offset+1),fill=color)
                if j+1 < len(points) and points[j]['stroke'] == points[j+1]['stroke']:
                    xx,yy = coords[j+1]
                    draw.line((x,y+offset,xx,yy+offset),fill=color,width=2)
        draw.text((20, 515), json.dumps(dict(counts)), font=small, fill='black')
        for row, detail in enumerate(details[:4]):
            draw.text((20, 540+row*17), detail, font=small, fill='black')
        name = f'{number:04d}_{Path(item["file"]).stem}.png'
        image.save(output/'lines'/name)
        thumb = image.copy();thumb.thumbnail((800,310));thumbnails.append(thumb)
        records.append({**item,'image':'lines/'+name,'diagnostics':dict(counts),'details':details})
    for start in range(0,len(thumbnails),16):
        sheet=Image.new('RGB',(1600,2480),'#dddddd')
        for j,thumb in enumerate(thumbnails[start:start+16]):
            sheet.paste(thumb,((j%2)*800,(j//2)*310))
        sheet.save(output/'sheets'/f'sheet_{start//16+1:03d}.jpg',quality=88)
    (output/'diagnostics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    rows=['<!doctype html><meta charset="utf-8"><title>IAM inspection</title><h1>IAM rejected lines</h1><p>Original point order preserved. Colors mark character ownership; red marks overlap, gray marks uncovered points.</p>']
    for r in records:
        rows.append(f'<details><summary>{html.escape(r["image"])} — {html.escape(r["error"])} </summary><p>{html.escape(r["file"])}</p><img loading="lazy" width="1200" src="{html.escape(r["image"])}"><pre>{html.escape(json.dumps(r["diagnostics"]))}</pre></details>')
    (output/'index.html').write_text('\n'.join(rows))
    print('Rendered',len(records),'lines;', (len(records)+15)//16,'contact sheets')
    print('Diagnostic totals:',dict(sum((Counter(r['diagnostics']) for r in records),Counter())))
    print('Nonmonotonic without overlap:',sum(r['diagnostics'].get('nonmonotonic_range',0)>0 and r['diagnostics'].get('overlap_points',0)==0 for r in records))


if __name__ == '__main__':
    main()
