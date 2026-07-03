#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter

def read_jsonl(path: Path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except Exception as e: raise SystemExit(f'{path}:{n}: {e}')
    return rows

def summarize(path: Path):
    rows=read_jsonl(path)
    ids=[r['id'] for r in rows]
    classes=Counter(r.get('expected_tool') or 'NO_CALL' for r in rows)
    tags=Counter(t for r in rows for t in r.get('tags', []))
    hist=sum(1 for r in rows if r.get('history'))
    intent=sum(1 for r in rows if r.get('expected_intent'))
    print(f'\n{path}')
    print(f'  rows: {len(rows)}')
    print(f'  duplicate ids: {len(ids)-len(set(ids))}')
    print(f'  rows with history: {hist}')
    print(f'  rows with expected_intent: {intent}')
    print('  class distribution:')
    for k,v in classes.most_common(): print(f'    {k}: {v}')
    print('  top tags:')
    for k,v in tags.most_common(12): print(f'    {k}: {v}')

if __name__ == '__main__':
    paths = [Path(p) for p in sys.argv[1:]] or [
        Path('data/eval/independent_holdout_300.jsonl'),
        Path('data/eval/manual_practical_100.jsonl'),
    ]
    for p in paths: summarize(p)
