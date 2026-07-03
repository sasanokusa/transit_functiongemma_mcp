#!/usr/bin/env python3
"""r7: station_departures / get_station / list_feeds のreplay例を増量する。

背景(artifacts/CLAUDE_R6_REVIEW.md, 非経路215評価):
  r6ではreverse_geocode/station_departuresの学習例がゼロで、非経路215件評価で
  station_departuresは30件中27件がresolve_route_requestに誤吸収された最重症
  クラスだった。generate_balanced_synthetic_dataset.py のテンプレートは
  STATIONS(21駅)×固定テンプレートの組み合わせ数が上限(departures最大105件、
  get_station最大84件、list_feeds最大96件程度)のため、単純に build_class() の
  per_class引数を増やしても目標件数まで届かない。

このスクリプトは generate_balanced_synthetic_dataset.py の
departures_case / station_detail_case / feeds_case と同一のテンプレート文言に
STYLE_SUFFIXES を掛け合わせてバリエーションを増やす(ジェネレータ本体は変更しない)。

使い方:
    python3 scripts/generate_r7_thin_class_replay.py \
        --output data/raw/thin_classes_r7.jsonl \
        --existing data/raw/intent_router_train_8000.jsonl \
        --existing data/raw/nonroute_replay_r7.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen.generate_synthetic_dataset import STATIONS
from datagen.generate_balanced_synthetic_dataset import STYLE_SUFFIXES, target, raw_row

DEFAULT_EVAL_FILES = [
    'data/eval/independent_holdout_300.jsonl',
    'data/eval/manual_practical_100.jsonl',
    'data/eval/intent_router_dev_950.jsonl',
    'data/eval/intent_router_stress_600.jsonl',
    'data/eval/operational_semantic_holdout_300.jsonl',
    'data/eval/operational_tokyo_routes_100.jsonl',
    'data/eval/operational_tokyo_routes.jsonl',
    'data/eval/operational_intent_raw_100.jsonl',
    'data/eval/eval_balanced.jsonl',
    'data/eval/eval_balanced_corrected.jsonl',
    'data/eval/eval_generated.jsonl',
    'data/eval/eval_real_user_japanese.jsonl',
    'data/eval/eval_template.jsonl',
]

DEPARTURES_TEMPLATES = [
    "{sid} の発車案内を10件",
    "駅ID {sid} の次の出発を表示",
    "{name}のID {sid} から出る便",
    "{sid} の明日9時以降の発車時刻",
    "発車標を確認したい: {sid}",
]

STATION_DETAIL_TEMPLATES = [
    "{sid} の駅詳細を見たい",
    "駅ID {sid} のホーム情報",
    "{name}の識別子 {sid} を確認",
    "{sid} に乗り入れる路線と駅情報",
]

FEEDS_TEMPLATES = [
    '対応してる鉄道会社を確認したい',
    'カバーしているエリアを知りたい',
    'このシステムが使ってるデータ元は？',
    '交通データの帰属表示を見せて',
    'フィードの更新頻度を確認したい',
    '対応路線一覧を出して',
    'データソースの信頼性を確認したい',
    '収録されている事業者数を教えて',
    'このAPIが参照しているデータを知りたい',
    '交通データの提供元一覧',
    'サポート対象のエリアを確認',
    '使用しているGTFSフィードを一覧化',
]
FEEDS_SUFFIXES = ['', 'お願いします', '検索してください', '確認したいです', '教えてください', '今すぐ知りたいです']


def load_prompts(files: list[str]) -> set[str]:
    prompts: set[str] = set()
    for path_str in files:
        path = Path(path_str)
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            u = obj.get('user') or obj.get('prompt')
            if u:
                prompts.add(u)
    return prompts


def build_departures(seen: set[str]) -> list[dict]:
    rows = []
    idx = 0
    for name, sid in STATIONS:
        for template in DEPARTURES_TEMPLATES:
            for suffix in STYLE_SUFFIXES:
                text = template.format(name=name, sid=sid) + suffix
                if text in seen:
                    continue
                seen.add(text)
                dated = "明日" in text
                args: dict = {"id": sid}
                explicit_limit = re.search(r"(\d+)件", text)
                if explicit_limit:
                    args["limit"] = int(explicit_limit.group(1))
                if dated:
                    args.update({"date": "20260629", "time": "9:00"})
                idx += 1
                rows.append(raw_row(f'r7-departures-{idx:05d}', target('station_departures', args), user=text))
    return rows


def build_station_detail(seen: set[str]) -> list[dict]:
    rows = []
    idx = 0
    for name, sid in STATIONS:
        for template in STATION_DETAIL_TEMPLATES:
            for suffix in STYLE_SUFFIXES:
                text = template.format(name=name, sid=sid) + suffix
                if text in seen:
                    continue
                seen.add(text)
                idx += 1
                rows.append(raw_row(f'r7-getstation-{idx:05d}', target('get_station', {'id': sid}), user=text))
    return rows


def build_feeds(seen: set[str]) -> list[dict]:
    rows = []
    idx = 0
    for template in FEEDS_TEMPLATES:
        for suffix in FEEDS_SUFFIXES:
            text = template + suffix
            if text in seen:
                continue
            seen.add(text)
            idx += 1
            rows.append(raw_row(f'r7-listfeeds-{idx:05d}', target('list_feeds', {}), user=text))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/raw/thin_classes_r7.jsonl'))
    parser.add_argument('--eval-file', action='append', default=[])
    parser.add_argument(
        '--existing', action='append', default=[],
        help='既に使用済みのuser文言を含むrawファイル(重複除去対象)。複数指定可。',
    )
    parser.add_argument('--departures-limit', type=int, default=200)
    parser.add_argument('--get-station-limit', type=int, default=220)
    args = parser.parse_args()

    eval_files = args.eval_file or DEFAULT_EVAL_FILES
    seen = load_prompts(eval_files)
    for path_str in args.existing:
        path = Path(path_str)
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            seen.add(json.loads(line)['user'])

    dep = build_departures(seen)[: args.departures_limit]
    gs = build_station_detail(seen)[: args.get_station_limit]
    lf = build_feeds(seen)

    print(f'station_departures extra: {len(dep)}')
    print(f'get_station extra: {len(gs)}')
    print(f'list_feeds extra: {len(lf)}')

    rows = dep + gs + lf
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'wrote {len(rows)} rows -> {args.output}')


if __name__ == '__main__':
    main()
