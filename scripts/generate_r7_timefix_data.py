#!/usr/bin/env python3
"""r7: 時刻表現の失敗例(午後X時/朝X時/X時半 の言い換え)を経路データへ追加する。

r6 review (artifacts/CLAUDE_R6_REVIEW.md) の指摘:
  - epoch2は「午後2時に出たい」を 22:00 + arrive_by と誤読した。
  - special_time は suffix のみで prefix 未検証(これは
    scripts/generate_intent_router_assets.py 側で既に修正済み)。
  - この不足は ARRIVAL_TIMES / DEPARTURE_TIMES に「午後X時に出たい」パターンの
    シンプル形が登録されていないために生じている。

scripts/generate_intent_router_assets.py の ROUTE_TEMPLATES / STATIONS /
resolve_args / suffix_phrase の各パターンに厳密に従い、決定的に生成する。

使い方:
    python3 scripts/generate_r7_timefix_data.py \
        --eval-prompts-out /tmp/eval_prompts.pkl \
        --output data/raw/timefix_r7.jsonl

--eval-prompts-out を省略した場合は data/eval/ 配下の主要な評価ファイルから
毎回 user 文言集合を再構築して重複除去に使う(train:eval の完全重複ゼロを保証)。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

STATIONS = [
    '東京', '新宿', '渋谷', '池袋', '上野', '品川', '秋葉原', '横浜', '大宮', '町田',
    '立川', '吉祥寺', '中野', '荻窪', '高円寺', '赤羽', '蒲田', '川崎', '武蔵小杉', '恵比寿',
    '目黒', '五反田', '大崎', '新橋', '有楽町', '神田', '御茶ノ水', '飯田橋', '四ツ谷', '高田馬場',
    '代々木', '原宿', '日暮里', '北千住', '押上', '浅草', '六本木', '大手町', '三田', '日本橋',
    '銀座', '表参道', '青山一丁目', '永田町', '溜池山王', '月島', '豊洲', '門前仲町', '錦糸町', '両国',
    '新木場', '羽田空港', '成田空港',
]

ROUTE_TEMPLATES = [
    '{origin}から{dest}まで{suffix}',
    '{origin}から{dest}まで行きたい{suffix}',
    '{origin}発で{dest}まで{suffix}',
    '{origin}駅から{dest}駅へ{suffix}',
    '{origin}→{dest}{suffix}',
    '{origin}から{dest}、{suffix_no_comma}',
    '{origin}から{dest}までお願い{suffix}',
    '{origin}から{dest}までどう行く？{suffix}',
    '{origin}出発で{dest}着{suffix}',
]

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


def resolve_args(origin=None, dest=None, **kw):
    args = {
        'origin_text': origin,
        'destination_text': dest,
        'via_station_texts': [],
        'avoid_station_texts': [],
        'avoid_line_texts': [],
        'preferred_line_texts': [],
        'allowed_operator_groups': [],
        'avoid_operator_groups': [],
        'avoid_modes': [],
        'priority': None,
        'time_mode': None,
        'date': None,
        'time': None,
        'graphical': False,
    }
    args.update(kw)
    return args


def suffix_phrase(parts):
    parts = [p for p in parts if p]
    if not parts:
        return ''
    return '、' + '、'.join(parts)


def call(tool, args):
    return {'tool_name': tool, 'arguments': args}


def make_row(prefix, idx, user, assistant, category, meta=None):
    r = {
        'id': f'{prefix}-{idx:05d}',
        'reference_datetime': '2026-07-01 11:00 Asia/Tokyo',
        'user': user,
        'assistant': assistant,
        'category': category,
    }
    if meta:
        r['metadata'] = meta
    return r


def pm(hour: int) -> int:
    """午後X時 -> 24時間表記。午後12時=12:00。"""
    return 12 if hour == 12 else 12 + hour


def am(hour: int) -> int:
    """午前X時 -> 24時間表記。午前12時(未使用)=0:00。"""
    return 0 if hour == 12 else hour


def build_time_phrase_pool() -> list[tuple[str, str, str]]:
    phrases: list[tuple[str, str, str]] = []

    # 午後X時 + departure ("に出たい" / "に出発" / "発で")。
    # r6実測失敗ケース「午後2時に出たい -> 22:00+arrive_by 誤読」を最優先で埋める。
    for h in range(1, 12):
        hhmm = f"{pm(h):02d}:00"
        phrases.append(('departure_at', hhmm, f'午後{h}時に出たい'))
        phrases.append(('departure_at', hhmm, f'午後{h}時に出発'))
        phrases.append(('departure_at', hhmm, f'午後{h}時発で'))
        phrases.append(('departure_at', hhmm, f'午後{h}時出発で'))

    # 午後X時 + arrival ("に着きたい" / "までに着きたい" / "着で")
    for h in range(1, 12):
        hhmm = f"{pm(h):02d}:00"
        phrases.append(('arrive_by', hhmm, f'午後{h}時に着きたい'))
        phrases.append(('arrive_by', hhmm, f'午後{h}時までに着きたい'))
        phrases.append(('arrive_by', hhmm, f'午後{h}時着で'))
        phrases.append(('arrive_by', hhmm, f'午後{h}時までに'))

    # 午前X時 + departure/arrival
    for h in range(1, 12):
        hhmm = f"{am(h):02d}:00"
        phrases.append(('departure_at', hhmm, f'午前{h}時に出たい'))
        phrases.append(('departure_at', hhmm, f'午前{h}時発で'))
        phrases.append(('arrive_by', hhmm, f'午前{h}時に着きたい'))
        phrases.append(('arrive_by', hhmm, f'午前{h}時までに'))

    # 朝X時 + departure/arrival(朝は午前の口語形、6-9時台中心)
    for h in (6, 7, 8, 9):
        hhmm = f"{h:02d}:00"
        phrases.append(('departure_at', hhmm, f'朝{h}時に出たい'))
        phrases.append(('departure_at', hhmm, f'朝{h}時発で'))
        phrases.append(('arrive_by', hhmm, f'朝{h}時に着きたい'))
        phrases.append(('arrive_by', hhmm, f'朝{h}時までに'))

    # 夜X時 + departure(夜は午後7時以降の口語形)
    for h in (7, 8, 9, 10, 11):
        hhmm = f"{pm(h):02d}:00"
        phrases.append(('departure_at', hhmm, f'夜{h}時に出たい'))
        phrases.append(('departure_at', hhmm, f'夜{h}時発で'))

    # X時半(30分)の言い換え。午後/午前/朝いずれも。
    for h in range(1, 12):
        hhmm = f"{pm(h):02d}:30"
        phrases.append(('departure_at', hhmm, f'午後{h}時半に出たい'))
        phrases.append(('arrive_by', hhmm, f'午後{h}時半に着きたい'))
    for h in range(1, 12):
        hhmm = f"{am(h):02d}:30"
        phrases.append(('departure_at', hhmm, f'午前{h}時半に出たい'))
    for h in (6, 7, 8, 9):
        hhmm = f"{h:02d}:30"
        phrases.append(('departure_at', hhmm, f'朝{h}時半に出たい'))
        phrases.append(('arrive_by', hhmm, f'朝{h}時半までに'))

    return phrases


def build(eval_prompts: set[str]) -> list[dict]:
    rows = []
    rng = random.Random(20260702)
    seen_texts: set[str] = set()
    idx = 0
    pool = build_time_phrase_pool()
    rng.shuffle(pool)
    # r6 review item 6: special-time/時刻表現は prefix 配置も検証する必要がある。
    # ここでは奇数/偶数インデックスで prefix と suffix を半々にする。
    for attempt, (time_mode, hhmm, phrase) in enumerate(pool, start=1):
        origin, dest = rng.sample(STATIONS, 2)
        args = resolve_args(origin, dest, time_mode=time_mode, time=hhmm)
        template = rng.choice(ROUTE_TEMPLATES)
        use_prefix = attempt % 2 == 0
        if use_prefix:
            body = template.format(origin=origin, dest=dest, suffix='', suffix_no_comma='')
            user = phrase + body
        else:
            suf = suffix_phrase([phrase])
            user = template.format(origin=origin, dest=dest, suffix=suf, suffix_no_comma=phrase)
        user = user.replace('、、', '、').replace('？、', '？').strip('、')
        if user in eval_prompts or user in seen_texts:
            continue
        seen_texts.add(user)
        idx += 1
        rows.append(
            make_row(
                'intent-train-timefix', idx, user,
                call('resolve_route_request', args),
                'departure' if time_mode == 'departure_at' else 'arrival',
                {'intent_family': 'route', 'r7_time_fix': True},
            )
        )
    return rows


def load_eval_prompts(eval_files: list[str]) -> set[str]:
    prompts: set[str] = set()
    for path_str in eval_files:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/raw/timefix_r7.jsonl'))
    parser.add_argument(
        '--eval-file', action='append', default=[],
        help='重複除去対象の評価ファイル(複数指定可)。省略時はdata/eval配下の既定リストを使う。',
    )
    args = parser.parse_args()
    eval_files = args.eval_file or DEFAULT_EVAL_FILES
    eval_prompts = load_eval_prompts(eval_files)
    rows = build(eval_prompts)
    print(f'generated time-phrase pool rows: {len(rows)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'wrote {len(rows)} rows -> {args.output}')


if __name__ == '__main__':
    main()
