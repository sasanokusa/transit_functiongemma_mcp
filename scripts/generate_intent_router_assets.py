import json, random, re, hashlib
from pathlib import Path
from collections import Counter
from datetime import date, timedelta

OUT = Path('/mnt/data/transit_intent_learning_assets')
REF = '2026-07-01 11:00 Asia/Tokyo'
TODAY = '20260701'
TOMORROW = '20260702'
random.seed(7341)

STATIONS = [
    '東京','新宿','渋谷','池袋','上野','品川','秋葉原','横浜','大宮','町田','立川','吉祥寺','中野','荻窪','高円寺','赤羽','蒲田','川崎','武蔵小杉','恵比寿','目黒','五反田','大崎','新橋','有楽町','神田','御茶ノ水','飯田橋','四ツ谷','高田馬場','代々木','原宿','日暮里','北千住','押上','浅草','六本木','大手町','三田','日本橋','銀座','表参道','青山一丁目','永田町','溜池山王','月島','豊洲','門前仲町','錦糸町','両国','新木場','羽田空港','成田空港'
]
AMBIGUOUS = ['浅草','三田','日本橋','大手町','押上','銀座','豊洲']
PLACES = ['東京タワー','東京スカイツリー','東京都庁','六本木ヒルズ','日本武道館','浅草寺','東京ビッグサイト','羽田空港第3ターミナル','渋谷区神南1丁目','秋葉原UDX','東京ドーム','国立競技場','明治神宮','上野動物園','お台場海浜公園']
LINES = ['山手線','中央線','中央線快速','総武線','京浜東北線','埼京線','銀座線','丸ノ内線','日比谷線','東西線','千代田線','有楽町線','半蔵門線','南北線','副都心線','浅草線','三田線','新宿線','大江戸線']
JR_LINES = ['山手線','中央線','中央線快速','総武線','京浜東北線','埼京線','常磐線','横須賀線','東海道線','京葉線']
SUBWAY_LINES = ['銀座線','丸ノ内線','日比谷線','東西線','千代田線','有楽町線','半蔵門線','南北線','副都心線','浅草線','三田線','新宿線','大江戸線']

PRIORITY_PHRASES = [
    ('fast',['なるべく早く','一番早く','急ぎで','最速で','早いやつ','できるだけ早め','時間優先で','サクッと']),
    ('cheap',['安いやつで','一番安く','安いルートで','料金安め','運賃安いやつ','安さ優先','安く済ませたい']),
    ('few_transfers',['乗換少なめ','乗り換え少なく','乗換え少ないやつ','乗換回数を抑えたい','なるべく乗り換えたくない','直通優先で']),
    ('less_walk',['歩きたくない','徒歩少なめ','歩く距離短め','あまり歩かないやつ','駅構内もあまり歩かない','徒歩短め','歩き少なめ']),
]
ARRIVAL_TIMES = [
    ('16:00','16:00着'),('16:00','午後4時着'),('18:00','18時着で'),('17:30','17:30までに着きたい'),('09:00','朝9時に到着したい'),('09:00','9時着'),('09:00','9時までに'),('09:30','午前9時半着'),('20:15','20:15に着くように'),('13:00','午後1時までに'),
]
DEPARTURE_TIMES = [
    ('16:00','16:00発'),('18:00','18時出発で'),('08:30','朝8時半に出たい'),('19:15','19:15に出発'),('09:00','9時に出る'),('09:00','9時発'),('15:45','午後3時45分出発'),('07:20','7:20発で'),('21:00','夜9時に出たい'),
]
SPECIAL_TIME_PHRASES = [
    ('first_train', None, '始発で'),('first_train', None, '始発に'),('first_train', None, '始発の電車で'),('first_train', None, '朝イチで'),('first_train', TOMORROW, '明日の始発で'),
    ('last_train', None, '終電で'),('last_train', None, '終電に'),('last_train', None, '最終電車で'),('last_train', TODAY, '今日の終電で'),('last_train', TOMORROW, '明日の終電で'),('last_train', None, '終電に間に合うように'),
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

def call(tool, args):
    return {'tool_name':tool, 'arguments':args}

def no_call(missing):
    q = '出発地と目的地を教えてください。'
    if missing == ['origin']:
        q = '出発地を教えてください。'
    elif missing == ['destination']:
        q = '目的地を教えてください。'
    return call('ask_clarification', {'missing':missing, 'question':q})

def resolve_args(origin=None,dest=None,**kw):
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

def make_row(prefix, idx, user, assistant, category, meta=None):
    r = {'id': f'{prefix}-{idx:05d}', 'reference_datetime': REF, 'user': user, 'assistant': assistant, 'category': category}
    if meta: r['metadata'] = meta
    return r

seen_texts = set()
def add(rows, prefix, user, assistant, category, meta=None):
    # allow same user with different prefix? keep only exact duplicates per dataset generation
    key = (prefix, user)
    if key in seen_texts:
        return False
    seen_texts.add(key)
    rows.append(make_row(prefix, len(rows)+1, user, assistant, category, meta))
    return True

def suffix_phrase(parts):
    parts = [p for p in parts if p]
    if not parts: return ''
    return '、' + '、'.join(parts)

def gen_route_rows(prefix, n, category_weights=None, stress=False):
    rows=[]
    attempts=0
    cats = list(category_weights.keys()) if category_weights else ['basic','arrival','departure','special_time','avoid_station','via_station','avoid_line','mode_constraint','priority','graphical','combo']
    weights = list(category_weights.values()) if category_weights else None
    while len(rows)<n and attempts<n*50:
        attempts += 1
        cat = random.choices(cats, weights=weights, k=1)[0]
        origin, dest = random.sample(STATIONS, 2)
        parts=[]
        args = resolve_args(origin, dest)
        meta = {'intent_family':'route'}
        special_prefix=None
        if cat=='basic':
            pass
        elif cat=='arrival':
            t, ph = random.choice(ARRIVAL_TIMES)
            parts.append(ph)
            args['time_mode']='arrive_by'; args['time']=t
        elif cat=='departure':
            t, ph = random.choice(DEPARTURE_TIMES)
            parts.append(ph)
            args['time_mode']='departure_at'; args['time']=t
        elif cat=='special_time':
            tm, d, ph = random.choice(SPECIAL_TIME_PHRASES)
            # Put the phrase before the route itself half the time. Merely
            # inserting it at the start of suffix parts still leaves it after
            # origin/destination and does not test prefix understanding.
            if random.random()<0.5:
                special_prefix=ph
            else:
                parts.append(ph)
            args['time_mode']=tm; args['date']=d
        elif cat=='avoid_station':
            avoid = random.choice([s for s in STATIONS if s not in (origin,dest)])
            ph = random.choice([f'{avoid}を避けて', f'{avoid}通らないで', f'{avoid}経由は嫌', f'{avoid}は通らないルートで', f'{avoid}には寄らずに', f'{avoid}を通らないやつ'])
            parts.append(ph); args['avoid_station_texts']=[avoid]
        elif cat=='via_station':
            via = random.choice([s for s in STATIONS if s not in (origin,dest)])
            ph = random.choice([f'{via}経由で', f'{via}に寄って', f'{via}を通って', f'{via}乗り換えで', f'{via}を経由したい'])
            parts.append(ph); args['via_station_texts']=[via]
        elif cat=='avoid_line':
            line = random.choice(LINES)
            ph = random.choice([f'{line}なしで', f'{line}使わないで', f'{line}は嫌', f'{line}を避けて', f'{line}に乗らないルートで'])
            parts.append(ph); args['avoid_line_texts']=[line]
        elif cat=='mode_constraint':
            kind = random.choice(['subway','tokyo_metro','toei_subway','avoid_jr','avoid_bus','rail_only'])
            if kind=='subway':
                ph=random.choice(['地下鉄だけで','地下鉄のみで','メトロか都営だけで']); args['allowed_operator_groups']=['subway']
            elif kind=='tokyo_metro':
                ph=random.choice(['東京メトロだけで','メトロだけで']); args['allowed_operator_groups']=['tokyo_metro']
            elif kind=='toei_subway':
                ph=random.choice(['都営地下鉄だけで','都営線だけで']); args['allowed_operator_groups']=['toei_subway']
            elif kind=='avoid_jr':
                ph=random.choice(['JR使わないで','JRなしで','JRは嫌']); args['avoid_operator_groups']=['JR']
            elif kind=='avoid_bus':
                ph=random.choice(['バスなしで','バス使わないで']); args['avoid_modes']=['bus']
            else:
                ph=random.choice(['電車だけで','バスなしで電車だけ']); args['avoid_modes']=['bus']
            parts.append(ph)
        elif cat=='priority':
            pr, phs = random.choice(PRIORITY_PHRASES)
            ph = random.choice(phs)
            parts.append(ph); args['priority']=pr
        elif cat=='graphical':
            ph = random.choice(['地図で','マップで','経路図付きで','グラフィカルに','図で見たい'])
            parts.append(ph); args['graphical']=True
        elif cat=='combo':
            # 2-4 constraints mixed.
            chosen=[]
            if random.random()<0.65:
                pr, phs=random.choice(PRIORITY_PHRASES); args['priority']=pr; chosen.append(random.choice(phs))
            if random.random()<0.5:
                via=random.choice([s for s in STATIONS if s not in (origin,dest)]); args['via_station_texts']=[via]; chosen.append(random.choice([f'{via}経由で',f'{via}に寄って']))
            if random.random()<0.5:
                av=random.choice([s for s in STATIONS if s not in (origin,dest) and s not in args['via_station_texts']]); args['avoid_station_texts']=[av]; chosen.append(random.choice([f'{av}を避けて',f'{av}通らないで']))
            if random.random()<0.35:
                line=random.choice(LINES); args['avoid_line_texts']=[line]; chosen.append(random.choice([f'{line}なしで', f'{line}は嫌']))
            if random.random()<0.35:
                t,ph=random.choice(ARRIVAL_TIMES+DEPARTURE_TIMES)
                if ph in [x[1] for x in ARRIVAL_TIMES]: args['time_mode']='arrive_by'
                else: args['time_mode']='departure_at'
                args['time']=t; chosen.append(ph)
            if random.random()<0.25:
                args['graphical']=True; chosen.append(random.choice(['地図で','経路図付きで']))
            parts.extend(chosen)
        # stress: force ambiguous names and odd phrasing
        if stress and random.random()<0.35:
            origin = random.choice(AMBIGUOUS); args['origin_text']=origin
        if stress and random.random()<0.25:
            dest = random.choice(AMBIGUOUS); args['destination_text']=dest
        template = random.choice(ROUTE_TEMPLATES)
        suf = suffix_phrase(parts)
        user = template.format(origin=origin, dest=dest, suffix=suf, suffix_no_comma='、'.join(parts) if parts else '')
        if special_prefix:
            user = special_prefix + user
        # Clean double punctuation
        user = user.replace('、、','、').replace('？、','？').strip('、')
        add(rows, prefix, user, call('resolve_route_request', args), cat, meta)
    return rows

def gen_nonroute_rows(prefix):
    rows=[]
    # Station search
    station_pats=['{x}駅を検索して','{x}の駅候補を出して','乗車駅として{x}を探して','{x}って駅ある？','{x}駅のID候補']
    for x in STATIONS:
        for pat in random.sample(station_pats, 3):
            add(rows,prefix,pat.format(x=x),call('suggest_stations',{'q':x + ('駅' if '駅' in pat and not x.endswith('駅') else ''),'limit':5}),'suggest_stations')
    place_pats=['{x}を場所として探して','{x}の最寄り候補','目的地が{x}なんだけど候補出して','{x}周辺の乗車地点を検索','施設として{x}を検索']
    for x in PLACES:
        for pat in random.sample(place_pats, 3):
            add(rows,prefix,pat.format(x=x),call('suggest_places',{'q':x,'limit':10}),'suggest_places')
    # No call / clarifications
    no_cases=[
        ('東京まで行きたい',['origin']),('新宿から行きたい',['destination']),('終電を調べて',['origin','destination']),('始発って何時？',['origin','destination']),('安いやつで行きたい',['origin','destination']),('ここから帰りたい',['origin','destination']),('駅まで行きたい',['origin','destination']),('何時に出ればいい？',['origin','destination']),('地図で見せて',['origin','destination']),('徒歩少なめがいい',['origin','destination'])
    ]
    for text, miss in no_cases*8:
        add(rows,prefix,text, no_call(miss), 'ask_clarification', {'intent_family':'no_call'})
    # list feeds and departures use ids to keep eval deterministic.
    for text in ['対応している交通データを見せて','利用できる交通フィード一覧','ライセンスとデータ出典を確認','どの交通データを使ってる？']*8:
        add(rows,prefix,text,call('list_feeds',{}),'list_feeds')
    ids=['feed:tokyo:station:tokyo','feed:tokyo:station:shinjuku','feed:tokyo:station:shibuya','feed:tokyo:station:asakusa','feed:tokyo:station:ikebukuro']
    for sid in ids:
        add(rows,prefix,f'{sid} の駅詳細',call('get_station',{'id':sid}),'get_station')
        add(rows,prefix,f'{sid} の発車案内を15件',call('station_departures',{'id':sid,'limit':15}),'station_departures')
        add(rows,prefix,f'{sid} の明日9時からの発車案内',call('station_departures',{'id':sid,'date':TOMORROW,'time':'09:00','limit':20}),'station_departures')
    return rows

def split_rows():
    global seen_texts
    seen_texts=set()
    train=[]
    train.extend(gen_route_rows('intent-train', 7200, {
        'basic': 0.7, 'arrival':1.1, 'departure':1.0, 'special_time':1.0, 'avoid_station':1.0, 'via_station':1.0, 'avoid_line':0.9, 'mode_constraint':1.0, 'priority':1.2, 'graphical':0.7, 'combo':1.4
    }))
    train.extend(gen_nonroute_rows('intent-train-nr'))
    random.shuffle(train)
    # re-id
    for i,r in enumerate(train,1): r['id']=f'intent-train-{i:05d}'

    seen_texts=set()
    dev=gen_route_rows('intent-dev', 800, {'basic':1,'arrival':1,'departure':1,'special_time':1,'avoid_station':1,'via_station':1,'avoid_line':1,'mode_constraint':1,'priority':1,'graphical':0.7,'combo':1.2}, stress=True)
    dev.extend(random.sample(gen_nonroute_rows('intent-dev-nr'), 150))
    random.shuffle(dev)
    for i,r in enumerate(dev,1): r['id']=f'intent-dev-{i:05d}'

    seen_texts=set()
    stress=gen_route_rows('intent-stress', 500, {'arrival':1.3,'special_time':1.2,'via_station':1.3,'mode_constraint':1.2,'priority':1.2,'combo':2.0,'avoid_line':1.1,'avoid_station':1.1,'departure':1,'basic':0.4,'graphical':0.6}, stress=True)
    stress.extend(random.sample(gen_nonroute_rows('intent-stress-nr'), 100))
    random.shuffle(stress)
    for i,r in enumerate(stress,1): r['id']=f'intent-stress-{i:05d}'
    return train, dev, stress

LOCAL_TOOLS = [
  {
    'name':'resolve_route_request',
    'description':'日本語の乗換・経路検索要求を、MCP実行前のローカルintent slotに変換する。経路事実は生成しない。',
    'inputSchema':{
      'type':'object','additionalProperties':False,
      'properties':{
        'origin_text':{'type':['string','null']},
        'destination_text':{'type':['string','null']},
        'via_station_texts':{'type':'array','items':{'type':'string'}},
        'avoid_station_texts':{'type':'array','items':{'type':'string'}},
        'avoid_line_texts':{'type':'array','items':{'type':'string'}},
        'preferred_line_texts':{'type':'array','items':{'type':'string'}},
        'allowed_operator_groups':{'type':'array','items':{'type':'string','enum':['subway','tokyo_metro','toei_subway']}},
        'avoid_operator_groups':{'type':'array','items':{'type':'string','enum':['JR']}},
        'avoid_modes':{'type':'array','items':{'type':'string','enum':['bus']}},
        'priority':{'type':['string','null'],'enum':['fast','cheap','few_transfers','less_walk',None]},
        'time_mode':{'type':['string','null'],'enum':['departure_at','arrive_by','first_train','last_train',None]},
        'date':{'type':['string','null']},
        'time':{'type':['string','null']},
        'graphical':{'type':'boolean'},
      },
      'required':['origin_text','destination_text','via_station_texts','avoid_station_texts','avoid_line_texts','preferred_line_texts','allowed_operator_groups','avoid_operator_groups','avoid_modes','priority','time_mode','date','time','graphical']
    }
  },
  {
    'name':'ask_clarification',
    'description':'出発地・目的地などの必須情報が不足している場合に、MCPを呼ばず確認するローカル疑似tool。',
    'inputSchema':{
      'type':'object','additionalProperties':False,
      'properties':{
        'missing':{'type':'array','items':{'type':'string','enum':['origin','destination','time','station_choice']}},
        'question':{'type':'string'}
      },
      'required':['missing','question']
    }
  }
]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',',':'))+'\n')

def summarize(rows):
    c=Counter(r['assistant']['tool_name'] if not r['assistant'].get('no_tool_call') else 'NO_CALL' for r in rows)
    cats=Counter(r.get('category','') for r in rows)
    return {'count':len(rows),'tools':dict(c),'categories':dict(cats)}

def main():
    if OUT.exists():
        import shutil; shutil.rmtree(OUT)
    (OUT/'data/raw').mkdir(parents=True)
    (OUT/'data/eval').mkdir(parents=True)
    (OUT/'tools').mkdir(parents=True)
    (OUT/'scripts').mkdir(parents=True)
    train,dev,stress=split_rows()
    # Generate the operational holdout independently instead of slicing stress.
    # Extra candidates absorb accidental cross-split matches and impossible
    # origin/destination constraints.
    seen_texts.clear()
    op_candidates=gen_route_rows('intent-holdout', 450, stress=True)
    used_eval_prompts={r['user'] for r in dev+stress}
    def executable(row):
        a=row['assistant']['arguments']; o=a['origin_text']; d=a['destination_text']
        via=set(a['via_station_texts']); avoid=set(a['avoid_station_texts'])
        return o!=d and o not in via and d not in via and o not in avoid and d not in avoid and not via&avoid
    op=[r for r in op_candidates if r['user'] not in used_eval_prompts and executable(r)][:300]
    if len(op)!=300: raise RuntimeError(f'independent holdout too small: {len(op)}')
    for i,r in enumerate(op,1): r['id']=f'intent-holdout-{i:05d}'
    # Enforce the boundary the evaluator assumes. Generation resets its local
    # de-duplication set per split, so cross-split duplicates must be removed.
    eval_prompts={r['user'] for r in dev+stress+op}
    train=[r for r in train if r['user'] not in eval_prompts]
    write_jsonl(OUT/'data/raw/intent_router_train_8000.jsonl', train)
    write_jsonl(OUT/'data/eval/intent_router_dev_950.jsonl', dev)
    write_jsonl(OUT/'data/eval/intent_router_stress_600.jsonl', stress)
    write_jsonl(OUT/'data/eval/operational_semantic_holdout_300.jsonl', op)
    (OUT/'tools/local_tools_schema.json').write_text(json.dumps({'tools':LOCAL_TOOLS},ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'train':summarize(train),'dev':summarize(dev),'stress':summarize(stress),'operational_semantic_holdout_300':summarize(op)}
    (OUT/'dataset_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # generator copy
    Path('/mnt/data/create_intent_assets.py').replace(OUT/'scripts/generate_intent_router_assets.py')
    # validation script
    (OUT/'scripts/validate_intent_assets.py').write_text('''#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\nfrom collections import Counter\npaths=[Path(p) for p in sys.argv[1:]] or list(Path("data").rglob("*.jsonl"))\nfor p in paths:\n    rows=[]\n    for i,l in enumerate(p.read_text(encoding="utf-8").splitlines(),1):\n        r=json.loads(l); rows.append(r)\n        assert "id" in r and "user" in r and "assistant" in r, (p,i)\n        a=r["assistant"]; assert "tool_name" in a and "arguments" in a, (p,i,a)\n        if a["tool_name"]=="resolve_route_request":\n            req=["origin_text","destination_text","via_station_texts","avoid_station_texts","avoid_line_texts","preferred_line_texts","allowed_operator_groups","avoid_operator_groups","avoid_modes","priority","time_mode","date","time","graphical"]\n            for k in req: assert k in a["arguments"], (p,i,k)\n    print(p, len(rows), Counter(r["assistant"]["tool_name"] for r in rows))\n''',encoding='utf-8')
    # README
    (OUT/'README.md').write_text('''# Transit FunctionGemma Intent Learning Assets\n\nFunctionGemmaに「日本語の曖昧表現 → intent slot」を学習させるための追加アセットです。\n前段の決定的parserで意味理解を増やしすぎず、LLM側に priority / avoid / via / mode / time_mode を吸収させる目的で作成しています。\n\n## ファイル\n\n- `data/raw/intent_router_train_8000.jsonl`: 追加SFT用。約8000件。\n- `data/eval/intent_router_dev_950.jsonl`: 開発評価用。\n- `data/eval/intent_router_stress_600.jsonl`: 口語・曖昧駅・複合条件多めのstress評価。\n- `data/eval/operational_semantic_holdout_300.jsonl`: 実運用寄りsemantic holdout。\n- `tools/local_tools_schema.json`: `resolve_route_request` / `ask_clarification` のローカル疑似tool schema。\n- `scripts/generate_intent_router_assets.py`: 再生成スクリプト。\n- `scripts/validate_intent_assets.py`: 形式検証。\n\n## 重要な設計\n\n経路系の自然文は、直接 `plan_journey` へ飛ばさず、まずローカル疑似tool `resolve_route_request` に変換します。\nその後、決定的plannerが駅解決、via multi-leg、avoid/filter、priority rerank、MCP callを行います。\n\nこの分離により、FunctionGemmaは自然言語理解に集中し、経路事実はMCPと決定的処理に任せられます。\n\n## 推奨評価\n\n必ず3段階で分けてください。\n\n1. raw model score: FunctionGemma単体のslot抽出精度\n2. normalized score: 表記正規化だけを通したslot抽出精度\n3. final pipeline score: planner/reranker/MCP/renderer込みのE2E精度\n\n`operational_100` は既に開発に使ったため、独立holdoutとしては扱わずregression testに降格してください。\n''', encoding='utf-8')
    # Codex prompt
    (OUT/'PROMPT_TRAIN_INTENT_ROUTER.md').write_text('''# Codexプロンプト: FunctionGemmaに自然言語の曖昧さを吸収させる追加学習\n\n現在、前段の決定的parserを厚くしすぎると、FunctionGemmaを使う意味が薄くなっています。\nこの追加アセットを使い、自然言語の意味理解をFunctionGemma側へ戻してください。\n\n## 方針\n\n- 前段normalizerは表記正規化に限定する\n  - Unicode正規化\n  - 全角半角\n  - 午前/午後や9時半などの時刻表記正規化\n  - 空白・句読点の軽い整理\n- `priority` / `avoid` / `via` / `mode_constraint` / `time_mode` の意味判断はFunctionGemmaに担当させる\n- ルールで増やした意味理解parserは棚卸しし、原則として削除またはオプション化する\n- 経路系は直接MCP toolへ飛ばさず、ローカル疑似tool `resolve_route_request` を出す\n- `ask_clarification` はMCPへ送らないローカル疑似toolとする\n\n## 取り込みファイル\n\n- `data/raw/intent_router_train_8000.jsonl`\n- `data/eval/intent_router_dev_950.jsonl`\n- `data/eval/intent_router_stress_600.jsonl`\n- `data/eval/operational_semantic_holdout_300.jsonl`\n- `tools/local_tools_schema.json`\n\n## 実装要求\n\n1. `prepare_sft.py` が local tools を扱えるようにする\n   - MCP schema由来toolsに加えて `tools/local_tools_schema.json` を読み込めるようにする\n   - `resolve_route_request` と `ask_clarification` を学習時toolsとして許可する\n\n2. runtimeにlocal tool dispatcherを追加する\n   - `resolve_route_request` はMCPに送らない\n   - station resolver / planner / reranker / MCP clientへ渡す\n   - `ask_clarification` は確認質問として返し、MCPを呼ばない\n\n3. 評価を3段階に分ける\n   - raw model score\n   - normalized score\n   - final pipeline score\n\n4. 意味理解parserの棚卸し\n   - 既存ルールを `surface_normalization` / `schema_safety` / `semantic_intent` に分類する\n   - `semantic_intent` はFunctionGemmaへ移す\n   - parserでテスト100件に合わせ込まない\n\n5. 学習\n   - GTX1650 4GB想定\n   - LoRA rank 4 or 8\n   - max_seq_length 512\n   - batch_size 1\n   - gradient_accumulation 16-32\n   - fp16, bf16禁止\n\n## 評価目標\n\n`intent_router_dev_950`:\n- raw tool/intent accuracy >= 90%\n- slot exact/inclusion >= 85%\n- no-call/ask_clarification >= 95%\n\n`intent_router_stress_600`:\n- raw tool/intent accuracy >= 80%\n- priority/avoid/via/mode/time slot F1 >= 80%\n\n`operational_semantic_holdout_300`:\n- raw route intent accuracy >= 80%\n- final pipeline success >= 90% after deterministic planner\n\n## 禁止\n\n- テスト文面を見て前段parserに個別表現を追加し続けること\n- FunctionGemmaの失敗をすべてregexで直すこと\n- LLMに経路、料金、時刻、路線名を生成させること\n\n## 成果物\n\n- 追加学習済みadapter\n- raw/normalized/finalの3段階評価レポート\n- parser棚卸しレポート\n- 失敗例と追加学習候補のJSONL\n''', encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
