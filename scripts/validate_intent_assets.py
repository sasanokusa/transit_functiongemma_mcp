#!/usr/bin/env python3
import json, sys
from pathlib import Path
from collections import Counter
paths=[Path(p) for p in sys.argv[1:]] or list(Path("data").rglob("*.jsonl"))
for p in paths:
    rows=[]
    for i,l in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
        r=json.loads(l); rows.append(r)
        assert "id" in r and "user" in r and "assistant" in r, (p,i)
        a=r["assistant"]; assert "tool_name" in a and "arguments" in a, (p,i,a)
        if a["tool_name"]=="resolve_route_request":
            req=["origin_text","destination_text","via_station_texts","avoid_station_texts","avoid_line_texts","preferred_line_texts","allowed_operator_groups","avoid_operator_groups","avoid_modes","priority","time_mode","date","time","graphical"]
            for k in req: assert k in a["arguments"], (p,i,k)
    print(p, len(rows), Counter(r["assistant"]["tool_name"] for r in rows))
