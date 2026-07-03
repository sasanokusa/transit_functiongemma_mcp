# Next release checklist

## Evaluation gates

- [x] `independent_holdout_300`: final parse >= 98% (100%)
- [x] `independent_holdout_300`: tool name >= 95% (100%)
- [x] `independent_holdout_300`: expected arguments inclusion >= 90% (100%)
- [x] `independent_holdout_300`: required arguments >= 95% (100%)
- [x] `independent_holdout_300`: no-call >= 95% (100%)
- [x] `independent_holdout_300`: schema valid = 100% (100%)
- [x] `manual_practical_100`: overall class/tool >= 90% (100%)
- [x] `manual_practical_100`: no-call >= 95% (100%)
- [x] `manual_practical_100`: intent slot macro F1 >= 85% (97.84%)
- [x] `manual_practical_100`: avoid/via extraction >= 85% (100%)
- [x] `manual_practical_100`: datetime normalization >= 90% (100%)
- [x] Fixed seven-scenario E2E success >= 90% (7/7, 100%)
- [x] E2E no-call cases perform zero MCP calls (100%)
- [x] Renderer source-only checks pass (100%)

Raw model parse accuracy and schema-constrained final accuracy are reported separately. A release
must not hide deterioration in the raw metric even when deterministic normalization repairs it.

## Runtime safety

- [x] Unknown tools and malformed calls are rejected before MCP execution
- [x] MCP input is validated against the saved `tools/list` JSON Schema
- [x] MCP timeout is finite and maps to a user-facing 504 response
- [x] Transient timeout/429/502/503/504 responses use bounded exponential retry
- [x] Station/place/schema responses use a short in-memory TTL cache
- [x] `fetch_tools.py --check` detects live schema changes before deployment
- [x] Per-client sliding-window rate limit is enabled
- [x] Every Web response has an `X-Request-Id`
- [x] Audit logs use hashed client/query identifiers by default
- [x] User query storage is off by default
- [x] Raw coordinate logging is off by default
- [ ] Configure a stable secret `TRANSIT_LOG_SALT` outside source control
- [ ] Put an edge/WAF rate limit in front of the application rate limit
- [ ] Add external uptime, p95 latency and error-rate alerts
- [ ] Define log retention and deletion periods

## Pre-deploy commands

```bash
python validate_eval_datasets.py
python fetch_tools.py --check
python -m unittest discover -s tests -v

python eval_toolcall.py \
  --dataset data/eval/independent_holdout_300.jsonl \
  --run-model --adapter outputs/functiongemma-transit-ja-real-r4 \
  --normalize-ja --bind-normalized-arguments --schema-constraint \
  --output artifacts/eval_independent_holdout_300.json \
  --markdown-output artifacts/eval_independent_holdout_300.md \
  --failures-output artifacts/failures_independent_holdout_300.jsonl

python eval_toolcall.py \
  --dataset data/eval/manual_practical_100.jsonl \
  --run-model --adapter outputs/functiongemma-transit-ja-real-r4 \
  --normalize-ja --bind-normalized-arguments --schema-constraint \
  --output artifacts/eval_manual_practical_100.json \
  --markdown-output artifacts/eval_manual_practical_100.md \
  --failures-output artifacts/failures_manual_practical_100.jsonl

python eval_pipeline.py \
  --adapter outputs/functiongemma-transit-ja-real-r4 \
  --max-routes 1
```

## Release decision

Do not assign the public-service grade solely from synthetic or repaired router metrics. Promote
only after the independent/manual gates, real-MCP E2E gates, privacy configuration, external
monitoring and retention policy are all checked.
