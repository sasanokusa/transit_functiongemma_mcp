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

## iPhone demo gates (2026-07-15)

- [x] v1.0.0 Q6_K GGUF is bundled by SHA-256 and runs on the connected iPhone
- [x] Signed arm64 device build succeeds with the pinned llama.cpp XCFramework
- [x] Japanese route smoke returns two options and six map segments
- [x] Route summary is visible while the slower map request is still running
- [x] A failed map request preserves the already returned route summary
- [x] Unsupported Cyrillic input performs no model/MCP call and shows localized guidance
- [x] Raw model/tool output is hidden unless the debug environment flag is explicit
- [x] Speech input starts recording on device without an authorization/runtime error
- [x] Keyboard can be dismissed by Done, scrolling, background tap, search, or voice input
- [x] In-app menu exposes map settings, permissions, privacy, data sources, licenses, and version
- [x] Gemma required notice and llama.cpp MIT license are available in-app
- [x] Privacy manifest declares no tracking/collection and app-only UserDefaults reason `CA92.1`
- [x] Bundle the prepared opaque icon at 120×120 and 180×180 for the wired-device demo
- [ ] Before App Store distribution, install an iOS Simulator runtime and move the 1024×1024 icon into the AppIcon asset catalog
- [ ] Re-run a visual pass on the installed icon, progressive map placeholder, and final timeline

Measured Transit MCP latency is currently dominated by server-side planning: `plan_journey`
roughly 11–30 seconds and `plan_route_map` roughly 15–27 seconds in observed runs. Network setup is below 250 ms
from the development Mac and below 100 ms from GB10. The progressive UI mitigates this for the demo,
but server-side p95 work remains a public-release requirement.

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
python evaluation/validate_eval_datasets.py
python scripts/fetch_tools.py --check
python -m unittest discover -s tests -v

python evaluation/eval_toolcall.py \
  --dataset data/eval/independent_holdout_300.jsonl \
  --run-model --adapter outputs/functiongemma-transit-ja-real-r4 \
  --normalize-ja --bind-normalized-arguments --schema-constraint \
  --output artifacts/eval_independent_holdout_300.json \
  --markdown-output artifacts/eval_independent_holdout_300.md \
  --failures-output artifacts/failures_independent_holdout_300.jsonl

python evaluation/eval_toolcall.py \
  --dataset data/eval/manual_practical_100.jsonl \
  --run-model --adapter outputs/functiongemma-transit-ja-real-r4 \
  --normalize-ja --bind-normalized-arguments --schema-constraint \
  --output artifacts/eval_manual_practical_100.json \
  --markdown-output artifacts/eval_manual_practical_100.md \
  --failures-output artifacts/failures_manual_practical_100.jsonl

python evaluation/eval_pipeline.py \
  --adapter outputs/functiongemma-transit-ja-real-r4 \
  --clarification-tool \
  --max-routes 1
```

## Release decision

Do not assign the public-service grade solely from synthetic or repaired router metrics. Promote
only after the independent/manual gates, real-MCP E2E gates, privacy configuration, external
monitoring and retention policy are all checked.
