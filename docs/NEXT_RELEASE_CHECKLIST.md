# Next release checklist

## Bundled model evaluation

The iPhone bundle contains **Tentetsu v1.0.0 Q6_K**, not the historical r4 adapter. GB10 reran the
pinned GGUF (`89eeb9d…aaabb`) through the current final-bound evaluator on 2026-07-11. These values
include deterministic normalization, schema constraint, and argument binding; they are not pure
model accuracy.

| Dataset | Semantic | Tool | Expected args | Parse | No-call | Schema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed dev | 87.42% | 93.85% | 86.15% | 100.00% | 95.24% | 100.00% |
| independent 300 | 73.33% | 91.85% | 71.85% | 96.30% | 86.67% | 96.99% |
| manual 100 | 76.00% | 96.47% | 72.94% | 98.82% | 93.33% | 99.00% |
| route 300 | 90.00% | 99.33% | 90.00% | 99.33% | n/a | 99.33% |

- [x] Evaluate the exact bundled Q6_K hash on all four current datasets
- [x] Compare rc11 fp32/Q6/Q8 against the same evaluator and datasets
- [x] Keep v1.0.0 Q6_K for the controlled iPhone demo: it retains the strongest quantized
  independent-holdout result and already passed the device route/failure/language smoke tests
- [x] Scope this decision to a controlled wired demo; unrestricted public release remains unapproved

The former 100% independent/manual and 7/7 E2E figures are 2026-06-29 r4 pipeline results. They
remain useful historical regression evidence, but they do not certify the bundled model. See
`docs/RC11_RELEASE_EVALUATION.md` for the like-for-like comparison and evaluator caveats.

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
- [x] Release configuration clean build, static analysis, Store bundle validation, and strict code-sign verification pass
- [x] Signed Release bundle contains v1.0.0 Q6_K with the pinned SHA-256 (`89eeb9d…aaabb`)
- [x] User visual pass covers the installed icon, menu screens, progressive map placeholder, and final timeline
- [x] About screen uses the same bundled Tentetsu icon as the installed app
- [ ] Before App Store distribution, install an iOS Simulator runtime and move the 1024×1024 icon into the AppIcon asset catalog

The unchecked App Store item does not block the current signed, wired-device demo. It is a
distribution requirement for a later public release.

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
python -m pytest -q

# On GB10: evaluates rc11 fp32/Q6/Q8 and the exact v1.0.0 Q6_K together.
# Use a new OUT directory for every non-resume release run.
OUT="$HOME/transit_work/eval_runs/$(date +%Y%m%dT%H%M%S)" \
  bash scripts/run_rc11_release_eval.sh
```

## Release decision

Do not assign the public-service grade solely from synthetic or repaired router metrics. Promote
only after the independent/manual gates, real-MCP E2E gates, privacy configuration, external
monitoring and retention policy are all checked.
