# v1.0.0 / rc11 release evaluation

## Scope and provenance

GB10 evaluated the exact iPhone v1.0.0 Q6_K and rc11 epoch-3 fp32/Q6/Q8 with the same
datasets and final-bound evaluator. Prediction generation was checked at commit `6e42565`;
the persisted quantized predictions were then rescored with the final 2026-07-15 time normalizer.
Evaluator and dataset hashes were independently compared between the development Mac and GB10
before using the results. The raw reports remain on GB10 under
`~/transit_work/eval_deterministic_v11/artifacts/rc11_release_eval/`.

New runs record full-content provenance for the runner, the complete repository Python source
tree used by evaluation, datasets, schemas, model inputs, converter, server, and quantizer.
`RESUME=1` fails closed if any recorded input changes.

Model hashes:

| Variant | SHA-256 |
| --- | --- |
| v1.0.0 Q6_K (iPhone bundle) | `89eeb9d467995a32e9935b26f8543fd7c758bce32a0f5b03391873e05d4aaabb` |
| rc11 epoch-3 adapter | `8848161e91ac4b7c3ff3acd38b72871b58ee77b943d365f5617263a3fb26abd6` |
| rc11 epoch-3 f16 GGUF | `62412bf2f79adcc99eb1c21d9814346ebcec140691c4f2d341c1aa96fb02bc77` |
| rc11 epoch-3 Q6_K | `481700bf410fca8a255fd06457fbeb185e6309389e8c82f39855a1b9f1d7a681` |
| rc11 epoch-3 Q8_0 | `9052aa1b2b8eaceacde5c449981e62467a557490045b0b652b22e8cf9cf9b55c` |

## Like-for-like results

Values are `semantic / tool / expected arguments / no-call`; `—` means the dataset has no
no-call subset.

| Variant | mixed dev | independent 300 | manual 100 | route 300 |
| --- | ---: | ---: | ---: | ---: |
| v1.0.0 Q6_K | 87.42 / 93.85 / 86.15 / 95.24 | 73.33 / 91.85 / 71.85 / 86.67 | 76.00 / 96.47 / 72.94 / 93.33 | 90.00 / 99.33 / 90.00 / — |
| rc11 fp32 | 86.09 / 89.23 / 83.85 / 100.00 | 70.00 / 83.70 / 67.78 / 90.00 | 87.00 / 98.82 / 85.88 / 93.33 | 88.33* / 99.33 / 88.33* / — |
| rc11 Q6_K | 84.11 / 90.00 / 81.54 / 100.00 | 69.33 / 84.44 / 67.04 / 90.00 | 85.00 / 97.65 / 83.53 / 93.33 | 90.33 / 99.67 / 90.33 / — |
| rc11 Q8_0 | 86.75 / 91.54 / 84.62 / 100.00 | 69.00 / 83.33 / 66.30 / 93.33 | 85.00 / 97.65 / 83.53 / 93.33 | 90.33 / 99.67 / 90.33 / — |

These are final-bound results after normalization, schema constraint, and argument binding. They
must not be presented as raw model accuracy. Quantized prediction files were rescored after the
`夜9時` PM-cue fix; `*` marks the fp32 route value that remains from the pre-fix run because that
run did not persist predictions. The route set also mixes intent-label expectations with
runtime-bound defaults for some date rows. The model slot score and effective MCP-argument score
need separate reporting in a future evaluator revision.

## Decision

rc11 improves the manual set, but all three rc11 forms regress the broader independent set and do
not provide a consistent promotion signal. v1.0.0 Q6_K also has the strongest mixed/independent
semantic result among the quantized candidates and has already passed the connected-iPhone route,
map-failure, unsupported-input, and speech smoke tests. It remains the controlled wired-demo
weight.

This is not an unrestricted public-release claim. The app limits unsupported input before model
or MCP execution, grounds route arguments against the user text, obtains transit facts only from
MCP, and labels the build as a prototype. Public distribution requires a new gate after evaluator
separation, broader device E2E, monitoring/retention work, and App Store packaging.
