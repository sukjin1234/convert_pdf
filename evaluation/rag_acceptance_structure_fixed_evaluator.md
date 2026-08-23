# RAG acceptance report

- Generated: 2026-08-23T04:20:04.648674+00:00
- Dataset: `0fd367d8-5003-440a-92bc-e256784f8507`
- Overall gate: FAIL
- Failed gates: boundary_accuracy

## Metrics

| Metric | Result | Target |
| --- | ---: | ---: |
| Boundary accuracy | 93.06% | >= 95% |
| Source boundary recall | 87.18% | diagnostic |
| Page coverage | 98.36% | diagnostic |
| End-to-end p95 latency | 0.000s | <= 10.0s |

## Per-document structure

- `2.  2027학년도 입학전형 수시·정시 모집요강.pdf`: boundary 99.83%, anchors 283/284, pages 22/22, segments 610/610
- `document.pdf`: boundary 89.31%, anchors 411/512, pages 59/61, segments 797/797

## Latency bottlenecks


## Failed quality cases

- None

## Interpretation

Boundary accuracy is a weighted structural score: 50% source heading/table/list/formula anchor recall, 25% meaningful-page coverage, and 25% valid segment-boundary precision. Answer and citation accuracy are scored independently. Expected content stays in the manifest; the evaluator contains no domain-specific field logic.
