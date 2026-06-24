# PARA Runtime and Trace Statistics

## Runtime Rows

| target | seed | method | max_path_queries | heldout_accuracy | heldout_recall | retrieval_seconds | retriever_total_seconds | proof_search_total_seconds | end_to_end_seconds |
|---|---|---|---|---|---|---|---|---|---|
| canCallClass | 0 | graphrag | - | 1.0000 | 1.0000 | 5.0243 | 6.0042 | 0.1840 | 190.8968 |
| canCallClass | 1 | graphrag | - | 1.0000 | 1.0000 | 9.5348 | 10.5127 | 0.1931 | 196.3500 |
| canCallClass | 2 | graphrag | - | 1.0000 | 1.0000 | 5.2289 | 6.4638 | 0.1790 | 199.6441 |
| isAllowedToUse | 0 | graphrag | - | 0.9400 | 0.8200 | 111.8323 | 112.8387 | 0.9128 | 298.9216 |
| isAllowedToUse | 1 | graphrag | - | 0.9422 | 0.8267 | 43.8209 | 44.9824 | 0.8847 | 236.5002 |
| isAllowedToUse | 2 | graphrag | - | 0.9422 | 0.8267 | 40.6067 | 41.5906 | 0.9741 | 228.6853 |
| overridesMethod | 0 | graphrag | - | 1.0000 | 1.0000 | 12.7710 | 13.8212 | 0.0473 | 213.9811 |
| overridesMethod | 1 | graphrag | - | 0.9978 | 1.0000 | 26.8978 | 27.9342 | 0.0415 | 233.7277 |
| overridesMethod | 2 | graphrag | - | 0.9978 | 1.0000 | 19.7799 | 20.8410 | 0.0410 | 221.3923 |
| canCallClass | 0 | strict-agentic | 5 | 1.0000 | 1.0000 | 0.0024 | 1.2312 | 0.1814 | 149.8674 |
| isAllowedToUse | 0 | strict-agentic | 5 | 0.9400 | 0.8200 | 0.0074 | 1.2183 | 0.9361 | 173.6337 |
| overridesMethod | 0 | strict-agentic | 5 | 1.0000 | 1.0000 | 0.0024 | 1.1409 | 0.0477 | 150.8703 |

## Trace Aggregate

```json
{
  "overall": {
    "final_accepted": 3,
    "rows": 3,
    "refiner_triggered": 1,
    "accepted_but_conservative": 1,
    "no_improvement_after_refiner": 1
  },
  "by_target": {
    "canCallClass": {
      "final_accepted": 1,
      "rows": 1
    },
    "isAllowedToUse": {
      "refiner_triggered": 1,
      "accepted_but_conservative": 1,
      "final_accepted": 1,
      "no_improvement_after_refiner": 1,
      "rows": 1
    },
    "overridesMethod": {
      "final_accepted": 1,
      "rows": 1
    }
  }
}
```

## Trace Rows

| target | seed | iterations | initial_f1 | best_f1 | initial_rejected | refiner_triggered | accepted_but_conservative | candidate_improved | final_accepted |
|---|---|---|---|---|---|---|---|---|---|
| canCallClass | 0 | 1 | 1.0000 | 1.0000 | False | False | False | False | True |
| isAllowedToUse | 0 | 2 | 0.9362 | 0.9362 | False | True | True | False | True |
| overridesMethod | 0 | 1 | 1.0000 | 1.0000 | False | False | False | False | True |