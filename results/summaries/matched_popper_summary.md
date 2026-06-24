# Matched Pure Popper Baseline

This experiment runs pure Popper on frozen PARA train splits and evaluates
any accepted rule with the same held-out bounded reasoner used by PARA.

## Aggregate Summary

- Cases: 3
- Accepted train rules: 1
- Completed reason-eval: 3

| Target | Cases | Accepted train rules | Held-out acc. | Supported recall | Negative non-support |
|---|---:|---:|---:|---:|---:|
| canCallClass | 1 | 1 | 1.000 | 1.000 | 1.000 |
| isAllowedToUse | 1 | 0 | 0.667 | 0.000 | 1.000 |
| overridesMethod | 1 | 0 | 0.667 | 0.000 | 1.000 |

## Case Details

| Project | Target | Seed | Train status | Train F1 | Held-out acc. | Held-out recall | Popper time/error |
|---|---|---:|---|---:|---:|---:|---|
| spring | canCallClass | 0 | ok | 1.000 | 1.000 | 1.000 | 26.481s |
| spring | isAllowedToUse | 0 | failed | -- | 0.667 | 0.000 | Popper timed out after 300s |
| spring | overridesMethod | 0 | failed | -- | 0.667 | 0.000 | Popper timed out after 300s |
