# Direct LLM Query Answering

The LLM receives each held-out query and a deterministic query-centered graph view. It does not generate or export a rule library.

**Comparison boundary.** Direct QA does not see held-out labels. PARA's evaluation harness uses explicit negative labels only to distinguish UNSUPPORTED from INCONCLUSIVE after proof search. Therefore supported precision/recall are the primary matched metrics; three-valued accuracy remains diagnostic.

| Model | Target | Queries | Supported precision | Supported recall | Unsupported recall | Inconclusive | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen27b | `canCallClass` | 450 | 1.000 | 0.720 | 0.000 | 0.756 | 0 |
| qwen27b | `isAllowedToUse` | 450 | 0.924 | 0.807 | 0.000 | 0.704 | 0 |
| qwen27b | `overridesMethod` | 450 | 1.000 | 0.007 | 0.180 | 0.878 | 0 |
| deepseek_v4_pro | `canCallClass` | 450 | 1.000 | 0.820 | 0.000 | 0.727 | 0 |
| deepseek_v4_pro | `isAllowedToUse` | 450 | 0.816 | 0.740 | 0.000 | 0.698 | 0 |
| deepseek_v4_pro | `overridesMethod` | 450 | 0.500 | 0.007 | 0.260 | 0.820 | 0 |
