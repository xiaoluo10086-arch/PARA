# PARA ICSE Revision Experiment Summary

Manifests: 18

| split_name | guide_provider | min_path_support | max_path_queries | learn_status | train_f1 | heldout_accuracy | heldout_precision | heldout_recall | inconclusive_rate | false_supported_negatives | end_to_end_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spring_canCallClass_train50_100_seed0_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 190.8968 |
| spring_canCallClass_train50_100_seed1_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 196.3500 |
| spring_canCallClass_train50_100_seed2_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 199.6441 |
| spring_isAllowedToUse_train50_100_seed0_graphrag_support2 | graphrag | 2 | - | ok | 0.9362 | 0.9400 | 1.0000 | 0.8200 | 0.0600 | 0 | 298.9216 |
| spring_isAllowedToUse_train50_100_seed1_graphrag_support2 | graphrag | 2 | - | ok | 0.9247 | 0.9422 | 1.0000 | 0.8267 | 0.0578 | 0 | 236.5002 |
| spring_isAllowedToUse_train50_100_seed2_graphrag_support2 | graphrag | 2 | - | ok | 0.9247 | 0.9422 | 1.0000 | 0.8267 | 0.0578 | 0 | 228.6853 |
| spring_overridesMethod_train50_100_seed0_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 213.9811 |
| spring_overridesMethod_train50_100_seed1_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 0.9978 | 0.9934 | 1.0000 | 0.0000 | 1 | 233.7277 |
| spring_overridesMethod_train50_100_seed2_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 0.9978 | 0.9934 | 1.0000 | 0.0000 | 1 | 221.3923 |
| teammates_canCallClass_mid_clean_seed0_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 1.8172 |
| teammates_canCallClass_small_clean_seed0_graphrag_support2 | graphrag | 2 | - | candidate_rejected | - | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0 | 0.5549 |
| teammates_canCallClass_xlarge_clean_seed0_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 1.7906 |
| teammates_isAllowedToUse_mid_clean_seed0_graphrag_support2 | graphrag | 2 | - | candidate_rejected | - | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0 | 0.5040 |
| teammates_isAllowedToUse_small_clean_seed0_graphrag_support2 | graphrag | 2 | - | candidate_rejected | - | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0 | 0.5106 |
| teammates_isAllowedToUse_xlarge_clean_seed0_graphrag_support2 | graphrag | 2 | - | candidate_rejected | - | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0 | 0.5121 |
| teammates_overridesMethod_mid_clean_seed0_graphrag_support2 | graphrag | 2 | - | candidate_rejected | - | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0 | 2.4536 |
| teammates_overridesMethod_small_clean_seed0_graphrag_support2 | graphrag | 2 | - | candidate_rejected | - | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0 | 0.4900 |
| teammates_overridesMethod_xlarge_clean_seed0_graphrag_support2 | graphrag | 2 | - | ok | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 5.5605 |

## Source Snapshot

```json
{
  "identity_rule": "sha256",
  "files": {
    "src/para/agentic_graph_rag.py": "e2ead9222aabda9b8d3da329c9b2aec9a2ea0032b10dc2cdcd1dbe2d13a4e897",
    "src/para/graph_rag.py": "4e79eec8a6d51e358654c339d42c3a95bb18abd78bcb8b66bf7d381d21f28bc3",
    "src/para/reasoner.py": "251fc7f97ec47500845a9a3d2f46d37f06b29fec90573de4660136cfcf64f4b1",
    "src/para/cli.py": "aaec9aaf7f011d39714821de3b3c7c76619742a2aaa84b281426470d60a71361",
    "scripts/run_learn_then_reason.py": "3850ad31b830ec79c48c2a762e0e0ab0651cfd95b1944934617034c3fbafcb84",
    "scripts/run_teammates_clean9_learn_then_reason.py": "832b640eed908647425122f9437b1e88d0501135143717f0cbee5b4033aa9a0b"
  }
}
```
