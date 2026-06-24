# Planner Evidence Ablation (Spring xlarge3, seed 0)

All arms share the frozen split, indexed plan-only executor, candidate compiler, symbolic verifier (train F1 >= 0.8), and held-out proof engine.

| Arm | Target | Train F1 | Held-out acc. | Supported precision | Supported recall | Positive abstention |
|---|---|---:|---:|---:|---:|---:|
| witness_only | canCallClass | 0.810 | 0.896 | 1.000 | 0.687 | 0.313 |
| witness_only | isAllowedToUse | 0.936 | 0.940 | 1.000 | 0.820 | 0.180 |
| witness_only | overridesMethod | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| schema_only | canCallClass | 0.810 | 0.896 | 1.000 | 0.687 | 0.313 |
| schema_only | isAllowedToUse | 0.936 | 0.940 | 1.000 | 0.820 | 0.180 |
| schema_only | overridesMethod | 0.795 | 0.667 | 0.000 | 0.000 | 1.000 |
| full_para | canCallClass | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| full_para | isAllowedToUse | 0.936 | 0.940 | 1.000 | 0.820 | 0.180 |
| full_para | overridesMethod | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Accepted rules

- `witness_only/canCallClass`: `canCallClass(A,B) :- importsClass(A,B) [0.814].`
- `witness_only/isAllowedToUse`: `isAllowedToUse(A,B) :- containsClass(A,V1),importsClass(V1,V2),containsClass(B,V2) [0.924].`
- `witness_only/overridesMethod`: `overridesMethod(A,B) :- containsMethod(V1,A),inheritsClass(V1,V2),containsMethod(V2,B),methodArity(A,C1),methodArity(B,C1),methodName(A,C2),methodName(B,C2) [0.980].`
- `schema_only/canCallClass`: `canCallClass(A,B) :- importsClass(A,B) [0.814].`
- `schema_only/isAllowedToUse`: `isAllowedToUse(A,B) :- containsClass(A,V1),importsClass(V1,V2),containsClass(B,V2) [0.924].`
- `schema_only/overridesMethod`: `None`
- `full_para/canCallClass`: `canCallClass(A,B) :- containsMethod(A,V1),callsMethod(V1,V2),containsMethod(B,V2) [0.980].`
- `full_para/isAllowedToUse`: `isAllowedToUse(A,B) :- containsClass(A,V1),importsClass(V1,V2),containsClass(B,V2) [0.924].`
- `full_para/overridesMethod`: `overridesMethod(A,B) :- containsMethod(V1,A),inheritsClass(V1,V2),containsMethod(V2,B),methodName(A,C1),methodName(B,C1),methodArity(A,C2),methodArity(B,C2) [0.980].`
