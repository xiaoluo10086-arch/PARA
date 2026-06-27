# PARA

**Path-Accountable Proof Trees for Software Architecture Relation Reasoning**

PARA turns software architecture relation queries into bounded, inspectable
proof obligations. A verifier-governed strategy agent proposes typed path
programs, the symbolic verifier admits only candidates supported by project
facts, and the proof engine answers relation queries with either:

- `SUPPORTED` and a finite proof tree grounded in project facts; or
- `INCONCLUSIVE` when no proof is found within the configured bounds.

The rule is an intermediate artifact. The proof tree and its decision contract
are the reasoning output.

## Repository Contents

```text
src/para/                 Core planner, verifier, compiler, and proof engine
scripts/                  Paper experiment and audit entry points
data/example/             Small self-contained runnable task
results/                  Curated paper summaries, proof contracts, and audits
figures/                  Figure source data, scripts, and final paper figures
docs/                     Reproduction, data, and repository-scope notes
tests/                    Public smoke tests
```

This repository intentionally excludes model weights, API credentials, raw
provider responses containing account metadata, third-party source checkouts,
build directories, and the approximately 90 GB of intermediate experiment
outputs. See [docs/DATA.md](docs/DATA.md) for the artifact boundary.

## Quick Start

```bash
git clone https://github.com/xiaoluo10086-arch/PARA.git
cd PARA
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

para inspect --task-dir data/example/can_call_class
para reason \
  --task-dir data/example/can_call_class \
  --rule-library data/example/can_call_class/rule_library.json \
  --query "canCallClass(order_service,payment_client)" \
  --json
```

The final command requires no LLM service. It executes the frozen example rule
and returns its fact-grounded proof.

Run all example queries:

```bash
para reason-eval \
  --task-dir data/example/can_call_class \
  --rule-library data/example/can_call_class/rule_library.json \
  --json
```

## LLM-Backed Planning

PARA supports local OpenAI-compatible servers and remote OpenAI, DeepSeek, or
Gemini endpoints. Pass the endpoint and model explicitly:

```bash
para learn \
  --task-dir PATH/TO/TASK \
  --output-dir runs/example \
  --guide-provider agent \
  --llm-base-url http://127.0.0.1:8000 \
  --llm-model YOUR_MODEL \
  --candidate-first \
  --strict-candidate-first
```

Credentials are read from environment variables such as `OPENAI_API_KEY`,
`DEEPSEEK_API_KEY`, or `GEMINI_API_KEY`. Never commit `.env` files or keys.

Popper is optional and is used only by the corresponding ILP baseline or hybrid
experiments. Set `PARA_POPPER_PATH=/path/to/popper.py` when needed.

## Reproducing the Paper

Start with:

- [Anonymous artifact README](docs/ANON_ARTIFACT_README.md)
- [Experiment evidence catalog](docs/EXPERIMENT_EVIDENCE_CATALOG.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data and artifact guide](docs/DATA.md)
- [Architecture description](docs/PARA_ARCHITECTURE_DESCRIPTION.md)
- [End-to-end case figure description](docs/PARA_END_TO_END_CASE_FIGURE_DESCRIPTION.md)

Curated summaries in `results/` are small enough for Git. The latest
paper-facing package is in `results/final_icse2027/`, including unified rerun
tables, high-complexity multi-seed summaries, and representative proof
contracts. Full frozen task splits and raw run artifacts are distributed
separately because they are large derived files.

## Review Anonymity

This is the official public repository. A double-anonymous submission should
not link directly to this named repository during review. See
[docs/PUBLICATION_AND_ANONYMITY.md](docs/PUBLICATION_AND_ANONYMITY.md).

## License

Code is released under the MIT License. Dataset components derived from
third-party projects retain their original licenses; see `docs/DATA.md`.
