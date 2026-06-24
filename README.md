# PARA

**Path Accountable Reasoning for Agentic Rule-Learning**

PARA uses an LLM to propose a bounded portfolio of typed path programs, verifies
the resulting Horn-rule candidates symbolically, and answers held-out relation
queries with either:

- `SUPPORTED` and a finite proof tree grounded in project facts; or
- `INCONCLUSIVE` when no proof is found within the configured bounds.

The rule is an intermediate artifact. The proof trace is the reasoning output.

## Repository Contents

```text
src/para/                 Core planner, verifier, compiler, and proof engine
scripts/                  Paper experiment and audit entry points
data/example/             Small self-contained runnable task
results/                  Curated paper summaries and proof-audit samples
figures/                  Figure source data and generation script
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

- [Experiment evidence catalog](docs/EXPERIMENT_EVIDENCE_CATALOG.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data and artifact guide](docs/DATA.md)

Curated summaries in `results/` are small enough for Git. Full frozen task
splits and raw run artifacts are distributed separately because they are large
derived files.

## Review Anonymity

This is the official public repository. A double-anonymous submission should
not link directly to this named repository during review. See
[docs/PUBLICATION_AND_ANONYMITY.md](docs/PUBLICATION_AND_ANONYMITY.md).

## License

Code is released under the MIT License. Dataset components derived from
third-party projects retain their original licenses; see `docs/DATA.md`.
