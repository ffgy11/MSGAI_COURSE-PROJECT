# Reveal Commitment Consistency (RCC)

Code release for **Reveal Commitment Consistency for Hallucination Detection in Diffusion Language Models**.

This repository contains **code only**. Experiment results, figures, cached trajectories, reports, datasets, and model checkpoints are intentionally excluded.

## Repository structure

```text
RCC_Code_Only/
├── src/rcc_dllm/              # reusable generation / scoring utilities
├── scripts/generate/          # LLaDA / Dream trajectory generation
├── scripts/final/             # RCC detection and selective mitigation
├── configs/                   # example configs
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You also need local access to the corresponding LLaDA / Dream checkpoints.

## Main workflow

### 1. Generate trajectories

```bash
python scripts/generate/generate_llada_trajectories.py --help
python scripts/generate/generate_dream_trajectories.py --help
```

### 2. Prepare labels

CommonsenseQA ground-truth labels:

```bash
python scripts/final/label_csqa_manifest.py --help
```

For TriviaQA / HotpotQA, merge externally prepared hallucination labels:

```bash
python scripts/final/merge_hallucination_labels.py --help
```

### 3. Run RCC detection

```bash
python scripts/final/test_pcs_selection_compact.py --help
python scripts/final/test_rcc_all_configs.py --help
```

### 4. Run selective mitigation

```bash
python scripts/final/analyze_rcc_selective_answering.py --help
python scripts/final/analyze_csqa64_rcc_selective.py --help
```

## Notes

- RCC does not fine-tune the base diffusion language model.
- RCC uses fixed trajectory features with a lightweight supervised classifier.
- Selective mitigation abstains / defers on high-risk examples.
- Model checkpoints, datasets, cached trajectories, results, figures, and reports are not included.
