# DeepD S288C chromosome III synteny demo

This directory is a lightweight notebook feature intended to live inside a larger repository. Runtime code is consolidated in `synteny_demo.py`; no sibling pipeline checkout is required.

## Runtime assets

Genome FASTA files, GFF3 annotations, BLASTP results, and DAGchainer outputs under `data/` are committed so the notebook runs from a clone. If `data/` is absent, `synteny_demo.ensure_demo_assets()` downloads the matching subtree from the Hugging Face dataset [`biomap-research/DeepD`](https://huggingface.co/datasets/biomap-research/DeepD), under the `synteny/` prefix. Existing local files are never downloaded or overwritten.

```text
biomap-research/DeepD
└── synteny/
    └── data/
```

Override the default only if you host a fork:

```bash
export DEEPD_DATA_REPO_ID=biomap-research/DeepD
export DEEPD_DATA_REVISION=main
export DEEPD_SYNTENY_DATA_PREFIX=synteny
```

## Run

```bash
python synteny_demo.py

# Or run the notebook interactively
jupyter lab synteny_workflow.ipynb
```

The demo audits the supplied production parameters and all 163 DAGchainer anchor rows, then writes the formal PDF and summary under `results/formal/`.

## Tracked layout

```text
synteny/
├── README.md
├── synteny_demo.py
├── synteny_workflow.ipynb
└── data/                  # FASTA, GFF3, BLASTP, DAGchainer anchors
```

Generated figures go under `results/` (gitignored).
