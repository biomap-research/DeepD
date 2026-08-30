# Interpretability demo

This directory is a notebook feature intended to live inside a larger repository. Model inference, verification, asset download, result staging, and notebook helpers are consolidated in `hbb_ovr_demo.py`. The original publication-style track renderer remains in one separate root-level file so that its plotting implementation stays easy to audit.

## Runtime assets

The small JSON/CSV inputs under `data/` (manifest, best-layer metrics, human annotations, reference predictions) and `models/manifest.csv` are committed to git. The large `data/human/features/*.npy` feature arrays and the model JSON files under `models/human/` are runtime assets and are re-downloaded automatically whenever they are missing locally.

When required feature arrays or models are absent, `hbb_ovr_demo.ensure_demo_assets()` downloads the missing `data/` and `models/` subtrees from the Hugging Face dataset [`biomap-research/DeepD`](https://huggingface.co/datasets/biomap-research/DeepD), under the `interpretability/` prefix. Existing local files are never downloaded or overwritten.

```text
biomap-research/DeepD
└── interpretability/
    ├── data/
    └── models/
```

Override the default only if you host a fork:

```bash
export DEEPD_DATA_REPO_ID=biomap-research/DeepD
export DEEPD_DATA_REVISION=main
export DEEPD_HBB_DATA_PREFIX=interpretability
```

Private datasets may use the standard `HF_TOKEN` environment variable.

## Run

```bash
python hbb_ovr_demo.py

# Or run the notebook interactively
jupyter lab DeepD_HBB_best_layer_OVR_demo.ipynb
```

The command verifies all 12 model hashes, checks eight `(20000, 576)` feature arrays, recomputes predictions for the complete 20,000-base HBB interval, compares them with the original probabilities, and generates separate 8 kb and 1M track figures.

## Outputs

All newly generated files are written under `generated/`:

- live predictions;
- interval summary metrics;
- the verification report;
- staged input for the original plotter;
- 8 kb and 1M SVG/PDF track figures;
- the full-test best-layer AUPRC chart.

## Minimal source layout

```text
interpretability/
├── DeepD_HBB_best_layer_OVR_demo.ipynb
├── README.md
├── data/                  # committed: manifest.json, best_layers.csv, human/*.csv inputs
├── hbb_ovr_demo.py
├── plot_ovr_full_prediction_tracks_stable_pdf.py
└── models/                # committed: manifest.csv; the model JSONs are re-downloaded
```

The committed `data/` inputs and `models/manifest.csv` reproduce the demo; the large `features/*.npy` arrays, model JSONs, `figures/`, and `generated/` outputs are handled by the downloader or are regenerated at runtime.

## Interpretation

- Metrics in `data/best_layers.csv` come from the original complete train/validation/test workflow and should be used for primary reporting.
- HBB interval metrics are local diagnostics and must not replace full-test metrics.
- `repeat_DNA_transposon` has no positive bases in this interval, so undefined interval AP/AUROC values are expected.
- The original command plots five tracks and excludes `cCRE`; the `cCRE` models are still verified numerically.
