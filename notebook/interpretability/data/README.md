# HBB data assets

- `manifest.json`: exact interval, row count, feature width, and model format.
- `best_layers.csv`: Human best layers and metrics from the original complete test workflow; server-specific absolute paths have been removed.
- `human/annotations.csv`: 20,000 continuous annotation rows for `chr11:5220000-5240000`, including `npy_row` indices from the original matrix.
- `human/features/*.npy`: eight real `(20000, 576)` float32 feature arrays from the actual best layers.
- `human/reference_predictions.csv`: probabilities from the corresponding rows of the original `all_probabilities.npy`.

Coordinates are 0-based and half-open. `npy_row` identifies the row in the original synchronized matrix, and `demo_row` is the contiguous 0-based row index within this repository.

Live predictions, interval summaries, verification reports, and plots are generated at runtime under `generated/`; they are not required input assets.
