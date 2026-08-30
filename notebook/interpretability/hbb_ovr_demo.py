from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


COLORS = {
    "1m_model": "#D55E00",
    "8kb_model": "#0072B2",
    "truth": "#1B1B1B",
}

# Reserved Hugging Face dataset; large npy/model files will be uploaded later.
DEFAULT_HF_DATASET_REPO = "biomap-research/DeepD"
DEFAULT_HF_REVISION = "main"
DEFAULT_HBB_REMOTE_PREFIX = "interpretability"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(y_true: Sequence[int], probability: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def roc_auc(y_true: Sequence[int], probability: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, len(p) + 1, dtype=np.float64)
    start = 0
    while start < len(p):
        stop = start + 1
        while stop < len(p) and sorted_p[stop] == sorted_p[start]:
            stop += 1
        ranks[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = ranks[np.argsort(order)][y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bar_chart_svg(
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    *,
    title: str,
    x_label: str,
    width: int = 900,
) -> str:
    labels = [str(x) for x in labels]
    values = np.asarray(values, dtype=float)
    row_h, left, right, top, bottom = 24, 225, 40, 55, 48
    height = top + bottom + row_h * len(labels)
    plot_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="600">{html.escape(title)}</text>',
    ]
    for tick in np.linspace(0, 1, 6):
        x = left + tick * plot_w
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top-5}" y2="{height-bottom}" stroke="#E6E6E6"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="11">{tick:.1f}</text>')
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = top + i * row_h
        bar_w = max(0.0, min(1.0, float(value))) * plot_w
        parts.append(f'<text x="{left-8}" y="{y+15}" text-anchor="end" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+3}" width="{bar_w:.1f}" height="15" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{min(left+bar_w+5,width-right+2):.1f}" y="{y+15}" font-family="sans-serif" font-size="10">{value:.3f}</text>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(x_label)}</text>')
    parts.append('</svg>')
    return "".join(parts)


# ----------------------------- model inference ---------------------------

def _as_float_vector(value: str | float | list[float]) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def predict_binary_json(model_path: str | Path, X: np.ndarray) -> np.ndarray:
    """Run the bundled binary XGBoost tree models with NumPy only."""
    with Path(model_path).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    learner = model["learner"]
    objective = learner["objective"]["name"]
    if objective != "binary:logistic":
        raise ValueError(f"Unsupported objective: {objective}")

    expected_features = int(learner["learner_model_param"]["num_feature"])
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != expected_features:
        raise ValueError(f"Expected X with shape (n, {expected_features}), got {X.shape}")

    base_probability = float(_as_float_vector(learner["learner_model_param"]["base_score"])[0])
    margin = np.full(
        X.shape[0],
        np.log(base_probability / (1.0 - base_probability)),
        dtype=np.float64,
    )
    trees = learner["gradient_booster"]["model"]["trees"]
    for tree in trees:
        left = np.asarray(tree["left_children"], dtype=np.int32)
        right = np.asarray(tree["right_children"], dtype=np.int32)
        feature = np.asarray(tree["split_indices"], dtype=np.int32)
        condition = np.asarray(tree["split_conditions"], dtype=np.float32)
        default_left = np.asarray(tree["default_left"], dtype=bool)
        node = np.zeros(X.shape[0], dtype=np.int32)
        while True:
            active = left[node] != -1
            if not np.any(active):
                break
            rows = np.flatnonzero(active)
            current = node[rows]
            values = X[rows, feature[current]]
            take_left = (values < condition[current]) | (np.isnan(values) & default_left[current])
            node[rows] = np.where(take_left, left[current], right[current])
        margin += condition[node]
    return (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)


def predict_with_xgboost(model_path: str | Path, X: np.ndarray) -> np.ndarray:
    """Run prediction with the optional official XGBoost runtime."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(model_path)
    return np.asarray(booster.predict(xgb.DMatrix(X)), dtype=np.float32)


# ----------------------------- asset download ----------------------------

HBB_CORE_ASSETS = (
    "data/manifest.json",
    "data/best_layers.csv",
    "data/human/annotations.csv",
    "data/human/reference_predictions.csv",
    "models/manifest.csv",
)


def _copy_absent_tree(source: Path, dest: Path) -> None:
    """Copy files from source into dest without replacing existing files."""
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = dest / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _missing_hbb_assets(root: Path) -> list[str]:
    missing = [relative for relative in HBB_CORE_ASSETS if not (root / relative).is_file()]
    manifest_path = root / "models/manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        for field in ("model_path", "feature_path"):
            for relative in manifest[field].dropna().astype(str).unique():
                if not (root / relative).is_file():
                    missing.append(relative)
    return sorted(set(missing))


def ensure_demo_assets(
    root: str | Path,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    remote_prefix: Optional[str] = None,
) -> dict[str, object]:
    """Download HBB data and models from a Hugging Face dataset if absent.

    Defaults to the ``biomap-research/DeepD`` dataset (override with arguments
    or ``DEEPD_DATA_REPO_ID`` / ``DEEPD_DATA_REVISION`` / ``DEEPD_HBB_DATA_PREFIX``).
    The remote prefix must contain ``data/`` and ``models/`` subdirectories
    matching the local layout. Existing local assets are never overwritten.
    """
    root = Path(root).resolve()
    missing = _missing_hbb_assets(root)
    if not missing:
        return {"downloaded": False, "root": str(root), "asset_count": len(HBB_CORE_ASSETS)}

    repo_id = repo_id or os.environ.get("DEEPD_DATA_REPO_ID") or DEFAULT_HF_DATASET_REPO
    revision = revision or os.environ.get("DEEPD_DATA_REVISION") or DEFAULT_HF_REVISION
    remote_prefix = remote_prefix or os.environ.get(
        "DEEPD_HBB_DATA_PREFIX", DEFAULT_HBB_REMOTE_PREFIX
    )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download missing demo assets. "
            "Install the dependencies from the repository root requirements.txt."
        ) from exc

    prefix = remote_prefix.strip("/")
    patterns = [f"{prefix}/data/**", f"{prefix}/models/**"] if prefix else ["data/**", "models/**"]
    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            token=os.environ.get("HF_TOKEN"),
        ))
    except Exception as exc:
        raise FileNotFoundError(
            f"HBB demo assets are missing locally and could not be downloaded "
            f"from Hugging Face dataset {repo_id}. Upload interpretability/data and "
            f"interpretability/models to that dataset, or place the files under {root}. "
            f"Missing: {', '.join(missing)}"
        ) from exc
    source_root = snapshot / prefix if prefix else snapshot
    for directory in ("data", "models"):
        _copy_absent_tree(source_root / directory, root / directory)

    missing = _missing_hbb_assets(root)
    if missing:
        raise FileNotFoundError(
            "The Hugging Face snapshot is incomplete. Missing files after download: "
            + ", ".join(missing)
        )
    return {
        "downloaded": True,
        "root": str(root),
        "repo_id": repo_id,
        "revision": revision,
        "remote_prefix": prefix,
        "asset_count": len(HBB_CORE_ASSETS),
    }


# ----------------------------- original plot adapter ---------------------

ORIGINAL_HBB_LABELS = [
    "CDS",
    "repeat_DNA_transposon",
    "repeat_LTR",
    "repeat_LINE",
    "repeat_SINE",
]


def _stage_predictions(root: Path, manifest: pd.DataFrame, predictions: pd.DataFrame) -> Path:
    results_root = root / "generated/original_plot_results"
    for row in manifest.itertuples(index=False):
        if row.annotation not in ORIGINAL_HBB_LABELS:
            continue
        task_dir = (
            results_root / "xgboost_fixed" / row.model_tag / row.best_layer
            / "one_vs_rest" / row.annotation
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        column = f"{row.model_tag}__{row.annotation}"
        np.save(task_dir / "all_probabilities.npy", predictions[column].to_numpy(dtype=np.float32), allow_pickle=False)
        summary = {
            "task_mode": "one_vs_rest",
            "target_label": row.annotation,
            "positive_mode": "membership",
            "layer": row.best_layer,
            "test_auprc": float(row.test_auprc),
            "test_auroc": float(row.test_auroc),
            "best_iteration": None,
        }
        (task_dir / "result_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return results_root


def _run_original_plot(root: Path, manifest: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Path]:
    results_root = _stage_predictions(root, manifest, predictions)
    output_dir = root / "generated/original_style_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "plot_ovr_full_prediction_tracks_stable_pdf.py"),
        "--species", "human",
        "--annotation_csv", str(root / "data/human/annotations.csv"),
        "--results_root", str(results_root),
        "--model_tags", "8kb_model,1m_model",
        "--annotations", ",".join(ORIGINAL_HBB_LABELS),
        "--truth_mode", "priority",
        "--priority_labels", ",".join(ORIGINAL_HBB_LABELS),
        "--region", "chr11",
        "--start", "5220000",
        "--end", "5240000",
        "--pos_col", "pos_0based",
        "--output_dir", str(output_dir),
        "--prefix", "human_HBB_raw_truth",
        "--formats", "svg,pdf",
        "--dpi", "600",
        "--rasterize_pred",
        "--pred_alpha", "1.0",
        "--pred_color", "#d2b3d5",
        "--truth_alpha", "1.0",
        "--truth_color", "#82b2d4",
        "--fig_width", "14",
        "--track_height", "0.72",
        "--hspace", "0.32",
        "--truth_ymin", "-0.21",
        "--truth_height", "0.16",
        "--mb_decimals", "3",
    ]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=root, check=True, env=env)
    stem = "human_HBB_raw_truth_{model}_chr11_5220000_5240000"
    return {
        "8kb_svg": output_dir / f"{stem.format(model='8kb_model')}.svg",
        "1m_svg": output_dir / f"{stem.format(model='1m_model')}.svg",
        "8kb_pdf": output_dir / f"{stem.format(model='8kb_model')}.pdf",
        "1m_pdf": output_dir / f"{stem.format(model='1m_model')}.pdf",
    }


# ----------------------------- demo orchestration ------------------------

def run_hbb_demo(
    root: str | Path,
    *,
    download_if_missing: bool = True,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    remote_prefix: Optional[str] = None,
    prefer_xgboost: bool = True,
) -> dict[str, object]:
    """Run all 12 models, numerical regression checks, and original plots."""
    started = time.perf_counter()
    root = Path(root).resolve()
    if download_if_missing:
        ensure_demo_assets(root, repo_id=repo_id, revision=revision, remote_prefix=remote_prefix)
    missing = _missing_hbb_assets(root)
    if missing:
        raise FileNotFoundError("Missing HBB demo assets: " + ", ".join(missing))

    manifest = pd.read_csv(root / "models/manifest.csv")
    annotations = pd.read_csv(root / "data/human/annotations.csv")
    reference = pd.read_csv(root / "data/human/reference_predictions.csv")
    if len(manifest) != 12:
        raise ValueError(f"Expected 12 model rows; found {len(manifest)}")
    if len(annotations) != 20_000:
        raise ValueError(f"Expected 20,000 annotation rows; found {len(annotations)}")
    expected_positions = np.arange(5_220_000, 5_240_000)
    if not annotations["chrom"].eq("chr11").all() or not np.array_equal(annotations["pos_0based"], expected_positions):
        raise ValueError("Annotation coordinates do not match chr11:5220000-5240000")

    predictor = predict_binary_json
    engine = "bundled NumPy predictor"
    if prefer_xgboost:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pass
        else:
            predictor = predict_with_xgboost
            engine = "official XGBoost"

    live = pd.DataFrame({"demo_row": annotations["demo_row"]})
    max_errors: list[float] = []
    summaries: list[dict[str, object]] = []
    for model in manifest.itertuples(index=False):
        model_path = root / model.model_path
        feature_path = root / model.feature_path
        if sha256_file(model_path) != model.model_sha256:
            raise ValueError(f"Model SHA-256 mismatch: {model.model_path}")
        X = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if X.shape != (20_000, 576):
            raise ValueError(f"Unexpected feature shape for {model.feature_path}: {X.shape}")
        probability = predictor(model_path, X)
        column = f"{model.model_tag}__{model.annotation}"
        original = reference[column].to_numpy(dtype=np.float32)
        error = float(np.max(np.abs(probability - original)))
        max_errors.append(error)
        live[column] = probability
        truth = annotations[model.annotation].to_numpy(dtype=np.int8)
        summaries.append({
            "model_tag": model.model_tag,
            "model_label": model.model_label,
            "annotation": model.annotation,
            "best_layer": model.best_layer,
            "truth_positive_bases": int(truth.sum()),
            "mean_probability": float(probability.mean()),
            "max_probability": float(probability.max()),
            "interval_average_precision": average_precision(truth, probability),
            "interval_roc_auc": roc_auc(truth, probability),
            "max_abs_error_vs_original": error,
        })

    maximum_error = max(max_errors)
    if maximum_error >= 2e-5:
        raise ValueError(f"Maximum probability error exceeds tolerance: {maximum_error}")
    generated = root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    live_path = generated / "verified_live_predictions.csv"
    summary_path = generated / "hbb_prediction_summary.csv"
    live.to_csv(live_path, index=False, lineterminator="\n")
    summary_df = pd.DataFrame(summaries).sort_values(["annotation", "model_tag"], kind="stable")
    summary_df.to_csv(summary_path, index=False, lineterminator="\n")

    plot_paths = _run_original_plot(root, manifest, live)
    if not all(path.is_file() for path in plot_paths.values()):
        raise FileNotFoundError("One or more original-style plot files were not generated")
    best = pd.read_csv(root / "data/best_layers.csv")
    labels = [f"{row.annotation} | {row.model_label} | {row.layer}" for row in best.itertuples(index=False)]
    best_chart = generated / "best_test_auprc.svg"
    best_chart.write_text(bar_chart_svg(
        labels,
        best["test_auprc"],
        [COLORS[row.model_tag] for row in best.itertuples(index=False)],
        title="Human best-layer OVR performance on the original held-out test set",
        x_label="AUPRC",
    ), encoding="utf-8")

    report = {
        "status": "passed",
        "region": "chr11:5220000-5240000",
        "rows": len(annotations),
        "models_checked_with_live_inference": len(manifest),
        "feature_arrays_checked": int(manifest["feature_path"].nunique()),
        "maximum_probability_error_vs_original_all_probabilities": maximum_error,
        "verification_engine": engine,
        "plotting_engine": "plot_ovr_full_prediction_tracks_stable_pdf.py",
        "plot_script_sha256": sha256_file(root / "plot_ovr_full_prediction_tracks_stable_pdf.py"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report_path = generated / "verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "report": report,
        "prediction_summary": summary_df,
        "plot_paths": plot_paths,
        "best_chart": best_chart,
        "report_path": report_path,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the self-contained DeepD HBB OVR demo.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_HF_DATASET_REPO})",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--remote-prefix", default=None)
    parser.add_argument("--numpy-only", action="store_true", help="Use the bundled NumPy predictor")
    args = parser.parse_args()
    result = run_hbb_demo(
        args.root,
        repo_id=args.repo_id,
        revision=args.revision,
        remote_prefix=args.remote_prefix,
        prefer_xgboost=not args.numpy_only,
    )
    print(json.dumps(result["report"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
