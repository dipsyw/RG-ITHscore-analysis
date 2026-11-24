from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def read_components(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"SampleID", "Subclone4", "Subclone5"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame[["SampleID", "Subclone4", "Subclone5"]].rename(
        columns={"Subclone4": "ECM_proportion", "Subclone5": "IR_proportion"}
    )
    for column in ["ECM_proportion", "IR_proportion"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.isna().any().any():
        raise ValueError("Missing values are not permitted in ECM/IR clustering")
    return data


def orient_labels(data: pd.DataFrame, labels_zero_based: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.vstack(
        [
            data.loc[labels_zero_based == label, ["ECM_proportion", "IR_proportion"]]
            .mean()
            .to_numpy(float)
            for label in [0, 1]
        ]
    )
    ecm_minus_ir = centers[:, 0] - centers[:, 1]
    ecm_label = int(np.argmax(ecm_minus_ir))
    labels = np.where(labels_zero_based == ecm_label, 1, 2)
    oriented_centers = np.vstack(
        [
            data.loc[labels == label, ["ECM_proportion", "IR_proportion"]]
            .mean()
            .to_numpy(float)
            for label in [1, 2]
        ]
    )
    return labels, oriented_centers


def fit_clusters(data: pd.DataFrame, seed: int, n_init: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = data[["ECM_proportion", "IR_proportion"]].to_numpy(float)
    fit = KMeans(n_clusters=2, n_init=n_init, random_state=seed).fit(x)
    labels, centers = orient_labels(data, fit.labels_)
    result = data.copy()
    result["Cluster"] = labels
    center_frame = pd.DataFrame(
        {
            "Cluster": [1, 2],
            "ECM_center": centers[:, 0],
            "IR_center": centers[:, 1],
            "phenotype": ["ECM-high / IR-low", "ECM-low / IR-high"],
            "n": [int((labels == 1).sum()), int((labels == 2).sum())],
        }
    )
    diagnostics = {
        "n": int(len(result)),
        "k": 2,
        "seed": seed,
        "n_init": n_init,
        "standardized_before_clustering": False,
        "silhouette_score": float(silhouette_score(x, labels)),
        "cluster_counts": {
            str(key): int(value)
            for key, value in result["Cluster"].value_counts().sort_index().items()
        },
    }
    return result, center_frame, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster ECM and IR NMF proportions with two-group k-means.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--n-init", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_components(args.input)
    clustered, centers, diagnostics = fit_clusters(data, args.seed, args.n_init)
    clustered.to_csv(args.output_dir / "ecm_ir_cluster_assignments.csv", index=False)
    centers.to_csv(args.output_dir / "ecm_ir_cluster_centroids.csv", index=False)
    (args.output_dir / "ecm_ir_cluster_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
