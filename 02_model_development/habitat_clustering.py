from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare k=3-6 voxel-wise radiomic habitat solutions and retain k=3."
    )
    parser.add_argument("--voxel-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-columns", nargs="*", default=["patient_id", "voxel_id"])
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    table = pd.read_csv(args.voxel_features)
    feature_columns = [c for c in table.columns if c not in set(args.id_columns)]
    if len(feature_columns) != 19:
        raise ValueError(f"Expected 19 voxel-wise features, found {len(feature_columns)}")
    values = table[feature_columns].apply(pd.to_numeric, errors="raise")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Voxel-wise feature matrix contains missing or non-finite values")
    scaled = StandardScaler().fit_transform(values)

    rows = []
    fits: dict[int, KMeans] = {}
    for k in range(3, 7):
        fit = KMeans(n_clusters=k, n_init=100, random_state=args.seed).fit(scaled)
        fits[k] = fit
        rows.append({"k": k, "calinski_harabasz_index": calinski_harabasz_score(scaled, fit.labels_)})
    diagnostics = pd.DataFrame(rows)
    selected_k = int(diagnostics.loc[diagnostics["calinski_harabasz_index"].idxmax(), "k"])
    if selected_k != 3:
        raise RuntimeError(f"Highest CH index occurred at k={selected_k}; manuscript specifies k=3")

    output = table[args.id_columns].copy()
    output["habitat"] = fits[3].labels_ + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "voxel_habitat_assignments.csv", index=False)
    diagnostics.to_csv(args.output_dir / "habitat_number_diagnostics.csv", index=False)
    (args.output_dir / "habitat_clustering_metadata.json").write_text(
        json.dumps({"selected_k": 3, "seed": args.seed, "n_init": 100, "features": feature_columns}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
