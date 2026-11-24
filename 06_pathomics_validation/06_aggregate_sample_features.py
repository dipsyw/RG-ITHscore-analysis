from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from common import coefficient_of_variation, finite_numeric_columns, gini, mad, read_table, shannon_from_counts, simpson_from_counts


EXCLUDE = {"sample_key", "patch_id", "patch_path", "x0", "y0", "x1", "y1", "patch_size_px", "target_mpp"}


def aggregate(values: pd.Series, name: str) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    histogram = np.histogram(x, bins=min(10, max(2, x.nunique())))[0] if len(x) else np.array([])
    near_constant = len(x) > 1 and np.isclose(float(x.std(ddof=0)), 0.0)
    return {
        f"{name}_mean": float(x.mean()) if len(x) else np.nan,
        f"{name}_median": float(x.median()) if len(x) else np.nan,
        f"{name}_sd": float(x.std(ddof=1)) if len(x) > 1 else (0.0 if len(x) else np.nan),
        f"{name}_iqr": float(x.quantile(.75) - x.quantile(.25)) if len(x) else np.nan,
        f"{name}_mad": mad(x),
        f"{name}_q10": float(x.quantile(.10)) if len(x) else np.nan,
        f"{name}_q25": float(x.quantile(.25)) if len(x) else np.nan,
        f"{name}_q75": float(x.quantile(.75)) if len(x) else np.nan,
        f"{name}_q90": float(x.quantile(.90)) if len(x) else np.nan,
        f"{name}_skew": 0.0 if near_constant or len(x) <= 2 else float(skew(x)),
        f"{name}_kurtosis": 0.0 if near_constant or len(x) <= 3 else float(kurtosis(x)),
        f"{name}_cv": coefficient_of_variation(x),
        f"{name}_gini": gini(x),
        f"{name}_entropy": shannon_from_counts(histogram),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 52 patch descriptors to 734 patient-level features and six phenotype proportions.")
    parser.add_argument("--patch-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    patches = read_table(args.patch_features)
    if "sample_key" not in patches.columns:
        raise ValueError("Patch features must contain sample_key")
    descriptors = finite_numeric_columns(patches, exclude=EXCLUDE)
    if len(descriptors) != 52:
        raise ValueError(f"Expected the prespecified 52 patch descriptors, found {len(descriptors)}")
    values = patches[descriptors].replace([np.inf, -np.inf], np.nan)
    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(values))
    rng = np.random.default_rng(args.seed)
    fit_index = rng.choice(len(patches), 200000, replace=False) if len(patches) > 200000 else np.arange(len(patches))
    model = KMeans(n_clusters=6, n_init=20, random_state=args.seed).fit(scaled[fit_index])
    patches["patch_phenotype"] = model.predict(scaled) + 1

    rows = []
    for sample, group in patches.groupby("sample_key", sort=True):
        row = {"sample_key": sample, "n_patches": len(group), "n_patch_features": len(descriptors)}
        for descriptor in descriptors:
            row.update(aggregate(group[descriptor], descriptor))
        counts = group["patch_phenotype"].value_counts().sort_index()
        proportions = counts / counts.sum()
        row.update({
            "patch_phenotype_shannon": shannon_from_counts(counts),
            "patch_phenotype_simpson": simpson_from_counts(counts),
            "patch_phenotype_dominant_fraction": float(proportions.max()),
            "patch_phenotype_n_observed": int((counts > 0).sum()),
        })
        for phenotype in range(1, 7):
            row[f"patch_phenotype_{phenotype}_fraction"] = float(proportions.get(phenotype, 0.0))
        rows.append(row)
    sample_features = pd.DataFrame(rows)
    phenotype_proportions = [f"patch_phenotype_{phenotype}_fraction" for phenotype in range(1, 7)]
    analysis_features = [c for c in sample_features if c not in {"sample_key", *phenotype_proportions}]
    if len(analysis_features) != 734:
        raise AssertionError(f"Expected 734 patient-level features, produced {len(analysis_features)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patches.to_csv(args.output_dir / "patch_features_with_six_phenotypes.csv", index=False)
    sample_features.to_csv(args.output_dir / "sample_features_734_plus_six_phenotypes.csv", index=False)


if __name__ == "__main__":
    main()
