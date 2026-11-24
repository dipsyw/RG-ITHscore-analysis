from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from common import bh_fdr, finite_numeric_columns, read_table


EXCLUDE = {"RG_ITHscore", "os_time", "os_status", "TNM_stage"}


def cliffs_delta(high: np.ndarray, low: np.ndarray) -> float:
    return float(sum((value > low).sum() - (value < low).sum() for value in high) / (len(high) * len(low)))


def group_tests(frame: pd.DataFrame, features: list[str], family: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        high = pd.to_numeric(frame.loc[frame["RG_ITH_group"] == "High", feature], errors="coerce").dropna().to_numpy(float)
        low = pd.to_numeric(frame.loc[frame["RG_ITH_group"] == "Low", feature], errors="coerce").dropna().to_numpy(float)
        result = mannwhitneyu(high, low, alternative="two-sided", method="auto")
        rows.append({
            "feature": feature,
            "analysis_family": family,
            "n_high": len(high), "n_low": len(low),
            "median_high": np.median(high), "median_low": np.median(low),
            "mann_whitney_u": float(result.statistic), "p_value": float(result.pvalue),
            "cliffs_delta_high_vs_low": cliffs_delta(high, low),
        })
    output = pd.DataFrame(rows)
    output["q_value_bh"] = bh_fdr(output["p_value"])
    return output.sort_values(["analysis_family", "q_value_bh", "p_value"])


def correlations(frame: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for feature in features:
            pair = frame[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
            rho, p_value = spearmanr(pair[feature], pair[target]) if len(pair) >= 5 else (np.nan, np.nan)
            rows.append({"feature": feature, "target": target, "n": len(pair), "spearman_rho": rho, "p_value": p_value})
    output = pd.DataFrame(rows)
    output["q_value_bh"] = bh_fdr(output["p_value"])
    return output.sort_values(["q_value_bh", "p_value"], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathomics high/low comparisons and patient-level CT-pathomics correlations.")
    parser.add_argument("--sample-features", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--correlation-targets", nargs="+", default=["RG_ITHscore"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    features = read_table(args.sample_features)
    manifest = read_table(args.sample_manifest)
    frame = manifest.merge(features, on="sample_key", how="inner", validate="one_to_one")
    if not {"RG_ITH_group", *args.correlation_targets}.issubset(frame.columns):
        raise ValueError("Manifest is missing RG_ITH_group or requested correlation targets")
    phenotype_features = [f"patch_phenotype_{phenotype}_fraction" for phenotype in range(1, 7)]
    candidates = finite_numeric_columns(features, exclude=EXCLUDE)
    pathomics_features = [c for c in candidates if c not in phenotype_features]
    if len(pathomics_features) != 734 or not all(c in candidates for c in phenotype_features):
        raise ValueError(f"Expected 734 pathomics features plus six phenotype proportions; found {len(pathomics_features)} plus {sum(c in candidates for c in phenotype_features)}")
    differential = pd.concat([
        group_tests(frame, pathomics_features, "734_pathomics_features"),
        group_tests(frame, phenotype_features, "six_phenotype_compositions"),
    ], ignore_index=True)
    association = correlations(frame, pathomics_features + phenotype_features, args.correlation_targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    differential.to_csv(args.output_dir / "pathomics_high_low_statistics.csv", index=False)
    association.to_csv(args.output_dir / "ct_pathomics_spearman_correlations.csv", index=False)


if __name__ == "__main__":
    main()
