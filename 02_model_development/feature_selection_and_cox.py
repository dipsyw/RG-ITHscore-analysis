from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


SEED = 20260604
ML_METHODS = [
    "logistic_l1",
    "logistic_elasticnet",
    "logistic_ridge",
    "linearsvm_l1",
    "linearsvm_l2",
    "svm_rbf",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "ada_boost",
    "knn",
    "mlp",
]
PENALTIES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
L1_RATIOS = [0.1, 0.5, 0.9]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_aligned(feature_path: Path, cohort_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(feature_path)
    cohort = pd.read_csv(cohort_path)
    required_cohort = {"ID", "cluster_binary", "os_time", "os_status"}
    missing = required_cohort.difference(cohort.columns)
    if missing:
        raise ValueError(f"Cohort table is missing columns: {sorted(missing)}")
    if "ID" not in features.columns:
        raise ValueError("Feature table must contain an ID column")
    if cohort["ID"].duplicated().any() or features["ID"].duplicated().any():
        raise ValueError("ID values must be unique in both input tables")
    aligned = cohort[["ID"]].merge(features, on="ID", how="left", validate="one_to_one")
    if aligned.drop(columns="ID").isna().all(axis=1).any():
        missing_ids = aligned.loc[aligned.drop(columns="ID").isna().all(axis=1), "ID"].tolist()
        raise ValueError(f"Feature rows are missing for IDs: {missing_ids[:10]}")
    cohort = cohort.set_index("ID").loc[aligned["ID"]].reset_index()
    return aligned, cohort


def preprocess_features(
    aligned: pd.DataFrame,
    labels: np.ndarray,
    missing_threshold: float = 0.20,
    correlation_cutoff: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    numeric = aligned.drop(columns="ID").apply(pd.to_numeric, errors="coerce")
    keep_missing = numeric.isna().mean() <= missing_threshold
    dropped_missing = keep_missing.index[~keep_missing].tolist()
    numeric = numeric.loc[:, keep_missing]
    medians = numeric.median(axis=0)
    numeric = numeric.fillna(medians)

    variances = numeric.var(axis=0, ddof=0)
    keep_variance = variances > 0
    dropped_variance = keep_variance.index[~keep_variance].tolist()
    numeric = numeric.loc[:, keep_variance]
    if numeric.empty:
        raise ValueError("No features remained after missingness and variance filtering")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(numeric)
    scaled = pd.DataFrame(scaled_array, columns=numeric.columns, index=numeric.index)

    f_values, _ = f_classif(scaled.to_numpy(float), labels)
    f_values = np.nan_to_num(f_values, nan=-np.inf)
    feature_order = [scaled.columns[index] for index in np.argsort(f_values)[::-1]]
    kept: list[str] = []
    dropped_correlation: list[str] = []
    correlation = scaled.corr().abs()
    for feature in feature_order:
        if any(correlation.loc[feature, selected] > correlation_cutoff for selected in kept):
            dropped_correlation.append(feature)
        else:
            kept.append(feature)
    scaled = scaled[kept]
    output = pd.concat([aligned[["ID"]].reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
    metadata = {
        "missing_threshold": missing_threshold,
        "correlation_cutoff": correlation_cutoff,
        "dropped_missing": dropped_missing,
        "dropped_zero_variance": dropped_variance,
        "dropped_correlation": dropped_correlation,
        "medians": {column: float(value) for column, value in medians.items()},
        "means": {column: float(value) for column, value in zip(numeric.columns, scaler.mean_)},
        "scales": {column: float(value) for column, value in zip(numeric.columns, scaler.scale_)},
        "retained_features": kept,
    }
    return output, metadata


def estimator(method: str):
    if method == "logistic_l1":
        return LogisticRegression(penalty="l1", C=0.2, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=SEED)
    if method == "logistic_elasticnet":
        return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.2, solver="saga", class_weight="balanced", max_iter=10000, random_state=SEED, n_jobs=-1)
    if method == "logistic_ridge":
        return LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=SEED)
    if method == "linearsvm_l1":
        return LinearSVC(penalty="l1", loss="squared_hinge", dual=False, C=0.05, class_weight="balanced", max_iter=10000, random_state=SEED)
    if method == "linearsvm_l2":
        return LinearSVC(penalty="l2", loss="squared_hinge", dual=False, C=0.1, class_weight="balanced", max_iter=10000, random_state=SEED)
    if method == "svm_rbf":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED)
    if method == "random_forest":
        return RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    if method == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
    if method == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=SEED)
    if method == "ada_boost":
        return AdaBoostClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED)
    if method == "knn":
        return KNeighborsClassifier(n_neighbors=9, weights="distance")
    if method == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.2, max_iter=1000, random_state=SEED)
    raise ValueError(method)


def rank_features(method: str, x: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    use_prefilter = method in {"svm_rbf", "knn", "mlp"} and x.shape[1] > 500
    subset = np.arange(x.shape[1])
    if use_prefilter:
        f_values, _ = f_classif(x, y)
        subset = np.argsort(np.nan_to_num(f_values, nan=-np.inf))[::-1][:500]
    model = estimator(method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x[:, subset], y)
    scores = np.zeros(x.shape[1], dtype=float)
    if hasattr(model, "coef_"):
        scores[subset] = np.abs(np.asarray(model.coef_)).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        scores[subset] = np.asarray(model.feature_importances_, dtype=float)
    else:
        importance = permutation_importance(model, x[:, subset], y, n_repeats=5, random_state=SEED, scoring="roc_auc", n_jobs=-1)
        scores[subset] = np.maximum(importance.importances_mean, 0.0)
    order = np.argsort(scores)[::-1]
    return pd.DataFrame(
        {
            "method": method,
            "feature": [names[index] for index in order],
            "rank": np.arange(1, len(order) + 1),
            "importance": scores[order],
            "used_prefilter": use_prefilter,
        }
    )


def select_by_vote(rankings: pd.DataFrame) -> pd.DataFrame:
    """Retain features appearing in the top 30 for at least 5 of 12 methods."""
    top = rankings[rankings["rank"] <= 30]
    summary = (
        top.groupby("feature")
        .agg(
            vote_count=("method", "nunique"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            supporting_methods=("method", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
        .sort_values(["vote_count", "mean_rank", "min_rank"], ascending=[False, True, True])
    )
    selected = summary[summary["vote_count"] >= 5].copy()
    if selected.empty:
        raise RuntimeError("No feature met the prespecified top-30 and five-vote rule")
    return selected


def run_selection(args: argparse.Namespace) -> None:
    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned, cohort = read_aligned(args.features, args.cohort)
    y = pd.to_numeric(cohort["cluster_binary"], errors="raise").astype(int).to_numpy()
    processed, metadata = preprocess_features(aligned, y)
    names = processed.columns.drop("ID").tolist()
    x = processed[names].to_numpy(float)
    rankings = pd.concat([rank_features(method, x, y, names) for method in ML_METHODS], ignore_index=True)
    selected = select_by_vote(rankings)
    selected.insert(0, "feature_set", args.name)
    selected.insert(1, "top_k_per_method", 30)
    processed.to_csv(args.output_dir / f"processed_features_{args.name}.csv", index=False)
    rankings.to_csv(args.output_dir / f"feature_rankings_{args.name}.csv", index=False)
    selected.to_csv(args.output_dir / f"selected_features_{args.name}.csv", index=False)
    metadata.update({"feature_set": args.name, "seed": SEED, "methods": ML_METHODS, "selected_count": int(len(selected)), "top_k_per_method": 30, "minimum_votes": 5})
    write_json(args.output_dir / f"preprocess_{args.name}.json", metadata)


def cindex(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index(times, -risk, events))


def fit_cox(x: pd.DataFrame, times: np.ndarray, events: np.ndarray, penalty: float, l1_ratio: float) -> CoxPHFitter:
    frame = x.copy()
    frame["os_time"] = times
    frame["os_status"] = events
    model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
    model.fit(frame, duration_col="os_time", event_col="os_status", show_progress=False)
    return model


def cross_validate(x: pd.DataFrame, times: np.ndarray, events: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for penalty in PENALTIES:
        for l1_ratio in L1_RATIOS:
            values = []
            for train_index, test_index in folds.split(x, events):
                model = fit_cox(x.iloc[train_index], times[train_index], events[train_index], penalty, l1_ratio)
                risk = model.predict_partial_hazard(x.iloc[test_index]).to_numpy(float)
                values.append(cindex(times[test_index], events[test_index], risk))
            rows.append({"penalty": penalty, "l1_ratio": l1_ratio, "mean_cv_cindex": float(np.nanmean(values)), "sd_cv_cindex": float(np.nanstd(values, ddof=1))})
    table = pd.DataFrame(rows).sort_values(["mean_cv_cindex", "sd_cv_cindex"], ascending=[False, True])
    best = table.iloc[0]
    return float(best["penalty"]), float(best["l1_ratio"]), table


def bootstrap_optimism(x: pd.DataFrame, times: np.ndarray, events: np.ndarray, apparent: float, penalty: float, l1_ratio: float, n_bootstrap: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(SEED)
    optimism = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(x), len(x))
        if np.unique(events[index]).size < 2:
            continue
        try:
            model = fit_cox(x.iloc[index].reset_index(drop=True), times[index], events[index], penalty, l1_ratio)
            boot_risk = model.predict_partial_hazard(x.iloc[index]).to_numpy(float)
            original_risk = model.predict_partial_hazard(x).to_numpy(float)
            optimism.append(cindex(times[index], events[index], boot_risk) - cindex(times, events, original_risk))
        except Exception:
            continue
    mean_optimism = float(np.mean(optimism)) if optimism else float("nan")
    return mean_optimism, apparent - mean_optimism, len(optimism)


def load_selected_matrix(processed_path: Path, selected_path: Path, prefix: str) -> pd.DataFrame:
    processed = pd.read_csv(processed_path)
    selected = pd.read_csv(selected_path)["feature"].tolist()
    missing = set(selected).difference(processed.columns)
    if missing:
        raise ValueError(f"Selected features missing from {processed_path}: {sorted(missing)[:10]}")
    matrix = processed[["ID", *selected]].copy()
    return matrix.rename(columns={feature: f"{prefix}__{feature}" for feature in selected})


def run_cox(args: argparse.Namespace) -> None:
    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort)
    radiomics = load_selected_matrix(args.radiomics_processed, args.radiomics_selected, "rad")
    deep = load_selected_matrix(args.deep_processed, args.deep_selected, args.deep_name)
    merged = cohort.merge(radiomics, on="ID", validate="one_to_one").merge(deep, on="ID", validate="one_to_one")
    times = pd.to_numeric(merged["os_time"], errors="raise").to_numpy(float)
    events = pd.to_numeric(merged["os_status"], errors="raise").astype(int).to_numpy()
    branches = {
        "radiomics": [column for column in merged if column.startswith("rad__")],
        f"deep_{args.deep_name}": [column for column in merged if column.startswith(f"{args.deep_name}__")],
    }
    branches[f"fusion_{args.deep_name}"] = branches["radiomics"] + branches[f"deep_{args.deep_name}"]
    summaries = []
    for name, columns in branches.items():
        x = merged[columns]
        penalty, l1_ratio, cv = cross_validate(x, times, events)
        cv.to_csv(args.output_dir / f"cox_cv_{name}.csv", index=False)
        model = fit_cox(x, times, events, penalty, l1_ratio)
        risk = model.predict_partial_hazard(x).to_numpy(float)
        apparent = cindex(times, events, risk)
        optimism, corrected, valid = bootstrap_optimism(x, times, events, apparent, penalty, l1_ratio, args.bootstrap)
        coefficients = model.params_.rename("coefficient").reset_index(names="feature")
        coefficients.to_csv(args.output_dir / f"cox_coefficients_{name}.csv", index=False)
        score_table = merged[["ID", "os_time", "os_status"]].copy()
        score_table["risk_score"] = risk
        score_table["risk_group"] = np.where(risk >= np.median(risk), "high", "low")
        score_table.to_csv(args.output_dir / f"risk_scores_{name}.csv", index=False)
        write_json(
            args.output_dir / f"risk_formula_{name}.json",
            {
                "model": name,
                "penalty": penalty,
                "l1_ratio": l1_ratio,
                "features": columns,
                "coefficients": dict(zip(coefficients["feature"], coefficients["coefficient"])),
                "training_median_partial_hazard_cutoff": float(np.median(risk)),
            },
        )
        summaries.append({"model": name, "n": len(merged), "events": int(events.sum()), "n_features": len(columns), "penalty": penalty, "l1_ratio": l1_ratio, "apparent_cindex": apparent, "bootstrap_optimism": optimism, "optimism_corrected_cindex": corrected, "valid_bootstrap": valid})
    pd.DataFrame(summaries).to_csv(args.output_dir / "cox_performance_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RG-ITHscore consensus feature selection and penalized Cox modelling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-features")
    select.add_argument("--features", type=Path, required=True)
    select.add_argument("--cohort", type=Path, required=True)
    select.add_argument("--name", required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.set_defaults(func=run_selection)

    cox = subparsers.add_parser("fit-cox")
    cox.add_argument("--cohort", type=Path, required=True)
    cox.add_argument("--radiomics-processed", type=Path, required=True)
    cox.add_argument("--radiomics-selected", type=Path, required=True)
    cox.add_argument("--deep-processed", type=Path, required=True)
    cox.add_argument("--deep-selected", type=Path, required=True)
    cox.add_argument("--deep-name", default="efficientnet_b0")
    cox.add_argument("--bootstrap", type=int, default=100)
    cox.add_argument("--output-dir", type=Path, required=True)
    cox.set_defaults(func=run_cox)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
