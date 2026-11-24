from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import auc, roc_curve


def parse_cohort(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=path.csv")
    name, path = value.split("=", 1)
    return name, Path(path)


def read_cohort(path: Path, score_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"ID", "os_time", "os_status", *score_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[["ID", "os_time", "os_status", *score_columns]].copy()
    for column in ["os_time", "os_status", *score_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or frame["ID"].duplicated().any():
        raise ValueError(f"{path} contains missing values or duplicate IDs")
    if not set(frame["os_status"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("os_status must be coded 0/1")
    frame["os_status"] = frame["os_status"].astype(int)
    return frame


def censoring_km(frame: pd.DataFrame) -> KaplanMeierFitter:
    return KaplanMeierFitter().fit(frame["os_time"], event_observed=1 - frame["os_status"])


def ipcw_roc(frame: pd.DataFrame, score: str, horizon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    times = frame["os_time"].to_numpy(float)
    events = frame["os_status"].to_numpy(int)
    values = frame[score].to_numpy(float)
    cases = (events == 1) & (times <= horizon)
    controls = times > horizon
    eligible = cases | controls
    if cases.sum() == 0 or controls.sum() == 0:
        raise ValueError("A horizon requires at least one case and one control")
    km = censoring_km(frame)
    g_horizon = float(km.predict(max(horizon - 1e-8, 0)))
    g_case = np.asarray(km.predict(np.maximum(times[cases] - 1e-8, 0)), dtype=float)
    if g_horizon <= 0 or np.any(g_case <= 0):
        raise ValueError("Censoring survival reached zero before the requested horizon")
    weights = np.zeros(len(frame), dtype=float)
    weights[cases] = 1.0 / g_case
    weights[controls] = 1.0 / g_horizon
    fpr, tpr, thresholds = roc_curve(cases[eligible].astype(int), values[eligible], sample_weight=weights[eligible])
    return fpr, tpr, thresholds, float(auc(fpr, tpr))


def bootstrap_auc(frame: pd.DataFrame, score: str, horizon: float, n_bootstrap: int, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))].reset_index(drop=True)
        try:
            values.append(ipcw_roc(sample, score, horizon)[3])
        except (ValueError, ZeroDivisionError):
            continue
    if not values:
        return np.nan, np.nan, 0
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def youden_threshold(frame: pd.DataFrame, score: str, horizon: float) -> float:
    fpr, tpr, thresholds, _ = ipcw_roc(frame, score, horizon)
    valid = np.isfinite(thresholds)
    index = np.flatnonzero(valid)[np.argmax((tpr - fpr)[valid])]
    return float(thresholds[index])


def weighted_metrics(frame: pd.DataFrame, score: str, horizon: float, threshold: float) -> dict[str, float]:
    times = frame["os_time"].to_numpy(float)
    events = frame["os_status"].to_numpy(int)
    pred = frame[score].to_numpy(float) >= threshold
    cases = (events == 1) & (times <= horizon)
    controls = times > horizon
    km = censoring_km(frame)
    weights = np.zeros(len(frame), dtype=float)
    weights[cases] = 1.0 / np.asarray(km.predict(np.maximum(times[cases] - 1e-8, 0)), dtype=float)
    weights[controls] = 1.0 / float(km.predict(max(horizon - 1e-8, 0)))
    tp = weights[cases & pred].sum()
    fn = weights[cases & ~pred].sum()
    fp = weights[controls & pred].sum()
    tn = weights[controls & ~pred].sum()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "f1": 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else np.nan,
        "youden_index": sensitivity + specificity - 1,
    }


def survival_statistics(frame: pd.DataFrame, score: str, cutoff: float) -> dict[str, float]:
    group = (frame[score] >= cutoff).astype(int)
    low = group == 0
    high = group == 1
    test = logrank_test(
        frame.loc[low, "os_time"], frame.loc[high, "os_time"],
        event_observed_A=frame.loc[low, "os_status"], event_observed_B=frame.loc[high, "os_status"],
    )
    model_data = frame[["os_time", "os_status"]].copy()
    model_data["high_group"] = group
    model = CoxPHFitter().fit(model_data, "os_time", "os_status")
    coefficient = model.summary.loc["high_group"]
    return {
        "n_low": int(low.sum()), "n_high": int(high.sum()),
        "events_low": int(frame.loc[low, "os_status"].sum()), "events_high": int(frame.loc[high, "os_status"].sum()),
        "logrank_p": float(test.p_value), "hazard_ratio_high_vs_low": float(np.exp(coefficient["coef"])),
        "hr_ci_lower_95": float(np.exp(coefficient["coef lower 95%"])),
        "hr_ci_upper_95": float(np.exp(coefficient["coef upper 95%"])),
        "c_index": float(concordance_index(frame["os_time"], -frame[score], frame["os_status"])),
    }


def calibration_rows(development: pd.DataFrame, cohort: pd.DataFrame, score: str, horizons: list[float]) -> list[dict]:
    model = CoxPHFitter().fit(development[["os_time", "os_status", score]], "os_time", "os_status")
    rows = []
    for horizon in horizons:
        predicted = 1 - model.predict_survival_function(cohort[[score]], times=[horizon]).iloc[0].to_numpy(float)
        bins = pd.qcut(predicted, q=min(5, len(np.unique(predicted))), duplicates="drop")
        for index, levels in enumerate(pd.Series(bins).groupby(bins, observed=True).groups.values(), start=1):
            subset = cohort.iloc[np.asarray(list(levels), dtype=int)]
            km = KaplanMeierFitter().fit(subset["os_time"], subset["os_status"])
            rows.append({
                "score": score, "horizon": horizon, "risk_group": index, "n": len(subset),
                "mean_predicted_risk": float(predicted[np.asarray(list(levels), dtype=int)].mean()),
                "km_observed_risk": float(1 - km.predict(horizon)),
            })
    return rows


def decision_curve_rows(development: pd.DataFrame, cohort: pd.DataFrame, score: str, horizons: list[float]) -> list[dict]:
    model = CoxPHFitter().fit(development[["os_time", "os_status", score]], "os_time", "os_status")
    rows = []
    for horizon in horizons:
        predicted = 1 - model.predict_survival_function(cohort[[score]], times=[horizon]).iloc[0].to_numpy(float)
        times = cohort["os_time"].to_numpy(float)
        events = cohort["os_status"].to_numpy(int)
        cases = (events == 1) & (times <= horizon)
        controls = times > horizon
        km = censoring_km(cohort)
        weights = np.zeros(len(cohort), dtype=float)
        weights[cases] = 1.0 / np.asarray(km.predict(np.maximum(times[cases] - 1e-8, 0)), dtype=float)
        weights[controls] = 1.0 / float(km.predict(max(horizon - 1e-8, 0)))
        for threshold in np.arange(0.05, 0.801, 0.01):
            positive = predicted >= threshold
            tp = weights[cases & positive].sum() / len(cohort)
            fp = weights[controls & positive].sum() / len(cohort)
            rows.append({"score": score, "horizon": horizon, "threshold_probability": threshold,
                         "net_benefit": tp - fp * threshold / (1 - threshold)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent validation analyses for fixed RG-ITHscore predictors.")
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", action="append", type=parse_cohort, required=True, help="Repeat NAME=path.csv")
    parser.add_argument("--scores", nargs="+", default=["habitat_radiomics", "deep_imaging", "RG_ITHscore", "TNM_stage", "RG_ITHscore_TNM"])
    parser.add_argument("--clinical-utility-scores", nargs="+", default=["RG_ITHscore", "TNM_stage", "RG_ITHscore_TNM"])
    parser.add_argument("--horizons", nargs="+", type=float, default=[12, 36, 60], help="Same time unit as os_time")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    development = read_cohort(args.development, args.scores)
    cohorts = {name: read_cohort(path, args.scores) for name, path in args.validation}
    cutoffs = {score: float(development[score].median()) for score in args.scores}
    operating = {(score, horizon): youden_threshold(development, score, horizon) for score in args.scores for horizon in args.horizons}
    survival_rows, auc_rows, metric_rows, calibration, dca = [], [], [], [], []
    for cohort_index, (name, frame) in enumerate(cohorts.items()):
        for score_index, score in enumerate(args.scores):
            survival_rows.append({"cohort": name, "score": score, "cutoff_from_development": cutoffs[score], **survival_statistics(frame, score, cutoffs[score])})
            for horizon_index, horizon in enumerate(args.horizons):
                try:
                    point = ipcw_roc(frame, score, horizon)[3]
                    lower, upper, valid = bootstrap_auc(frame, score, horizon, args.bootstrap, 20260604 + cohort_index * 1000 + score_index * 100 + horizon_index)
                    auc_rows.append({"cohort": name, "score": score, "horizon": horizon, "ipcw_auc": point, "ci_lower_95": lower, "ci_upper_95": upper, "valid_bootstrap": valid})
                    metric_rows.append({"cohort": name, "score": score, "horizon": horizon, **weighted_metrics(frame, score, horizon, operating[(score, horizon)])})
                except ValueError:
                    auc_rows.append({"cohort": name, "score": score, "horizon": horizon, "ipcw_auc": np.nan, "ci_lower_95": np.nan, "ci_upper_95": np.nan, "valid_bootstrap": 0})
        for score in args.clinical_utility_scores:
            calibration.extend({"cohort": name, **row} for row in calibration_rows(development, frame, score, args.horizons))
            dca.extend({"cohort": name, **row} for row in decision_curve_rows(development, frame, score, args.horizons))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(survival_rows).to_csv(args.output_dir / "survival_statistics.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(args.output_dir / "time_dependent_ipcw_auc.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "ipcw_operating_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(args.output_dir / "calibration_by_risk_quintile.csv", index=False)
    pd.DataFrame(dca).to_csv(args.output_dir / "decision_curve_net_benefit.csv", index=False)
    (args.output_dir / "fixed_development_parameters.json").write_text(
        json.dumps({"risk_group_cutoffs": cutoffs, "ipcw_youden_thresholds": {f"{s}@{h}": v for (s, h), v in operating.items()}}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
