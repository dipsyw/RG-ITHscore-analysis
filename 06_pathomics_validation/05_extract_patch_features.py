from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from common import SUBDIRS, mad, read_table, shannon_from_counts, write_table


NUCLEI_COLUMNS = [
    "patch_id",
    "nucleus_id",
    "x",
    "y",
    "area_px2",
    "perimeter_px",
    "class_name",
    "cell_class_binary",
    "type_probability",
    "source_file",
    "sample_key",
]


def polygon_area(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 3:
        return np.nan
    x = np.asarray([p[0] for p in coords])
    y = np.asarray([p[1] for p in coords])
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def polygon_perimeter(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return np.nan
    xy = np.asarray(coords, dtype=float)
    diffs = np.diff(np.vstack([xy, xy[0]]), axis=0)
    return float(np.sqrt((diffs * diffs).sum(axis=1)).sum())


def centroid(coords: list[tuple[float, float]]) -> tuple[float, float]:
    if not coords:
        return np.nan, np.nan
    xy = np.asarray(coords, dtype=float)
    return float(xy[:, 0].mean()), float(xy[:, 1].mean())


def class_to_binary(value: Any) -> str:
    text = str(value).strip().lower()
    if "tumor" in text or "tumour" in text or "epithelial" in text:
        return "Tumor"
    if text in {"1", "tumor_cell"}:
        return "Tumor"
    return "Other"


def extract_geojson_nuclei(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        features = data
    elif isinstance(data, dict) and data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif isinstance(data, dict) and data.get("type") == "Feature":
        features = [data]
    else:
        raise ValueError(f"Not a GeoJSON feature file: {path}")
    fallback_patch_id = path.parent.name if path.stem in {"cells", "cell_detection"} else path.stem
    rows = []
    for i, feature in enumerate(features):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {}) or {}
        coords_raw = []
        if geom.get("type") == "Polygon":
            rings = geom.get("coordinates", [])
            coords_raw = rings[0] if rings else []
        elif geom.get("type") == "MultiPolygon":
            polys = geom.get("coordinates", [])
            coords_raw = polys[0][0] if polys and polys[0] else []
        coords = [(float(x), float(y)) for x, y, *_ in coords_raw]
        cx, cy = centroid(coords)
        class_name = (
            props.get("class_name")
            or props.get("cell_type")
            or props.get("type")
            or (props.get("classification", {}) or {}).get("name")
            or props.get("label")
            or "Other"
        )
        rows.append(
            {
                "patch_id": props.get("patch_id") or fallback_patch_id,
                "nucleus_id": props.get("id") or f"{fallback_patch_id}_{i}",
                "x": cx,
                "y": cy,
                "area_px2": polygon_area(coords),
                "perimeter_px": polygon_perimeter(coords),
                "class_name": class_name,
                "cell_class_binary": class_to_binary(class_name),
                "source_file": str(path),
            }
        )
    return rows


def extract_cellvit_json_nuclei(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise ValueError(f"Not a CellViT cells.json file: {path}")
    patch_id = path.parent.name
    type_map = {str(k): v for k, v in (data.get("type_map") or {}).items()}
    rows = []
    for i, cell in enumerate(data["cells"]):
        coords_raw = cell.get("contour") or []
        coords = []
        for point in coords_raw:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                coords.append((float(point[0]), float(point[1])))
        if cell.get("centroid") and len(cell["centroid"]) >= 2:
            cx, cy = float(cell["centroid"][0]), float(cell["centroid"][1])
        else:
            cx, cy = centroid(coords)
        type_value = cell.get("type", "Other")
        class_name = type_map.get(str(type_value), str(type_value))
        rows.append(
            {
                "patch_id": patch_id,
                "nucleus_id": cell.get("id") or f"{patch_id}_{i}",
                "x": cx,
                "y": cy,
                "area_px2": polygon_area(coords),
                "perimeter_px": polygon_perimeter(coords),
                "class_name": class_name,
                "cell_class_binary": class_to_binary(class_name),
                "type_probability": cell.get("type_prob", np.nan),
                "source_file": str(path),
            }
        )
    return rows


def normalize_nuclei_table(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "centroid_x": "x",
        "centroid_y": "y",
        "x_centroid": "x",
        "y_centroid": "y",
        "area": "area_px2",
        "nucleus_area": "area_px2",
        "perimeter": "perimeter_px",
        "nucleus_perimeter": "perimeter_px",
        "type": "class_name",
        "label": "class_name",
        "predicted_class": "class_name",
        "cell_type": "class_name",
    }
    out = df.rename(columns={c: aliases.get(c, c) for c in df.columns}).copy()
    if "patch_id" not in out.columns:
        if "patch_name" in out.columns:
            out["patch_id"] = out["patch_name"].astype(str).map(lambda s: Path(s).stem)
        elif "source_file" in out.columns:
            out["patch_id"] = out["source_file"].astype(str).map(lambda s: Path(s).stem)
        else:
            raise ValueError("Nuclei table must contain patch_id, patch_name, or source_file.")
    if "class_name" not in out.columns:
        out["class_name"] = "Other"
    out["cell_class_binary"] = out["class_name"].map(class_to_binary)
    if "nucleus_id" not in out.columns:
        out["nucleus_id"] = [f"{pid}_{i}" for i, pid in enumerate(out["patch_id"].astype(str))]
    for col in ["x", "y", "area_px2", "perimeter_px"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_nuclei(nuclei_table: Path | None, cellvit_root: Path) -> pd.DataFrame:
    if nuclei_table:
        return normalize_nuclei_table(read_table(nuclei_table))
    rows: list[dict] = []
    parsed_patch_ids: set[str] = set()
    for path in sorted(cellvit_root.rglob("cells.json")):
        try:
            rows.extend(extract_cellvit_json_nuclei(path))
            parsed_patch_ids.add(path.parent.name)
        except Exception:
            continue
    for path in sorted(cellvit_root.rglob("*.geojson")):
        try:
            if path.parent.name not in parsed_patch_ids:
                rows.extend(extract_geojson_nuclei(path))
        except Exception:
            continue
    if not rows:
        raise FileNotFoundError(
            "No CellViT nuclei were found. Provide --nuclei-table or place GeoJSON/JSON outputs under cellvit_outputs."
        )
    return normalize_nuclei_table(pd.DataFrame(rows))


def series_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "median", "sd", "q10", "q25", "q75", "q90", "iqr", "mad"]}
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_q10": float(x.quantile(0.10)),
        f"{prefix}_q25": float(x.quantile(0.25)),
        f"{prefix}_q75": float(x.quantile(0.75)),
        f"{prefix}_q90": float(x.quantile(0.90)),
        f"{prefix}_iqr": float(x.quantile(0.75) - x.quantile(0.25)),
        f"{prefix}_mad": mad(x),
    }


def nearest_neighbor_stats(group: pd.DataFrame, mpp: float) -> dict[str, float]:
    xy = group[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(xy) < 2:
        return {"nn_dist_um_mean": np.nan, "nn_dist_um_median": np.nan, "nn_dist_um_sd": np.nan}
    dist, _ = cKDTree(xy).query(xy, k=2)
    nn = dist[:, 1] * mpp
    return {
        "nn_dist_um_mean": float(np.mean(nn)),
        "nn_dist_um_median": float(np.median(nn)),
        "nn_dist_um_sd": float(np.std(nn, ddof=1)) if len(nn) > 1 else 0.0,
    }


def image_features(path: Path) -> dict[str, float]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feats = {}
    for idx, name in enumerate(["r", "g", "b"]):
        channel = arr[:, :, idx]
        feats[f"rgb_{name}_mean"] = float(channel.mean())
        feats[f"rgb_{name}_sd"] = float(channel.std(ddof=1))
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-6)
    feats["saturation_mean"] = float(sat.mean())
    feats["brightness_mean"] = float(maxc.mean())

    try:
        from skimage.feature import graycomatrix, graycoprops  # type: ignore
    except ImportError:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops  # type: ignore
    gray = np.asarray(img.convert("L").resize((256, 256)), dtype=np.uint8)
    quant = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(quant, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop)[0, 0])
    return feats


def patch_feature_row(patch: pd.Series, group: pd.DataFrame, include_image: bool) -> dict:
    patch_id = str(patch["patch_id"])
    target_mpp = float(patch.get("target_mpp", 0.50))
    patch_size = float(patch.get("patch_size_px", 1024))
    area_mm2 = ((patch_size * target_mpp) / 1000.0) ** 2
    row = {
        "sample_key": patch.get("sample_key", ""),
        "patch_id": patch_id,
        "patch_path": patch.get("patch_path", ""),
        "n_nuclei": int(len(group)),
        "patch_area_mm2": area_mm2,
        "nuclei_density_per_mm2": float(len(group) / area_mm2) if area_mm2 > 0 else np.nan,
        "mask_coverage": patch.get("mask_coverage", np.nan),
        "tissue_fraction": patch.get("tissue_fraction", np.nan),
    }
    counts = group["cell_class_binary"].value_counts() if len(group) else pd.Series(dtype=int)
    n_tumor = int(counts.get("Tumor", 0))
    n_other = int(counts.get("Other", 0))
    denom = max(n_tumor + n_other, 1)
    row.update(
        {
            "n_tumor_cells": n_tumor,
            "n_other_cells": n_other,
            "tumor_cell_fraction": float(n_tumor / denom),
            "other_cell_fraction": float(n_other / denom),
            "cell_class_shannon": shannon_from_counts(np.array([n_tumor, n_other], dtype=float)),
        }
    )
    if len(group):
        area_um2 = group["area_px2"] * (target_mpp**2)
        perim_um = group["perimeter_px"] * target_mpp
        circ = 4.0 * math.pi * area_um2 / (perim_um * perim_um + 1e-6)
        row.update(series_stats(area_um2, "nucleus_area_um2"))
        row.update(series_stats(perim_um, "nucleus_perimeter_um"))
        row.update(series_stats(circ.clip(lower=0, upper=1.5), "nucleus_circularity"))
        row.update(nearest_neighbor_stats(group, target_mpp))
    else:
        for prefix in ["nucleus_area_um2", "nucleus_perimeter_um", "nucleus_circularity"]:
            row.update(series_stats(pd.Series(dtype=float), prefix))
        row.update({"nn_dist_um_mean": np.nan, "nn_dist_um_median": np.nan, "nn_dist_um_sd": np.nan})
    if include_image:
        patch_path = Path(str(patch.get("patch_path", "")))
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch image not found: {patch_path}")
        row.update(image_features(patch_path))
    return row


def empty_nuclei_table() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in NUCLEI_COLUMNS})


def normalize_nuclei_chunk(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_nuclei_table()
    df = normalize_nuclei_table(pd.DataFrame(rows))
    df["sample_key"] = df["patch_id"].astype(str).str.split("_").str[0]
    for col in NUCLEI_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    out = df[NUCLEI_COLUMNS].copy()
    for col in ["patch_id", "nucleus_id", "class_name", "cell_class_binary", "source_file", "sample_key"]:
        out[col] = out[col].astype(str)
    for col in ["x", "y", "area_px2", "perimeter_px", "type_probability"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


class StreamingNucleiWriter:
    def __init__(self, path: Path, chunk_rows: int) -> None:
        self.path = path
        self.chunk_rows = int(chunk_rows)
        self.buffer: list[dict] = []
        self.n_rows = 0
        self._writer = None
        self._fallback_csv = path.with_suffix(".csv")
        self._fallback_active = False
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.buffer.extend(rows)
        if len(self.buffer) >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        df = normalize_nuclei_chunk(self.buffer)
        self.buffer = []
        self.n_rows += int(len(df))
        if len(df) == 0:
            return
        if self._fallback_active:
            df.to_csv(self._fallback_csv, mode="a", header=not self._fallback_csv.exists(), index=False, encoding="utf-8-sig")
            return
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
            self._writer.write_table(table)
        except Exception:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._fallback_active = True
            if self.path.exists():
                self.path.unlink()
            df.to_csv(self._fallback_csv, mode="a", header=not self._fallback_csv.exists(), index=False, encoding="utf-8-sig")

    def close(self) -> Path:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._fallback_csv if self._fallback_active else self.path


def write_patch_feature_chunk(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def completed_patch_ids_from_partial(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["patch_id"], chunksize=200000):
        done.update(chunk["patch_id"].astype(str))
    return done


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_features_streaming(
    patches: pd.DataFrame,
    cellvit_root: Path,
    prefix: str,
    include_image: bool,
    resume: bool,
    nuclei_chunk_rows: int,
    feature_chunk_rows: int,
) -> dict[str, Any]:
    features_dir = SUBDIRS["features"]
    logs_dir = SUBDIRS["logs"]
    nuclei_target = features_dir / f"{prefix}cellvit_nuclei.parquet"
    patch_partial = features_dir / f"{prefix}patch_features.partial.csv"
    patch_csv = features_dir / f"{prefix}patch_features.csv"
    patch_parquet = features_dir / f"{prefix}patch_features.parquet"
    progress_json = logs_dir / f"{prefix}patch_feature_extraction_progress.json"

    if not resume:
        for path in [nuclei_target, nuclei_target.with_suffix(".csv"), patch_partial, patch_csv, patch_parquet, progress_json]:
            if path.exists():
                path.unlink()

    completed = completed_patch_ids_from_partial(patch_partial) if resume else set()
    writer = StreamingNucleiWriter(nuclei_target, nuclei_chunk_rows)
    empty_nuclei = empty_nuclei_table()
    feature_buffer: list[dict] = []
    n_missing_cells_json = 0
    n_processed_now = 0
    n_nuclei_now = 0
    total = int(len(patches))

    for i, patch in patches.iterrows():
        patch_id = str(patch["patch_id"])
        if patch_id in completed:
            continue
        cells_path = cellvit_root / patch_id / "cells.json"
        nuclei_rows: list[dict] = []
        if cells_path.exists():
            try:
                nuclei_rows = extract_cellvit_json_nuclei(cells_path)
            except Exception:
                nuclei_rows = []
        else:
            n_missing_cells_json += 1
        nuclei_df = normalize_nuclei_chunk(nuclei_rows) if nuclei_rows else empty_nuclei
        writer.add(nuclei_rows)
        n_nuclei_now += int(len(nuclei_rows))
        feature_buffer.append(patch_feature_row(patch, nuclei_df, include_image=include_image))
        n_processed_now += 1
        done_total = len(completed) + n_processed_now

        if len(feature_buffer) >= feature_chunk_rows:
            write_patch_feature_chunk(feature_buffer, patch_partial)
            feature_buffer = []
        if done_total % 1000 == 0 or done_total == total:
            writer.flush()
            write_patch_feature_chunk(feature_buffer, patch_partial)
            feature_buffer = []
            progress = {
                "total_patches": total,
                "already_completed_at_start": len(completed),
                "processed_now": n_processed_now,
                "done_total": done_total,
                "percent": round(100.0 * done_total / total, 4) if total else 100.0,
                "nuclei_rows_written_this_run": n_nuclei_now,
                "missing_cells_json_this_run": n_missing_cells_json,
                "patch_features_partial": str(patch_partial),
                "nuclei_output_target": str(nuclei_target),
            }
            write_progress(progress_json, progress)
            print(f"Processed {done_total}/{total} patches ({progress['percent']}%)")

    write_patch_feature_chunk(feature_buffer, patch_partial)
    nuclei_out = writer.close()

    features = pd.read_csv(patch_partial)
    features_out = write_table(features, patch_parquet)
    features.to_csv(patch_csv, index=False, encoding="utf-8-sig")
    summary = {
        "n_patches": int(len(features)),
        "n_nuclei_written_this_run": int(n_nuclei_now),
        "n_missing_cells_json_this_run": int(n_missing_cells_json),
        "nuclei_output": str(nuclei_out),
        "patch_features_output": str(features_out),
        "patch_features_csv": str(patch_csv),
        "patch_features_partial": str(patch_partial),
        "resume": bool(resume),
    }
    write_progress(progress_json, {**summary, "done_total": int(len(features)), "total_patches": total, "percent": 100.0})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract patch-level pathomics features from CellViT nuclei outputs.")
    parser.add_argument("--patch-manifest", type=Path, default=SUBDIRS["manifests"] / "patch_manifest.csv")
    parser.add_argument("--nuclei-table", type=Path, default=None)
    parser.add_argument("--cellvit-root", type=Path, default=SUBDIRS["cellvit_outputs"])
    parser.add_argument("--resume", action="store_true", help="Resume from patch_features.partial.csv if present.")
    parser.add_argument("--nuclei-chunk-rows", type=int, default=250000)
    parser.add_argument("--feature-chunk-rows", type=int, default=1000)
    args = parser.parse_args()

    patches = read_table(args.patch_manifest)
    prefix = ""
    if args.nuclei_table is None:
        summary = extract_features_streaming(
            patches=patches,
            cellvit_root=args.cellvit_root,
            prefix=prefix,
            include_image=True,
            resume=bool(args.resume),
            nuclei_chunk_rows=int(args.nuclei_chunk_rows),
            feature_chunk_rows=int(args.feature_chunk_rows),
        )
        with (SUBDIRS["logs"] / f"{prefix}patch_feature_extraction_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    nuclei = load_nuclei(args.nuclei_table, args.cellvit_root)
    nuclei["sample_key"] = nuclei["patch_id"].astype(str).str.split("_").str[0]
    nuclei_out = write_table(nuclei, SUBDIRS["features"] / f"{prefix}cellvit_nuclei.parquet")

    nuclei_groups = {str(k): g.copy() for k, g in nuclei.groupby(nuclei["patch_id"].astype(str), sort=False)}
    empty_nuclei = nuclei.iloc[0:0].copy()
    rows = []
    for i, patch in patches.iterrows():
        patch_id = str(patch["patch_id"])
        rows.append(
            patch_feature_row(
                patch,
                nuclei_groups.get(patch_id, empty_nuclei),
                include_image=True,
            )
        )
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} patches")
    features = pd.DataFrame(rows)
    features_out = write_table(features, SUBDIRS["features"] / f"{prefix}patch_features.parquet")
    features.to_csv(SUBDIRS["features"] / f"{prefix}patch_features.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_patches": int(len(features)),
        "n_nuclei": int(len(nuclei)),
        "nuclei_output": str(nuclei_out),
        "patch_features_output": str(features_out),
    }
    with (SUBDIRS["logs"] / f"{prefix}patch_feature_extraction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
