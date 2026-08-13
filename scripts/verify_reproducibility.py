#!/usr/bin/env python3
"""Repository-level reproducibility checks for the MK-Ensemble submission.

The script verifies that released CSV/JSON source files support the manuscript
and supporting-information summary values. It intentionally avoids heavyweight
model retraining because Chemprop checkpoints and serialized feature caches are
not part of the public archive.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "data/01_dataset/antioxidant_dataset.csv",
    "data/01_dataset/saponins_annotated.csv",
    "data/01_dataset/scaffold_split.json",
    "data/02_structures/saponins_3d.sdf",
    "data/03_targets/targets_predicted.csv",
    "data/03_targets/antioxidant_pathway_genes.csv",
    "data/04_results/model_predictions.csv",
    "data/04_results/pathway_enrichment.csv",
    "data/04_results/applicability_domain_summary.csv",
    "data/04_results/fragment_attribution_validation.csv",
    "data/04_results/y_randomization_summary.csv",
    "data/04_results/learning_curve_summary.csv",
    "data/04_results/train_cv_performance.csv",
    "data/04_results/ablation_summary.csv",
    "data/04_results/frap_exploratory_results.csv",
    "data/04_results/statistical_model_comparison.csv",
    "data/04_results/bace_train_test_gap.csv",
    "data/04_results/descriptor_importance_summary.csv",
    "data/04_results/descriptor_class_importance.csv",
    "data/Table1_dataset_statistics.csv",
    "data/Table2_model_performance.csv",
    "data/Table3_docking_summary.csv",
    "external_validation/data/bace.csv",
    "external_validation/results/bace_external_validation_results.csv",
    "external_validation/results/bace_predictions.json",
]


EXPECTED_PERFORMANCE = {
    ("MK-Ensemble", "DPPH"): (0.846, 0.154, 0.121, 0.831),
    ("RandomForest", "DPPH"): (0.822, 0.165, 0.128, 0.804),
    ("XGBoost", "DPPH"): (0.805, 0.172, 0.135, 0.791),
    ("Chemprop", "DPPH"): (0.778, 0.184, 0.142, 0.763),
    ("SVR-Tanimoto", "DPPH"): (0.791, 0.178, 0.139, 0.775),
    ("MK-Ensemble", "ABTS"): (0.920, 0.089, 0.068, 0.907),
    ("RandomForest", "ABTS"): (0.795, 0.142, 0.112, 0.778),
    ("XGBoost", "ABTS"): (0.842, 0.125, 0.094, 0.829),
    ("Chemprop", "ABTS"): (0.818, 0.134, 0.105, 0.801),
    ("SVR-Tanimoto", "ABTS"): (0.862, 0.117, 0.091, 0.848),
    ("MK-Ensemble", "FRAP"): (0.779, 0.140, 0.120, math.nan),
}


EXPECTED_AD = {
    "DPPH": (54, 9, 0.333, 8, 1, 0.333, 0.259, 0.340, 0.258),
    "ABTS": (31, 6, 0.581, 5, 1, 0.411, 0.253, 0.143, 0.112),
    "FRAP": (12, 4, 1.500, 4, 0, 0.140, 0.106, 0.140, 0.106),
}


EXPECTED_BACE = {
    "SVR-Tanimoto": (0.36315829566202484, 1.0177586592166432, 0.7686059430661807),
    "V2-E: Consensus": (0.32628935086070565, 1.0468049417949454, 0.7809265250173372),
    "V2-B: AdaptiveKernel (Dice)": (0.3103664321937524, 1.059103137005166, 0.7984099637737021),
    "SVR-Dice": (0.292559518421155, 1.0726894564914968, 0.8124066859193261),
    "V2-D: DomainAdapted": (0.2848060351155539, 1.078551727192125, 0.8032091111506536),
    "V2-A: Dice-MKL": (0.282460570959306, 1.0803188238862926, 0.8035128698259424),
    "XGBoost": (0.27049104039173955, 1.0892921482333526, 0.8112257097548438),
    "V2-C: StackedKernel": (0.262361833664454, 1.0953445404101467, 0.8235492582224739),
    "RF": (0.24479647867560095, 1.1083095138253063, 0.839608726072245),
    "Chemprop": (0.1938043787350957, 1.1451154536430113, 0.9029336455335968),
}


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-6) -> None:
    if math.isnan(expected):
        return
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise AssertionError("Missing required files:\n" + "\n".join(missing))


def check_dataset_counts() -> None:
    df = pd.read_csv(ROOT / "data/01_dataset/antioxidant_dataset.csv")
    if len(df) != 128:
        raise AssertionError(f"Activity records: expected 128, observed {len(df)}")
    if df["CID"].nunique() != 91:
        raise AssertionError(f"Unique compounds: expected 91, observed {df['CID'].nunique()}")
    counts = df["assay_type"].value_counts().to_dict()
    expected = {"DPPH": 70, "ABTS": 42, "FRAP": 16}
    if counts != expected:
        raise AssertionError(f"Assay counts: expected {expected}, observed {counts}")

    with (ROOT / "data/01_dataset/scaffold_split.json").open(encoding="utf-8") as handle:
        split = json.load(handle)
    if (split["train_n"], split["val_n"], split["test_n"]) != (100, 16, 12):
        raise AssertionError("Scaffold split counts do not match 100/16/12")


def check_performance_table() -> None:
    df = pd.read_csv(ROOT / "data/Table2_model_performance.csv")
    for (model, assay), expected in EXPECTED_PERFORMANCE.items():
        rows = df[(df["Model"] == model) & (df["Assay"] == assay)]
        if rows.empty:
            raise AssertionError(f"Missing performance row: {model} {assay}")
        row = rows.iloc[0]
        for col, value in zip(["R2", "RMSE", "MAE", "Q2CV"], expected):
            if col == "Q2CV" and pd.isna(row[col]):
                continue
            assert_close(row[col], value, f"{model} {assay} {col}", tol=1e-3)


def check_supporting_tables() -> None:
    required_rows = {
        "train_cv_performance.csv": 10,
        "ablation_summary.csv": 5,
        "frap_exploratory_results.csv": 4,
        "statistical_model_comparison.csv": 8,
        "bace_train_test_gap.csv": 7,
        "descriptor_importance_summary.csv": 10,
        "descriptor_class_importance.csv": 5,
    }
    for filename, expected_rows in required_rows.items():
        path = ROOT / "data/04_results" / filename
        with path.open(newline="", encoding="utf-8") as handle:
            observed_rows = sum(1 for _ in csv.DictReader(handle))
        if observed_rows != expected_rows:
            raise AssertionError(f"{filename}: expected {expected_rows} rows, observed {observed_rows}")

    bace_results = pd.read_csv(ROOT / "external_validation/results/bace_external_validation_results.csv")
    bace_gap = pd.read_csv(ROOT / "data/04_results/bace_train_test_gap.csv")
    method_map = {
        "MK-Ensemble (Consensus)": "V2-E: Consensus",
        "MK-Ensemble (Adaptive)": "V2-B: AdaptiveKernel (Dice)",
        "MK-Ensemble (Domain-Adapted)": "V2-D: DomainAdapted",
        "Random Forest": "RF",
        "Chemprop (MPNN)": "Chemprop",
    }
    for _, gap_row in bace_gap.iterrows():
        result_name = method_map.get(gap_row["method"], gap_row["method"])
        result_row = bace_results[bace_results["method"] == result_name].iloc[0]
        expected_test = float(f"{result_row['R2']:.3f}")
        assert_close(gap_row["test_r2"], expected_test, f"{gap_row['method']} BACE gap test_r2", tol=1e-6)
        expected_delta = float(f"{gap_row['train_r2'] - expected_test:.3f}")
        assert_close(gap_row["delta_r2"], expected_delta, f"{gap_row['method']} BACE gap delta_r2", tol=1e-6)

    pathway = pd.read_csv(ROOT / "data/04_results/pathway_enrichment.csv")
    nrf2 = pathway[pathway["pathway_name"] == "Nrf2/HO-1 Antioxidant Response"].iloc[0]
    assert_close(nrf2["overlap"], 4, "Nrf2/HO-1 overlap", tol=0)
    assert_close(nrf2["p_adj"], 0.003, "Nrf2/HO-1 FDR", tol=1e-6)


def check_ad_and_y_randomization() -> None:
    ad = pd.read_csv(ROOT / "data/04_results/applicability_domain_summary.csv")
    for assay, expected in EXPECTED_AD.items():
        row = ad[ad["assay"] == assay].iloc[0]
        columns = [
            "train_n",
            "test_n",
            "h_star",
            "within_ad",
            "outside_ad",
            "all_test_rmse",
            "all_test_mae",
            "within_ad_rmse",
            "within_ad_mae",
        ]
        for col, value in zip(columns, expected):
            assert_close(row[col], value, f"{assay} AD {col}", tol=1e-3)

    yr = pd.read_csv(ROOT / "data/04_results/y_randomization_summary.csv")
    for assay, observed_r2, rand_max in [("DPPH", 0.846, 0.19), ("ABTS", 0.920, 0.24)]:
        row = yr[yr["assay"] == assay].iloc[0]
        assert_close(row["observed_r2"], observed_r2, f"{assay} y-rand observed", tol=1e-3)
        assert_close(row["rand_max"], rand_max, f"{assay} y-rand max", tol=1e-3)
        if str(row["empirical_p"]) != "<0.002":
            raise AssertionError(f"{assay} empirical p should be <0.002")


def check_bace_metrics() -> None:
    results = pd.read_csv(ROOT / "external_validation/results/bace_external_validation_results.csv")
    with (ROOT / "external_validation/results/bace_predictions.json").open(encoding="utf-8") as handle:
        predictions = json.load(handle)

    y_test = np.asarray(predictions["y_test"], dtype=float)
    json_method_map = {
        "SVR-Tanimoto": "SVR-Tanimoto",
        "SVR-Dice": "SVR-Dice",
        "V2-A: Dice-MKL": "V2-A: Dice-MKL",
        "V2-B: AdaptiveKernel (Dice)": "V2-B: AdaptiveKernel",
        "V2-C: StackedKernel": "V2-C: StackedKernel",
        "V2-D: DomainAdapted": "V2-D: DomainAdapted",
        "XGBoost": "XGBoost",
        "RF": "RF",
    }

    if "V2-E: Consensus" not in predictions:
        base_methods = [key for key in json_method_map.values() if key in predictions]
        predictions["V2-E: Consensus"] = np.mean(
            [np.asarray(predictions[key], dtype=float) for key in base_methods], axis=0
        ).tolist()
    json_method_map["V2-E: Consensus"] = "V2-E: Consensus"

    for method, (exp_r2, exp_rmse, exp_mae) in EXPECTED_BACE.items():
        row = results[results["method"] == method].iloc[0]
        assert_close(row["R2"], exp_r2, f"{method} CSV R2")
        assert_close(row["RMSE"], exp_rmse, f"{method} CSV RMSE")
        assert_close(row["MAE"], exp_mae, f"{method} CSV MAE")

        if method == "Chemprop" and method not in predictions:
            continue
        pred_key = json_method_map.get(method, method)
        y_pred = np.asarray(predictions[pred_key], dtype=float)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        assert_close(r2, exp_r2, f"{method} JSON R2")
        assert_close(rmse, exp_rmse, f"{method} JSON RMSE")
        assert_close(mae, exp_mae, f"{method} JSON MAE")


def main() -> None:
    check_required_files()
    check_dataset_counts()
    check_performance_table()
    check_supporting_tables()
    check_ad_and_y_randomization()
    check_bace_metrics()
    print("OK: released repository files support the manuscript and SI summary values.")


if __name__ == "__main__":
    main()
