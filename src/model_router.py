"""
Model router for per-assay champion inference.

Usage:
    router = ModelRouter()
    y_pred = router.predict(assay="DPPH", smiles=["CCO", ...])
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.optimized_svr_v2 import (
    generate_maccs_keys,
    generate_saponin_domain_features,
    tanimoto_kernel,
)


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_DIR = ROOT / "results" / "champion"
UNCERTAINTY_DIR = ROOT / "results" / "uncertainty"


def morgan_ecfp4(smiles_list: List[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append([0] * nbits)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        fps.append(list(fp))
    return np.asarray(fps, dtype=float)


class ModelRouter:
    def __init__(self) -> None:
        self.champions: Dict[str, str] = {}
        self.assay_similarity_thresholds: Dict[str, float] = {}
        for assay in ["DPPH", "ABTS", "FRAP"]:
            p = CHAMPION_DIR / f"{assay}_champion.json"
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    self.champions[assay] = json.load(f)["selected_model"]

        ad_file = UNCERTAINTY_DIR / "applicability_domain.csv"
        if ad_file.exists():
            with ad_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    assay = row.get("assay")
                    thr = row.get("similarity_threshold")
                    if assay and thr and assay not in self.assay_similarity_thresholds:
                        self.assay_similarity_thresholds[assay] = float(thr)

    def _predict_svr_tanimoto(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        K_train = tanimoto_kernel(X_train)
        K_test = tanimoto_kernel(X_test, X_train)
        model = SVR(kernel="precomputed", C=10.0, epsilon=0.05)
        model.fit(K_train, y_train)
        return model.predict(K_test)

    def _predict_bayesian_ridge(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = BayesianRidge()
        model.fit(X_train_s, y_train)
        return model.predict(X_test_s)

    def _predict_rf(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        model = RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def _predict_domain_adapted(self, X_train: np.ndarray, y_train: np.ndarray, smiles_train: List[str], X_test: np.ndarray, smiles_test: List[str]) -> np.ndarray:
        X_maccs_tr = generate_maccs_keys(smiles_train)
        X_maccs_te = generate_maccs_keys(smiles_test)
        X_sap_tr = generate_saponin_domain_features(smiles_train)
        X_sap_te = generate_saponin_domain_features(smiles_test)

        # Fixed weights from best-known setting for stable inference.
        w1, w2, w3 = 0.6, 0.2, 0.2
        scaler = StandardScaler()
        Xs_tr = scaler.fit_transform(X_sap_tr)
        Xs_te = scaler.transform(X_sap_te)

        from src.optimized_svr_v2 import dice_kernel, rbf_kernel

        K1_tr = dice_kernel(X_train)
        K1_te = dice_kernel(X_test, X_train)
        K2_tr = tanimoto_kernel(X_maccs_tr)
        K2_te = tanimoto_kernel(X_maccs_te, X_maccs_tr)
        K3_tr = rbf_kernel(Xs_tr)
        K3_te = rbf_kernel(Xs_te, Xs_tr)

        K_tr = w1 * K1_tr + w2 * K2_tr + w3 * K3_tr
        K_te = w1 * K1_te + w2 * K2_te + w3 * K3_te

        model = SVR(kernel="precomputed", C=10.0, epsilon=0.05)
        model.fit(K_tr, y_train)
        return model.predict(K_te)

    def predict(
        self,
        assay: str,
        smiles: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        smiles_train: List[str],
    ) -> np.ndarray:
        """
        Predict values for an assay using selected champion model.

        Notes:
        - Router assumes caller provides the training set (X_train, y_train, smiles_train).
        - This keeps the router stateless and easy to test.
        """
        if assay not in self.champions:
            raise ValueError(f"No champion selected for assay={assay}. Run scripts/m1_select_champions.py first.")

        model_name = self.champions[assay]
        X_test = morgan_ecfp4(smiles)

        if model_name == "SVR-Tanimoto":
            return self._predict_svr_tanimoto(X_train, y_train, X_test)
        if model_name == "BayesianRidge":
            return self._predict_bayesian_ridge(X_train, y_train, X_test)
        if model_name == "RF":
            return self._predict_rf(X_train, y_train, X_test)
        if model_name == "V2-DomainAdapted":
            return self._predict_domain_adapted(X_train, y_train, smiles_train, X_test, smiles)

        raise ValueError(f"Unsupported champion model: {model_name}")

    def predict_with_guardrails(
        self,
        assay: str,
        smiles: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        smiles_train: List[str],
        similarity_threshold: Optional[float] = None,
        blend_strength: float = 0.35,
    ) -> Dict[str, object]:
        """
        Conservative prediction mode for OOD safety.

        If max Tanimoto similarity to training set is below threshold, blend champion
        prediction with a conservative fallback estimate.
        """
        X_test = morgan_ecfp4(smiles)
        raw = self.predict(assay, smiles, X_train, y_train, smiles_train)

        K = tanimoto_kernel(X_test, X_train)
        max_sim = K.max(axis=1)

        thr = similarity_threshold
        if thr is None:
            thr = self.assay_similarity_thresholds.get(assay, 0.5)

        # Fallback: Bayesian ridge prediction + train mean anchor.
        br_pred = self._predict_bayesian_ridge(X_train, y_train, X_test)
        mean_anchor = np.full(len(raw), float(np.mean(y_train)))
        fallback = 0.5 * br_pred + 0.5 * mean_anchor

        adjusted = raw.copy()
        warnings: List[str] = []
        for i, s in enumerate(max_sim):
            if s < thr:
                adjusted[i] = (1.0 - blend_strength) * raw[i] + blend_strength * fallback[i]
                warnings.append("OUT_OF_APPLICABILITY_DOMAIN")
            else:
                warnings.append("")

        return {
            "pred_raw": raw,
            "pred_adjusted": adjusted,
            "max_similarity": max_sim,
            "threshold": float(thr),
            "warnings": warnings,
        }
