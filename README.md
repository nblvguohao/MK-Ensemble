# MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable SAR Modeling

Computational framework for small-sample natural-product structure-activity relationship (SAR) modeling using multi-kernel regression, fragment-level interpretation, domain-aware variants, and stacking.

## Publication

This repository accompanies the accepted article:

> **MK-Ensemble: fragment-based multi-kernel ensemble for interpretable structure-activity relationship modeling of steroidal saponins**  
> Guohao Lv, Yingchun Xia, Huichao Liu, Xiaolei Zhu, Shuai Yang, Qingyong Wang, Lichuan Gu  
> *Journal of Cheminformatics* (2026)  
> DOI: **10.1186/s13321-026-01270-x**

The accepted manuscript reports 91 unique compounds and 128 activity records across DPPH (n=70), ABTS (n=42), and FRAP (n=16). The primary headline results are R²=0.846 for DPPH and R²=0.920 for ABTS; FRAP is exploratory because of its small sample size.

## Published method versus development code

The **published analysis** is the four-stage workflow described in the accepted manuscript:

1. Base multi-kernel SVR with BRICS-derived fragment information.
2. Hybrid enhancement with molecular descriptors/fingerprints.
3. Domain-aware/domain-adapted model variants.
4. Ridge stacking with out-of-fold meta-features and nested evaluation.

The repository also contains **development/experimental neural code** (`src/model.py`, `src/train_hybrid_fragmoe.py`, and related GIN/MoE utilities). Those files document model-development experiments and **must not be interpreted as the exact architecture used to generate every headline value in the accepted manuscript**. The paper-aligned kernel/stacking utilities are primarily in `src/optimized_svr_v2.py`, `src/model_router.py`, `src/ensemble_models.py`, and `external_validation/`.

This distinction is now made explicit to avoid conflating development prototypes with the accepted-paper pipeline.

## Paper-reported performance

| Assay | n | R² | RMSE | Status |
|---|---:|---:|---:|---|
| DPPH | 70 | 0.846 | 0.154 | Primary |
| ABTS | 42 | 0.920 | 0.089 | Primary |
| FRAP | 16 | 0.779 | 0.140 | Exploratory |

For exact paper-reported values and known repository/manuscript differences, see [`PAPER_ALIGNMENT.md`](PAPER_ALIGNMENT.md).

## Repository layout

```text
MK-Ensemble/
|-- PAPER_ALIGNMENT.md             Accepted-paper provenance and discrepancy log
|-- data/
|   |-- 01_dataset/                Activity data and split metadata
|   |-- 02_structures/             Molecular structures
|   |-- 03_targets/                Predicted targets/pathway inputs
|   |-- 04_results/                Released result summaries
|   `-- paper_reported_values.csv  Values transcribed from the accepted manuscript
|-- external_validation/           BACE external-validation code/results
|-- scripts/
|   `-- verify_reproducibility.py  Repository-level consistency checks
|-- src/                            Modeling/development code
|-- requirements.txt
`-- LICENSE
```

## Reproducibility scope

Run:

```bash
python scripts/verify_reproducibility.py
```

This performs **repository-level consistency checks** on released CSV/JSON files and selected manuscript-level summary values. It is **not a full end-to-end retraining certificate**: heavyweight checkpoints, serialized feature caches, and some trainer artifacts are not part of the public archive.

Where a published number cannot currently be regenerated from the released artifact set, it is listed explicitly in `PAPER_ALIGNMENT.md` rather than silently replacing the underlying result file.

## Dataset

Current public files contain:

- 91 unique compounds
- 128 activity records
- DPPH: 70 records
- ABTS: 42 records
- FRAP: 16 records (exploratory)

The accepted manuscript additionally describes the dataset as 24 steroidal saponins plus 67 reference antioxidants. Because the currently released artifact set contains version-sensitive annotations/splits that do not unambiguously reconstruct every figure-level category count, this manuscript statement is recorded in `paper_reported_values.csv` and the remaining mismatch is documented in `PAPER_ALIGNMENT.md` rather than being forced into the source data.

## External validation: MoleculeNet BACE

The accepted manuscript reports a BACE scaffold-split experiment with n_train=203 and n_test=1265. The ranking and R² values in the released result file agree with the accepted manuscript for the main methods (for example, SVR-Tanimoto R²≈0.363 and MK-Ensemble Consensus R²≈0.326).

The currently released prediction file recomputes different RMSE values from those printed in the accepted manuscript Table 2. This unresolved scale/version discrepancy is documented in `PAPER_ALIGNMENT.md`; the prediction-derived values are retained unchanged to preserve provenance.

## Fragment attribution and mechanistic analyses

The repository releases summary attribution statistics and pathway/docking outputs. The accepted manuscript contains figure-level values/labels that are not all reconstructible from the current summary files alone. These are therefore treated as **paper-reported results** unless a raw per-fragment/per-pathway provenance file is present.

In particular, `data/04_results/fragment_attribution_validation.csv` is a summary table and is not a replacement for the full Figure 3 attribution matrix.

## Requirements

- Python >= 3.10
- RDKit >= 2022.03.1
- scikit-learn >= 1.0
- PyTorch >= 1.10
- SHAP >= 0.40

See `requirements.txt` for the full dependency list.

## Citation

```bibtex
@article{lv2026mkensemble,
  title={MK-Ensemble: fragment-based multi-kernel ensemble for interpretable structure-activity relationship modeling of steroidal saponins},
  author={Lv, Guohao and Xia, Yingchun and Liu, Huichao and Zhu, Xiaolei and Yang, Shuai and Wang, Qingyong and Gu, Lichuan},
  journal={Journal of Cheminformatics},
  year={2026},
  doi={10.1186/s13321-026-01270-x}
}
```

## License

MIT License. See `LICENSE` for code licensing; dataset licensing is described under `data/`.
