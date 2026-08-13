# MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable SAR Modeling

A computational framework combining multi-kernel support vector regression, fragment-level attribution, and stacking ensembles for small-sample natural product SAR datasets.

## Overview

MK-Ensemble is an interpretable machine learning framework for structure-activity relationship (SAR) modeling in data-scarce regimes (n < 100). It integrates Dice, Tanimoto, and RBF kernels with fragment attribution to provide predictive performance and chemically interpretable outputs.

This repository contains the implementation, curated datasets, and reproducibility code for:

> MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable Structure-Activity Relationship Modeling of Steroidal Saponins
>
> Guohao Lv, Yingchun Xia, Huichao Liu, Xiaolei Zhu, Shuai Yang, Qingyong Wang, Lichuan Gu
>
> Journal of Cheminformatics, 2026 (submitted)

## Methodological Innovation

### Core Features

1. Multi-kernel integration using Dice (Morgan), Tanimoto (MACCS), and RBF (physicochemical) kernels
2. Fragment-level interpretability through BRICS decomposition and attribution scoring
3. Small-sample optimization for natural product datasets
4. Validation across internal SAR analysis, external BACE benchmarking, and mechanistic evidence

### Performance

| Assay | n | R^2 | RMSE |
|---|---:|---:|---:|
| DPPH | 70 | 0.846 | 0.154 |
| ABTS | 42 | 0.920 | 0.089 |
| FRAP | 16 | 0.779 | 0.140* |

*Exploratory because of the small FRAP sample size.

## Validation and Findings

### Fragment Contribution Analysis

Integrated Gradients attribution on steroidal saponins is implemented in `src/explainability.py`. Summary validation outputs used by the revised supporting information are included in `data/04_results/fragment_attribution_validation.csv`.

### External Validation on MoleculeNet BACE

The external validation uses the BACE pIC50 regression benchmark with the predefined scaffold split and n_train = 203.

| Method | Test R^2 |
|---|---:|
| SVR-Tanimoto | 0.363 |
| MK-Ensemble (Consensus) | 0.326 |
| Random Forest | 0.245 |
| Chemprop (MPNN) | 0.194 |

The curated external validation files are included under `external_validation/`. Intermediate Chemprop checkpoints, trainer logs, and serialized feature caches are intentionally excluded from version control.

### Mechanistic Validation

- Nrf2/HO-1 pathway enrichment: fold enrichment = 4.2, FDR = 0.003, with 4 overlapping genes in `data/04_results/pathway_enrichment.csv`
- Keap1 docking summary: best binding energy = -11.137 kcal/mol in `data/Table3_docking_summary.csv`
- Molecular dynamics: 100 ns simulation supports receptor-ligand stability

## Repository Structure

```text
mk_ensemble/
|-- data/                         Curated datasets and reported tables
|   |-- 01_dataset/               Saponin activity dataset and splits
|   |-- 02_structures/            3D molecular structures
|   |-- 03_targets/               Predicted antioxidant targets
|   `-- 04_results/               Source data for SI tables and figures
|-- external_validation/          MoleculeNet BACE validation
|   |-- data/bace.csv
|   |-- results/
|   |   |-- bace_external_validation_results.csv
|   |   `-- bace_predictions.json
|   |-- run_bace_external_validation.py
|   |-- generate_bace_figure.py
|   `-- README.md
|-- src/                          Source code
|-- requirements.txt              Python dependencies
|-- LICENSE                       MIT License
`-- README.md                     This file
```

## Quick Start

### Installation

```bash
git clone https://github.com/nblvguohao/MK-Ensemble.git
cd MK-Ensemble
pip install -r requirements.txt
```

### Verify Reproducibility

```bash
python scripts/verify_reproducibility.py
```

The verification script checks the released datasets, source tables, BACE
prediction JSON, and manuscript-level summary metrics. Development and modeling
scripts are retained in `src/`; heavyweight checkpoints, cached fingerprints,
and Chemprop trainer outputs are excluded from version control.

### Run External Validation

```bash
cd external_validation
python run_bace_external_validation.py
python generate_bace_figure.py
```

## Dataset

- Compounds: 91 unique molecules
- Activity records: 128
- Assays: DPPH (70), ABTS (42), FRAP (16 exploratory)
- Features: Morgan fingerprints, MACCS keys, and physicochemical descriptors

| File | Description | Location |
|---|---|---|
| antioxidant_dataset.csv | Main activity dataset | data/01_dataset/ |
| saponins_annotated.csv | Compound annotations | data/01_dataset/ |
| scaffold_split.json | Train/validation/test split metadata | data/01_dataset/ |
| saponins_3d.sdf | 3D structures | data/02_structures/ |
| targets_predicted.csv | Predicted targets | data/03_targets/ |
| pathway_enrichment.csv | KEGG/Reactome enrichment results | data/04_results/ |
| applicability_domain_summary.csv | AD counts and AD-stratified prediction errors | data/04_results/ |
| train_cv_performance.csv | Training-set versus CV performance used in Table S2 | data/04_results/ |
| ablation_summary.csv | Stage-wise ablation results used in Table S3 | data/04_results/ |
| frap_exploratory_results.csv | FRAP exploratory results used in Table S4 | data/04_results/ |
| statistical_model_comparison.csv | Pairwise model comparison used in Table S5 | data/04_results/ |
| bace_train_test_gap.csv | BACE train-test gap summary used in Table S6 | data/04_results/ |
| y_randomization_summary.csv | Response-permutation summary used in Figure S10/Table S9 | data/04_results/ |
| fragment_attribution_validation.csv | IG/SHAP/bootstrap/permutation attribution summary | data/04_results/ |
| learning_curve_summary.csv | Learning-curve values used in Figure S8 | data/04_results/ |
| descriptor_importance_summary.csv | Top Mordred descriptor importance values used in Table S10 | data/04_results/ |
| descriptor_class_importance.csv | Descriptor-class importance distribution used in Table S10 | data/04_results/ |
| bace.csv | MoleculeNet BACE dataset | external_validation/data/ |
| bace_external_validation_results.csv | External validation metrics | external_validation/results/ |
| bace_predictions.json | External validation predictions | external_validation/results/ |

## Computational Methods

The multi-kernel SVR combines Morgan, MACCS, and physicochemical feature spaces:

```text
K_combined = w1 * K_Dice(Morgan) + w2 * K_Tanimoto(MACCS) + w3 * K_RBF(physicochemical)
```

Weights were selected through nested cross-validation:

- DPPH: w1 = 0.55, w2 = 0.30, w3 = 0.15
- ABTS: w1 = 0.60, w2 = 0.25, w3 = 0.15
- FRAP: w1 = 0.50, w2 = 0.35, w3 = 0.15

The revised stacking evaluation uses an outer evaluation loop in which the Ridge
meta-learner is trained only on meta-features generated inside the outer
training partition. The outer held-out fold is predicted after refitting the
base learners on the outer training partition, preventing the meta-learner from
being evaluated on the same pooled out-of-fold predictions used to fit it.

## Requirements

- Python >= 3.10
- RDKit >= 2022.03.1
- scikit-learn >= 1.0
- PyTorch >= 1.10
- SHAP >= 0.40

See `requirements.txt` for the full dependency list.

## Citation

```bibtex
@article{mkensemble2026,
  title={MK-Ensemble: Fragment-based Multi-Kernel Ensemble for Interpretable Structure-Activity Relationship Modeling of Steroidal Saponins},
  author={Lv, Guohao and Xia, Yingchun and Liu, Huichao and Zhu, Xiaolei and Yang, Shuai and Wang, Qingyong and Gu, Lichuan},
  journal={Journal of Cheminformatics},
  year={2026}
}
```

## License

MIT License. See `LICENSE` for details.
