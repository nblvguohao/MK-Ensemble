# External Validation: MoleculeNet BACE

This directory contains the curated external validation experiment for MK-Ensemble on the MoleculeNet BACE pIC50 regression benchmark.

## Purpose

The experiment tests whether the MK-Ensemble framework generalizes beyond the steroidal saponin dataset to a different chemical space, biological target, and scaffold-based train/test split.

## Dataset

- Source: MoleculeNet BACE
- Task: pIC50 regression
- Split: predefined scaffold split
- Training size: 203 compounds
- Test size: 1265 compounds
- Data file: `data/bace.csv`

## Files Included in Git

| File | Description |
|---|---|
| `run_bace_external_validation.py` | Main evaluation script |
| `generate_bace_figure.py` | Figure generation script |
| `data/bace.csv` | MoleculeNet BACE dataset |
| `results/bace_external_validation_results.csv` | Final numerical metrics |
| `results/bace_predictions.json` | Final test-set predictions for baselines and MK-Ensemble variants plus `y_test` |

Intermediate Chemprop checkpoints, trainer logs, event files, serialized feature caches, and debug outputs are excluded from version control. The released metrics CSV includes the Chemprop result; the prediction JSON contains the lightweight scikit-learn/MK-Ensemble predictions, and the consensus prediction is reproducible as their mean.

## Key Result

| Method | Test R^2 |
|---|---:|
| SVR-Tanimoto | 0.363 |
| MK-Ensemble (Consensus) | 0.326 |
| Random Forest | 0.245 |
| Chemprop (MPNN) | 0.194 |

## Reproduce

```bash
python run_bace_external_validation.py
python generate_bace_figure.py
```
