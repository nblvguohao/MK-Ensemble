# External Validation: MoleculeNet BACE

This directory contains the public BACE external-validation artifacts associated with the accepted MK-Ensemble manuscript.

## Dataset

- Source: MoleculeNet BACE
- Task: pIC50 regression
- Split: scaffold split used by the released experiment
- Training size: 203 compounds
- Test size: 1265 compounds
- Data file: `data/bace.csv`

## Released files

| File | Description |
|---|---|
| `run_bace_external_validation.py` | Evaluation script |
| `generate_bace_figure.py` | Figure generation script |
| `data/bace.csv` | BACE dataset |
| `results/bace_external_validation_results.csv` | Released numerical metrics |
| `results/bace_predictions.json` | Released test-set predictions and `y_test` |

Intermediate Chemprop checkpoints, trainer logs, serialized feature caches, and debug outputs are not included in the public archive.

## R² alignment with the accepted manuscript

The main method ranking and R² values agree with the accepted manuscript to rounding:

| Method | Accepted-manuscript R² | Released-result R² |
|---|---:|---:|
| SVR-Tanimoto | 0.363 | 0.363158... |
| MK-Ensemble (Consensus) | 0.326 | 0.326289... |
| Random Forest | 0.245 | 0.244796... |
| Chemprop (MPNN) | 0.194 | 0.193804... |

## RMSE provenance warning

The accepted manuscript Table 2 prints RMSE values such as **0.781** for SVR-Tanimoto and **0.804** for MK-Ensemble Consensus. The currently released prediction file, when evaluated on its stored `y_test`, yields approximately **1.018** and **1.047**, respectively.

Because the R² values agree while the RMSE values do not, this likely reflects a **label-scaling, inverse-transform, or analysis-version difference**. The released predictions/results are intentionally left unchanged until the original preprocessing/scaling metadata for the accepted-manuscript Table 2 is recovered.

Do not manually edit the result CSV to the printed RMSE values without provenance.

For the complete audit trail, see `../PAPER_ALIGNMENT.md` and `../data/paper_reported_values.csv`.

## Reproduce the released external-validation artifacts

```bash
python run_bace_external_validation.py
python generate_bace_figure.py
```

These commands reproduce the **released experiment version**. Exact reproduction of every accepted-manuscript Table 2 RMSE value requires the missing preprocessing/version provenance noted above.
