# Accepted-paper alignment and provenance log

This document records how the public repository maps to the accepted *Journal of Cheminformatics* manuscript (DOI `10.1186/s13321-026-01270-x`). It is intended to preserve provenance while making version differences explicit.

## Principle

The repository must not be made to “match” the manuscript by manually overwriting source data or computed outputs without provenance. Where the accepted manuscript and current public artifacts disagree, both are recorded and the discrepancy is left open until the original analysis artifact or generation script is recovered.

## Alignment status

| Topic | Accepted manuscript | Current public repository | Status / action |
|---|---|---|---|
| Publication status | Accepted 22 Jul 2026, DOI 10.1186/s13321-026-01270-x | README previously said “submitted” | **Fixed in README** |
| Primary dataset size | 91 unique compounds, 128 activity records; DPPH 70, ABTS 42, FRAP 16 | `antioxidant_dataset.csv` and verification script report the same totals | **Aligned** |
| Dataset composition | 24 steroidal saponins + 67 reference antioxidants | Current released annotation/split artifacts do not unambiguously reconstruct all figure-level category counts | **Open; do not alter raw data without provenance** |
| Figure 1 category count | Manuscript text describes 24+67; Figure 1A displays 16+75 | No exact Figure 1 source-generation artifact is released | **Manuscript-internal mismatch; open** |
| Molecular-weight mean | Manuscript text: 604.8 ± 156.3 Da; Figure 1C displays 408.1 Da | No exact Figure 1 source-generation artifact is released | **Manuscript-internal mismatch; open** |
| DPPH headline performance | R² 0.846, RMSE 0.154, MAE 0.121, Q²CV 0.831 | `data/Table2_model_performance.csv` agrees | **Aligned** |
| ABTS headline performance | R² 0.920, RMSE 0.089, MAE 0.068, Q²CV 0.907 | `data/Table2_model_performance.csv` agrees for MK-Ensemble | **Aligned** |
| ABTS XGBoost | Manuscript Table 1: MAE 0.098, Q²CV 0.826 | Released summary: MAE 0.094, Q²CV 0.829 | **Open version difference** |
| Four-stage method | Base multi-kernel → hybrid descriptors → domain-adapted → nested Ridge stacking | Kernel/domain/stacking utilities exist, but repository also contains a separate GIN/MoE development model | **README now separates published pipeline from experimental code** |
| Integrated Gradients | Manuscript describes attribution on an intermediate differentiable fragment representation and notes approximation relative to SVR | `src/explainability.py` operates on the GIN/MoE development model; released `fragment_attribution_validation.csv` is only a summary | **Open; exact paper attribution pipeline/raw matrix not fully archived** |
| Figure 3 attribution matrix | Figure-level per-compound/per-fragment values | Only summary statistics are public | **Open; raw matrix required for full reproduction** |
| Figure 4 | Manuscript figure/caption contains compound-label inconsistencies | No exact Figure 4 source-generation artifact is public | **Manuscript figure-version issue; open** |
| Pathway enrichment / Figure 5 | Manuscript Figure 5 visually labels several non-Nrf2 pathways with 1 overlapping gene | `pathway_enrichment.csv` currently has overlap=0 and FDR=1 for those pathways; Nrf2/HO-1 remains overlap=4, FDR=0.003 | **High-priority open discrepancy; do not edit CSV without original enrichment output** |
| BACE R² ranking | SVR-Tanimoto ≈0.363; Consensus ≈0.326 | Released predictions/results reproduce the same ranking and R² | **Aligned** |
| BACE RMSE | Manuscript Table 2 prints 0.781 (SVR-Tanimoto), 0.804 (Consensus) | Released predictions recompute to ≈1.018 and ≈1.047 | **High-priority open scale/version discrepancy** |
| Applicability domain | Manuscript reports DPPH 3/7 within AD and ABTS 3/4 within AD | `applicability_domain_summary.csv` currently reports DPPH 8/9 and ABTS 5/6 | **High-priority open version discrepancy** |
| Y-randomization | DPPH observed 0.846, null max 0.19; ABTS observed 0.920, null max 0.24; 500 permutations | `y_randomization_summary.csv` agrees | **Aligned at summary level** |

## What is safe to treat as paper-aligned now

The following currently have a clear manuscript-to-repository mapping at summary level:

- 91 unique compounds / 128 activity records.
- DPPH n=70, ABTS n=42, FRAP n=16.
- MK-Ensemble DPPH headline metrics.
- MK-Ensemble ABTS headline metrics.
- Stage-wise ablation summary values currently released.
- Y-randomization summary values currently released.
- BACE method ranking and R² values for the principal methods.
- Nrf2/HO-1 enrichment summary (4 overlapping genes, FDR=0.003).

## What must not be “fixed” by hand

Do **not** manually change the following source/result files merely to reproduce the printed manuscript value:

- `data/04_results/pathway_enrichment.csv`
- `data/04_results/applicability_domain_summary.csv`
- `external_validation/results/bace_external_validation_results.csv`
- `external_validation/results/bace_predictions.json`
- raw activity values in `data/01_dataset/antioxidant_dataset.csv`

A change to any of these should be backed by one of:

1. the original raw analysis output,
2. an executable generation script that regenerates the corrected file from the archived inputs, or
3. a documented correction to the accepted manuscript.

## Missing provenance artifacts to recover

To complete exact accepted-paper reproduction, the archive still needs, if they exist:

1. Figure 1 generation script and exact input table used for the 16/75 (or 24/67) category display and molecular-weight mean.
2. Full per-compound/per-fragment attribution matrix used for Figure 3, plus the exact model checkpoint/configuration used for IG/SHAP.
3. Figure 4 generation source and exact compound/fragment mapping.
4. Original pathway enrichment output that generated Figure 5, including the gene-set database/version and background universe.
5. Original AD test split and per-compound leverage/residual table used for the manuscript’s 3/7 and 3/4 counts.
6. BACE preprocessing/label-scaling metadata that explains why the manuscript RMSE differs from the prediction-derived RMSE while R² agrees.
7. Exact fold assignments and fold-level predictions used for the reported corrected resampled t-tests and Bayesian comparisons.

## Resolution policy

When a missing artifact is recovered, update this file first with its provenance (filename, generating script, commit, and relevant random seed), then update the affected summary table. This keeps the public record auditable and avoids introducing a second undocumented version.
