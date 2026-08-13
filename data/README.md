# MK-Ensemble Dataset

This directory contains the curated datasets and result tables needed to reproduce the repository-level analyses.

## Directory Structure

```text
data/
|-- 01_dataset/          Main activity dataset and scaffold split metadata
|-- 02_structures/       3D molecular structures in SDF format
|-- 03_targets/          Predicted targets and antioxidant pathway gene sets
|-- 04_results/          Model predictions and pathway enrichment results
|-- Table1_dataset_statistics.csv
|-- Table2_model_performance.csv
|-- Table3_docking_summary.csv
`-- LICENSE.txt
```

## Dataset Statistics

- Total unique compounds: 91
- Total activity records: 128
- DPPH assay records: 70
- ABTS assay records: 42
- FRAP assay records: 16
- Predicted protein targets: 127

## Key Files

| File | Description |
|---|---|
| `01_dataset/antioxidant_dataset.csv` | Main activity dataset |
| `01_dataset/saponins_annotated.csv` | Compound annotations |
| `01_dataset/scaffold_split.json` | Split metadata |
| `02_structures/saponins_3d.sdf` | 3D molecular structures |
| `03_targets/targets_predicted.csv` | Predicted protein targets |
| `03_targets/antioxidant_pathway_genes.csv` | Antioxidant pathway gene sets |
| `04_results/model_predictions.csv` | Model predictions for all compounds |
| `04_results/pathway_enrichment.csv` | KEGG/Reactome enrichment results |
| `04_results/applicability_domain_summary.csv` | AD counts and AD-stratified prediction errors |
| `04_results/train_cv_performance.csv` | Training-set versus CV performance used in Table S2 |
| `04_results/ablation_summary.csv` | Stage-wise ablation results used in Table S3 |
| `04_results/frap_exploratory_results.csv` | FRAP exploratory results used in Table S4 |
| `04_results/statistical_model_comparison.csv` | Pairwise model comparison used in Table S5 |
| `04_results/bace_train_test_gap.csv` | BACE train-test gap summary used in Table S6 |
| `04_results/y_randomization_summary.csv` | Response-permutation summary used in Figure S10/Table S9 |
| `04_results/fragment_attribution_validation.csv` | IG/SHAP/bootstrap/permutation attribution summary |
| `04_results/learning_curve_summary.csv` | Learning-curve values used in Figure S8 |
| `04_results/descriptor_importance_summary.csv` | Top Mordred descriptor importance values used in Table S10 |
| `04_results/descriptor_class_importance.csv` | Descriptor-class importance distribution used in Table S10 |
| `Table1_dataset_statistics.csv` | Dataset summary |
| `Table2_model_performance.csv` | Model performance summary |
| `Table3_docking_summary.csv` | Docking summary |

## License

Datasets are released under CC-BY-4.0. See `LICENSE.txt`.
