# MK-Ensemble Dataset and Result Artifacts

This directory contains the public datasets and summary result tables associated with the accepted *Journal of Cheminformatics* article (DOI `10.1186/s13321-026-01270-x`).

## Important provenance note

The files under `data/` preserve the currently released data and computed summaries. They are **not overwritten merely to match a printed manuscript number** when the corresponding raw provenance is missing. Accepted-paper values are recorded separately in `paper_reported_values.csv`, and unresolved differences are documented in the repository-level `PAPER_ALIGNMENT.md`.

## Directory structure

```text
data/
|-- 01_dataset/                 Main activity data and split metadata
|-- 02_structures/              3D molecular structures
|-- 03_targets/                 Predicted targets and pathway inputs
|-- 04_results/                 Released model/statistical summaries
|-- paper_reported_values.csv   Accepted-paper values and alignment status
|-- Table1_dataset_statistics.csv
|-- Table2_model_performance.csv
|-- Table3_docking_summary.csv
`-- LICENSE.txt
```

## Public dataset statistics

- Total unique compounds: 91
- Total activity records: 128
- DPPH assay records: 70
- ABTS assay records: 42
- FRAP assay records: 16
- Predicted protein targets: 127

The accepted manuscript additionally describes the dataset as **24 steroidal saponins + 67 reference antioxidants**. Because the exact Figure 1 category-generation source is not currently archived, this statement is recorded as a paper-reported value rather than being forced into the released data files.

## Key files

| File | Description |
|---|---|
| `01_dataset/antioxidant_dataset.csv` | Main activity dataset |
| `01_dataset/saponins_annotated.csv` | Compound annotations |
| `01_dataset/scaffold_split.json` | Split metadata |
| `02_structures/saponins_3d.sdf` | 3D molecular structures |
| `03_targets/targets_predicted.csv` | Predicted protein targets |
| `03_targets/antioxidant_pathway_genes.csv` | Antioxidant pathway gene sets |
| `04_results/model_predictions.csv` | Released prediction table |
| `04_results/pathway_enrichment.csv` | Released pathway-enrichment summary |
| `04_results/applicability_domain_summary.csv` | Released AD summary |
| `04_results/train_cv_performance.csv` | Training-vs-CV summary |
| `04_results/ablation_summary.csv` | Stage-wise ablation summary |
| `04_results/frap_exploratory_results.csv` | FRAP exploratory results |
| `04_results/statistical_model_comparison.csv` | Pairwise statistical comparison summary |
| `04_results/bace_train_test_gap.csv` | BACE train-test gap summary |
| `04_results/y_randomization_summary.csv` | Y-randomization summary |
| `04_results/fragment_attribution_validation.csv` | Attribution validation summary only; not the full Figure 3 matrix |
| `04_results/learning_curve_summary.csv` | Learning-curve summary |
| `04_results/descriptor_importance_summary.csv` | Top descriptor importance values |
| `04_results/descriptor_class_importance.csv` | Descriptor-class importance distribution |
| `paper_reported_values.csv` | Machine-readable accepted-paper/repository comparison |
| `Table1_dataset_statistics.csv` | Dataset summary |
| `Table2_model_performance.csv` | Released model-performance summary |
| `Table3_docking_summary.csv` | Docking summary |

## Known version-sensitive artifacts

Three areas currently require original provenance files before exact paper-level reproduction can be claimed:

1. **Figure 1** category counts and molecular-weight mean.
2. **Figure 5** non-Nrf2 pathway overlap counts.
3. **Applicability-domain** test-set counts and within-domain classifications.

See `../PAPER_ALIGNMENT.md` for the exact differences and the files needed to resolve them.

## License

Datasets are released under CC-BY-4.0. See `LICENSE.txt`.
