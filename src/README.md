# Source-code map for the accepted MK-Ensemble paper

The `src/` directory contains both paper-related kernel/ensemble utilities and model-development prototypes. This file documents that distinction so the repository does not imply that every development module generated the accepted-paper headline results.

## Closest mapping to the accepted-paper workflow

- `optimized_svr_v2.py`: kernel utilities and several multi-kernel/domain-aware variants.
- `model_router.py`: per-assay inference wrappers using SVR/kernel/domain-aware components.
- `ensemble_models.py`: Ridge stacking utilities and baseline ensemble experiments.
- `fragment.py`: BRICS/Murcko fragment decomposition utilities.

The accepted manuscript describes a four-stage progression: base multi-kernel SVR, hybrid descriptor enhancement, domain adaptation, and nested Ridge stacking. The files above contain components of that workflow, but the current archive does not contain a single frozen end-to-end training entry point that regenerates every accepted-paper primary result from raw inputs.

## Development / experimental neural modules

- `model.py`: GIN + router + mixture-of-experts neural model.
- `train_hybrid_fragmoe.py`: training code for the neural/fragment-MoE development branch.
- `trainer.py`: associated training utilities.
- `explainability.py`: Integrated Gradients implementation for the differentiable neural development model.

These modules should be treated as **development code**, not as proof that the accepted manuscript used a GIN/MoE architecture for its headline multi-kernel SVR results.

## Explainability provenance

The accepted manuscript reports approximate fragment attribution associated with the hybrid/paper pipeline. The current `explainability.py` computes IG through the differentiable GIN/MoE development model. Therefore, the exact paper-level Figure 3 attribution generation path is not fully represented by this file alone.

The public `data/04_results/fragment_attribution_validation.csv` contains summary validation statistics but not the complete per-compound/per-fragment matrix used in Figure 3.

## Reproducibility statement

Until the original frozen paper-training entry point, model artifacts, and figure-level intermediate outputs are recovered, repository-level checks should be interpreted as consistency checks on released artifacts rather than a complete end-to-end recreation of every manuscript figure and statistic.

See `../PAPER_ALIGNMENT.md` for the current provenance checklist.
