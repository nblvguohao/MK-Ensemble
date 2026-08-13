#!/usr/bin/env python3
"""Generate Figure for BACE external validation results."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR

df = pd.read_csv(RESULTS_DIR / "bace_external_validation_results.csv")

# Sort by R2 descending
df = df.sort_values('R2', ascending=True)

colors = []
for method in df['method']:
    if 'MK-Ensemble' in method or method.startswith('V2-'):
        colors.append('#2E86AB')  # MK-Ensemble blue
    elif method == 'Chemprop':
        colors.append('#A23B72')  # Chemprop purple
    else:
        colors.append('#F18F01')  # Baselines orange

# Better color mapping
method_colors = {
    'RF': '#F18F01',
    'XGBoost': '#F18F01',
    'SVR-Tanimoto': '#F18F01',
    'SVR-Dice': '#F18F01',
    'Chemprop': '#A23B72',
    'V2-A: Dice-MKL': '#2E86AB',
    'V2-B: AdaptiveKernel (Dice)': '#2E86AB',
    'V2-C: StackedKernel': '#2E86AB',
    'V2-D: DomainAdapted': '#2E86AB',
    'V2-E: Consensus': '#1B4965',
}
colors = [method_colors.get(m, '#888888') for m in df['method']]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(df['method'], df['R2'], color=colors, edgecolor='white', height=0.7)

# Add value labels
for bar, r2, rmse in zip(bars, df['R2'], df['RMSE']):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f"{r2:.3f}", va='center', ha='left', fontsize=9)

ax.set_xlabel("Test $R^2$", fontsize=12)
ax.set_title("External validation: BACE pIC50 regression\n(scaffold split: $n_{\\mathrm{train}}=203$, $n_{\\mathrm{test}}=1265$)", fontsize=12)
ax.set_xlim(0, max(df['R2']) * 1.22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#F18F01', label='Baseline methods'),
    Patch(facecolor='#A23B72', label='Chemprop (MPNN)'),
    Patch(facecolor='#2E86AB', label='MK-Ensemble variants'),
    Patch(facecolor='#1B4965', label='MK-Ensemble Consensus'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure6_bace_external_validation.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURE_DIR / "Figure6_bace_external_validation.png", dpi=300, bbox_inches='tight')
print(f"Figure saved to {FIGURE_DIR / 'Figure6_bace_external_validation.pdf'}")
