"""Plot final metric comparison between our method and ProteinMPNN.

The values are extracted from the final annotations shown in the provided
training/evaluation figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "Training\naccuracy",
    "Validation\naccuracy",
    "AUROC",
    "Macro\nprecision",
    "Macro\nrecall",
]

OURS = np.array([0.433, 0.446, 0.892, 0.439, 0.402])
PROTEIN_MPNN = np.array([0.362, 0.401, 0.870, 0.411, 0.320])


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    x = np.arange(len(METRICS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ours_bars = ax.bar(
        x - width / 2,
        OURS,
        width,
        label="Ours",
        color="#2f66d0",
    )
    protein_mpnn_bars = ax.bar(
        x + width / 2,
        PROTEIN_MPNN,
        width,
        label="ProteinMPNN",
        color="#e83b2e",
    )

    ax.set_title("Final Metric Comparison: Ours vs ProteinMPNN")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in (ours_bars, protein_mpnn_bars):
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    output_path = output_dir / "final_metric_comparison.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
