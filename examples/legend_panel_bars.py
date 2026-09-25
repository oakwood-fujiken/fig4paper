"""Per-dataset bars with a dedicated legend panel and grouped datasets
(ports of figure_Cflows/plot_comparison_GeneRegulatory.py and plot_comparison_Trajectory.py)."""
import numpy as np

import figures4papers as fp

methods = ["TrajectoryNet", "OT-CFM", "SB-CFM", "BEMIOflow (ours)"]
colors = [fp.PALETTE["green_1"], fp.PALETTE["green_2"], fp.PALETTE["green_3"], fp.PALETTE["blue_secondary"]]
results = {
    "PHATE Space RMSE": {
        "Bifurcation": np.array([6.16, 2.98, 2.83, 2.60]) * 1e-3,
        "Cycle": np.array([2.49, 3.79, 1.58, 0.777]) * 1e-3,
        "Unidirectional": np.array([8.08, 5.33, 5.67, 3.40]) * 1e-3,
    },
    "Gene Space RMSE": {
        "Bifurcation": np.array([0.150, 0.0842, 0.0835, 0.0773]),
        "Cycle": np.array([0.151, 0.209, 0.141, 0.118]),
        "Unidirectional": np.array([0.118, 0.0960, 0.0978, 0.0853]),
    },
    "Interpolation EMD": {
        "Bifurcation": np.array([1.07, 0.495, 0.516, 0.465]) * 1e-2,
        "Cycle": np.array([0.710, 0.818, 0.526, 0.465]) * 1e-2,
        "Unidirectional": np.array([1.50, 1.32, 1.28, 0.935]) * 1e-2,
    },
}

if __name__ == "__main__":
    fp.apply_publication_style("bar")
    fig, axes = fp.create_subplots(1, 4, figsize=(36, 6))
    for ax, (metric, groups) in zip(axes, results.items()):
        fp.make_clustered_bars(ax, groups, colors=colors, labels=methods)
        fp.style_axis(ax, xlabel="Dataset", ylabel=metric, label_size=36, sci_y=True)
    fp.legend_panel(axes[-1], source=axes[0], loc="lower left", fontsize=30)
    fp.finalize_figure(fig, "figures/legend_panel_bars")
