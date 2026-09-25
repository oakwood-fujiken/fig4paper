"""Ablation bars with in-bar values and drop arrows (port of figure_CellSpliceNet/plot_ablation.py)."""
import numpy as np
from matplotlib import pyplot as plt

import figures4papers as fp

methods = ["CellSpliceNet", "No Expression", "No Structure", "No ROI", "No Sequence"]
colors = ["#0F4D92", "#B4E6B4", "#AFE6E6", "#FFE080", "#D3D3D3"]
result = np.array([0.88, 0.84, 0.82, 0.81, 0.74])

if __name__ == "__main__":
    fp.apply_publication_style("bar")
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_ylim(0, result.max() + 0.5)
    bars = fp.make_bars(ax, result, colors=colors, labels=methods, annotate=True,
                        annotate_kwargs=dict(position="inside", fontsize=32, padding=0.03))
    fp.add_delta_arrows(ax, bars, reference=0, fontsize=24)
    fp.style_axis(ax, ylabel="Spearman correlation", label_size=54, yticks=[0, 0.25, 0.5, 0.75, 1.0],
                  tick_size=36, tick_length=10, tick_width=2)
    ax.legend(bbox_to_anchor=(0.5, 1.08), loc="upper left", fontsize=36)
    fp.finalize_figure(fig, "figures/ablation_bars")
