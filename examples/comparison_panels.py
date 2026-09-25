"""Multi-metric comparison with mean±std over runs (port of figure_CellSpliceNet/plot_comparison.py)."""
import numpy as np

import figures4papers as fp

methods = ["CellSpliceNet", "Pangolin", "SpliceTransformer", "SpliceAI", "SpliceFinder", "ViT", "AlphaGenome", "ESM2"]
# (n_methods, n_runs) per metric: the library reduces to mean and std.
results = {
    "Spearman correlation": np.array([
        [0.909, 0.872, 0.906], [0.813, 0.825, 0.832], [0.786, 0.799, 0.815], [0.751, 0.715, 0.753],
        [0.725, 0.715, 0.752], [0.689, 0.722, 0.705], [0.637, 0.637, 0.636], [0.642, 0.649, 0.596]]),
    "Pearson correlation": np.array([
        [0.929, 0.863, 0.918], [0.846, 0.867, 0.870], [0.827, 0.844, 0.862], [0.751, 0.708, 0.768],
        [0.728, 0.721, 0.761], [0.694, 0.723, 0.713], [0.646, 0.648, 0.645], [0.616, 0.635, 0.599]]),
    r"R$^2$ score": np.array([
        [0.863, 0.745, 0.843], [0.634, 0.672, 0.689], [0.604, 0.609, 0.714], [0.542, 0.483, 0.532],
        [0.354, 0.359, 0.440], [0.375, 0.415, 0.347], [0.399, 0.401, 0.401], [0.353, 0.387, 0.348]]),
}

if __name__ == "__main__":
    fp.apply_publication_style("bar")
    colors = fp.ours_vs_baselines(len(methods) - 1)
    fig, axes = fp.metric_panels(results, methods, colors, legend="first", annotate=True,
                                 panel_size=(15, 12), ylabel_size=54, capsize=15,
                                 annotate_kwargs=dict(fontsize=32),
                                 legend_kwargs=dict(bbox_to_anchor=(0.02, 1.08), loc="upper left",
                                                    fontsize=36, ncols=2, columnspacing=0.6))
    for ax in axes:
        fp.style_axis(ax, ylim=(0, 1.5), yticks=[0, 0.25, 0.5, 0.75, 1.0], tick_size=36, tick_length=10, tick_width=2)
    fp.finalize_figure(fig, "figures/comparison_panels")
