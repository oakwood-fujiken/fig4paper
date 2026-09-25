"""Annotated heatmaps (ports of figure_ophthal_review/plot_composition.py and
the per-column table in figure_RNAGenScape/plot_comparison.py)."""
import numpy as np

import figures4papers as fp

stages = ["Benchmark\nEvaluation", "Expert\nEvaluation", "Retrospective\nClinical\nValidation",
          "Prospective\nPilot Study", "Full\nClinical Trial"]
tasks = ["Screening or Diagnosis", "Report Generation", "Treatment Planning", "Patient QA", "Exam Taking"]
counts = np.array([[10, 7, 10, 3, 0], [2, 3, 3, 0, 0], [2, 3, 3, 2, 0], [5, 13, 1, 1, 0], [19, 5, 0, 0, 0]])

methods = ["VAE", "DDPM", "LDM", "FM", "Ours"]
columns = ["OpenVaccine (+)", "OpenVaccine (-)", "Zebrafish (+)", "Zebrafish (-)"]
delta = np.array([[-0.23, -0.23, -0.01, -0.01], [-0.33, -0.33, 0.29, 0.29], [-0.07, -0.07, -0.95, -0.95],
                  [-0.34, -0.34, 0.32, 0.32], [0.54, -2.81, 0.77, -1.29]])

if __name__ == "__main__":
    fp.apply_publication_style("compact")
    fig, (ax1, ax2) = fp.create_subplots(1, 2, figsize=(26, 8), gridspec_kw={"width_ratios": [1.3, 1]})
    fp.make_heatmap(ax1, counts, x_labels=stages, y_labels=tasks, cmap="Reds", vmin=0, vmax=20,
                    annotate=True, fmt="{:.0f}", cbar_ticks=[0, 5, 10, 15, 20])

    best = np.where([True, False, True, False], delta[:-1].max(0), delta[:-1].min(0))
    improvement = 100 * (delta[-1] - best) / np.abs(best)
    fp.make_column_heatmap(ax2, delta, x_labels=columns, y_labels=methods + ["Improvement"],
                           cmaps=["Reds", "Blues", "Reds", "Blues"], higher_is_better=[True, False, True, False],
                           summary_row=improvement)
    fp.finalize_figure(fig, "figures/heatmaps")
