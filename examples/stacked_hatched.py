"""Stacked bars with hatch-encoded subtypes and legend-only panels
(port of one row of figure_Brainteaser/plot_brute_force.py)."""
import numpy as np

import figures4papers as fp

methods = ["DeepSeek R1 Distill Qwen 1.5B", "DeepSeek R1 Distill Qwen 14B", "DeepSeek R1 Distill Llama 70B",
           "deepseek-chat (Deepseek-V3)", "deepseek-reasoner (Deepseek-R1)", "gemini-2.5-flash-preview-04-17",
           "OpenAI o3"]
subtypes = [r"$\bf{Only}$ $\bf{Model}$ brute force", r"$\bf{Only}$ $\bf{Human}$ brute force",
            r"$\bf{Neither}$ brute force", r"$\bf{Both}$ brute force"]
hatches = ["/", "\\", "", "x"]
result = {
    "CoT Prompt": np.array([[26.8, 3.6, 60.0, 9.6], [27.6, 3.2, 59.2, 10.0], [24.4, 4.0, 62.4, 9.2],
                            [31.2, 3.6, 55.6, 9.6], [14.0, 7.2, 72.8, 6.0], [16.9, 5.9, 70.0, 7.2],
                            [9.5, 7.5, 79.4, 3.5]]) / 100,
    "Math Prompt": np.array([[27.2, 4.8, 59.6, 8.4], [25.6, 4.8, 61.2, 8.4], [24.4, 4.8, 62.4, 8.4],
                             [28.4, 3.2, 58.4, 10.0], [10.0, 5.6, 76.7, 7.6], [12.6, 5.2, 74.8, 7.4],
                             [4.2, 7.9, 85.7, 2.1]]) / 100,
}

if __name__ == "__main__":
    fp.apply_publication_style("bar")
    fig, axes = fp.create_subplots(1, 4, figsize=(42, 7))
    for ax, (prompt, matrix) in zip(axes, result.items()):
        fp.make_stacked_bar(ax, matrix, colors=fp.BASELINES_THEN_OURS, hatches=hatches,
                            alpha=[1, 0.8, 0.8, 0.8], annotate_layer=0, annotate_kwargs=dict(fontsize=20))
        fp.style_axis(ax, title=prompt, title_size=36, title_pad=36, ylabel="Probability", label_size=30,
                      ylim=(0, 1.01))
    fp.legend_panel(axes[2], fp.patch_handles(methods, fp.BASELINES_THEN_OURS, edgecolor="black", linewidth=2),
                    fontsize=24)
    fp.legend_panel(axes[3], fp.patch_handles(subtypes, "white", hatches, linewidth=2), fontsize=24)
    fp.finalize_figure(fig, "figures/stacked_hatched")
