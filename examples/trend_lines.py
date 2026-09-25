"""Line panels: hyper-parameter sweep with a baseline and a gradient-alpha training curve
(ports of figure_VIGIL/plot_ablation.py and plot_posttraining.py)."""
import numpy as np

import figures4papers as fp

methods = ["DPO", "DA-DPO", "VIGIL (Ours)"]
colors = [fp.PALETTE["red_3"], fp.PALETTE["green_3"], fp.PALETTE["blue_main"]]
beta = [0.05, 0.1, 0.2, 0.5]
by_beta = np.array([[79.5, 82.8, 81.0, 76.2], [83.2, 84.2, 83.5, 81.0], [86.8, 86.9, 86.5, 85.8]])
steps = [0, 200, 400, 600, 800]
by_step = np.array([[22.0, 25.5, 28.2, 29.5, 30.2], [22.0, 33.5, 38.2, 39.8, 40.5], [22.0, 52.5, 56.8, 57.9, 58.5]])

if __name__ == "__main__":
    fp.apply_publication_style("bar")
    fig, (ax1, ax2) = fp.create_subplots(1, 2, figsize=(20, 8))

    fp.add_reference_line(ax1, 82.1, label="SFT only")
    fp.make_trend(ax1, beta, by_beta, methods, colors, categorical_x=True, markersize=10, alpha=1)
    fp.style_axis(ax1, xlabel=r"Hyperparameter $\beta$", ylabel=r"POPE$_{Adv}\uparrow$", label_size=28,
                  tick_size=24, tick_length=8, tick_width=1.5)
    ax1.legend(fontsize=22, loc="lower center", ncols=2)

    x = np.arange(len(steps))
    fp.add_reference_line(ax2, by_step[0, 0])
    for y, c in zip(by_step, colors):
        fp.make_gradient_line(ax2, x, y, c)
    ax2.set_xticks(x)
    ax2.set_xticklabels(steps)
    fp.style_axis(ax2, xlabel="Post-training steps", ylabel="Vision-dependent tasks" + r"$\uparrow$",
                  label_size=28, yticks=[0, 20, 40, 60], tick_size=20, tick_length=8, tick_width=1.5)
    handles = fp.line_handles(["SFT only"], "black", linestyles="--", linewidth=4, alpha=0.3)
    handles += fp.line_handles(methods, colors, markers="o")
    ax2.legend(handles=handles, fontsize=20, loc="lower right", ncols=2)
    fp.finalize_figure(fig, "figures/trend_lines")
