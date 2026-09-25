"""Radar chart with per-benchmark scales (port of figure_VIGIL/plot_comparison_radar.py)."""
import numpy as np
from matplotlib import pyplot as plt

import figures4papers as fp

methods = ["DPO", "DA-DPO", "VIGIL (Ours)"]
colors = [fp.PALETTE["red_3"], fp.PALETTE["green_3"], fp.PALETTE["blue_main"]]
results = {
    "Qwen2.5-VL-7B\nPOPE$_{Adv}$": [82.8, 84.2, 86.9],
    "LLaVA-OneVision-7B\nPOPE$_{Adv}$": [82.8, 84.2, 86.9],
    "InternVL2.5-26B\nPOPE$_{Adv}$": [85.5, 86.8, 89.4],
    "Qwen2.5-VL-72B\nPOPE$_{Adv}$": [84.5, 87.4, 89.8],
    "Qwen2.5-VL-7B\nMathVista": [48.0, 48.8, 49.5],
    "LLaVA-OneVision-7B\nMathVista": [50.8, 51.5, 52.8],
    "InternVL2.5-26B\nMathVista": [57.9, 58.8, 60.1],
    "Qwen2.5-VL-72B\nMathVista": [54.1, 55.4, 56.6],
    "Qwen2.5-VL-7B\nMMBench": [71.2, 72.0, 72.5],
    "LLaVA-OneVision-7B\nMMBench": [72.5, 73.0, 73.8],
    "InternVL2.5-26B\nMMBench": [79.5, 80.1, 81.3],
    "Qwen2.5-VL-72B\nMMBench": [77.2, 77.8, 78.5],
}
ticks = {"POPE$_{Adv}$": [75, 80, 85, 91], "MathVista": [30, 40, 50, 61], "MMBench": [40, 55, 70, 85]}

if __name__ == "__main__":
    fp.apply_publication_style("bar")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="polar")
    fp.make_radar(ax, results, methods, colors, spoke_ticks=ticks, label_family="monospace",
                  legend_kwargs=dict(fontsize=15))
    fp.finalize_figure(fig, "figures/radar", tight_bbox=True)
