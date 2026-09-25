"""Cumulative counts with event annotations (port of figure_ophthal_review/plot_trend.py)."""
import numpy as np

import figures4papers as fp

months = [f"{2022 + (10 + i) // 12}-{(10 + i) % 12 + 1:02d}" for i in range(33)]
monthly = np.array([0, 0, 4, 0, 0, 6, 1, 3, 2, 0, 5, 0, 4, 0, 9, 1, 3, 4, 7, 6, 7, 6, 6, 1, 6, 4, 6, 2, 0, 0, 0, 1, 1])
events = {"2022-11": "ChatGPT\n(GPT-3.5)", "2023-03": "GPT-4*", "2023-07": "LlaMA 2", "2023-12": "Gemini 1.0",
          "2024-04": "LlaMA 3", "2024-12": "Gemini 2.0", "2025-06": "Gemini 2.5*"}

if __name__ == "__main__":
    fp.apply_publication_style("compact")
    fig, (ax,) = fp.create_subplots(1, 1, figsize=(14, 5))
    x = np.arange(len(months))
    cumulative = np.cumsum(monthly)
    ax.fill_between(x, 0, cumulative, color="#ffa8a6", label="Evaluation / Application")
    ax.plot(x, cumulative, lw=3, color="#850c0a")
    ax.set_ylim(0, 105)
    fp.mark_events(ax, months, cumulative, events)
    ax.set_xticks(x[2::6])
    ax.set_xticklabels(months[2::6])
    ax.set_ylabel("Cumulative\nPublication Count")
    ax.legend(loc="upper left")
    fp.finalize_figure(fig, "figures/timeline")
