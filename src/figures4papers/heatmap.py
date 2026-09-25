"""Heatmaps with readable cell annotations."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes

from .palette import text_color_for


def make_heatmap(
    ax: Axes,
    matrix,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    cmap="magma",
    vmin=None,
    vmax=None,
    cbar: bool = True,
    cbar_label: Optional[str] = None,
    cbar_ticks=None,
    annotate: bool = False,
    fmt: Union[str, Callable] = "{:.2f}",
    annot_fontsize: Optional[float] = None,
    gridlines: Optional[str] = "white",
    gridwidth: float = 1,
    x_rotation: float = 0,
):
    """Heatmap via ``pcolormesh`` (no seaborn needed).

    Cell text switches between black and white depending on the cell color.
    ``gridlines`` draws separators between cells (as seaborn's ``linewidths``).
    NaN cells are left blank.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    n_rows, n_cols = matrix.shape
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax,
                         edgecolors=gridlines if gridlines else "face", linewidth=gridwidth if gridlines else 0)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(x_labels if x_labels is not None else [""] * n_cols, rotation=x_rotation)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(y_labels if y_labels is not None else [""] * n_rows)

    colorbar = None
    if cbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax)
        colorbar.outline.set_visible(False)
        if cbar_label:
            colorbar.set_label(cbar_label)
        if cbar_ticks is not None:
            colorbar.set_ticks(cbar_ticks)

    if annotate:
        formatter = fmt.format if isinstance(fmt, str) else fmt
        for (i, j), val in np.ndenumerate(matrix):
            if np.isnan(val):
                continue
            color = text_color_for(mesh.cmap(mesh.norm(val)))
            ax.text(j + 0.5, i + 0.5, formatter(val), ha="center", va="center", color=color, fontsize=annot_fontsize)
    return mesh, colorbar


def make_column_heatmap(
    ax: Axes,
    matrix,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    cmaps="Reds",
    higher_is_better: Union[bool, Sequence[bool]] = True,
    vmin=None,
    fmt: Union[str, Callable] = "{:.2f}",
    annot_fontsize: Optional[float] = None,
    summary_row: Optional[Sequence[float]] = None,
    summary_fmt: Union[str, Callable] = "{:+.1f}%",
    summary_colors=("forestgreen", "darkred"),
    x_rotation: float = 30,
):
    """Table-like heatmap where every column has its own color scale (RNAGenScape comparison).

    ``higher_is_better`` (per column) decides whether the dark end of the
    colormap sits at the column max or min. ``vmin`` can clip the light end
    (scalar or per column). ``summary_row`` adds an uncolored last row, e.g.
    improvement over the best baseline, drawn green if >= 0 else red.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_rows, n_cols = matrix.shape
    cmaps = [cmaps] * n_cols if isinstance(cmaps, str) or not isinstance(cmaps, Sequence) else list(cmaps)
    better = [higher_is_better] * n_cols if isinstance(higher_is_better, bool) else list(higher_is_better)
    vmins = [vmin] * n_cols if vmin is None or np.isscalar(vmin) else list(vmin)
    formatter = fmt.format if isinstance(fmt, str) else fmt
    total_rows = n_rows + (summary_row is not None)

    for j in range(n_cols):
        col = matrix[:, j]
        cmap = mpl.colormaps[cmaps[j]] if isinstance(cmaps[j], str) else cmaps[j]
        lo, hi = col.min(), col.max()
        if better[j]:
            if vmins[j] is not None:
                lo = max(vmins[j], lo)
        else:
            # Dark end at the column minimum.
            cmap = cmap.reversed()
        norm = mpl.colors.Normalize(vmin=lo, vmax=hi)
        ax.imshow(col[:, None], cmap=cmap, norm=norm, aspect="auto",
                  extent=[j - 0.5, j + 0.5, n_rows, 0])
        for i, val in enumerate(col):
            ax.text(j, i + 0.5, formatter(val), ha="center", va="center",
                    color=text_color_for(cmap(norm(val))), fontsize=annot_fontsize)

    if summary_row is not None:
        sfmt = summary_fmt.format if isinstance(summary_fmt, str) else summary_fmt
        for j, val in enumerate(summary_row):
            ax.text(j, n_rows + 0.5, sfmt(val), ha="center", va="center",
                    color=summary_colors[0] if val >= 0 else summary_colors[1], fontsize=annot_fontsize)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(total_rows, 0)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels if x_labels is not None else [""] * n_cols, rotation=x_rotation)
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)) + 0.5)
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_frame_on(False)
    return ax
