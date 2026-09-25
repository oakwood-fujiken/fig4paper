"""Figure layout and legend helpers."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """``plt.subplots`` that always returns ``(fig, axes)`` with ``axes`` flattened to 1D.

    ``figsize`` defaults to ``(9 * ncols, 7 * nrows)``, matching the scripts' panel size.
    """
    if figsize is None:
        figsize = (9 * ncols, 7 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kwargs)
    return fig, axes.ravel()


def patch_handles(
    labels: Sequence[str],
    colors=None,
    hatches=None,
    edgecolor=None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
) -> list[Patch]:
    """Legend handles for bars, built without drawing (and removing) dummy bars.

    ``colors`` / ``hatches`` may be a single value or one per label.
    """
    n = len(labels)
    colors = _broadcast(colors if colors is not None else "white", n)
    hatches = _broadcast(hatches if hatches is not None else "", n)
    if edgecolor is None:
        # Hatches are drawn in the edge color, so they need one to be visible.
        edgecolor = "black" if any(hatches) else None
    kw = {}
    if linewidth is not None:
        kw["linewidth"] = linewidth
    if alpha is not None:
        kw["alpha"] = alpha
    return [
        Patch(facecolor=c, hatch=h, edgecolor=edgecolor, label=lab, **kw)
        for lab, c, h in zip(labels, colors, hatches)
    ]


def line_handles(
    labels: Sequence[str],
    colors,
    linestyles="-",
    markers=None,
    linewidth: float = 3,
    markersize: float = 10,
    alpha: Optional[float] = None,
) -> list[Line2D]:
    """Legend handles for lines (optionally with markers)."""
    n = len(labels)
    colors = _broadcast(colors, n)
    linestyles = _broadcast(linestyles, n)
    markers = _broadcast(markers, n)
    return [
        Line2D([0], [0], color=c, linestyle=ls, marker=m, linewidth=linewidth,
               markersize=markersize, alpha=alpha, label=lab)
        for lab, c, ls, m in zip(labels, colors, linestyles, markers)
    ]


def legend_panel(
    ax: Axes,
    handles=None,
    labels=None,
    source: Optional[Axes] = None,
    loc: str = "center",
    **legend_kwargs,
):
    """Turn ``ax`` into a legend-only panel.

    Handles come from ``handles``/``labels`` or, if omitted, from ``source``
    (another axis). Keeps data panels free of legend boxes.
    """
    if handles is None:
        if source is None:
            raise ValueError("Pass either handles or source")
        handles, auto_labels = source.get_legend_handles_labels()
        labels = labels if labels is not None else auto_labels
    ax.set_axis_off()
    legend_kwargs.setdefault("frameon", False)
    if labels is None:
        return ax.legend(handles=handles, loc=loc, **legend_kwargs)
    return ax.legend(handles, labels, loc=loc, **legend_kwargs)


def style_axis(
    ax: Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    label_size: Optional[float] = None,
    title_size: Optional[float] = None,
    tick_size: Optional[float] = None,
    tick_length: Optional[float] = None,
    tick_width: Optional[float] = None,
    labelpad: float = 12,
    title_pad: Optional[float] = None,
    hide_xticks: bool = False,
    ylim=None,
    yticks=None,
    sci_y: bool = False,
) -> Axes:
    """Apply the common per-axis cosmetics in one call."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=labelpad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=labelpad)
    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=title_pad)
    tick_kw = {k: v for k, v in dict(labelsize=tick_size, length=tick_length, width=tick_width).items()
               if v is not None}
    if tick_kw:
        ax.tick_params(**tick_kw)
    if hide_xticks:
        ax.set_xticks([])
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return ax


def tight_ylim(ax: Axes, values, margin: float = 0.5, floor: Optional[float] = None, ceil: Optional[float] = None):
    """Set y-limits to ``[min - margin*std, max + margin*std]`` so differences stay visible."""
    values = np.asarray(values, dtype=float)
    spread = values.std() or (abs(values.max()) or 1.0) * 0.1
    lo, hi = values.min() - margin * spread, values.max() + margin * spread
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    ax.set_ylim(lo, hi)
    return lo, hi


def _broadcast(value, n: int) -> list:
    if value is None or isinstance(value, (str, tuple)) or np.isscalar(value):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"Expected {n} values, got {len(value)}")
    return value
