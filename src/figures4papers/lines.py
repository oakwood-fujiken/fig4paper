"""Line, trend and scatter helpers."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from .layout import _broadcast
from .palette import DEFAULT_COLORS, with_alpha


def _as_series(y_series) -> list[np.ndarray]:
    if isinstance(y_series, np.ndarray) and y_series.ndim == 1:
        return [y_series]
    return [np.asarray(y, dtype=float) for y in y_series]


def make_trend(
    ax: Axes,
    x,
    y_series,
    labels: Optional[Sequence[str]] = None,
    colors=None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    errors=None,
    show_shadow: bool = True,
    shadow_alpha: float = 0.15,
    linewidth: float = 3,
    marker: Optional[str] = "o",
    markersize: float = 8,
    alpha: float = 0.9,
    categorical_x: bool = False,
    **plot_kwargs,
):
    """Several lines on one axis, optionally with a ±error band.

    ``y_series`` is a list of 1D arrays (same length as ``x``) or a 2D array
    ``(n_series, n_points)``. A series given as 2D ``(n_points, n_runs)``
    inside the list is reduced to mean ± std. ``categorical_x=True`` places
    points at equal spacing and uses ``x`` as tick labels (e.g. beta = 0.05,
    0.1, 0.2, 0.5 in the VIGIL ablations).
    """
    series = _as_series(y_series)
    n = len(series)
    colors = _broadcast(colors, n) if colors is not None else [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]
    labels = _broadcast(labels, n) if labels is not None else [None] * n
    errors = [None] * n if errors is None else [None if e is None else np.asarray(e, dtype=float) for e in errors]

    x = np.asarray(x)
    xpos = np.arange(len(x)) if categorical_x else x
    lines = []
    for y, err, color, label in zip(series, errors, colors, labels):
        if y.ndim == 2:
            y, err = y.mean(axis=1), (y.std(axis=1) if err is None else err)
        if len(y) != len(x):
            raise ValueError(f"Series of length {len(y)} does not match x of length {len(x)}")
        (line,) = ax.plot(xpos, y, color=color, label=label, linewidth=linewidth, marker=marker,
                          markersize=markersize, alpha=alpha, **plot_kwargs)
        if show_shadow and err is not None:
            ax.fill_between(xpos, y - err, y + err, color=color, alpha=shadow_alpha, linewidth=0)
        lines.append(line)
    if categorical_x:
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(v) for v in x])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return lines


def make_gradient_line(
    ax: Axes,
    x,
    y,
    color,
    alpha_range=(0.3, 0.9),
    linewidth: float = 3,
    marker: Optional[str] = "o",
    markersize: float = 10,
    label: Optional[str] = None,
):
    """A line whose opacity increases from left to right (VIGIL post-training figure).

    The returned proxy artist carries ``label`` so the line shows up in legends.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pts = np.column_stack([x, y])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = [with_alpha(color, a) for a in np.linspace(*alpha_range, len(segments))]
    ax.add_collection(LineCollection(segments, colors=seg_colors, linewidths=linewidth, capstyle="round"))
    (proxy,) = ax.plot(x, y, color=color, linewidth=0, marker=marker, markersize=markersize, label=label)
    ax.autoscale_view()
    return proxy


def add_reference_line(
    ax: Axes,
    y: float,
    label: Optional[str] = None,
    color="black",
    alpha: float = 0.3,
    linewidth: float = 4,
    linestyle="--",
    horizontal: bool = True,
    **kwargs,
):
    """Dashed baseline (e.g. "SFT only") across the axis."""
    fn = ax.axhline if horizontal else ax.axvline
    return fn(y, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label, **kwargs)


def mark_events(
    ax: Axes,
    x,
    y,
    events: Mapping,
    dy: float = 0.1,
    fontsize: float = 11,
    lift_char: str = "*",
    lift: float = 0.8,
    arrow_kwargs: Optional[dict] = None,
):
    """Annotate points of a curve with labelled arrows (e.g. model releases on a timeline).

    ``events`` maps an x value (must be present in ``x``) to a label. Each
    ``lift_char`` in a label raises it by an extra ``lift * dy`` to avoid
    overlap; the characters are stripped from the drawn text.
    """
    index = {v: i for i, v in enumerate(list(x))}
    y = np.asarray(y)
    y0, y1 = ax.get_ylim()
    arrowprops = dict(arrowstyle="-|>", lw=1.3, color="black", shrinkA=0, shrinkB=0, mutation_scale=15)
    arrowprops.update(arrow_kwargs or {})
    texts = []
    for key, label in events.items():
        if key not in index:
            continue
        i = index[key]
        xi = i if isinstance(key, str) else key
        height = (1 + lift * label.count(lift_char)) * dy * (y1 - y0)
        texts.append(ax.annotate(label.replace(lift_char, ""), xy=(xi, y[i]), xytext=(xi, y[i] + height),
                                 ha="center", va="bottom", fontsize=fontsize, arrowprops=arrowprops))
    return texts


def make_scatter(
    ax: Axes,
    x,
    y,
    label: Optional[str] = None,
    color=None,
    size: float = 50,
    alpha: float = 0.7,
    edgecolor=None,
    **kwargs,
):
    """Single-series scatter with house defaults."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same length")
    return ax.scatter(x, y, label=label, color=color or DEFAULT_COLORS[0], s=size, alpha=alpha,
                      edgecolors=edgecolor, **kwargs)
