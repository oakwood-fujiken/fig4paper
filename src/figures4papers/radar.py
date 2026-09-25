"""Radar (spider) charts where each spoke may have its own scale."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from .palette import DEFAULT_COLORS


def _benchmark_of(spoke: str) -> str:
    # 'Qwen2.5-VL-7B\nMathVista' -> 'MathVista'
    return spoke.split("\n", 1)[-1]


def make_radar(
    ax: Axes,
    results: Mapping[str, Sequence[float]],
    methods: Sequence[str],
    colors=None,
    spoke_ticks: Optional[Mapping[str, Sequence[float]]] = None,
    group_of=_benchmark_of,
    inner: float = 45,
    outer: float = 90,
    linewidth: float = 2,
    fill_alpha: float = 0.05,
    marker_size: float = 18,
    label_fontsize: float = 14,
    tick_fontsize: float = 12,
    label_offset=(8, 10),
    label_family: Optional[str] = None,
    legend: bool = True,
    legend_kwargs: Optional[dict] = None,
):
    """Radar chart on a polar ``ax`` (create it with ``projection="polar"``).

    ``results`` maps spoke name -> one value per method. Spokes are grouped
    by ``group_of(spoke)`` (default: text after the first newline, i.e. the
    benchmark), and ``spoke_ticks[group]`` gives that group's tick values; the
    smallest/largest tick map to ``inner``/``outer`` display radius. Spokes
    without ticks use the data range. Ported from the VIGIL radar figure.
    """
    spokes = list(results)
    values = np.array([np.asarray(results[s], dtype=float) for s in spokes])  # (n_spokes, n_methods)
    n_spokes, n_methods = values.shape
    if len(methods) != n_methods:
        raise ValueError(f"{n_methods} values per spoke but {len(methods)} methods")
    colors = colors or [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n_methods)]
    groups = [group_of(s) for s in spokes]
    spoke_ticks = dict(spoke_ticks or {})
    for s, g in zip(spokes, groups):
        if g not in spoke_ticks:
            row = values[spokes.index(s)]
            spoke_ticks[g] = [np.nanmin(row), np.nanmax(row)]

    def to_display(v, group):
        ticks = spoke_ticks[group]
        lo, hi = min(ticks), max(ticks)
        if hi <= lo:
            return (inner + outer) / 2
        return inner + (outer - inner) * np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    angles = np.linspace(2 * np.pi, 0, n_spokes, endpoint=False)
    closed = np.append(angles, angles[0])

    for m in range(n_methods):
        vals = values[:, m].copy()
        if np.isnan(vals).any():
            vals[np.isnan(vals)] = 0.0 if np.isnan(vals).all() else np.nanmean(vals)
        r = np.array([to_display(v, g) for v, g in zip(vals, groups)])
        r_closed = np.append(r, r[0])
        ax.plot(closed, r_closed, color=colors[m], linewidth=linewidth, label=methods[m])
        ax.fill(closed, r_closed, color=colors[m], alpha=fill_alpha)
        ax.scatter(angles, r, color=colors[m], s=marker_size, zorder=5, edgecolors="none")

    ax.set_ylim(inner, outer)
    ax.set_theta_zero_location("N")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.plot(closed, np.full_like(closed, outer), color="k", linewidth=0.8, zorder=4)
    for a in angles:
        ax.plot([a, a], [inner, outer], color="gray", linewidth=0.5, zorder=4)

    # Contour polygons: level k connects the k-th tick of every spoke.
    n_levels = max(len(spoke_ticks[g]) for g in groups)
    for k in range(n_levels):
        r = [to_display(spoke_ticks[g][min(k, len(spoke_ticks[g]) - 1)], g) for g in groups]
        ax.plot(closed, np.append(r, r[0]), color="k", linewidth=0.6, zorder=4, label="_nolegend_")

    ax.set_yticks([outer])
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, g in zip(angles, groups):
        for tick in sorted(spoke_ticks[g])[1:]:  # skip the innermost to avoid clutter
            lbl = f"{tick:.0f}" if float(tick).is_integer() else f"{tick:.1f}"
            ax.text(angle, to_display(tick, g) + 1, lbl, fontsize=tick_fontsize, ha="center", va="center",
                    rotation=np.degrees(angle), rotation_mode="anchor", clip_on=False)
    for angle, spoke in zip(angles, spokes):
        r = outer + label_offset[0] + label_offset[1] * abs(np.sin(angle))
        ax.text(angle, r, spoke, fontsize=label_fontsize, ha="center", va="center", clip_on=False,
                fontfamily=label_family)
    if legend:
        kw = dict(loc="upper right", bbox_to_anchor=(1.40, 0.05), frameon=False)
        kw.update(legend_kwargs or {})
        ax.legend(**kw)
    return ax
