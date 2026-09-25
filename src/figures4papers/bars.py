"""Bar-chart helpers: the most common figure type in the figure_* scripts."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from .layout import _broadcast, create_subplots, legend_panel, patch_handles
from .palette import DEFAULT_COLORS, PALETTE, text_color_for

ERROR_KW = {"elinewidth": 2, "capthick": 2}


def summarize(values, err: Optional[str] = "std"):
    """Reduce per-run results to ``(mean, error)``.

    1D input is returned as-is with ``error=None``. 2D input of shape
    ``(n_items, n_runs)`` is reduced along the last axis; ``err`` is
    ``"std"``, ``"sem"`` or ``None``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values, None
    if values.ndim != 2:
        raise ValueError(f"Expected 1D or 2D values, got shape {values.shape}")
    mean = values.mean(axis=1)
    if err is None:
        return mean, None
    std = values.std(axis=1)
    if err == "std":
        return mean, std
    if err == "sem":
        return mean, std / np.sqrt(values.shape[1])
    raise ValueError(f"err must be 'std', 'sem' or None, got {err!r}")


def make_bars(
    ax: Axes,
    values,
    colors=None,
    labels: Optional[Sequence[str]] = None,
    yerr=None,
    err: Optional[str] = "std",
    annotate: bool = False,
    fmt: str = "{:.2f}",
    annotate_kwargs: Optional[dict] = None,
    horizontal: bool = False,
    positions=None,
    width: float = 0.8,
    hatch=None,
    edgecolor=None,
    linewidth: Optional[float] = None,
    capsize: float = 8,
    error_kw: Optional[dict] = None,
    hide_ticks: bool = True,
    **bar_kwargs,
) -> BarContainer:
    """One bar per method: the building block of every comparison panel.

    ``values`` is 1D (one value per method) or 2D ``(n_methods, n_runs)``,
    in which case the mean is plotted and ``err`` ("std"/"sem") is used as
    the error bar unless ``yerr`` is given. ``labels`` become legend entries
    (the scripts hide x ticks and identify methods via the legend).
    """
    mean, auto_err = summarize(values, err)
    if yerr is None:
        yerr = auto_err
    n = len(mean)
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]
    positions = np.arange(n) if positions is None else np.asarray(positions)

    kw = dict(color=colors, label=labels, **bar_kwargs)
    if hatch is not None:
        kw["hatch"] = hatch
        if edgecolor is None:
            edgecolor = "black"
    if edgecolor is not None:
        kw["edgecolor"] = edgecolor
    if linewidth is not None:
        kw["linewidth"] = linewidth
    if yerr is not None:
        kw["capsize"] = capsize
        kw["error_kw"] = {**ERROR_KW, **(error_kw or {})}

    if horizontal:
        bars = ax.barh(positions, mean, height=width, xerr=yerr, **kw)
        if hide_ticks:
            ax.set_yticks([])
    else:
        bars = ax.bar(positions, mean, width=width, yerr=yerr, **kw)
        if hide_ticks:
            ax.set_xticks([])

    if annotate:
        annotate_bars(ax, bars, fmt=fmt, errors=yerr, **(annotate_kwargs or {}))
    return bars


def annotate_bars(
    ax: Axes,
    bars: BarContainer,
    fmt: str = "{:.2f}",
    position: str = "top",
    fontsize: Optional[float] = None,
    color: str = "auto",
    padding: Optional[float] = None,
    errors=None,
    outline: Optional[str] = None,
    outline_width: float = 4,
    log_factor: Optional[float] = None,
    **text_kwargs,
):
    """Write each bar's value on it.

    ``position``: ``"top"`` (above the bar and its error bar), ``"inside"``
    (just below the top edge) or ``"center"``. ``color="auto"`` picks white
    or black depending on the bar color for inside/center labels.
    ``outline`` draws a stroke around the text (e.g. gold text with a black
    outline, as in the Brainteaser brute-force figure). For log-scaled axes
    pass ``log_factor`` (e.g. 1.1): top labels go at ``value * log_factor``.
    """
    if position not in ("top", "inside", "center"):
        raise ValueError("position must be 'top', 'inside' or 'center'")
    horizontal = getattr(bars, "orientation", "vertical") == "horizontal"
    n = len(bars)
    errors = np.zeros(n) if errors is None else np.broadcast_to(np.asarray(errors, dtype=float), (n,))
    if padding is None:
        lo, hi = ax.get_xlim() if horizontal else ax.get_ylim()
        padding = 0.02 * (hi - lo)

    effects = None
    if outline is not None:
        effects = [patheffects.Stroke(linewidth=outline_width, foreground=outline), patheffects.Normal()]

    texts = []
    for bar, err in zip(bars, errors):
        value = bar.get_width() if horizontal else bar.get_height()
        base = bar.get_x() if horizontal else bar.get_y()
        center = (bar.get_y() + bar.get_height() / 2) if horizontal else (bar.get_x() + bar.get_width() / 2)
        end = base + value
        if position == "top":
            sign = 1 if value >= 0 else -1
            if log_factor is not None:
                along = end * log_factor
            else:
                along = end + sign * (err + padding)
            txt_color = "black" if color == "auto" else color
            align = ("left" if sign > 0 else "right") if horizontal else ("bottom" if sign > 0 else "top")
        elif position == "inside":
            along = end - np.sign(value) * padding
            txt_color = text_color_for(bar.get_facecolor()) if color == "auto" else color
            align = "right" if horizontal else "top"
        else:
            along = base + value / 2
            txt_color = text_color_for(bar.get_facecolor()) if color == "auto" else color
            align = "center"

        kw = dict(color=txt_color, fontsize=fontsize, path_effects=effects, **text_kwargs)
        if horizontal:
            t = ax.text(along, center, fmt.format(value), ha=align, va="center", **kw)
        else:
            t = ax.text(center, along, fmt.format(value), ha="center", va=align, **kw)
        texts.append(t)
    return texts


def add_delta_arrows(
    ax: Axes,
    bars: BarContainer,
    reference: int = 0,
    color: str = "red",
    fmt: str = "{:+.2f}",
    fontsize: Optional[float] = None,
    line_color=None,
    linewidth: float = 4,
    arrow_width: float = 4,
    show_line: bool = True,
):
    """Ablation-style annotation: dashed line at the reference bar and arrows to every other bar.

    Each arrow starts at the reference height, ends at the bar top, and is
    labelled with the difference (``bar - reference``).
    """
    heights = np.array([b.get_height() for b in bars])
    ref = heights[reference]
    if show_line:
        ax.axhline(ref, color=line_color or bars[reference].get_facecolor(), linestyle="--",
                   linewidth=linewidth, alpha=0.7)
    for i, bar in enumerate(bars):
        if i == reference:
            continue
        x = bar.get_x() + bar.get_width()
        ax.annotate("", xy=(x, heights[i]), xytext=(x, ref),
                    arrowprops=dict(arrowstyle="->", color=color, lw=arrow_width))
        delta = heights[i] - ref
        ax.text(x, ref, fmt.format(delta).replace("-", "\N{MINUS SIGN}"),
                ha="right", va="bottom" if delta < 0 else "top", fontsize=fontsize, color=color)


def make_grouped_bar(
    ax: Axes,
    categories: Sequence[str],
    series,
    labels: Sequence[str],
    ylabel: Optional[str] = "Value",
    colors=None,
    annotate: bool = False,
    fmt: str = "{:.2f}",
    yerr=None,
    hatches=None,
    color_by: str = "series",
    width: Optional[float] = None,
    gap: float = 0.0,
    edgecolor=None,
    linewidth: Optional[float] = None,
    show_categories: bool = True,
) -> BarContainer:
    """Grouped bars: ``series`` (n_series x n_categories) side by side per category.

    ``color_by="series"`` colors each series (``colors`` per series).
    ``color_by="category"`` colors each category and distinguishes series by
    ``hatches`` instead (Brainteaser rewriting figure). Returns the last
    BarContainer, as in the skill API.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series[None, :]
    n_series, n_cat = series.shape
    if n_cat != len(categories):
        raise ValueError(f"{len(categories)} categories but series have {n_cat} columns")
    if len(labels) != n_series:
        raise ValueError(f"{n_series} series but {len(labels)} labels")
    if width is None:
        width = 0.8 / n_series
    yerr = [None] * n_series if yerr is None else list(np.asarray(yerr, dtype=float))
    hatches = _broadcast(hatches, n_series) if hatches is not None else [None] * n_series

    if color_by == "series":
        colors = colors or DEFAULT_COLORS
        bar_colors = [colors[i % len(colors)] for i in range(n_series)]
    elif color_by == "category":
        colors = colors or DEFAULT_COLORS
        bar_colors = [[colors[j % len(colors)] for j in range(n_cat)]] * n_series
    else:
        raise ValueError("color_by must be 'series' or 'category'")

    x = np.arange(n_cat)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width * (1 + gap)
    bars = None
    for i in range(n_series):
        kw = {}
        if hatches[i]:
            kw["hatch"] = hatches[i]
            edgecolor = edgecolor or "black"
        if edgecolor is not None:
            kw["edgecolor"] = edgecolor
        if linewidth is not None:
            kw["linewidth"] = linewidth
        if yerr[i] is not None:
            kw.update(capsize=5, error_kw=ERROR_KW)
        bars = ax.bar(x + offsets[i], series[i], width=width, yerr=yerr[i],
                      color=bar_colors[i], label=labels[i], **kw)
        if annotate:
            annotate_bars(ax, bars, fmt=fmt, errors=yerr[i])
    if show_categories:
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
    else:
        ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel)
    return bars


def make_stacked_bar(
    ax: Axes,
    matrix,
    colors=None,
    labels: Optional[Sequence[str]] = None,
    hatches=None,
    color_by: str = "bar",
    edgecolor: str = "black",
    linewidth: float = 2,
    alpha: Union[float, Sequence[float]] = 1.0,
    annotate_layer: Optional[int] = None,
    annotate_kwargs: Optional[dict] = None,
    hide_ticks: bool = True,
) -> list[BarContainer]:
    """Stacked bars from ``matrix`` of shape ``(n_bars, n_layers)``.

    ``color_by="bar"``: one color per bar (method), layers told apart by
    ``hatches`` (Brainteaser brute-force figure). ``color_by="layer"``: one
    color per layer. ``annotate_layer`` writes the values of that layer at its
    center (default style: gold text with black outline).
    """
    matrix = np.asarray(matrix, dtype=float)
    n_bars, n_layers = matrix.shape
    hatches = _broadcast(hatches, n_layers) if hatches is not None else [""] * n_layers
    alphas = _broadcast(alpha, n_layers)
    colors = colors or DEFAULT_COLORS
    x = np.arange(n_bars)
    bottoms = np.concatenate([np.zeros((n_bars, 1)), np.cumsum(matrix, axis=1)[:, :-1]], axis=1)

    containers = []
    for k in range(n_layers):
        if color_by == "bar":
            c = [colors[i % len(colors)] for i in range(n_bars)]
            label = labels if k == 0 else None
        elif color_by == "layer":
            c = colors[k % len(colors)]
            label = labels[k] if labels is not None else None
        else:
            raise ValueError("color_by must be 'bar' or 'layer'")
        bars = ax.bar(x, matrix[:, k], bottom=bottoms[:, k], color=c, label=label, hatch=hatches[k],
                      edgecolor=edgecolor, linewidth=linewidth, alpha=alphas[k])
        containers.append(bars)

    if annotate_layer is not None:
        kw = dict(position="center", fmt="{:.3f}", color=PALETTE["highlight"], outline="black")
        kw.update(annotate_kwargs or {})
        annotate_bars(ax, containers[annotate_layer], **kw)
    if hide_ticks:
        ax.set_xticks([])
    return containers


def make_clustered_bars(
    ax: Axes,
    groups: Mapping[str, Sequence[float]],
    colors=None,
    labels: Optional[Sequence[str]] = None,
    gap: float = 1.0,
    yerr: Optional[Mapping[str, Sequence[float]]] = None,
    **bar_kwargs,
) -> list[BarContainer]:
    """Methods side by side within each group (dataset), groups along x.

    ``groups`` maps group name -> one value per method. Tick labels are the
    group names at each cluster center (Cflows trajectory figure).
    """
    names = list(groups)
    n_methods = len(np.asarray(groups[names[0]]))
    containers, centers = [], []
    for g, name in enumerate(names):
        pos = np.arange(n_methods) + g * (n_methods + gap)
        err = None if yerr is None else yerr[name]
        containers.append(make_bars(ax, groups[name], colors=colors, labels=labels if g == 0 else None,
                                    yerr=err, positions=pos, hide_ticks=False, **bar_kwargs))
        centers.append(pos.mean())
    ax.set_xticks(centers)
    ax.set_xticklabels(names)
    return containers


def metric_panels(
    results: Union[Mapping[str, object], np.ndarray],
    methods: Sequence[str],
    colors=None,
    metrics: Optional[Sequence[str]] = None,
    std: Optional[Union[Mapping[str, object], np.ndarray]] = None,
    err: Optional[str] = "std",
    ncols: Optional[int] = None,
    legend: Union[bool, str] = "panel",
    legend_kwargs: Optional[dict] = None,
    panel_size=(9, 7),
    ylabel_size: Optional[float] = None,
    ylims: Optional[Mapping[str, tuple]] = None,
    annotate: bool = False,
    fmt: str = "{:.2f}",
    sci_y: bool = False,
    **bar_kwargs,
):
    """One bar panel per metric plus a legend-only panel. Returns ``(fig, axes)``.

    ``results`` is either a mapping ``metric -> values`` (values 1D per method,
    or 2D ``(n_methods, n_runs)``) or an array ``(n_methods, n_metrics)`` with
    ``metrics`` naming its columns. ``std`` has the same layout (optional).
    ``legend``: ``"panel"`` (extra axis, default), ``"first"`` (inside the first
    panel) or ``False``. ``axes`` contains only the data panels.

    Covers CellSpliceNet / Cflows / ImmunoStruct / Brainteaser comparison figures.
    """
    if isinstance(results, Mapping):
        metrics = list(metrics) if metrics is not None else list(results)
        get = lambda m: results[m]  # noqa: E731
        get_std = (lambda m: std[m]) if std is not None else (lambda m: None)  # noqa: E731
    else:
        arr = np.asarray(results, dtype=float)
        if metrics is None:
            metrics = [f"Metric {j + 1}" for j in range(arr.shape[1])]
        std_arr = None if std is None else np.asarray(std, dtype=float)
        get = lambda m: arr[:, metrics.index(m)]  # noqa: E731
        get_std = (lambda m: std_arr[:, metrics.index(m)]) if std_arr is not None else (lambda m: None)  # noqa: E731

    n_panels = len(metrics) + (1 if legend == "panel" else 0)
    ncols = ncols or n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = create_subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows))

    for ax, metric in zip(axes, metrics):
        make_bars(ax, get(metric), colors=colors, labels=list(methods), yerr=get_std(metric), err=err,
                  annotate=annotate, fmt=fmt, **bar_kwargs)
        ax.set_ylabel(metric, fontsize=ylabel_size, labelpad=12)
        if ylims and metric in ylims:
            ax.set_ylim(ylims[metric])
        if sci_y:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    lkw = dict(legend_kwargs or {})
    if legend == "panel":
        legend_panel(axes[len(metrics)], source=axes[0], **lkw)
    elif legend == "first":
        axes[0].legend(**lkw)
    for ax in axes[n_panels:]:
        ax.set_axis_off()
    return fig, axes[: len(metrics)]


__all__ = [
    "summarize",
    "make_bars",
    "annotate_bars",
    "add_delta_arrows",
    "make_grouped_bar",
    "make_stacked_bar",
    "make_clustered_bars",
    "metric_panels",
]
