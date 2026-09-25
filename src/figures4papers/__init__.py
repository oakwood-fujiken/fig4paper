"""Publication-quality matplotlib helpers extracted from the figures4papers scripts.

Typical use::

    import figures4papers as fp

    fp.apply_publication_style("bar")
    fig, axes = fp.metric_panels(results, methods, colors, metrics=["AUROC", "AUPRC"])
    fp.finalize_figure(fig, "figures/comparison")  # -> comparison.png + comparison.pdf
"""
from .bars import (
    add_delta_arrows,
    annotate_bars,
    make_bars,
    make_clustered_bars,
    make_grouped_bar,
    make_stacked_bar,
    metric_panels,
    summarize,
)
from .heatmap import make_column_heatmap, make_heatmap
from .illustration import (
    Arrow3D,
    arrow3d,
    arrow_legend_handle,
    clean_3d_axes,
    draw_3d_frame,
    draw_geodesic,
    lift_to_sphere,
    make_sphere_illustration,
    sample_points_in_disk,
    slerp,
    sphere_shading,
    text_box,
)
from .io import finalize_figure
from .layout import create_subplots, legend_panel, line_handles, patch_handles, style_axis, tight_ylim
from .lines import add_reference_line, make_gradient_line, make_scatter, make_trend, mark_events
from .palette import (
    BASELINES_THEN_OURS,
    DEFAULT_COLORS,
    GREENS,
    PALETTE,
    REDS,
    alpha_ramp,
    gradient,
    is_dark,
    luminance,
    make_cmap,
    ours_vs_baselines,
    text_color_for,
    with_alpha,
)
from .radar import make_radar
from .style import PRESETS, FigureStyle, apply_publication_style, publication_style, style_rcparams

__version__ = "0.1.0"
