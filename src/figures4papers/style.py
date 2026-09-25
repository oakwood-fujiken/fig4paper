"""rcParams presets shared by every figure_* script."""
from __future__ import annotations

import contextlib
import shutil
import warnings
from dataclasses import dataclass, replace
from typing import Iterator, Union

import matplotlib as mpl


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple = ("Helvetica", "Arial", "DejaVu Sans", "sans-serif")
    # Keep text editable in SVG output.
    svg_editable_text: bool = True
    # Embed TrueType fonts in PDF/PS (journals often reject Type 3 fonts).
    truetype_fonts: bool = True


PRESETS = {
    # Large comparison bar panels (font 24, spine 3).
    "bar": FigureStyle(font_size=24, axes_linewidth=3),
    # Paper subfigures / compact analytic plots.
    "compact": FigureStyle(font_size=16, axes_linewidth=2),
    # Conceptual illustrations.
    "concept": FigureStyle(font_size=18, axes_linewidth=1.5),
}

StyleLike = Union[FigureStyle, str, None]


def _resolve(style: StyleLike, **overrides) -> FigureStyle:
    if style is None:
        style = FigureStyle()
    elif isinstance(style, str):
        try:
            style = PRESETS[style]
        except KeyError:
            raise ValueError(f"Unknown style preset {style!r}; choose from {sorted(PRESETS)}") from None
    return replace(style, **overrides) if overrides else style


def style_rcparams(style: StyleLike = None, **overrides) -> dict:
    """The rcParams dict for ``style`` (a FigureStyle, a preset name or None)."""
    style = _resolve(style, **overrides)
    use_tex = style.use_tex
    if use_tex and shutil.which("latex") is None:
        warnings.warn("use_tex=True but no `latex` executable was found; falling back to mathtext.")
        use_tex = False

    params = {
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family) + list(mpl.rcParamsDefault["font.sans-serif"]),
        "font.size": style.font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": style.axes_linewidth,
        "legend.frameon": False,
        "text.usetex": use_tex,
        "savefig.facecolor": "white",
    }
    if style.svg_editable_text:
        params["svg.fonttype"] = "none"
    if style.truetype_fonts:
        params["pdf.fonttype"] = 42
        params["ps.fonttype"] = 42
    return params


def apply_publication_style(style: StyleLike = None, **overrides) -> FigureStyle:
    """Set the house style globally. Returns the resolved FigureStyle.

    >>> apply_publication_style("bar")
    >>> apply_publication_style(FigureStyle(font_size=24, axes_linewidth=3))
    >>> apply_publication_style("compact", use_tex=True)
    """
    resolved = _resolve(style, **overrides)
    # Missing fonts in the fallback list (e.g. Helvetica on Linux) are expected; silence the spam.
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(style_rcparams(resolved))
    return resolved


@contextlib.contextmanager
def publication_style(style: StyleLike = None, **overrides) -> Iterator[FigureStyle]:
    """Context-manager version of :func:`apply_publication_style`."""
    resolved = _resolve(style, **overrides)
    with mpl.rc_context(style_rcparams(resolved)):
        yield resolved
