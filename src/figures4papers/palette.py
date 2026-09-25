"""Colors used across the figure_* scripts and small color utilities."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba

PALETTE = {
    # Proposed method / key result.
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#B8C9E5",
    "blue_mid": "#7097CA",
    # Improvements / related positives.
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    # Baselines / contrasts.
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_3": "#D88F8A",
    "red_strong": "#B64342",
    # Neutrals.
    "neutral": "#CFCECE",
    "gray_1": "#767676",
    "gray_2": "#4D4D4D",
    "gray_3": "#272727",
    # Accents.
    "highlight": "#FFD700",
    "yellow_light": "#FFF6CC",
    "magenta": "#EA84DD",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "navy": "#0C2458",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]

# Ready-made method color sequences. The proposed method comes LAST, as in the scripts.
GREENS = [PALETTE["green_1"], PALETTE["green_2"], PALETTE["green_3"]]
REDS = [PALETTE["red_1"], PALETTE["red_2"], PALETTE["red_3"]]
BASELINES_THEN_OURS = GREENS + REDS[:2] + [PALETTE["yellow_light"], PALETTE["blue_secondary"]]


def luminance(color) -> float:
    """Perceived luminance in [0, 1] (ITU-R BT.601 weights)."""
    r, g, b, _ = to_rgba(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def is_dark(color, threshold: float = 0.5) -> bool:
    """True if ``color`` is dark enough that white text should be drawn on it."""
    return luminance(color) < threshold


def text_color_for(color, light: str = "white", dark: str = "black") -> str:
    """Readable text color to put on top of ``color``."""
    return light if is_dark(color) else dark


def with_alpha(color, alpha: float) -> tuple:
    """``color`` as an RGBA tuple with the given alpha."""
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def alpha_ramp(color, n: int, lo: float = 0.2, hi: float = 1.0) -> list[tuple]:
    """``n`` copies of ``color`` with alpha from ``lo`` to ``hi``.

    Used for ablations where opacity encodes how complete the method is.
    """
    return [with_alpha(color, a) for a in np.linspace(lo, hi, n)]


def gradient(start, end, n: int) -> list[str]:
    """``n`` hex colors interpolated linearly from ``start`` to ``end``."""
    s, e = np.array(to_rgba(start)), np.array(to_rgba(end))
    return [to_hex(s + (e - s) * t) for t in np.linspace(0, 1, n)]


def ours_vs_baselines(
    n_baselines: int,
    ours: str = PALETTE["blue_main"],
    start: str = "#D4685F",
    end: str = "#FCEEED",
    ours_first: bool = True,
) -> list[str]:
    """Proposed method in blue plus baselines fading from ``start`` to ``end``.

    Mirrors the CellSpliceNet comparison figure.
    """
    baselines = gradient(start, end, n_baselines)
    return [ours] + baselines if ours_first else baselines + [ours]


def make_cmap(colors: Sequence, name: str = "custom", n: int = 256) -> LinearSegmentedColormap:
    """Linear colormap through ``colors``."""
    return LinearSegmentedColormap.from_list(name, list(colors), N=n)
