"""Saving figures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SUPPORTED_FORMATS = ("pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff")
DEFAULT_FORMATS = ("png", "pdf")


def finalize_figure(
    fig: Figure,
    out_path,
    formats: Optional[Iterable[str]] = None,
    dpi: int = 300,
    close: bool = True,
    pad: Optional[float] = 2.0,
    tight_bbox: bool = False,
    pad_inches: float = 0.1,
    **savefig_kwargs,
) -> list[Path]:
    """Lay out ``fig`` and save it in one or more formats.

    ``out_path`` may carry an extension (``figures/ablation.png``) or not
    (``figures/ablation``). Without an extension and without ``formats``,
    PNG and PDF are written. Parent directories are created.

    ``pad`` is passed to ``fig.tight_layout``; set ``pad=None`` to skip it
    (e.g. after a manual ``subplots_adjust``). ``tight_bbox=True`` saves with
    ``bbox_inches='tight'`` (useful for radar charts and text outside axes).
    """
    out_path = Path(out_path)
    suffix = out_path.suffix.lstrip(".").lower()
    if suffix in SUPPORTED_FORMATS:
        base = out_path.with_suffix("")
        formats = list(formats) if formats is not None else [suffix]
    else:
        base = out_path
        formats = list(formats) if formats is not None else list(DEFAULT_FORMATS)

    formats = [f.lstrip(".").lower() for f in formats]
    unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
    if unknown:
        raise ValueError(f"Unsupported format(s) {unknown}; supported: {SUPPORTED_FORMATS}")

    if pad is not None:
        fig.tight_layout(pad=pad)
    if tight_bbox:
        savefig_kwargs.setdefault("bbox_inches", "tight")
        savefig_kwargs.setdefault("pad_inches", pad_inches)

    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = base.parent / f"{base.name}.{fmt}"
        fig.savefig(path, dpi=dpi, **savefig_kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved
