"""Building blocks for conceptual (non-data) figures: shaded spheres, geodesics, 3D arrows."""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

EPSILON = 1e-6


def sphere_shading(
    resolution: int = 512,
    light_dir=(-0.5, 0.5, 0.8),
    ambient: float = 0.3,
    diffuse: float = 0.9,
    background: float = 1.0,
) -> np.ndarray:
    """Lambert-shaded unit disk as a ``(resolution, resolution)`` image in [0, 1].

    Pixels outside the disk are ``background`` (white by default).
    """
    xs = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(xs, xs)
    r2 = x**2 + y**2
    mask = r2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])
    normal = np.stack([x, y, z]) / (np.sqrt(x**2 + y**2 + z**2) + EPSILON)
    light = np.asarray(light_dir, dtype=float)
    light /= np.linalg.norm(light)
    intensity = np.maximum(0.0, np.tensordot(light, normal, axes=1))
    img = np.full_like(x, background)
    img[mask] = np.clip(ambient + diffuse * intensity, 0, 1)[mask]
    return img


def make_sphere_illustration(
    ax: Axes,
    light_dir=(-0.5, 0.5, 0.8),
    resolution: int = 512,
    alpha: float = 0.5,
    ambient: float = 0.3,
    diffuse: float = 0.9,
    cmap: str = "gray",
    center=(0.0, 0.0),
    radius: float = 1.0,
    axis_off: bool = True,
):
    """Draw a shaded disk that reads as a 3D sphere (Dispersion figures)."""
    img = sphere_shading(resolution, light_dir, ambient, diffuse)
    cx, cy = center
    im = ax.imshow(img, cmap=cmap, origin="lower", vmin=0, vmax=1, alpha=alpha,
                   extent=[cx - radius, cx + radius, cy - radius, cy + radius])
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    return im


def sample_points_in_disk(n: int, theta_range: float = 2 * np.pi, r_max: float = 0.95,
                          center_angle: float = np.pi / 2, rng=None) -> np.ndarray:
    """``n`` random 2D points in a disk sector (sqrt-radius for uniform area density)."""
    rng = np.random.default_rng(rng)
    r = np.sqrt(rng.uniform(0, r_max, n))
    theta = rng.uniform(center_angle - theta_range / 2, center_angle + theta_range / 2, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def lift_to_sphere(xy) -> np.ndarray:
    """Map 2D points in the unit disk to the upper unit hemisphere."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    z = np.sqrt(np.clip(1.0 - (xy**2).sum(axis=1), 0.0, 1.0))
    p = np.column_stack([xy, z])
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + EPSILON)


def slerp(p, q, n: int = 200) -> np.ndarray:
    """Great-circle arc from ``p`` to ``q`` on the unit sphere (``n`` points)."""
    p = np.asarray(p, dtype=float) / (np.linalg.norm(p) + EPSILON)
    q = np.asarray(q, dtype=float) / (np.linalg.norm(q) + EPSILON)
    theta = np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
    if theta < EPSILON:
        return np.repeat(p[None, :], n, axis=0)
    t = np.linspace(0.0, 1.0, n)[:, None]
    arc = (np.sin((1 - t) * theta) * p + np.sin(t * theta) * q) / (np.sin(theta) + EPSILON)
    return arc / (np.linalg.norm(arc, axis=1, keepdims=True) + EPSILON)


def draw_geodesic(
    ax: Axes,
    a2d,
    b2d,
    color="blue",
    lw: float = 2.0,
    linestyle="-",
    alpha: float = 0.8,
    arrows: bool = True,
    arrow_scale: float = 30,
    shorten: float = 0.04,
    n: int = 300,
):
    """Draw the projected great-circle arc between two disk points, with arrowheads at both ends."""
    arc = slerp(lift_to_sphere(a2d)[0], lift_to_sphere(b2d)[0], n=n)
    x, y = arc[:, 0], arc[:, 1]
    k = int(shorten * n)
    body = int(2 * k) if arrows else k
    ax.plot(x[body:n - body], y[body:n - body], color=color, lw=lw, solid_capstyle="round",
            alpha=alpha, linestyle=linestyle)
    if arrows:
        xs, ys = x[k:n - k], y[k:n - k]
        for tail, head in (((xs[1], ys[1]), (xs[0], ys[0])), ((xs[-2], ys[-2]), (xs[-1], ys[-1]))):
            ax.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>", color=color,
                                         mutation_scale=arrow_scale, lw=0))
    return ax


def text_box(ax: Axes, x, y, text: str, color="black", pad: float = 0.2, **kwargs):
    """Text on a white rounded box (label over busy illustrations)."""
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    bbox = dict(facecolor="white", alpha=1, edgecolor="none", boxstyle=f"round,pad={pad}")
    return ax.text(x, y, text, color=color, bbox=bbox, **kwargs)


def arrow_legend_handle(label: str, color, symbol: str = r"$\rightarrow$", markersize: float = 35,
                        alpha: float = 0.8) -> Line2D:
    """Legend entry drawn as an arrow glyph (e.g. "Decorrelation")."""
    return Line2D([], [], color=color, alpha=alpha, marker=symbol, linestyle="None",
                  markersize=markersize, label=label)


class Arrow3D(FancyArrowPatch):
    """Arrow between two 3D points on an ``Axes3D`` (add with ``ax.add_artist``)."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        from mpl_toolkits.mplot3d import proj3d

        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def arrow3d(ax, start, end, **kwargs) -> Arrow3D:
    """Add an :class:`Arrow3D` from ``start`` to ``end`` (3-vectors)."""
    kwargs.setdefault("arrowstyle", "->")
    kwargs.setdefault("mutation_scale", 16)
    arrow = Arrow3D(*zip(start, end), **kwargs)
    ax.add_artist(arrow)
    return arrow


def clean_3d_axes(ax, box_aspect=(1, 1, 0.5), elev: Optional[float] = None, azim: Optional[float] = None):
    """Hide panes, axis lines and ticks of an ``Axes3D`` (surface/manifold plots)."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color((0, 0, 0, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)
    return ax


def draw_3d_frame(ax, length: float, color="black", fontsize: float = 36, labels=("x", "y", "z"),
                  scales=(1.0, 1.0, 1.2)):
    """Draw x/y/z arrows from the origin on a cleaned ``Axes3D`` (Dispersion l2-repel panel)."""
    sx, sy, sz = scales
    ax.quiver(0, 0, 0, 0, -sx * length, 0, color=color, linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, sy * length, 0, 0, color=color, linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, sz * length, color=color, linewidth=2, arrow_length_ratio=0.1)
    ax.text(0, -sx * length * 1.2, 0, labels[0], color=color, fontsize=fontsize)
    ax.text(sy * length * 1.05, -0.2, 0, labels[1], color=color, fontsize=fontsize)
    ax.text(-0.2, 0, sz * length * 1.05, labels[2], color=color, fontsize=fontsize)
    return ax
