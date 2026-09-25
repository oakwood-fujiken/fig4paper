"""Conceptual sphere figures (ports of figure_Dispersion/plot_idea.py and plot_illustration.py)."""
import numpy as np
from matplotlib import pyplot as plt

import figures4papers as fp

RED, TEAL, NAVY = fp.PALETTE["red_strong"], fp.PALETTE["teal"], fp.PALETTE["navy"]

if __name__ == "__main__":
    fp.apply_publication_style("concept")
    fig = plt.figure(figsize=(24, 8))

    # Condensed vs. spread points on a sphere.
    for i, (spread, color) in enumerate([(np.pi / 4, "#6a98cb"), (2 * np.pi, "#cde5f8")]):
        ax = fig.add_subplot(1, 4, i + 1)
        fp.make_sphere_illustration(ax, alpha=0.3, ambient=-0.5, diffuse=2.0)
        pts = fp.sample_points_in_disk(16, theta_range=spread, rng=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=80, facecolor=color, edgecolor="black", alpha=0.8)
        for p in pts:
            ax.plot([p[0], 0], [p[1], 0], "--", color="black", linewidth=1, alpha=0.8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

    # Geodesic "disperse" arrows between points.
    ax = fig.add_subplot(1, 4, 3)
    fp.make_sphere_illustration(ax)
    pts = np.array([[-0.2, 0.6], [0.9, 0.0], [-0.75, -0.4]])
    ax.scatter(pts[:, 0], pts[:, 1], s=80, color=NAVY, alpha=0.5)
    fp.draw_geodesic(ax, pts[0], pts[1], color=RED, lw=4)
    fp.draw_geodesic(ax, pts[0], pts[2], color=RED, lw=4)
    fp.draw_geodesic(ax, pts[1], pts[2], color=TEAL, lw=4, linestyle="--", arrows=False, alpha=0.5)
    fp.text_box(ax, 0.45, 0.4, "acute angle,\ndisperse", color=RED, fontsize=20)
    fp.text_box(ax, 0.15, -0.36, "obtuse angle,\ndo nothing", color=TEAL, fontsize=20)
    ax.set_ylim(-1.6, 1.2)
    ax.legend(handles=[fp.arrow_legend_handle("Orthogonalization", RED, r"$\leftarrow\rightarrow$", 50)],
              loc="lower center", fontsize=22)

    # 3D arrows.
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    fp.clean_3d_axes(ax, box_aspect=None, elev=30, azim=-60)
    fp.draw_3d_frame(ax, 4, fontsize=24)
    Z = np.array([[3.0, -3.0, 0.0], [1.0, -3.0, 0.0], [-3.0, -2.0, -1.0], [0.0, 2.0, 2.0], [4.0, 0.0, 2.0]])
    ax.scatter(*Z.T, s=80, color=NAVY, alpha=0.5)
    for p in Z:
        fp.arrow3d(ax, p, p * 0.7, color=fp.PALETTE["violet"], lw=3, alpha=0.8)

    fp.finalize_figure(fig, "figures/illustration")
