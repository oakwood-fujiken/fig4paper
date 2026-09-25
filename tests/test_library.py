import os
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import figures4papers as fp  # noqa: E402

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_is_dark():
    assert fp.is_dark("#0F4D92")
    assert not fp.is_dark("#FFD700")
    assert fp.text_color_for("black") == "white"


def test_palette_helpers():
    assert len(fp.alpha_ramp("#3775BA", 5)) == 5
    g = fp.gradient("#000000", "#ffffff", 3)
    assert g[0] == "#000000" and g[-1] == "#ffffff"
    colors = fp.ours_vs_baselines(4)
    assert colors[0] == fp.PALETTE["blue_main"] and len(colors) == 5


def test_style_presets():
    with fp.publication_style("bar") as s:
        assert matplotlib.rcParams["font.size"] == 24
        assert matplotlib.rcParams["axes.spines.top"] is False
    assert s.axes_linewidth == 3
    with pytest.raises(ValueError):
        fp.apply_publication_style("nope")


def test_finalize_formats(tmp_path):
    fig, ax = plt.subplots()
    paths = fp.finalize_figure(fig, tmp_path / "sub" / "a")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    fig, ax = plt.subplots()
    paths = fp.finalize_figure(fig, tmp_path / "b.svg")
    assert [p.name for p in paths] == ["b.svg"] and paths[0].exists()
    with pytest.raises(ValueError):
        fp.finalize_figure(plt.figure(), tmp_path / "c", formats=["docx"])


def test_summarize():
    mean, err = fp.summarize(np.array([[1.0, 3.0], [2.0, 2.0]]))
    np.testing.assert_allclose(mean, [2, 2])
    np.testing.assert_allclose(err, [1, 0])
    mean, err = fp.summarize([1, 2, 3])
    assert err is None


def test_make_bars_and_annotations():
    fig, ax = plt.subplots()
    bars = fp.make_bars(ax, [0.9, 0.8, 0.7], labels=["a", "b", "c"], annotate=True)
    assert len(bars) == 3 and len(ax.texts) == 3
    fp.add_delta_arrows(ax, bars)
    assert len(ax.texts) == 3 + 2 * 2  # arrow annotations are texts too
    fig, ax = plt.subplots()
    bars = fp.make_bars(ax, [1, 2], horizontal=True)
    texts = fp.annotate_bars(ax, bars, position="center")
    assert texts[0].get_text() == "1.00"


def test_grouped_bar_validation():
    fig, ax = plt.subplots()
    fp.make_grouped_bar(ax, ["A", "B"], [[1, 2], [2, 1]], ["x", "y"], annotate=True)
    with pytest.raises(ValueError):
        fp.make_grouped_bar(ax, ["A", "B", "C"], [[1, 2]], ["x"])


def test_stacked_and_clustered():
    fig, ax = plt.subplots()
    containers = fp.make_stacked_bar(ax, np.full((3, 2), 0.5), hatches=["/", ""], annotate_layer=0)
    assert len(containers) == 2
    np.testing.assert_allclose([b.get_y() for b in containers[1]], 0.5)
    fig, ax = plt.subplots()
    fp.make_clustered_bars(ax, {"d1": [1, 2], "d2": [3, 4]}, labels=["m1", "m2"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["d1", "d2"]


def test_metric_panels_array_and_mapping():
    fig, axes = fp.metric_panels(np.random.rand(3, 2), ["a", "b", "c"], metrics=["m1", "m2"])
    assert len(axes) == 2 and len(fig.axes) == 3
    fig, axes = fp.metric_panels({"m": np.random.rand(3, 5)}, ["a", "b", "c"], legend=False)
    assert len(fig.axes) == 1


def test_trend_and_lines():
    fig, ax = plt.subplots()
    lines = fp.make_trend(ax, [1, 2, 3], [np.random.rand(3, 4), [1, 2, 3]], ["a", "b"])
    assert len(lines) == 2
    with pytest.raises(ValueError):
        fp.make_trend(ax, [1, 2], [[1, 2, 3]])
    fp.make_gradient_line(ax, [0, 1, 2], [1, 2, 3], "red", label="g")
    texts = fp.mark_events(ax, ["a", "b", "c"], [1, 2, 3], {"b": "B*", "z": "missing"})
    assert len(texts) == 1 and texts[0].get_text() == "B"


def test_heatmaps():
    fig, ax = plt.subplots()
    fp.make_heatmap(ax, np.arange(6).reshape(2, 3), annotate=True, fmt="{:.0f}")
    assert len(ax.texts) == 6
    fig, ax = plt.subplots()
    fp.make_column_heatmap(ax, np.random.rand(3, 2), higher_is_better=[True, False], summary_row=[1, -1])
    assert len(ax.texts) == 8


def test_radar():
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    fp.make_radar(ax, {"x\nA": [1, 2], "y\nA": [2, 3], "z\nB": [5, 6]}, ["m1", "m2"])
    with pytest.raises(ValueError):
        fp.make_radar(ax, {"x": [1, 2]}, ["only one"])


def test_illustration():
    img = fp.sphere_shading(32)
    assert img.shape == (32, 32) and img[0, 0] == 1.0
    arc = fp.slerp([1, 0, 0], [0, 1, 0], n=5)
    np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 1, atol=1e-5)
    fig, ax = plt.subplots()
    fp.make_sphere_illustration(ax)
    fp.draw_geodesic(ax, [0.1, 0.2], [0.5, -0.3])


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_examples_run(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")
    assert list((tmp_path / "figures").glob("*.png"))
