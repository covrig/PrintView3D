"""The Measure tool: distances, angles and circles on the model."""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from printview3d.settings import Settings  # noqa: E402
from printview3d.ui import measure as ms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CUBE = os.path.join(HERE, "data", "cube_binary.stl")        # 20 mm cube, 0..20 on every axis
GCODE = os.path.join(HERE, "data", "cylinder.gcode")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    from PySide6.QtCore import QEvent
    from printview3d.ui import main_window as mw
    w = mw.MainWindow(Settings(str(tmp_path / "settings.json")))
    w.resize(1200, 760)
    w.show()
    yield w
    w.close()
    w.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def _screen(w, p):
    """Display position of a world point."""
    ren = w.plotter.renderer
    ren.SetWorldPoint(p[0], p[1], p[2], 1.0)
    ren.WorldToDisplay()
    d = ren.GetDisplayPoint()
    return int(round(d[0])), int(round(d[1]))


def _measuring(w, mode="distance"):
    w.load_sync(CUBE)
    w.plotter.ren_win.SetSize(800, 600)           # the offscreen window has no size until shown for real
    w.scene.set_view("iso", render=True)
    w.choose_measure_mode(mode)
    w.plotter.render()
    return w.measure


def _click(w, p):
    w.measure.click(*_screen(w, p))


# Corners of the cube that face the iso camera (front-right, from above): all but (0, 20, 0).
A, B, C, D = (0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 20.0, 0.0), (20.0, 20.0, 20.0)
TOP = (0.0, 0.0, 20.0)


# ------------------------------------------------------------ geometry
def test_distances():
    d, dx, dy, dz = ms.distances((0, 0, 0), (3, 4, 12))
    assert d == pytest.approx(13.0)
    assert (dx, dy, dz) == (3.0, 4.0, 12.0)
    assert ms.distances((1, 1, 1), (0, 1, 3))[1:] == (-1.0, 0.0, 2.0)       # signed: end minus start


def test_angle_at():
    assert ms.angle_at((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    assert ms.angle_at((1, 0, 0), (0, 0, 0), (-1, 0, 0)) == pytest.approx(180.0)
    assert ms.angle_at((1, 0, 0), (0, 0, 0), (1, 1, 0)) == pytest.approx(45.0)
    assert ms.angle_at((0, 0, 0), (0, 0, 0), (1, 0, 0)) is None            # no side


def test_circle_through_three_points():
    c, r, n = ms.circle_through((0, 0, 0), (2, 0, 0), (0, 2, 0))
    assert np.allclose(c, (1, 1, 0)) and r == pytest.approx(2 ** 0.5)
    assert abs(n[2]) == pytest.approx(1.0)
    # a tilted circle in space: three points of a 10 mm radius circle around (5, 5, 5)
    pts = [np.array((5, 5, 5)) + 10 * np.array((math.cos(t), math.sin(t) * 0.6, math.sin(t) * 0.8))
           for t in (0.3, 2.0, 4.4)]
    c, r, _ = ms.circle_through(*pts)
    assert np.allclose(c, (5, 5, 5), atol=1e-9) and r == pytest.approx(10.0)
    assert ms.circle_through((0, 0, 0), (1, 1, 1), (2, 2, 2)) is None      # on a line


def test_arc_and_circle_points_lie_where_they_should():
    arc = ms.arc_points((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0)
    assert np.allclose(np.linalg.norm(arc, axis=1), 2.0)
    assert np.allclose(arc[0], (2, 0, 0)) and np.allclose(arc[-1], (0, 2, 0))
    ring = ms.circle_points((1, 2, 3), 4.0, (0, 0, 1))
    assert np.allclose(np.linalg.norm(ring - (1, 2, 3), axis=1), 4.0) and np.allclose(ring[0], ring[-1])


def test_shapes_and_labels():
    s = ms.shape_of("distance", [(0, 0, 0), (3, 4, 0)])
    assert s.values["distance"] == pytest.approx(5.0) and "5" in s.label[1] and len(s.dist_lines) == 1
    assert ms.shape_of("distance", [(0, 0, 0)]).values is None
    s = ms.shape_of("angle", [(1, 0, 0), (0, 0, 0), (0, 1, 0)])
    assert s.values["angle"] == pytest.approx(90.0) and "90" in s.label[1] and "°" in s.label[1]
    assert not s.dist_lines and len(s.lines) == 3                         # two sides and the arc
    s = ms.shape_of("circle", [(0, 0, 0), (2, 0, 0), (0, 2, 0)])
    assert s.values["diameter"] == pytest.approx(2 * 2 ** 0.5)
    assert "Ø" in s.label[1] and "R" in s.label[1] and not s.dist_lines
    assert ms.shape_of("angle", [(1, 0, 0), (0, 0, 0)]).label is None     # one side so far


def test_is_valid():
    assert not ms.is_valid("circle", [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    assert ms.is_valid("circle", [(0, 0, 0), (1, 1, 1)])                  # not finished: fine
    assert not ms.is_valid("angle", [(0, 0, 0), (0, 0, 0), (1, 0, 0)])
    assert ms.is_valid("distance", [(0, 0, 0), (0, 0, 0)])


# ------------------------------------------------------------ the button and the panel
def test_measure_is_off_without_a_model_and_for_gcode(window):
    w = window
    assert not w.act_measure.isEnabled() and not w.measure_btn.isEnabled() and w.measure_box.isHidden()
    w.load_sync(CUBE)
    assert w.act_measure.isEnabled() and w.measure_btn.isEnabled()
    w.load_sync(GCODE)
    assert not w.act_measure.isEnabled() and not w.measure_btn.isEnabled()


def test_toolbar_button_sits_next_to_view_and_offers_three_modes(window):
    w = window
    tb = w.view_btn.parentWidget()
    assert w.measure_btn.parentWidget() is tb
    kids = [c for c in tb.findChildren(type(w.view_btn)) if c in (w.view_btn, w.measure_btn)]
    assert kids == [w.view_btn, w.measure_btn]                            # in toolbar order
    assert w.measure_btn.defaultAction() is w.act_measure
    assert w.measure_btn.menu() is w.measure_menu
    assert list(w.measure_actions) == ["distance", "angle", "circle"]
    assert w.act_measure.shortcut().toString() == "M"
    assert w.measure_actions["distance"].isChecked()


def test_turning_it_on_shows_the_panel_and_off_hides_it(window):
    w = window
    m = _measuring(w)
    assert m.active and w.measure_box.isVisible() and w.act_measure.isChecked()
    assert w.controls.measure is m
    w.measure_close_btn.click()
    assert not m.active and not w.act_measure.isChecked() and w.measure_box.isHidden()


def test_choosing_a_mode_turns_measure_on_and_is_remembered(window, tmp_path):
    w = window
    w.load_sync(CUBE)
    assert not w.measure.active
    w.measure_actions["angle"].trigger()
    assert w.measure.active and w.measure.mode == "angle" and w.act_measure.isChecked()
    assert w.settings.get("measure_mode") == "angle"
    assert "Angle" in w.measure_box.title() and "–" in w.measure_info.text()
    w.measure_actions["circle"].trigger()
    assert w.measure.mode == "circle" and "Diameter" in w.measure_info.text()


def test_mode_is_restored_from_settings(app, tmp_path):
    from printview3d.ui import main_window as mw
    s = Settings(str(tmp_path / "settings.json"))
    s.set("measure_mode", "circle")
    s.set("measure_dotted", True)
    w = mw.MainWindow(s)
    try:
        assert w.measure.mode == "circle" and w.measure_actions["circle"].isChecked()
        assert w.measure.dotted and w.measure_dotted_cb.isChecked()
    finally:
        w.close()


def test_closing_the_file_turns_it_off(window):
    w = window
    m = _measuring(w)
    w.go_home()
    assert not m.active and not w.act_measure.isChecked()


# ------------------------------------------------------------ distance
def test_two_clicks_measure_a_corner_to_corner_line(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    assert len(m.current.points) == 1
    _click(w, D)
    d, dx, dy, dz = (m.current_values()[k] for k in ("distance", "dx", "dy", "dz"))
    assert d == pytest.approx(20 * 3 ** 0.5, abs=1e-3)
    assert (dx, dy, dz) == pytest.approx((20.0, 20.0, 20.0), abs=1e-3)
    assert "34.64" in w.measure_info.text() and "ΔZ" in w.measure_delta.text()


def test_live_line_follows_the_mouse_until_the_second_click(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    m.hover(*_screen(w, B))
    assert m.current_values()["distance"] == pytest.approx(20.0, abs=1e-3)
    m.hover(*_screen(w, C))
    assert m.current_values()["distance"] == pytest.approx(20 * 2 ** 0.5, abs=1e-3)
    _click(w, B)
    m.hover(*_screen(w, D))                                # finished: the mouse no longer matters
    assert m.current_values()["distance"] == pytest.approx(20.0, abs=1e-3)


def test_value_is_written_next_to_the_line(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, D)
    shown = [a for a in m._labels if a.GetVisibility()]
    assert len(shown) == 1 and "34.64 mm" in shown[0].GetInput()
    m.clear()
    assert not any(a.GetVisibility() for a in m._labels)


# ------------------------------------------------------------ several measurements
def test_measurements_stay_and_the_next_click_starts_another(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, B)
    _click(w, B)
    _click(w, C)
    assert len(m.items) == 2 and all(i.complete for i in m.items)
    assert np.allclose(m.items[0].points[1], B, atol=1e-3)
    assert sum(1 for a in m._labels if a.GetVisibility()) == 2
    assert w.measure_clear_btn.isEnabled()
    w.measure_clear_btn.click()
    assert m.items == [] and not w.measure_clear_btn.isEnabled()


def test_click_on_nothing_does_nothing(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    m.click(1, 1)
    assert len(m.items) == 1 and len(m.items[0].points) == 1


def test_changing_the_mode_drops_the_unfinished_measurement_only(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, B)
    _click(w, C)                                           # a second one, unfinished
    assert len(m.items) == 2
    w.measure_actions["angle"].trigger()
    assert len(m.items) == 1 and m.items[0].kind == "distance"


# ------------------------------------------------------------ angle
def test_angle_on_a_cube_corner(window):
    w = window
    m = _measuring(w, "angle")
    _click(w, B)                                           # one side
    _click(w, A)                                           # the corner
    _click(w, TOP)                                         # the other side
    v = m.current_values()
    assert v["kind"] == "angle" and v["angle"] == pytest.approx(90.0, abs=1e-3)
    assert "90.0°" in w.measure_info.text()
    shown = [a for a in m._labels if a.GetVisibility()]
    assert len(shown) == 1 and shown[0].GetInput() == "90°" or "90" in shown[0].GetInput()


def test_angle_live_preview(window):
    w = window
    m = _measuring(w, "angle")
    _click(w, B)
    _click(w, A)
    m.hover(*_screen(w, C))
    assert m.current_values()["angle"] == pytest.approx(45.0, abs=1e-3)


# ------------------------------------------------------------ circle
def test_circle_through_three_cube_corners(window):
    w = window
    m = _measuring(w, "circle")
    _click(w, A)
    _click(w, B)
    _click(w, C)                                           # three corners of a 20 mm square
    v = m.current_values()
    assert v["diameter"] == pytest.approx(20 * 2 ** 0.5, abs=1e-3)
    assert v["radius"] == pytest.approx(10 * 2 ** 0.5, abs=1e-3)
    assert "28.28" in w.measure_info.text() and "14.14" in w.measure_delta.text()
    assert any("Ø" in a.GetInput() for a in m._labels if a.GetVisibility())


def test_three_points_on_a_line_make_no_circle(window):
    w = window
    m = _measuring(w, "circle")
    _click(w, A)
    _click(w, B)
    _click(w, (10.0, 0.0, 0.0))                            # between them, on the same edge
    assert len(m.items) == 1 and len(m.items[0].points) == 2


# ------------------------------------------------------------ snapping
def test_snap_jumps_to_the_corner_and_can_be_turned_off(window):
    w = window
    m = _measuring(w)
    x, y = _screen(w, (20.0, 0.0, 20.0))
    m.click(x + 5, y + 3)                                  # a few pixels off the corner
    assert np.allclose(m.items[0].points[0], (20, 0, 20), atol=1e-3)
    m.clear()
    w.measure_snap_cb.setChecked(False)
    assert w.settings.get("measure_snap") is False
    m.click(x + 5, y + 3)
    assert not np.allclose(m.items[0].points[0], (20, 0, 20), atol=1e-3)


def test_snap_to_the_middle_of_an_edge(window):
    w = window
    m = _measuring(w)
    x, y = _screen(w, (10.0, 0.0, 0.0))                    # the middle of the bottom front edge
    pos, how = m._pick(x + 3, y + 2)
    assert how == "mid" and np.allclose(pos, (10, 0, 0), atol=1e-3)


def test_snap_to_a_point_on_an_edge(window):
    w = window
    m = _measuring(w)
    x, y = _screen(w, (5.0, 0.0, 0.0))                     # a quarter along the same edge
    pos, how = m._pick(x, y + 3)
    assert how == "edge" and np.allclose(pos[1:], (0, 0), atol=1e-3) and 3.5 < pos[0] < 6.5


def test_a_flat_face_has_no_edge_to_snap_to(window):
    w = window
    m = _measuring(w)
    x, y = _screen(w, (10.0, 10.0, 20.0))                  # the middle of the top face: the
    pos, how = m._pick(x, y)                               # triangulation's diagonal runs here
    assert how == "free" and pos[2] == pytest.approx(20.0, abs=1e-3)


# ------------------------------------------------------------ moving a point
def test_dragging_a_placed_point_moves_it(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, B)
    sx, sy = _screen(w, B)
    assert m.grab(sx + 2, sy + 1)
    m.drag_to(*_screen(w, C))
    m.drop()
    assert np.allclose(m.items[0].points[1], C, atol=1e-3)
    assert m.current_values()["distance"] == pytest.approx(20 * 2 ** 0.5, abs=1e-3)


def test_a_press_away_from_the_points_grabs_nothing(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, B)
    sx, sy = _screen(w, D)
    assert not m.grab(sx, sy)


def test_a_drag_cannot_turn_a_circle_into_a_line(window):
    w = window
    m = _measuring(w, "circle")
    _click(w, A)
    _click(w, B)
    _click(w, C)
    sx, sy = _screen(w, C)
    assert m.grab(sx, sy)
    m.drag_to(*_screen(w, (10.0, 0.0, 0.0)))               # onto the line A-B: refused
    m.drop()
    assert np.allclose(m.items[0].points[2], C, atol=1e-3)


# ------------------------------------------------------------ dotted
def test_dotted_checkbox_dots_distance_angle_and_circle(window):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, D)
    assert m._lines_poly.GetNumberOfLines() == 1                           # solid: one line
    w.measure_dotted_cb.setChecked(True)
    assert w.settings.get("measure_dotted") is True
    assert m._lines_poly.GetNumberOfLines() > 10                           # dotted: many dashes
    w.measure.clear()
    for mode, pts in (("angle", (B, A, TOP)), ("circle", (A, B, C))):
        w.measure_actions[mode].trigger()
        for p in pts:
            _click(w, p)
        assert m._lines_poly.GetNumberOfLines() > 20, mode
        w.measure.clear()
    w.measure_dotted_cb.setChecked(False)
    w.measure_actions["angle"].trigger()
    for p in (B, A, TOP):
        _click(w, p)
    assert m._lines_poly.GetNumberOfLines() == 3                           # sides + arc, solid again


def test_overlapping_values_are_stacked(window):
    w = window
    m = _measuring(w, "circle")
    for _ in range(2):                                     # the same circle twice: same centre
        for p in (A, B, C):
            _click(w, p)
    shown = [a for a in m._labels if a.GetVisibility()]
    assert len(shown) == 2
    offsets = {tuple(a.GetDisplayOffset()) for a in shown}
    assert len(offsets) == 2                               # not on top of each other


# ------------------------------------------------------------ click vs. drag in the controls
def _patch_iren(w, monkeypatch, pos):
    monkeypatch.setattr(w.controls.iren, "GetEventPosition", lambda: pos[0], raising=False)
    monkeypatch.setattr(w.controls.iren, "GetShiftKey", lambda: 0, raising=False)


def test_click_places_a_point_and_does_not_set_the_pivot(window, monkeypatch):
    w = window
    m = _measuring(w)
    pos = [_screen(w, A)]
    _patch_iren(w, monkeypatch, pos)
    w.controls._left_down()
    assert w.controls.mode == "measure"
    pos[0] = (pos[0][0] + 1, pos[0][1])                    # a wobble is still a click
    w.controls._move()
    w.controls._button_up()
    assert len(m.items) == 1 and w.controls.mode is None


def test_drag_still_turns_the_view_and_places_nothing(window, monkeypatch):
    w = window
    m = _measuring(w)
    pos = [_screen(w, A)]
    _patch_iren(w, monkeypatch, pos)
    before = np.array(w.plotter.camera.position)
    w.controls._left_down()
    pos[0] = (pos[0][0] + 60, pos[0][1])
    w.controls._move()
    assert w.controls.mode == "rotate"
    w.controls._button_up()
    assert m.items == []
    assert not np.allclose(before, w.plotter.camera.position)


def test_drag_from_a_placed_point_moves_the_point_not_the_view(window, monkeypatch):
    w = window
    m = _measuring(w)
    _click(w, A)
    _click(w, B)
    pos = [_screen(w, B)]
    _patch_iren(w, monkeypatch, pos)
    before = np.array(w.plotter.camera.position)
    w.controls._left_down()
    pos[0] = _screen(w, C)
    w.controls._move()
    assert w.controls.mode == "measure_drag"
    w.controls._button_up()
    assert w.controls.mode is None
    assert np.allclose(m.items[0].points[1], C, atol=1e-3)
    assert np.allclose(before, w.plotter.camera.position)
