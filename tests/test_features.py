from __future__ import annotations

from cyber_trainer.features import angle_deg, ema
from cyber_trainer.pose import Point


def test_angle_right():
    # right angle at B
    a = Point(0, 0, 1.0)
    b = Point(0, 1, 1.0)
    c = Point(1, 1, 1.0)
    ang = angle_deg(a, b, c)
    assert 89.0 <= ang <= 91.0


def test_ema():
    assert ema(None, 10.0, 0.5) == 10.0
    assert abs(ema(10.0, 14.0, 0.5) - 12.0) < 1e-6
