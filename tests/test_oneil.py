"""O'Neil rules: each base type is recognised from a hand-built price path, wrong shapes are rejected,
breakouts need volume, and the market-direction machine needs a follow-through day."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oneil as ON  # noqa: E402


def path(points):
    """Piecewise-linear closes through (bar, price) points; small bars around the close."""
    c = []
    for (i0, p0), (i1, p1) in zip(points, points[1:]):
        for k in range(i0, i1):
            c.append(p0 + (p1 - p0) * (k - i0) / (i1 - i0))
    c.append(points[-1][1])
    h = [x * 1.005 for x in c]
    l = [x * 0.995 for x in c]
    v = [1e6] * len(c)
    return h, l, c, v


def run_up(n=200, start=40.0, end=100.0):
    return [(0, start), (n, end)]


class Bases(unittest.TestCase):
    def test_cup_with_handle(self):
        pts = run_up() + [(215, 85), (235, 75), (250, 76), (270, 97), (277, 92), (285, 94)]  # 7+ wk cup, U bottom, handle
        h, l, c, v = path(pts)
        b = ON.detect_base(h, l, c, v, len(c) - 1)
        self.assertEqual(b["type"], "cup_handle")
        self.assertAlmostEqual(b["pivot"], round(max(h[270:]) + 0.10, 2), places=2)
        self.assertTrue(12 <= b["depth"] <= 33)

    def test_v_shape_rejected_as_cup(self):
        pts = run_up() + [(220, 75), (222, 74), (245, 98), (252, 93), (258, 95)]   # sharp V bottom
        h, l, c, v = path(pts)
        b = ON.detect_base(h, l, c, v, len(c) - 1)
        self.assertTrue(b is None or b["type"] not in ("cup_handle", "cup"))

    def test_too_deep_rejected(self):
        pts = run_up() + [(230, 50), (250, 50), (280, 97), (288, 93), (295, 95)]  # 50 % deep
        h, l, c, v = path(pts)
        self.assertIsNone(ON.detect_base(h, l, c, v, len(c) - 1))

    def test_flat_base(self):
        pts = run_up() + [(210, 92), (220, 99), (232, 91), (240, 98)]
        h, l, c, v = path(pts)
        self.assertEqual(ON.detect_base(h, l, c, v, len(c) - 1)["type"], "flat")

    def test_double_bottom(self):
        pts = run_up() + [(212, 78), (225, 90), (238, 77), (248, 88)]   # second low undercuts the first
        h, l, c, v = path(pts)
        b = ON.detect_base(h, l, c, v, len(c) - 1)
        self.assertEqual(b["type"], "double_bottom")
        self.assertAlmostEqual(b["pivot"], round(h[b["middle"]] + 0.10, 2), places=2)

    def test_high_tight_flag(self):
        pts = [(0, 30), (150, 30), (180, 40), (215, 85), (225, 72), (236, 78)]  # +100 % in < 8 weeks, then flag
        h, l, c, v = path(pts)
        self.assertEqual(ON.detect_base(h, l, c, v, len(c) - 1)["type"], "htf")

    def test_no_future_bars_used(self):
        pts = run_up() + [(215, 85), (235, 75), (250, 76), (270, 97), (277, 92), (285, 94), (300, 140)]
        h, l, c, v = path(pts)
        a = ON.detect_base(h, l, c, v, 285)
        b = ON.detect_base(h[:286], l[:286], c[:286], v[:286], 285)
        self.assertEqual(a, b)


class Breakout(unittest.TestCase):
    def test_needs_volume(self):
        pts = run_up() + [(215, 85), (235, 75), (250, 76), (270, 97), (277, 92), (285, 94)]
        h, l, c, v = path(pts)
        b = ON.detect_base(h, l, c, v, len(c) - 1)
        piv = b["pivot"]
        h2, l2, c2, v2 = h + [piv * 1.02], l + [piv * 0.99], c + [piv * 1.015], v + [1.2e6]
        self.assertFalse(ON.breakout_at(h2, l2, c2, v2, len(c2) - 1, b)["ok"])     # x1.2 volume: not enough
        v2[-1] = 1.6e6
        self.assertTrue(ON.breakout_at(h2, l2, c2, v2, len(c2) - 1, b)["ok"])
        c2[-1] = piv * 1.08
        self.assertFalse(ON.breakout_at(h2, l2, c2, v2, len(c2) - 1, b)["ok"])     # 8 % above: extended


class Market(unittest.TestCase):
    def test_correction_then_follow_through(self):
        c = [100 + i * 0.1 for i in range(80)]
        c += [c[-1] * (1 - 0.01 * k) for k in range(1, 13)]           # 12 down days: correction
        low = c[-1]
        c += [low * 1.003, low * 1.006, low * 1.008, low * 1.008 * 1.02]   # day 4 of the rally: +2 %
        v = [1e8] * len(c)
        v[-1] = 1.3e8                                                   # on higher volume
        st = ON.market_direction(c, v)
        self.assertEqual(st[-6]["state"], "correction")
        self.assertEqual(st[-1]["state"], "uptrend")
        self.assertEqual(st[-1]["ftd"], len(c) - 1)

    def test_no_ftd_without_volume(self):
        c = [100 + i * 0.1 for i in range(80)] 
        c += [c[-1] * (1 - 0.01 * k) for k in range(1, 13)]
        low = c[-1]
        c += [low * 1.003, low * 1.006, low * 1.008, low * 1.008 * 1.02]
        v = [1e8] * len(c)
        self.assertNotEqual(ON.market_direction(c, v)[-1]["state"], "uptrend")


class MarketPressure(unittest.TestCase):
    def test_distribution_without_price_damage_is_pressure_not_correction(self):
        c, v = [100.0], [1e8]
        for i in range(1, 140):                       # a rising market with many small high-volume down days
            down = i % 4 == 0
            c.append(c[-1] * (0.997 if down else 1.004))
            v.append(v[-1] * (1.1 if down else 0.95))
        st = ON.market_direction(c, v)
        self.assertNotIn(st[-1]["state"], ("correction", "rally"))
        self.assertTrue(any(x and x["state"] == "pressure" for x in st))


class FollowThroughResets(unittest.TestCase):
    def test_no_relapse_right_after_a_follow_through_day(self):
        c = [100 + i * 0.1 for i in range(80)]
        v = [1e8] * 80
        for k in range(12):                       # a correction on heavy, rising volume (many distribution days)
            c.append(c[-1] * 0.99)
            v.append(v[-1] * 1.05)
        low = c[-1]
        for x in (1.003, 1.006, 1.008):
            c.append(low * x)
            v.append(v[-1] * 0.9)
        c.append(c[-1] * 1.02)                    # day 4: follow-through on higher volume
        v.append(v[-1] * 1.3)
        for k in range(3):                        # quiet up days, still under the 50-day line
            c.append(c[-1] * 1.002)
            v.append(v[-1] * 0.95)
        st = ON.market_direction(c, v)
        self.assertEqual(st[-4]["state"], "uptrend")
        self.assertEqual(st[-1]["state"], "uptrend")


class Canslim(unittest.TestCase):
    def test_unknowns_are_none_not_false(self):
        items = ON.canslim()
        self.assertEqual(len(items), 7)
        self.assertTrue(all(i["ok"] is None for i in items))

    def test_c_requires_sales_too(self):
        ev = {"eps_yoy": 0.40, "rev_yoy": 0.05, "rev_yoy_prev": [0.10, 0.08], "end": "2026-09-30"}
        c = ON.canslim(ev)[0]
        self.assertFalse(c["ok"])
        ev["rev_yoy"] = 0.30
        self.assertTrue(ON.canslim(ev)[0]["ok"])


if __name__ == "__main__":
    unittest.main()


class PointInTimeGrowth(unittest.TestCase):
    def test_growth_known_only_after_filing(self):
        import secpit
        f = lambda s, e, v, filed, form="10-Q": {"start": s, "end": e, "val": v, "filed": filed, "form": form}
        q = [("2024-01-01", "2024-03-31", "2024-05-01"), ("2024-04-01", "2024-06-30", "2024-08-01"),
             ("2025-01-01", "2025-03-31", "2025-05-01"), ("2025-04-01", "2025-06-30", "2025-08-01")]
        eps = [f(a, b, x, c) for (a, b, c), x in zip(q, [1.0, 1.0, 1.5, 1.2])]
        rev = [f(a, b, x, c) for (a, b, c), x in zip(q, [100, 100, 140, 115])]
        eps.append(f("2025-01-01", "2025-03-31", 9.9, "2026-02-01"))   # a later restatement: ignored
        cf = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": eps}}, "Revenues": {"units": {"USD": rev}}}}}
        ev = secpit.growth_events(cf)
        self.assertIsNone(secpit.known_at(ev, "2025-05-01"))             # filed that day: usable from the next
        e1 = secpit.known_at(ev, "2025-05-02")
        self.assertAlmostEqual(e1["eps_yoy"], 0.5)
        self.assertAlmostEqual(e1["rev_yoy"], 0.4)
        self.assertAlmostEqual(secpit.known_at(ev, "2025-09-01")["eps_yoy"], 0.2)
