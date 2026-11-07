"""Unit and edge tests for the v2 engine. Run: python -m unittest discover -s tests -v"""
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import engine as EN  # noqa: E402
import features as FX  # noqa: E402
import market as MK  # noqa: E402


def bars(n=600, seed=1, drift=0.0005, vol=0.02, split_at=None):
    rnd = random.Random(seed)
    t0, p = 1_600_000_000, 100.0
    T, O, H, L, C, V = [], [], [], [], [], []
    for i in range(n):
        o = p * (1 + rnd.gauss(0, vol / 3))
        c = o * (1 + rnd.gauss(drift, vol))
        h, l = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2))), min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        if split_at is not None and i < split_at:  # data adjusted for a 1:4 split: older prices / 4
            pass
        T.append(t0 + i * 86400); O.append(o); H.append(h); L.append(l); C.append(c); V.append(rnd.uniform(1e6, 3e6))
        p = c
    return T, O, H, L, C, V


class LookAhead(unittest.TestCase):
    def test_no_feature_uses_future_bars(self):
        """Every feature at bar t must equal the value computed from bars[:t+1] only."""
        T, O, H, L, C, V = bars(420, seed=3)
        spy = [x * 1.01 for x in C[::-1]][::-1]
        full = FX.compute(T, O, H, L, C, V, {"SPY": spy})
        for t in (260, 300, 350, 419):
            part = FX.compute(T[:t + 1], O[:t + 1], H[:t + 1], L[:t + 1], C[:t + 1], V[:t + 1], {"SPY": spy[:t + 1]})
            for k, s in full.items():
                a, b = s[t], part[k][t]
                if isinstance(a, float) and isinstance(b, float):
                    self.assertTrue(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), "%s differs at %d: %r vs %r" % (k, t, a, b))
                else:
                    self.assertEqual(a, b, "%s differs at %d" % (k, t))

    def test_pivot_confirmed_only_after_k_bars(self):
        h = [1, 2, 3, 10, 3, 2, 1, 1, 1]
        l = [x - 0.5 for x in h]
        c = [x - 0.2 for x in h]
        a = [1.0] * len(h)
        st = FX.structure(h, l, c, a, k=3)
        self.assertIsNone(st["last_ph"][5])      # pivot at 3 is not known at bar 5
        self.assertEqual(st["last_ph"][6], 10)   # confirmed at 3 + 3


class Indicators(unittest.TestCase):
    def test_adx_bounds_and_trend(self):
        n = 200
        h = [100 + i + 1 for i in range(n)]
        l = [100 + i - 1 for i in range(n)]
        c = [100 + i + 0.5 for i in range(n)]
        a, p, m = FX.adx(h, l, c)
        self.assertGreater(a[-1], 50)            # a steady trend has a strong ADX
        self.assertGreater(p[-1], m[-1])         # and it points up
        self.assertTrue(all(0 <= x <= 100 for x in a if x is not None))

    def test_rsi_matches_definition(self):
        c = [float(x) for x in range(1, 60)]
        self.assertEqual(round(FX.rsi(c)[-1]), 100)

    def test_avwap_and_profile(self):
        T, O, H, L, C, V = bars(200)
        av = FX.avwap(H, L, C, V, 150)
        self.assertIsNone(av[149])
        self.assertTrue(min(L[150:]) <= av[-1] <= max(H[150:]))
        vp = FX.volume_profile(H, L, V)
        self.assertTrue(vp["approx"])
        self.assertTrue(vp["va_low"] <= vp["poc"] <= vp["va_high"])

    def test_relative_strength(self):
        c = [100 * 1.01 ** i for i in range(130)]
        b = [100.0] * 130
        self.assertGreater(FX.rel_strength(c, b, 60)[-1], 0)
        self.assertAlmostEqual(FX.rel_strength(b, b, 60)[-1], 0)


class Regime(unittest.TestCase):
    def test_labels(self):
        up = [100 * 1.002 ** i for i in range(400)]
        dn = [100 * 0.998 ** i for i in range(400)]
        self.assertEqual(MK.regime_series(up, up, [14.0] * 400)[1][-1], "on")
        self.assertEqual(MK.regime_series(dn, dn, [35.0] * 400)[1][-1], "off")

    def test_missing_vix_is_skipped_not_guessed(self):
        up = [100 * 1.002 ** i for i in range(400)]
        p1 = MK.regime_series(up, up, None)[0][-1]
        p2 = MK.regime_series(up, up, [None] * 400)[0][-1]
        self.assertEqual(p1, p2)

    def test_asset_types(self):
        self.assertEqual(MK.asset_type("RDTL", {"instrumentType": "ETF", "longName": "Defiance Daily Target 2X Long RDDT ETF"})["type"], "leveraged_etf")
        self.assertEqual(MK.asset_type("SQQQ", {"instrumentType": "ETF", "longName": "ProShares UltraPro Short QQQ -3x"})["type"], "inverse_etf")
        self.assertEqual(MK.asset_type("SPY", {"instrumentType": "ETF", "longName": "SPDR S&P 500 ETF Trust"})["type"], "etf")
        self.assertEqual(MK.asset_type("BTC-USD", {"instrumentType": "CRYPTOCURRENCY"})["type"], "crypto")
        self.assertEqual(MK.asset_type("NVDA", {"instrumentType": "EQUITY", "longName": "NVIDIA Corporation"})["type"], "stock")


class Score(unittest.TestCase):
    def x_full(self):
        T, O, H, L, C, V = bars(400, seed=5)
        f = FX.compute(T, O, H, L, C, V, {"SPY": C})
        x = {k: FX.last(f, k) for k in f}
        x.update(price=C[-1], chg1=0.5, regime="on", sector_trend=1, sector_etf="XLK")
        return x

    def test_range_and_coverage(self):
        s = EN.indicator_score(self.x_full(), {"rev_growth": 0.2, "op_margin": 0.2})
        self.assertTrue(0 <= s["value"] <= 10)
        self.assertEqual(s["coverage"], 1.0)

    def test_missing_family_is_not_zero(self):
        x = self.x_full()
        with_f = EN.indicator_score(x, None)  # no fundamentals: renormalised, not zero
        self.assertIsNotNone(with_f["value"])
        self.assertLess(with_f["coverage"], 1.0)
        fam = [c for c in with_f["components"] if c["key"] == "fundamentals"][0]
        self.assertIsNone(fam["value"])

    def test_low_coverage_is_na(self):
        s = EN.indicator_score({"price": 10}, None)
        self.assertIsNone(s["value"])
        self.assertEqual(s["label"], "N/A")

    def test_losses_not_floored_to_zero(self):
        v, _ = EN.fam_fundamentals({"op_margin": -0.5, "roe": -0.3})
        self.assertGreaterEqual(v, 1.5)

    def test_weights_from_config(self):
        cfg = {"indicator_score": {"weights": {"trend": 100}, "min_coverage": 0.5}}
        s = EN.indicator_score(self.x_full(), None, cfg=cfg)
        self.assertEqual(s["value"], [c for c in s["components"] if c["key"] == "trend"][0]["value"])


class Leveraged(unittest.TestCase):
    def test_daily_reset_path_effect(self):
        """A 2x daily product over a choppy path loses vs 2x the base return (volatility drag)."""
        base = [100.0]
        lev = [100.0]
        for i in range(40):
            r = 0.05 if i % 2 == 0 else -0.047
            base.append(base[-1] * (1 + r))
            lev.append(lev[-1] * (1 + 2 * r))
        naive = 100 * (1 + 2 * (base[-1] / 100 - 1))
        self.assertLess(lev[-1], naive)


if __name__ == "__main__":
    unittest.main()
