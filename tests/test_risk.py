"""Portfolio risk and point-in-time fundamentals."""
import json
import os
import random
import sys
import tempfile
import unittest
import importlib.util

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import risk as RK  # noqa: E402


def rows(seed, n=400, drift=0.0004, vol=0.015, base=None, beta=1.0):
    rnd = random.Random(seed)
    p, out = 100.0, []
    for i in range(n):
        r = beta * (base[i] if base else 0) + rnd.gauss(drift, vol)
        p *= 1 + r
        out.append((1_600_000_000 + i * 86400, p, p * 1.01, p * 0.99, p, 1e6))
    return out


_R = random.Random(9)
MKT = [_R.gauss(0.0004, 0.01) for _ in range(400)]
DATA = {"SPY": ({"instrumentType": "ETF", "longName": "SPDR S&P 500 ETF Trust"}, rows(1, base=MKT, beta=1.0, vol=0.002)),
        "AAA": ({"instrumentType": "EQUITY", "longName": "Aaa Inc"}, rows(2, base=MKT, beta=1.5, vol=0.005)),
        "BBB": ({"instrumentType": "EQUITY", "longName": "Bbb Inc"}, rows(3, base=MKT, beta=1.5, vol=0.005)),
        "LEV": ({"instrumentType": "ETF", "longName": "Daily Target 2X Long AAA ETF"}, rows(4)[-100:])}
fetch = lambda s: DATA[s]


class Portfolio(unittest.TestCase):
    def test_no_holdings_means_nothing(self):
        self.assertIsNone(RK.portfolio({"cash": 0, "positions": []}, fetch))
        self.assertIsNone(RK.read_holdings(os.path.join(tempfile.gettempdir(), "nope_%d.json" % random.random())))

    def test_core_numbers(self):
        h = {"cash": 1000.0, "positions": [{"symbol": "AAA", "qty": 10, "stop": None, "cost": None},
                                           {"symbol": "BBB", "qty": 5, "stop": 1.0, "cost": 50.0},
                                           {"symbol": "LEV", "qty": 20, "stop": None, "cost": None}]}
        p = RK.portfolio(h, fetch, mc_paths=300)
        tot = sum(x["value"] for x in p["positions"]) + 1000
        self.assertAlmostEqual(p["equity"], tot, places=1)
        self.assertEqual(p["leveraged"], ["LEV"])
        lev = [x for x in p["positions"] if x["symbol"] == "LEV"][0]
        self.assertEqual(lev["eff"], round(lev["value"] * 2, 2))           # 2x counts twice
        self.assertGreater(lev["synthetic_days"], 0)                       # short history extended from AAA
        self.assertIn("AAA", p["without_stop"])
        self.assertTrue(any(a == "AAA" and b == "BBB" or a == "BBB" and b == "AAA" for a, b, c in p["corr_high"]))
        self.assertLess(p["stress"][0]["pnl"], 0)
        self.assertIn("10", p["mc"])
        self.assertLessEqual(p["mc"]["10"]["p05"], p["mc"]["10"]["median"])

    def test_leveraged_path_drag(self):
        prod, naive = RK.leveraged_path_demo([0.05, -0.047] * 10, 2, 20)
        self.assertLess(prod, naive)

    def test_read_holdings_skips_bad_rows(self):
        f = os.path.join(tempfile.gettempdir(), "h_test.json")
        with open(f, "w") as fh:
            json.dump({"cash": 5, "positions": [{"symbol": "aaa", "qty": "3"}, {"symbol": "x"}, {"symbol": "z", "qty": 0}]}, fh)
        h = RK.read_holdings(f)
        self.assertEqual([p["symbol"] for p in h["positions"]], ["AAA"])


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas not installed (runs in the research workflow)")
class PointInTime(unittest.TestCase):
    def test_first_filed_and_q4(self):
        import backtest4 as B4
        f = lambda s, e, v, filed, form="10-Q": {"start": s, "end": e, "val": v, "filed": filed, "form": form}
        cf = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            f("2023-01-01", "2023-03-31", 100, "2023-05-01"), f("2023-01-01", "2023-03-31", 999, "2024-05-01"),  # restated later
            f("2023-04-01", "2023-06-30", 110, "2023-08-01"), f("2023-07-01", "2023-09-30", 120, "2023-11-01"),
            f("2023-01-01", "2023-12-31", 460, "2024-02-20", "10-K")]}}}}}
        q = B4.pit_quarters(cf, ["Revenues"])
        self.assertEqual(q["2023-03-31"], (100.0, "2023-05-01"))   # the first report, not the restatement
        self.assertEqual(q["2023-12-31"], (130.0, "2024-02-20"))   # Q4 = year - Q1..Q3, known at the 10-K


if __name__ == "__main__":
    unittest.main()
