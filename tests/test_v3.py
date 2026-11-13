"""Tests for v3: indicator formulas, data checks, levels, the two scores, the plan and the decision."""
import datetime
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataqa as DQ  # noqa: E402
import engine as EN  # noqa: E402
import engine3 as EN3  # noqa: E402
import features as F  # noqa: E402


def series(n=400, seed=1, drift=0.0005, vol=0.02):
    rnd = random.Random(seed)
    c, p = [], 100.0
    for _ in range(n):
        p *= 1 + rnd.gauss(drift, vol)
        c.append(p)
    o = [c[0]] + c[:-1]
    h = [max(a, b) * (1 + abs(rnd.gauss(0, 0.006))) for a, b in zip(o, c)]
    l = [min(a, b) * (1 - abs(rnd.gauss(0, 0.006))) for a, b in zip(o, c)]
    v = [1e6 * (1 + rnd.random()) for _ in range(n)]
    return o, h, l, c, v


def wilder_ref(x, n):
    out, s = [None] * len(x), None
    first = next(i for i, v in enumerate(x) if v is not None)
    for i in range(len(x)):
        if i == first + n - 1:
            s = sum(x[first:i + 1]) / n
        elif i > first + n - 1:
            s = s + (x[i] - s) / n
        out[i] = s
    return out


GOOD = {"price": 100.0, "atr": 2.0, "sma20": 99, "sma50": 96, "sma150": 92, "sma200": 90, "slope50": 1.0, "adx": 30,
        "rsi": 48, "cci": -20, "macd_h": 0.05, "atr_pctile": 40, "atr_pct": 2.0, "gap": 0.1, "state": 2, "bos_up": False,
        "bos_dn": False, "failed_bo": False, "swing_low": 97.5, "ext_atr": 0.5, "rvol": 1.1, "updown": 1.4, "obv20": 1,
        "clv": 0.6, "rs_spy_60": 6.0, "regime": "on", "move_atr": 0.2}
LV = {"support": [{"price": 97.5, "touches": 3, "last": 380, "flip": True, "dist_atr": -1.25}],
      "resistance": [{"price": 112.0, "touches": 2, "last": 300, "flip": False, "dist_atr": 6.0}], "gap_up": None, "gap_down": None}
HIST_OK = {"value": 7.2, "edge": 0.2, "edge_ci": [0.05, 0.35], "significant": True, "stable": True, "test_edge": 0.15}
HIST_NS = {"value": 5.7, "edge": 0.06, "edge_ci": [-0.04, 0.17], "significant": False, "stable": True, "test_edge": 0.05}
EV_OK = {"earnings_known": True, "earnings_days": 40}


class Indicators(unittest.TestCase):
    def test_rsi_atr_match_wilder_from_the_first_value(self):
        o, h, l, c, v = series(120)
        up = [None] + [max(c[i] - c[i - 1], 0) for i in range(1, len(c))]
        dn = [None] + [max(c[i - 1] - c[i], 0) for i in range(1, len(c))]
        au, ad = wilder_ref(up, 14), wilder_ref(dn, 14)
        r = F.rsi(c)
        for i in range(15, 120):
            self.assertAlmostEqual(r[i], 100 - 100 / (1 + au[i] / ad[i]), places=6)
        tr = F.true_range(h, l, c)
        a, ref = F.atr(h, l, c), wilder_ref(tr, 14)
        self.assertIsNone(a[12])
        for i in range(13, 120):
            self.assertAlmostEqual(a[i], ref[i], places=6)

    def test_sma_and_short_history(self):
        o, h, l, c, v = series(150)
        s = F.sma(c, 200)
        self.assertTrue(all(x is None for x in s))  # not enough history: no value, not a guess
        self.assertAlmostEqual(F.sma(c, 50)[-1], sum(c[-50:]) / 50, places=9)

    def test_closed_bar_values_do_not_change_when_a_bar_is_added(self):
        o, h, l, c, v = series(300)
        a = F.compute(list(range(300)), o, h, l, c, v)
        b = F.compute(list(range(299)), o[:-1], h[:-1], l[:-1], c[:-1], v[:-1])
        for k in ("rsi", "atr", "cci", "sma50", "macd_h", "state"):
            self.assertEqual(a[k][298], b[k][298], k)


class Data(unittest.TestCase):
    def test_duplicates_and_bad_bars_are_dropped_not_repaired(self):
        t0 = int(datetime.datetime(2026, 1, 5, 15).timestamp())
        rows = [(t0, 10, 11, 9, 10.5, 100), (t0 + 3600, 10, 11, 9, 10.6, 120), (t0 + 86400, 10, 9.5, 9, 9.8, 100),
                (t0 + 2 * 86400, 10, 10.5, 9.5, 10.2, 0)]
        clean, issues = DQ.clean_bars(rows)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean[0][4], 10.6)  # the later bar of the same day wins
        self.assertEqual(len(issues), 2)

    def test_stale_and_short_history_levels(self):
        now = datetime.datetime(2026, 11, 13, 18, tzinfo=datetime.timezone.utc)
        day = 86400
        fresh = [(int(now.timestamp()) - (300 - i) * day, 1, 1.1, 0.9, 1, 10) for i in range(300)]
        self.assertEqual(DQ.check(fresh, now=now)["level"], "ok")
        old = [(t - 10 * day, *r) for t, *r in fresh]
        self.assertEqual(DQ.check(old, now=now)["level"], "bad")
        short = fresh[-120:]
        r = DQ.check(short, now=now)
        self.assertEqual(r["level"], "partial")
        self.assertEqual(DQ.check(fresh[-40:], now=now)["level"], "bad")


class Levels(unittest.TestCase):
    def test_at_most_three_each_side_and_none_invented(self):
        o, h, l, c, v = series(400, seed=4)
        a = F.atr(h, l, c)
        lv = F.levels(h, l, c, a)
        self.assertLessEqual(len(lv["support"]), 3)
        self.assertLessEqual(len(lv["resistance"]), 3)
        self.assertTrue(all(x["price"] < c[-1] for x in lv["support"]))
        self.assertTrue(all(x["price"] > c[-1] for x in lv["resistance"]))
        up = [x * 1.003 ** i for i, x in enumerate([100] * 300)]  # a straight rise: no swing high above the price
        lv2 = F.levels([x * 1.001 for x in up], [x * 0.999 for x in up], up, F.atr([x * 1.001 for x in up], [x * 0.999 for x in up], up))
        self.assertEqual(lv2["resistance"], [])

    def test_levels_use_only_bars_known_at_end(self):
        o, h, l, c, v = series(400, seed=5)
        a = F.atr(h, l, c)
        self.assertEqual(F.levels(h, l, c, a, end=300), F.levels(h[:301], l[:301], c[:301], a[:301]))

    def test_unfilled_gap(self):
        c = [100.0] * 50 + [110.0] * 30
        h = [x + 0.5 for x in c]
        l = [x - 0.5 for x in c]
        a = F.atr(h, l, c)
        g = F.levels(h, l, c, a)["gap_up"]
        self.assertEqual((g["low"], g["high"]), (100.5, 109.5))


class Scores(unittest.TestCase):
    def test_bounds_and_missing_data_never_helps(self):
        t = EN3.technical_score(GOOD)
        self.assertTrue(0 <= t["value"] <= 10)
        part = {k: v for k, v in GOOD.items() if k not in ("updown", "obv20", "rvol", "rs_spy_60", "regime")}
        p = EN3.technical_score(part)
        self.assertLess(abs(p["value"] - 5), abs(t["value"] - 5) + 1e-9)
        self.assertIsNone(EN3.technical_score({"price": 10})["value"])

    def test_momentum_is_one_family(self):
        names = [c["key"] for c in EN3.technical_score(GOOD)["components"]]
        self.assertEqual(names, ["trend", "structure", "momentum", "volume", "rs", "vol"])

    def test_no_structural_stop_no_rr_and_capped(self):
        x = dict(GOOD, swing_low=None)
        p = EN3.plan_v3(x, {"support": [], "resistance": []})
        self.assertIsNone(p["stop"])
        self.assertIsNone(p["rr"])
        e = EN3.entry_score(x, p, HIST_OK, EV_OK, {"support": []})
        self.assertLessEqual(e["value"], 4.5)

    def test_a_jump_cannot_make_a_high_entry_score(self):
        p0 = EN3.plan_v3(GOOD, LV)
        e0 = EN3.entry_score(GOOD, p0, HIST_OK, EV_OK, LV)["value"]
        jumped = EN3.plan_v3(GOOD, LV, {"price": 104.0, "session_open": True})  # +2 ATR today
        e1 = EN3.entry_score(GOOD, jumped, HIST_OK, EV_OK, LV)["value"]
        self.assertLessEqual(e1, 4.0)
        self.assertGreater(e0, e1)
        big_bar = dict(GOOD, move_atr=2.5)
        self.assertLessEqual(EN3.entry_score(big_bar, EN3.plan_v3(big_bar, LV), HIST_OK, EV_OK, LV)["value"], 4.0)

    def test_structural_stop_and_targets(self):
        p = EN3.plan_v3(GOOD, LV)
        self.assertAlmostEqual(p["stop"], 97.3, places=2)  # 0.1 ATR under the 97.5 support
        self.assertEqual(p["t2"], round(100 + 3 * (100 - 97.3), 2))
        self.assertIsNone(p["t1"])  # 112 is beyond 3R (108.1)
        self.assertEqual(p["rr"], 3.0)


class Decision(unittest.TestCase):
    def dec(self, x=GOOD, hist=HIST_OK, ev=EV_OK, live=None, lv=LV, held=False, data="ok", blockers=None):
        p = EN3.plan_v3(x, lv, live)
        t = EN3.technical_score(x)
        e = EN3.entry_score(x, p, hist, ev, lv)
        return EN3.decide_v3(x, t, e, p, hist, ev, blockers, {"type": "stock"}, held, data), e

    def test_go_needs_everything(self):
        d, e = self.dec()
        self.assertEqual(d["code"], "go", (d["missing"], e["value"]))
        self.assertEqual(self.dec(hist=HIST_NS)[0]["code"], "wait")  # not significant -> no entry
        self.assertEqual(self.dec(data="bad")[0]["code"], "wait")
        self.assertEqual(self.dec(ev={"earnings_known": True, "earnings_days": 5})[0]["code"], "no")
        self.assertEqual(self.dec(ev={})[0]["code"], "wait")  # unknown earnings date: not confirmed

    def test_sndk_case_jump_after_signal(self):
        d, _ = self.dec(live={"price": 104.0, "session_open": True})
        self.assertEqual(d["code"], "no")
        self.assertTrue(any("התרחק" in h for h in d["hard"]))

    def test_falling_knife_is_not_an_entry(self):
        x = dict(GOOD, state=-2, price=85.0, sma20=88, sma50=95, sma200=99, rsi=27, swing_low=83.0)
        lv = {"support": [{"price": 83.0, "touches": 2, "last": 390, "flip": False, "dist_atr": -1}], "resistance": []}
        d, _ = self.dec(x=x, lv=lv)
        self.assertEqual(d["code"], "no")
        self.assertTrue(d["knife"])

    def test_reduce_only_for_a_held_position(self):
        x = dict(GOOD, state=-2, price=85.0, sma20=88, sma50=95, sma200=99, swing_low=83.0, bos_dn=True)
        lv = {"support": [{"price": 83.0, "touches": 2, "last": 390, "flip": False, "dist_atr": -1}], "resistance": []}
        self.assertEqual(self.dec(x=x, lv=lv, held=True)[0]["code"], "reduce")
        self.assertNotEqual(self.dec(x=x, lv=lv, held=False)[0]["code"], "reduce")

    def test_no_contradictions_on_random_inputs(self):
        rnd = random.Random(7)
        for _ in range(400):
            x = {k: (v * rnd.uniform(0.7, 1.3) if isinstance(v, float) else v) for k, v in GOOD.items()}
            x.update(state=rnd.choice([2, 0, -2, None]), rsi=rnd.uniform(10, 90), move_atr=rnd.uniform(-3, 3),
                     swing_low=rnd.choice([None, x["price"] - rnd.uniform(0.2, 8)]))
            lv = {"support": [{"price": x["price"] - rnd.uniform(0.2, 8), "touches": rnd.randint(1, 4), "last": 1, "flip": rnd.random() < 0.3, "dist_atr": 0}],
                  "resistance": [{"price": x["price"] + rnd.uniform(0.5, 15), "touches": rnd.randint(1, 4), "last": 1, "flip": False, "dist_atr": 0}]}
            live = {"price": x["price"] * rnd.uniform(0.95, 1.05), "session_open": rnd.random() < 0.5}
            d, e = self.dec(x=x, hist=rnd.choice([HIST_OK, HIST_NS, None]), lv=lv, live=live)
            self.assertNotIn("contradictions", d)
            p = EN3.plan_v3(x, lv, live)
            self.assertEqual(EN3.consistency(d, e, p), [])
            if d["code"] == "go":
                self.assertGreaterEqual(e["value"], 6.5)
                self.assertTrue(p["in_zone"] and p["stop"] and p["rr"] >= 2)

    def test_weight_sensitivity_is_smooth(self):
        base = EN3.technical_score(GOOD)["value"]
        old = dict(EN3.CFG.get("technical_score") or {})
        try:
            for k in EN3.TECH_W:
                w = dict(EN3.TECH_W)
                w[k] += 5
                EN3.CFG["technical_score"] = {"weights": w}
                self.assertLess(abs(EN3.technical_score(GOOD)["value"] - base), 0.6)
        finally:
            EN3.CFG["technical_score"] = old


if __name__ == "__main__":
    unittest.main()


class RsScope(unittest.TestCase):
    def test_threshold_only_against_the_whole_market(self):
        import oneil as ON
        L = lambda meta: next(i for i in ON.canslim(rs_pct=94, rs_meta=meta) if i["letter"] == "L")
        self.assertTrue(L({"scope": "universe", "n": 5000, "date": "2026-11-12"})["ok"])
        self.assertIsNone(L({"scope": "narrow", "label": "87 מניות גדולות בלבד"})["ok"])  # not comparable to IBD's 80
        self.assertIn("לא דירוג IBD", L({"scope": "universe", "n": 5000})["detail"])
