"""Tests for the historical score v2, the tested plan and the decision rules."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as EN  # noqa: E402

ROW = lambda edge, n_eff, ci, ve, te: {"n": n_eff * 4, "n_eff": n_eff, "edge": edge, "exp_R": edge + 0.05, "edge_ci": ci,
                                      "val_edge": ve, "test_edge": te, "profit": 40.0, "target": 15.0, "gap_loss": 3.0}
TAB = {"base_R": 0.05, "test_from": "2025-04-03", "stocks": 87, "from": "2017-11-07", "to": "2026-10-23",
       "buckets": {"<20": ROW(0.30, 40, [0.1, 0.5], 0.2, 0.3), "20-25": ROW(0.30, 2000, [0.12, 0.48], 0.25, 0.35),
                   "25-30": ROW(0.06, 1500, [-0.02, 0.14], 0.06, 0.05), "30-35": ROW(0.06, 3000, [0.01, 0.11], 0.07, -0.02),
                   "35-40": ROW(0.03, 4000, [-0.01, 0.07], 0.03, 0.03), "40-50": ROW(0.0, 9000, [-0.02, 0.02], 0.0, 0.01),
                   "50-60": ROW(0.0, 9000, [-0.02, 0.02], 0.0, 0.0), "60-70": ROW(-0.03, 6000, [-0.06, 0.0], -0.02, -0.07),
                   ">70": {"n": 10}}}


class Hist(unittest.TestCase):
    def test_not_covered_assets_are_na(self):
        for t in ("leveraged_etf", "inverse_etf", "crypto"):
            self.assertIsNone(EN.hist_score(22, {"type": t}, tab=TAB)["value"])

    def test_significant_and_stable_can_exceed_cap(self):
        s = EN.hist_score(22, {"type": "stock"}, tab=TAB)
        self.assertGreater(s["value"], 6.4)
        self.assertTrue(s["significant"] and s["stable"])

    def test_ci_with_zero_is_capped(self):
        s = EN.hist_score(27, {"type": "stock"}, tab=dict(TAB, buckets=dict(TAB["buckets"], **{"25-30": ROW(0.5, 5000, [-0.01, 0.9], 0.4, 0.5)})))
        self.assertLessEqual(s["value"], 6.4)

    def test_unstable_sign_is_capped(self):
        s = EN.hist_score(32, {"type": "stock"}, tab=dict(TAB, buckets=dict(TAB["buckets"], **{"30-35": ROW(0.5, 5000, [0.2, 0.8], 0.6, -0.1)})))
        self.assertLessEqual(s["value"], 5.5)

    def test_small_group_pooled_or_na(self):
        s = EN.hist_score(15, {"type": "stock"}, tab=TAB)   # <20 has n_eff 40 -> pooled with 20-25
        self.assertEqual(s["bucket"], "<25")
        s2 = EN.hist_score(80, {"type": "stock"}, tab=TAB)  # >70 has no data -> N/A, not a made-up value
        self.assertIsNone(s2["value"])

    def test_blockers_cap_at_5(self):
        s = EN.hist_score(22, {"type": "stock"}, ["דוח כספי בעוד 3 ימים"], tab=TAB)
        self.assertEqual(s["value"], 5.0)

    def test_neutral_is_five(self):
        self.assertEqual(EN.hist_score(45, {"type": "stock"}, tab=TAB)["value"], 5.0)


X = {"price": 100.0, "atr": 2.0, "rsi": 22.0, "res": 110.0, "swing_low": 95.0, "sma20": 105, "sma50": 108, "sma150": 110,
     "sma200": 112, "regime": "neutral"}


class PlanDecision(unittest.TestCase):
    def test_plan_stop_fixed_and_targets(self):
        p = EN.plan_v2(X)
        self.assertEqual(p["stop"], 98.0)
        self.assertEqual(p["t2"], 106.0)
        self.assertIsNone(p["t1"])          # resistance 110 is beyond 3R: no first target there
        p2 = EN.plan_v2(dict(X, res=101.0))
        self.assertTrue(p2["res_blocks"])  # resistance under 1R: no quality trade
        self.assertEqual(p2["stop"], 98.0)  # the stop was not moved to improve R/R

    def test_entry_possible_only_when_all_known_gates_pass(self):
        h = EN.hist_score(22, {"type": "stock"}, tab=TAB)
        d = EN.decide(X, h, {"value": 4.0}, EN.plan_v2(X), {"earnings_known": True, "earnings_days": 40, "sec_known": False},
                      {"type": "stock"})
        self.assertEqual(d["code"], "go")
        self.assertIsNone([i for i in d["items"] if "SEC" in i["label"]][0]["ok"])  # unknown is not a failure

    def test_wait_when_resistance_blocks(self):
        h = EN.hist_score(22, {"type": "stock"}, tab=TAB)
        d = EN.decide(dict(X, res=101.0), h, None, EN.plan_v2(dict(X, res=101.0)), {}, {"type": "stock"})
        self.assertEqual(d["code"], "wait")
        self.assertIn("התנגדות", d["missing"])

    def test_good_conditions_without_edge_is_wait_not_entry(self):
        x = dict(X, rsi=60.0, price=120.0)
        h = EN.hist_score(60, {"type": "stock"}, tab=TAB)
        d = EN.decide(x, h, {"value": 7.2}, EN.plan_v2(x), {}, {"type": "stock"})
        self.assertNotEqual(d["code"], "go")

    def test_no_sell_instruction_wording(self):
        x = dict(X, rsi=55.0)
        d = EN.decide(x, EN.hist_score(55, {"type": "stock"}, tab=TAB), {"value": 2.5}, EN.plan_v2(x), {}, {"type": "stock"})
        self.assertEqual(d["code"], "exposure")
        self.assertTrue(d["verdict"].startswith("אם אתה מחזיק"))

    def test_max_seven_items(self):
        d = EN.decide(X, None, None, None, {"premarket_below_stop": False}, {"type": "stock"})
        self.assertLessEqual(len(d["items"]), 7)


if __name__ == "__main__":
    unittest.main()
