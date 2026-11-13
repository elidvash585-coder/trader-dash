"""Event layer: halts feed, abnormal move, severity rules and the decision override."""
import datetime
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import events as EV  # noqa: E402

HALT_XML = """<?xml version="1.0"?><rss xmlns:ndaq="http://www.nasdaqtrader.com/"><channel><item>
<title>NVDA</title><ndaq:HaltDate>11/09/2026</ndaq:HaltDate><ndaq:HaltTime>10:31:02</ndaq:HaltTime>
<ndaq:IssueSymbol>NVDA</ndaq:IssueSymbol><ndaq:IssueName>NVIDIA Corp</ndaq:IssueName><ndaq:ReasonCode>T1</ndaq:ReasonCode>
<ndaq:ResumptionDate></ndaq:ResumptionDate><ndaq:ResumptionTradeTime></ndaq:ResumptionTradeTime></item></channel></rss>"""


def market(n=200, shock=0.0, vol_mult=1.0, seed=3):
    rnd = random.Random(seed)
    t0, spy, stk, cand = 1_700_000_000, [100.0], [50.0], []
    st = [t0]
    for i in range(1, n):
        m = rnd.gauss(0.0003, 0.01)
        r = 1.2 * m + rnd.gauss(0, 0.01) + (shock if i == n - 1 else 0)
        spy.append(spy[-1] * (1 + m))
        stk.append(stk[-1] * (1 + r))
        st.append(t0 + i * 86400)
    for i in range(n):
        v = 1e6 * (vol_mult if i == n - 1 else 1)
        cand.append([st[i], stk[i - 1] if i else stk[0], stk[i] * 1.01, stk[i] * 0.99, stk[i], v])
    return cand, (st, spy)


class Halts(unittest.TestCase):
    def test_parse(self):
        h = EV.parse_halts(HALT_XML)
        self.assertEqual(h["NVDA"][0]["code"], "T1")
        self.assertIn("חדשות", h["NVDA"][0]["reason"])
        self.assertEqual(EV.parse_halts("not xml"), {})


class Reaction(unittest.TestCase):
    def test_extreme_move_detected(self):
        c, spy = market(shock=-0.18, vol_mult=5)
        rx = EV.abnormal(c, *spy)
        self.assertLessEqual(rx["z"], -4)
        self.assertGreaterEqual(rx["rvol"], 3)

    def test_normal_day_is_quiet(self):
        c, spy = market()
        self.assertLess(abs(EV.abnormal(c, *spy)["z"]), 3.5)


def today_of(c):
    return datetime.datetime.utcfromtimestamp(c[-1][0]).date()


class Severity(unittest.TestCase):
    def test_extreme_move_is_critical_even_without_news(self):
        c, spy = market(shock=-0.18, vol_mult=5)
        e = EV.assess("X", c, spy, today=today_of(c))
        self.assertEqual(e["level"], "critical")
        self.assertEqual(e["verify"], "תנועה חריגה בלי הסבר בחדשות")

    def test_sec_502_is_critical_and_verified(self):
        d = datetime.date(2026, 11, 9)
        e = EV.assess("X", sec={"filings": [{"form": "8-K", "date": "2026-11-08", "items": ["5.02", "9.01"]}]}, today=d)
        self.assertEqual((e["level"], e["verify"]), ("critical", "מאומת"))

    def test_halt_is_critical(self):
        e = EV.assess("NVDA", halts=EV.parse_halts(HALT_XML))
        self.assertEqual(e["level"], "critical")

    def test_single_headline_alone_raises_nothing(self):
        now = datetime.datetime(2026, 11, 9, 15, 0)
        news = [{"title": "Acme CEO arrested in espionage case, report says", "time": "2026-11-09 14:00"}]
        self.assertIsNone(EV.assess("ACME", news=news, now=now, fund={"name": "Acme Corp"}))

    def test_two_serious_headlines_are_high_and_labelled(self):
        now = datetime.datetime(2026, 11, 9, 15, 0)
        news = [{"title": "Acme CEO arrested in espionage case", "time": "2026-11-09 14:00"},
                {"title": "Acme shares halted after CEO arrest report", "time": "2026-11-09 14:30"},
                {"title": "Acme CEO arrested in espionage case", "time": "2026-11-09 14:40"}]  # duplicate: counted once
        e = EV.assess("ACME", news=news, now=now, fund={"name": "Acme Corp"})
        self.assertEqual(e["level"], "high")
        self.assertEqual(e["verify"], "מדווח בכמה כותרות")
        self.assertEqual(e["headlines"], 2)

    def test_old_filing_ignored(self):
        e = EV.assess("X", sec={"filings": [{"form": "8-K", "date": "2026-10-01", "items": ["5.02"]}]}, today=datetime.date(2026, 11, 9))
        self.assertIsNone(e)


class Override(unittest.TestCase):
    def test_critical_event_suspends_entry_and_caps_score(self):
        import dashboard as D
        import engine as EN
        D.DEMO = True
        old = EN.TABLE
        EN.TABLE = {"base_R": 0.04, "test_from": "2025-04-03", "stocks": 1, "from": "x", "to": "y",
                    "buckets": {b: {"n": 4000, "n_eff": 2000, "edge": 0.3, "exp_R": 0.34, "edge_ci": [0.1, 0.5], "val_edge": 0.3,
                                    "test_edge": 0.3, "profit": 45.0, "target": 20.0} for b in
                                ["<20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-60", "60-70", ">70"]}}
        try:
            a = D.analyze("NVDA")
            h, p, d = D.engine_block(a["tech"], a["fund"], None, None, {"value": 5.0}, {"level": "critical", "verify": "מאומת"})
            self.assertLessEqual(h["value"], 5.0)
            self.assertIn(d["code"], ("no", "reduce"))  # v3: a material event inside the window -> do not enter
            self.assertTrue(any("אירוע מהותי" in x for x in d["hard"]))
            h2, _, d2 = D.engine_block(a["tech"], a["fund"], None, None, {"value": 5.0}, None)
            self.assertFalse(any("אירוע מהותי" in x for x in d2["hard"]))
            self.assertFalse(any("אירוע מהותי" in x for x in h2["caps"]))
        finally:
            EN.TABLE = old
            D.DEMO = False


if __name__ == "__main__":
    unittest.main()


class Relevance(unittest.TestCase):
    def test_headline_must_be_about_the_company(self):
        now = datetime.datetime(2026, 11, 9, 19, 0)
        t = ["NVIDIA Falls 5% as Board Probes Jensen Huang's Ties to Alleged Chinese Spy; AMD Rises 4%, Intel Holds Steady",
             "Nvidia's Huang scandal rattles AI industry built around him",
             "Nvidia Stock Drops On Spy Scandal, CEO Probe"]
        news = [{"title": x, "time": "2026-11-09 15:00"} for x in t]
        self.assertEqual(len(EV.headline_cluster(news, now, about=EV.name_pattern("NVDA", "NVIDIA Corporation"))), 3)
        self.assertEqual(len(EV.headline_cluster(news, now, about=EV.name_pattern("INTC", "Intel Corporation"))), 0)
        self.assertEqual(len(EV.headline_cluster(news, now, about=EV.name_pattern("AAPL", "Apple Inc."))), 0)
