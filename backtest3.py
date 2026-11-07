#!/usr/bin/env python3
"""Score table for the site: how did the model trade do, by RSI(14) at the close?

    python backtest3.py          # real data
    python backtest3.py --demo   # synthetic data

The trade is the site's model: buy the next open, stop 1 ATR under the entry, target 3R,
exit after 10 sessions, 0.10 % round-trip cost. It is simulated on EVERY day of every stock,
then grouped by the RSI at the signal close. For each group: trades, share that hit the target
first, mean R, and the edge = mean R of the group minus mean R of all days in the same period
(so the market's drift is not counted). 95 % intervals come from a block bootstrap over months.
Shown for the full period and for the out-of-sample period only (after the strategy-grid
validation date). The site turns the shrunk edge into a 0-10 score; nothing else is fitted here.
"""
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402
import backtest2 as B2  # noqa: E402
import dashboard as D  # noqa: E402

BINS = [0, 20, 25, 30, 35, 40, 50, 60, 70, 101]
LABELS = ["<20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-60", "60-70", ">70"]
TEST_FROM = "2025-04-03"


def table(ev):
    base = ev.R.mean()
    rows = []
    rng = np.random.default_rng(3)
    months = ev.month.unique()
    gm = ev.groupby("month")["R"].agg(["sum", "count"]).reindex(months)
    for lab in LABELS:
        d = ev[ev.bucket == lab]
        if len(d) < 30:
            rows.append({"rsi": lab, "n": int(len(d))})
            continue
        gd = d.groupby("month")["R"].agg(["sum", "count"]).reindex(months).fillna(0)
        vals = []
        for _ in range(400):
            p = rng.integers(0, len(months), len(months))
            a, b = gd.iloc[p], gm.iloc[p]
            if a["count"].sum() > 0:
                vals.append(a["sum"].sum() / a["count"].sum() - b["sum"].sum() / b["count"].sum())
        rows.append({"rsi": lab, "n": int(len(d)), "win": round(float((d.win == 1).mean() * 100), 1),
                     "profit": round(float((d.R > 0).mean() * 100), 1), "R": round(float(d.R.mean()), 3), "edge": round(float(d.R.mean() - base), 3),
                     "edge_ci": [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]})
    return {"base_R": round(float(base), 3), "base_win": round(float((ev.win == 1).mean() * 100), 1),
            "base_profit": round(float((ev.R > 0).mean() * 100), 1), "n": int(len(ev)), "rows": rows}


def main():
    D.DEMO = "--demo" in sys.argv
    wl = [s for s in D.read_watchlist()["symbols"] if s not in B.SKIP]
    syms = list(dict.fromkeys(wl + B.EXTRA))[: (20 if D.DEMO else None)]
    parts = []
    for s in syms:
        try:
            df = B.load(s)
            if len(df) < 400:
                raise RuntimeError("short history")
            f = B.features(df)
            R = B2.simulate(df, f["atr"], "every_day", 1.0, 3.0, 10)
            # win = the target was reached: R close to +3 after costs
            ev = pd.DataFrame({"date": df["date"], "sym": s, "rsi": f["rsi"], "R": R})
            ev["win"] = (ev.R > 2.5).astype(int)
            ev = ev.iloc[252:].dropna()
            parts.append(ev)
            print("ok   %-6s %d" % (s, len(ev)), flush=True)
        except Exception as e:
            print("FAIL %-6s %s" % (s, str(e)[:100]), flush=True)
        if not D.DEMO:
            time.sleep(0.3)
    ev = pd.concat(parts, ignore_index=True)
    ev["month"] = ev.date.dt.to_period("M")
    ev["bucket"] = pd.cut(ev.rsi, BINS, labels=LABELS, right=False).astype(str)
    test_from = TEST_FROM if not D.DEMO else str(ev.date.quantile(0.6).date())
    res = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
           "trade": {"entry": "next open", "stop_atr": 1.0, "target_r": 3.0, "horizon": 10, "cost": 0.001},
           "stocks": len(parts), "from": str(ev.date.min().date()), "to": str(ev.date.max().date()), "test_from": test_from,
           "all": table(ev), "test": table(ev[ev.date >= test_from])}
    os.makedirs(B.OUT, exist_ok=True)
    with open(os.path.join(B.OUT, "score_table.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    L = ["# Score table: model trade by RSI(14)", "", "%d stocks, %s to %s. Next open, stop 1 ATR, target 3R, 10 days, 0.1%% cost." % (
        res["stocks"], res["from"], res["to"])]
    for k in ("all", "test"):
        t = res[k]
        L += ["", "## %s (all days: n %d, win %.1f%%, mean R %+.3f)" % (k, t["n"], t["base_win"], t["base_R"]), "",
              "| RSI | n | target hit | in profit | mean R | edge [95% CI] |", "|---|---|---|---|---|---|"]
        for r in t["rows"]:
            if "edge" in r:
                L.append("| %s | %d | %.1f | %.1f | %+.3f | %+.3f [%+.3f, %+.3f] |" % (r["rsi"], r["n"], r["win"], r["profit"], r["R"], r["edge"], r["edge_ci"][0], r["edge_ci"][1]))
    md = "\n".join(L) + "\n"
    with open(os.path.join(B.OUT, "REPORT3.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
