#!/usr/bin/env python3
"""Which trade structure, if any, has an edge? A pre-registered grid, honest selection.

    python backtest2.py            # real data (Yahoo, 5 years)
    python backtest2.py --demo     # synthetic data, to test the code

The grid is fixed before any result is seen:
  setups   brk_high   next day trades above today's high (today's site plan)
           brk_20d    close above the highest close of the 20 sessions before; buy next open
           ma20_up    close crosses above its 20-day average; buy next open
           pullback   uptrend (close > 200-day average) and RSI(14) < 40; buy next open
           oversold   RSI(14) < 30; buy next open
           every_day  buy next open every day: the placebo every setup must beat
  stops    1, 1.5, 2, 3 ATR(14)      targets 1, 1.5, 2, 3 R      horizons 10, 20 sessions
  costs    0.10 % of price per round trip, charged in R

Edge = mean R of a setup minus mean R of every_day with the same exit, same stocks, same days
(the market's drift is not an edge). Periods: train (first 45 % of dates), validation (next 15 %),
test (last 40 %). A config is chosen by its validation edge (at least 300 trades); the test period
is looked at once, for the chosen config, with a month-block bootstrap interval. Then a filter
model (L2 logistic regression on the 21 indicators) is fitted on train for that config and kept
only if its top 40 % beats the unfiltered config on validation.
"""
import datetime
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402  (features, data loading, universe)
import dashboard as D  # noqa: E402

SETUPS = ["brk_high", "brk_20d", "ma20_up", "pullback", "oversold", "every_day"]
SETUPS_HE = {"brk_high": "פריצת השיא של אתמול", "brk_20d": "פריצת שיא סגירה של 20 יום", "ma20_up": "חציה מעל ממוצע 20",
             "pullback": "תיקון במגמה עולה (RSI<40 מעל ממוצע 200)", "oversold": "מכירת יתר (RSI<30)", "every_day": "קנייה בכל יום (בסיס)"}
STOPS, TARGETS, HORIZONS = [1.0, 1.5, 2.0, 3.0], [1.0, 1.5, 2.0, 3.0], [10, 20]
COST = 0.001
FEATS = B.RULES + B.CONT


def setups(df, f):
    c, h = df.c, df.h
    s20, s200 = c.rolling(20).mean(), c.rolling(200).mean()
    nxt_h = h.shift(-1)
    return {"brk_high": (nxt_h >= h + 0.01),
            "brk_20d": c > c.shift(1).rolling(20).max(),
            "ma20_up": (c > s20) & (c.shift() <= s20.shift()),
            "pullback": (c > s200) & (f.rsi < 40),
            "oversold": f.rsi < 30,
            "every_day": pd.Series(True, index=c.index)}


def simulate(df, atr, setup, stop_k, tgt_r, hz):
    """R per signal day (NaN where no trade). Entry on day t+1; bars t+1 .. t+hz."""
    o, h, l, c = (df[k].to_numpy() for k in ("o", "h", "l", "c"))
    a = atr.to_numpy()
    n = len(c)
    m = n - hz - 1
    if m <= 0:
        return np.full(n, np.nan)
    e = np.arange(m) + 1
    if setup == "brk_high":
        trig = h[:m] + 0.01
        entry = np.maximum(o[e], trig)
        base = trig
    else:
        entry = o[e]
        base = entry
    stop = base - stop_k * a[:m]
    risk = entry - stop
    target = entry + tgt_r * (base - stop)
    W_o = sliding_window_view(o[1:], hz)[:m]
    W_h = sliding_window_view(h[1:], hz)[:m]
    W_l = sliding_window_view(l[1:], hz)[:m]
    sh = W_l <= stop[:, None]
    th = W_h >= target[:, None]
    th[:, 0] &= setup != "brk_high"  # a breakout fill bar can only stop out (conservative)
    fs = np.where(sh.any(1), sh.argmax(1), hz)
    ft = np.where(th.any(1), th.argmax(1), hz)
    idx = np.arange(m)
    o_fs = W_o[idx, np.minimum(fs, hz - 1)]
    o_ft = W_o[idx, np.minimum(ft, hz - 1)]
    stop_px = np.where((fs > 0) & (o_fs <= stop), o_fs, stop)
    if setup == "brk_high":
        stop_px = np.where(fs == 0, stop, stop_px)
    tgt_px = np.where((ft > 0) & (o_ft >= target), o_ft, target)
    last = c[e + hz - 1]
    exit_px = np.where((fs < hz) & (fs <= ft), stop_px, np.where(ft < hz, tgt_px, last))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (exit_px - entry) / risk - COST * entry / risk
    ok = np.isfinite(a[:m]) & (a[:m] > 0) & (risk > 0)
    out = np.full(n, np.nan)
    out[:m] = np.where(ok, R, np.nan)
    return out


def boot_ci(df, col, n=400, seed=11):
    rng = np.random.default_rng(seed)
    g = df.groupby("month")[col].agg(["sum", "count"])
    s, k = g["sum"].to_numpy(), g["count"].to_numpy()
    vals = []
    for _ in range(n):
        p = rng.integers(0, len(s), len(s))
        vals.append(s[p].sum() / max(1, k[p].sum()))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D.DEMO = "--demo" in sys.argv
    wl = [s for s in D.read_watchlist()["symbols"] if s not in B.SKIP]
    syms = list(dict.fromkeys(wl + B.EXTRA))[: (20 if D.DEMO else None)]
    configs = list(itertools.product(SETUPS, STOPS, TARGETS, HORIZONS))
    rows, failed = [], []
    for s in syms:
        try:
            df = B.load(s)
            if len(df) < 400:
                raise RuntimeError("short history")
            f = B.features(df)
            sig = setups(df, f)
            base = pd.DataFrame({"date": df["date"], "sym": s})
            for k in FEATS:
                base[k] = f[k].astype(float)
            for name in SETUPS:
                base["sig_" + name] = sig[name].fillna(False).astype(bool)
            for (st, sk, tr, hz) in configs:
                base["R|%s|%s|%s|%s" % (st, sk, tr, hz)] = simulate(df, f["atr"], st, sk, tr, hz)
            base = base.iloc[252:].dropna(subset=B.CONT)
            rows.append(base)
            print("ok   %-6s %d days" % (s, len(base)), flush=True)
        except Exception as e:
            failed.append(s)
            print("FAIL %-6s %s" % (s, str(e)[:100]), flush=True)
        if not D.DEMO:
            time.sleep(0.3)
    ev = pd.concat(rows, ignore_index=True)
    ev["month"] = ev.date.dt.to_period("M")
    dates = np.sort(ev.date.unique())
    c1, c2 = dates[int(len(dates) * 0.45)], dates[int(len(dates) * 0.60)]
    per = np.where(ev.date < c1, "train", np.where(ev.date < c2, "val", "test"))
    ev["period"] = per

    grid = []
    for (st, sk, tr, hz) in configs:
        rc = "R|%s|%s|%s|%s" % (st, sk, tr, hz)
        bc = "R|every_day|%s|%s|%s" % (sk, tr, hz)
        row = {"setup": st, "stop": sk, "target": tr, "horizon": hz}
        for p in ("train", "val", "test"):
            d = ev[(ev.period == p) & ev["sig_" + st] & ev[rc].notna()]
            b = ev[(ev.period == p) & ev[bc].notna()]
            row[p + "_n"] = int(len(d))
            row[p + "_R"] = round(float(d[rc].mean()), 4) if len(d) else None
            row[p + "_win"] = round(float((d[rc] > 0).mean() * 100), 1) if len(d) else None
            row[p + "_edge"] = round(float(d[rc].mean() - b[bc].mean()), 4) if len(d) else None
        grid.append(row)
    G = pd.DataFrame(grid)
    cand = G[(G.setup != "every_day") & (G.val_n >= 300) & (G.train_n >= 300)]
    # choose on validation edge, require the train edge to point the same way
    cand = cand[cand.train_edge > 0].sort_values("val_edge", ascending=False)
    chosen = cand.iloc[0].to_dict() if len(cand) else None

    result = {"generated": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
              "symbols": len(rows), "failed": failed, "configs": len(configs), "cost_round_trip": COST,
              "periods": {"train_to": str(pd.Timestamp(c1).date()), "val_to": str(pd.Timestamp(c2).date()),
                          "from": str(pd.Timestamp(dates[0]).date()), "to": str(pd.Timestamp(dates[-1]).date())},
              "grid": grid, "chosen": None, "model": None}
    # best per setup (by validation) so the report can show every setup's best shot
    best = []
    for st in SETUPS:
        g = G[(G.setup == st) & (G.val_n >= 300)].sort_values("val_edge" if st != "every_day" else "val_R", ascending=False)
        if len(g):
            best.append(g.iloc[0].to_dict())
    result["best_per_setup"] = best

    if chosen:
        st, sk, tr, hz = chosen["setup"], chosen["stop"], chosen["target"], chosen["horizon"]
        rc = "R|%s|%s|%s|%s" % (st, sk, tr, hz)
        bc = "R|every_day|%s|%s|%s" % (sk, tr, hz)
        te = ev[(ev.period == "test") & ev["sig_" + st] & ev[rc].notna()].copy()
        teb = ev[(ev.period == "test") & ev[bc].notna()].copy()
        te["x"] = te[rc]
        teb["x"] = teb[bc]
        # edge CI: bootstrap months, difference of means
        rng = np.random.default_rng(5)
        ms = te.month.unique()
        gs = te.groupby("month")["x"].agg(["sum", "count"]).reindex(ms)
        gb = teb.groupby("month")["x"].agg(["sum", "count"]).reindex(ms)
        diffs = []
        for _ in range(400):
            p = rng.integers(0, len(ms), len(ms))
            a1, a2 = gs.iloc[p], gb.iloc[p]
            diffs.append(a1["sum"].sum() / a1["count"].sum() - a2["sum"].sum() / a2["count"].sum())
        chosen["test_edge_ci"] = [round(float(np.nanpercentile(diffs, 2.5)), 3), round(float(np.nanpercentile(diffs, 97.5)), 3)]
        chosen["test_R_ci"] = boot_ci(te, "x")
        chosen["name"] = SETUPS_HE[st]
        result["chosen"] = chosen
        # filter model for the chosen config
        tr_d = ev[(ev.period == "train") & ev["sig_" + st] & ev[rc].notna()]
        va_d = ev[(ev.period == "val") & ev["sig_" + st] & ev[rc].notna()]
        if len(tr_d) > 500:
            sc = StandardScaler().fit(tr_d[FEATS].to_numpy())
            mdl = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(tr_d[FEATS].to_numpy()), (tr_d[rc] > 0).astype(int))
            def top(d):
                p = mdl.predict_proba(sc.transform(d[FEATS].to_numpy()))[:, 1]
                thr = np.quantile(p, 0.6)
                return d[p >= thr]
            v_top, t_top = top(va_d), top(te)
            keep = bool(v_top[rc].mean() > va_d[rc].mean() + 0.02)
            result["model"] = {"kept": keep, "val_R_all": round(float(va_d[rc].mean()), 4), "val_R_top40": round(float(v_top[rc].mean()), 4),
                               "test_R_all": round(float(te[rc].mean()), 4), "test_R_top40": round(float(t_top[rc].mean()), 4),
                               "coef": sorted([{"feature": k, "weight": round(float(w), 3)} for k, w in zip(FEATS, mdl.coef_[0])],
                                              key=lambda x: -abs(x["weight"])),
                               "mean": [float(x) for x in sc.mean_], "scale": [float(x) for x in sc.scale_],
                               "intercept": float(mdl.intercept_[0]), "features": FEATS}
    os.makedirs(B.OUT, exist_ok=True)
    with open(os.path.join(B.OUT, "strategies.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1, default=str)
    md = report(result)
    with open(os.path.join(B.OUT, "REPORT2.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


def report(r):
    p = r["periods"]
    L = ["# Strategy grid", "", "%d stocks, %d configs, %s to %s. Train to %s, validation to %s, test after. Costs %.2f%% round trip." % (
        r["symbols"], r["configs"], p["from"], p["to"], p["train_to"], p["val_to"], r["cost_round_trip"] * 100), "",
         "## Best config per setup (chosen on validation)", "",
         "| setup | stop ATR | target R | days | train edge | val edge | test n | test R | test win | test edge |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in r["best_per_setup"]:
        L.append("| %s | %s | %s | %s | %+.3f | %+.3f | %d | %+.3f | %.1f | %+.3f |" % (
            b["setup"], b["stop"], b["target"], b["horizon"], b["train_edge"] or 0, b["val_edge"] or 0, b["test_n"],
            b["test_R"] or 0, b["test_win"] or 0, b["test_edge"] or 0))
    c = r["chosen"]
    L += ["", "## Chosen", ""]
    if c:
        L.append("%s, stop %s ATR, target %sR, %s days: test n %d, mean R %+.3f %s, edge vs every day %+.3f %s" % (
            c["setup"], c["stop"], c["target"], c["horizon"], c["test_n"], c["test_R"], c["test_R_ci"], c["test_edge"], c["test_edge_ci"]))
    else:
        L.append("No setup had a positive edge on both train and validation.")
    m = r["model"]
    if m:
        L += ["", "Filter model kept: %s. Validation R all %+.3f vs top 40%% %+.3f; test R all %+.3f vs top 40%% %+.3f" % (
            m["kept"], m["val_R_all"], m["val_R_top40"], m["test_R_all"], m["test_R_top40"]),
              "", "Weights: " + ", ".join("%s %+.2f" % (x["feature"], x["weight"]) for x in m["coef"][:10])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
