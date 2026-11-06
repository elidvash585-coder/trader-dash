#!/usr/bin/env python3
"""Backtest of the Trader Dash entry checklist: how well did each rule predict, and what weights
does the data support?

    python backtest.py                 # real data (Yahoo, 5 years of daily bars)
    python backtest.py --demo          # synthetic data, to test the code

Method (all decisions fixed before looking at results):
  1. Universe: the watchlist plus a fixed list of large US stocks from every sector (more,
     and more varied, data = less overfitting to one theme). Crypto and ETFs are left out.
  2. Every trading day t is a candidate. Features use data up to the close of day t only.
  3. The trade is the dashboard's plan: buy if day t+1 trades above day t's high (fill at the
     higher of the open and the trigger), stop 1 ATR(14) under the trigger, target 2R.
     Walk forward 20 sessions. Stop is checked before target in every bar (conservative);
     a gap through a level fills at the open. Not reached by day 20: exit at that close.
     Outcome: R multiple, and win = target hit first.
  4. Time split: the first 60% of dates train, the last 40% test. Nothing about the test
     period is used to choose weights.
  5. Per rule: win rate and mean R when the rule passes vs fails, with 95% intervals from a
     block bootstrap over calendar months (consecutive days of one stock are not independent).
  6. Weights: L2 logistic regression on the train period (standardized features), scored on
     the test period with AUC and with mean R of the top fifth of scores, side by side with the
     current equal-weight checklist. A weight is only worth using if it holds up out of sample.

Writes research/backtest.json and research/REPORT.md. Not financial advice: past behaviour
of these rules says nothing certain about the future.
"""
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "research")
sys.path.insert(0, HERE)
import dashboard as D  # noqa: E402

EXTRA = ["JPM", "BAC", "WFC", "GS", "V", "MA", "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "XOM", "CVX",
         "COP", "PG", "KO", "PEP", "COST", "HD", "LOW", "MCD", "NKE", "SBUX", "DIS", "NFLX", "CMCSA",
         "T", "VZ", "CAT", "DE", "GE", "HON", "UPS", "LMT", "RTX", "ORCL", "CRM", "ADBE", "CSCO",
         "IBM", "QCOM", "TXN", "AVGO", "MU", "AMAT", "NOW", "UBER", "SHOP", "PYPL", "F", "GM",
         "NEE", "DUK", "AMT", "PLD", "LIN", "SHW", "MMM", "BMY", "GILD", "ISRG", "TMO", "ABT", "BKNG"]
SKIP = {"BTC-USD", "ETH-USD", "IBIT", "ETHA", "CIBR"}
HORIZON, TARGET_R, ATR_STOP = 20, 2.0, 1.0
RULES = ["trend", "channel", "momentum", "not_overbought", "volume", "room"]
RULES_HE = {"trend": "מגמה ראשית", "channel": "תעלת מגמה", "momentum": "מומנטום (CCI + MACD)",
            "not_overbought": "לא קנייה ביתר (RSI<70)", "volume": "נפח מאשר", "room": "מקום עד ההתנגדות"}
CONT = ["rsi", "cci", "macd_h", "d_s50", "d_s200", "ch_move", "ch_pos", "rvol", "updown", "obv20",
        "room_r", "atr_pct", "ret5", "ret20", "d_hi52"]
CONT_HE = {"rsi": "RSI", "cci": "CCI", "macd_h": "MACD מול אות", "d_s50": "מרחק מממוצע 50",
           "d_s200": "מרחק מממוצע 200", "ch_move": "שיפוע התעלה", "ch_pos": "מיקום בתעלה", "rvol": "נפח יחסי",
           "updown": "נפח עלייה/ירידה", "obv20": "שינוי OBV", "room_r": "מקום עד התנגדות (R)",
           "atr_pct": "תנודתיות (ATR%)", "ret5": "תשואה 5 ימים", "ret20": "תשואה 20 ימים", "d_hi52": "מרחק משיא שנתי"}


# ---------------------------------------------------------------- data

def load(sym):
    if D.DEMO:
        _, rows = D.demo_chart(sym)
    else:
        doc = D.yahoo("/v8/finance/chart/%s?range=5y&interval=1d" % D.ysym(sym))
        res = doc["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        rows = [(t, o, h, l, c, v) for t, o, h, l, c, v in
                zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"]) if None not in (o, h, l, c)]
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["v"] = df["v"].fillna(0).astype(float)
    df["date"] = pd.to_datetime(df["t"], unit="s").dt.normalize()
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------- features (same definitions as dashboard.py)

def wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def features(df):
    c, h, l, o, v = df.c, df.h, df.l, df.o, df.v
    f = pd.DataFrame(index=df.index)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    d = c.diff()
    rsi = 100 - 100 / (1 + wilder(d.clip(lower=0), 14) / wilder((-d).clip(lower=0), 14))
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    cci = (tp - sma_tp) / (0.015 * md)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    # trend channel: regression of the last 60 closes
    n = 60
    x = np.arange(n, dtype=float)
    xm, sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    arr = c.to_numpy()
    slope = np.full(len(c), np.nan)
    pos = np.full(len(c), np.nan)
    move = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        y = arr[i - n + 1:i + 1]
        b = ((x - xm) * (y - y.mean())).sum() / sxx
        a = y.mean() - b * xm
        fit = a + b * x
        sd = np.sqrt(((y - fit) ** 2).mean())
        up, lo = fit[-1] + 2 * sd, fit[-1] - 2 * sd
        slope[i] = b
        move[i] = b * (n - 1) / fit[0] * 100 if fit[0] else 0
        pos[i] = (y[-1] - lo) / (up - lo) if up > lo else 0.5
    up_day = c >= c.shift()
    upv = (v * up_day).rolling(20).sum()
    dnv = (v * ~up_day).rolling(20).sum()
    obv = (np.sign(d).fillna(0) * v).cumsum()
    # resistance: nearest confirmed swing high (5 bars each side) above the trigger in the last 120 bars
    hh = h.to_numpy()
    piv = np.zeros(len(hh), bool)
    for i in range(5, len(hh) - 5):
        piv[i] = hh[i] == hh[i - 5:i + 6].max()
    trig = h + 0.01
    risk = ATR_STOP * atr
    room = np.full(len(c), np.nan)
    for i in range(130, len(c)):
        lo_i = i - 120
        cand = hh[lo_i:i - 4][piv[lo_i:i - 4]]  # a pivot at j is only known at j+5
        cand = cand[cand > trig.iloc[i]]
        room[i] = (cand.min() - trig.iloc[i]) / risk.iloc[i] if len(cand) and risk.iloc[i] > 0 else 9.0
    f["rsi"], f["cci"] = rsi, cci
    f["macd_h"] = (macd - sig) / atr
    f["d_s50"], f["d_s200"] = c / s50 - 1, c / s200 - 1
    f["ch_move"], f["ch_pos"] = move, pos
    f["rvol"] = v / v.shift().rolling(20).mean()
    f["updown"] = (upv / dnv.replace(0, np.nan)).clip(upper=5)
    f["obv20"] = np.sign(obv - obv.shift(20))
    f["room_r"] = np.minimum(room, 9.0)
    f["atr_pct"] = atr / c * 100
    f["ret5"], f["ret20"] = c / c.shift(5) - 1, c / c.shift(20) - 1
    f["d_hi52"] = c / h.rolling(252, min_periods=120).max() - 1
    # the dashboard's rules, exactly
    f["trend"] = (c > s50) & (s50 > s200)
    f["channel"] = (f.ch_move > 5) | ((f.ch_move.abs() <= 5) & (f.ch_pos < 0.35))
    f["momentum"] = (cci > 0) & (macd > sig)
    f["not_overbought"] = rsi < 70
    f["volume"] = (f.updown > 1) & (f.obv20 > 0)
    f["room"] = f.room_r >= 1.5
    f["atr"] = atr
    return f


# ---------------------------------------------------------------- outcomes

def outcomes(df, atr):
    o, h, l, c = (df[k].to_numpy() for k in ("o", "h", "l", "c"))
    a = atr.to_numpy()
    n = len(c)
    R = np.full(n, np.nan)
    win = np.full(n, np.nan)
    for t in range(n - HORIZON - 2):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        trig = h[t] + 0.01
        if h[t + 1] < trig:
            continue  # never filled
        entry = max(o[t + 1], trig)
        stop = trig - ATR_STOP * a[t]
        risk = entry - stop
        target = entry + TARGET_R * (trig - stop)
        res = None
        if l[t + 1] <= stop:  # same bar as the fill: assume the stop came first
            res = (stop - entry) / risk
            w = 0
        for j in range(t + 2, t + 2 + HORIZON - 1):
            if res is not None:
                break
            if o[j] <= stop:
                res, w = (o[j] - entry) / risk, 0
            elif l[j] <= stop:
                res, w = (stop - entry) / risk, 0
            elif o[j] >= target:
                res, w = (o[j] - entry) / risk, 1
            elif h[j] >= target:
                res, w = (target - entry) / risk, 1
        if res is None:
            res, w = (c[t + HORIZON] - entry) / risk, 0
        R[t], win[t] = res, w
    return R, win


# ---------------------------------------------------------------- statistics

def block_boot(df, fn, n=500, seed=7):
    """95% interval of fn(sample) resampling whole calendar months."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("month")]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(fn(pd.concat([groups[i] for i in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rule_stats(ev, rule):
    a, b = ev[ev[rule]], ev[~ev[rule]]
    diff = lambda s: s[s[rule]]["R"].mean() - s[~s[rule]]["R"].mean()
    lo, hi = block_boot(ev, diff)
    return {"rule": rule, "name": RULES_HE[rule], "share_pass": round(len(a) / len(ev) * 100, 1),
            "win_pass": round(a.win.mean() * 100, 1), "win_fail": round(b.win.mean() * 100, 1),
            "R_pass": round(a.R.mean(), 3), "R_fail": round(b.R.mean(), 3),
            "R_diff": round(a.R.mean() - b.R.mean(), 3), "R_diff_ci": [round(lo, 3), round(hi, 3)],
            "significant": bool(lo > 0 or hi < 0)}


def buckets(ev, score, q=5):
    s = ev.assign(score=score)
    s["b"] = pd.qcut(s.score.rank(method="first"), q, labels=False)
    return [{"bucket": int(k) + 1, "n": int(len(g)), "win": round(g.win.mean() * 100, 1), "R": round(g.R.mean(), 3)}
            for k, g in s.groupby("b")]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    D.DEMO = "--demo" in sys.argv
    wl = [s for s in D.read_watchlist()["symbols"] if s not in SKIP]
    syms = list(dict.fromkeys(wl + EXTRA))
    if D.DEMO:
        syms = syms[:25]
    parts, failed = [], []
    for s in syms:
        try:
            df = load(s)
            if len(df) < 400:
                raise RuntimeError("short history")
            f = features(df)
            R, win = outcomes(df, f["atr"])
            ev = f.assign(R=R, win=win, date=df["date"], sym=s).dropna(subset=["R"] + CONT)
            ev = ev[ev.index >= 252]  # warm-up for the 200-day average and the 52-week high
            parts.append(ev)
            print("ok   %-6s %5d trades" % (s, len(ev)), flush=True)
        except Exception as e:
            failed.append(s)
            print("FAIL %-6s %s" % (s, str(e)[:100]), flush=True)
        if not D.DEMO:
            time.sleep(0.3)
    ev = pd.concat(parts, ignore_index=True)
    for r in RULES:
        ev[r] = ev[r].astype(bool)
    ev["month"] = ev.date.dt.to_period("M")
    ev["eq_score"] = ev[RULES].sum(axis=1)
    dates = np.sort(ev.date.unique())
    cut = dates[int(len(dates) * 0.6)]
    tr, te = ev[ev.date < cut], ev[ev.date >= cut]

    base = {"n": int(len(ev)), "symbols": len(parts), "failed": failed, "win": round(ev.win.mean() * 100, 1),
            "R": round(ev.R.mean(), 3), "breakeven_win": round(100 / (1 + TARGET_R), 1),
            "from": str(pd.Timestamp(dates[0]).date()), "to": str(pd.Timestamp(dates[-1]).date()),
            "split": str(pd.Timestamp(cut).date())}
    rules = [rule_stats(ev, r) for r in RULES]
    rules_train = {r: rule_stats(tr, r)["R_diff"] for r in RULES}
    rules_test = {r: rule_stats(te, r)["R_diff"] for r in RULES}
    for x in rules:
        x["R_diff_train"], x["R_diff_test"] = rules_train[x["rule"]], rules_test[x["rule"]]
        x["stable"] = np.sign(x["R_diff_train"]) == np.sign(x["R_diff_test"])

    # current checklist: equal weights, as on the site (technical rules only)
    eq = {"by_count": [{"passed": int(k), "n": int(len(g)), "win": round(g.win.mean() * 100, 1), "R": round(g.R.mean(), 3)}
                       for k, g in te.groupby("eq_score")],
          "auc_test": round(roc_auc_score(te.win, te.eq_score), 3)}

    # learned weights: rules only, and rules + continuous values
    out_models = {}
    for name, cols in (("rules", RULES), ("full", RULES + CONT)):
        Xtr, Xte = tr[cols].astype(float).to_numpy(), te[cols].astype(float).to_numpy()
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xtr), tr.win)
        p_te = m.predict_proba(sc.transform(Xte))[:, 1]
        p_tr = m.predict_proba(sc.transform(Xtr))[:, 1]
        coefs = sorted(zip(cols, m.coef_[0]), key=lambda x: -abs(x[1]))
        out_models[name] = {"auc_train": round(roc_auc_score(tr.win, p_tr), 3), "auc_test": round(roc_auc_score(te.win, p_te), 3),
                            "coef": [{"feature": k, "name": RULES_HE.get(k) or CONT_HE.get(k), "weight": round(float(w), 3)} for k, w in coefs],
                            "buckets_test": buckets(te, p_te)}
    out_models["equal_weight_buckets_test"] = buckets(te, te.eq_score + np.random.default_rng(1).random(len(te)) * 1e-6)

    res = {"generated": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
           "method": {"horizon": HORIZON, "target_r": TARGET_R, "atr_stop": ATR_STOP}, "baseline": base,
           "rules": rules, "equal_weight": eq, "models": out_models}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "backtest.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    md = report(res)
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


def report(r):
    b = r["baseline"]
    L = ["# Backtest of the entry checklist", "",
         "%s trades, %d stocks, %s to %s (train until %s, test after). Plan: buy above the day's high, stop %.0f ATR, target %.0fR, %d days." % (
             format(b["n"], ","), b["symbols"], b["from"], b["to"], b["split"], r["method"]["atr_stop"], r["method"]["target_r"], r["method"]["horizon"]),
         "", "Baseline: win %.1f%% (break-even %.1f%%), mean R %+.3f" % (b["win"], b["breakeven_win"], b["R"]), "",
         "## Each rule (all years)", "", "| rule | passes | win pass / fail | R pass / fail | R diff [95%% CI] | train / test diff |", "|---|---|---|---|---|---|"]
    for x in r["rules"]:
        L.append("| %s | %.0f%% | %.1f / %.1f | %+.3f / %+.3f | %+.3f [%+.3f, %+.3f]%s | %+.3f / %+.3f%s |" % (
            x["rule"], x["share_pass"], x["win_pass"], x["win_fail"], x["R_pass"], x["R_fail"], x["R_diff"],
            x["R_diff_ci"][0], x["R_diff_ci"][1], " *" if x["significant"] else "", x["R_diff_train"], x["R_diff_test"], "" if x["stable"] else " (flips)"))
    L += ["", "## Out of sample (test period)", "", "Equal weights AUC %.3f; learned rules AUC %.3f; learned full AUC %.3f (0.5 = no skill)" % (
        r["equal_weight"]["auc_test"], r["models"]["rules"]["auc_test"], r["models"]["full"]["auc_test"]), "",
          "| rules passed | n | win | R |", "|---|---|---|---|"]
    for x in r["equal_weight"]["by_count"]:
        L.append("| %d | %d | %.1f | %+.3f |" % (x["passed"], x["n"], x["win"], x["R"]))
    for name in ("rules", "full"):
        L += ["", "Model %s, score fifths (test):" % name, "", "| fifth | n | win | R |", "|---|---|---|---|"]
        for x in r["models"][name]["buckets_test"]:
            L.append("| %d | %d | %.1f | %+.3f |" % (x["bucket"], x["n"], x["win"], x["R"]))
        L += ["", "Weights: " + ", ".join("%s %+.2f" % (c["feature"], c["weight"]) for c in r["models"][name]["coef"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
