#!/usr/bin/env python3
"""Research v3: do the two v3 scores predict anything, out of sample, on the whole US market?

    python backtest7_scores.py          # real data (GitHub Actions)
    python backtest7_scores.py --demo   # synthetic universe, to test the code

Universe and data: as backtest6_universe.py (every listed US common stock, liquid on the day only:
close >= $10 and 50-day dollar volume >= $10M), 10 years of daily bars, features from features.py on
CLOSED bars, scores from engine3.py -- the same code the site runs. Every 3rd liquid day of every stock.

The trade (what the site now shows): signal at the close, entry at the NEXT open (+0.05 % slippage),
structural stop from engine3.plan_v3 (0.1 ATR under the nearest confirmed support 0.5-3 ATR below),
target 3R, exit at the close of session 10. 0.10 % round-trip cost. Stop first when one bar touches both.
No trade when there is no structural stop, or when the next open is already within 0.2 ATR of the stop.

Not available point-in-time, so left out (and said so): earnings dates, the sector ETF comparison.

Outputs:
  research/score_table_v3.json  the RSI-bucket history table the site reads (full period, val/test edges)
  research/scores_v3.json       the evidence: IC and buckets of each score, val (< TEST_FROM) and test;
                                the entry score's history part comes from a table built on validation only;
                                old vs new trade and old vs new indicator score on the same days;
                                falling-knife test; weight and threshold sensitivity
  research/REPORT7.md
Survivorship: delisted companies are missing (see backtest6_universe.py).
"""
import datetime
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest4 as B4  # noqa: E402
import backtest6_universe as B6  # noqa: E402
import dashboard as D  # noqa: E402
import engine as EN  # noqa: E402
import engine3 as EN3  # noqa: E402
import features as FX  # noqa: E402
import market as MK  # noqa: E402

OUT = os.path.join(HERE, "research")
TEST_FROM = B4.TEST_FROM
COST, SLIP, HZ, STEP = B4.COST, B4.SLIP, 10, 3
TW, EW = EN3.TECH_W, EN3.ENTRY_W
G = {}  # market series shared with the worker processes (fork)


def sim(o, h, l, c, i, stop, a, tgt_r=3.0, hz=HZ):
    """One trade from the signal at the close of bar i. Returns (R, kind, days, ret) or None (no trade).
    kind: 1 target, 2 stop, 3 time, 5 stop and target on the same bar (counted as stop)."""
    e = i + 1
    if e + hz > len(c):
        return None
    entry = o[e] * (1 + SLIP)
    if stop is None or entry - stop < 0.2 * a:
        return None
    risk = entry - stop
    tgt = entry + tgt_r * risk
    for j in range(e, e + hz):
        sh, th = l[j] <= stop, h[j] >= tgt
        if sh:
            px = min(o[j], stop) if j > e else stop
            ex = px * (1 - SLIP)
            return (ex - entry) / risk - COST * entry / risk, 5 if th else 2, j - e + 1, ex / entry - 1
        if th:
            ex = max(o[j], tgt) if j > e else tgt
            return (ex - entry) / risk - COST * entry / risk, 1, j - e + 1, ex / entry - 1
    ex = c[e + hz - 1]
    return (ex - entry) / risk - COST * entry / risk, 3, hz, ex / entry - 1


def work(item):
    sym, path = item
    try:
        z = np.load(path)
    except Exception:
        return None
    t, o, h, l, c, v = (z[k] for k in ("t", "o", "h", "l", "c", "v"))
    n = len(c)
    if n < 300:
        return None
    dv = pd.Series(c * v).rolling(50).mean().shift(1).to_numpy()
    liquid = (c >= B6.MIN_PRICE) & (dv >= B6.MIN_DOLLAR_VOL)
    idx = np.flatnonzero(liquid & (np.arange(n) >= 260) & (np.arange(n) < n - HZ - 1))[::STEP]
    if not len(idx):
        return None
    days = (t // 86400).astype(np.int64)
    T, O, H, L, C, V = t.tolist(), o.tolist(), h.tolist(), l.tolist(), c.tolist(), v.tolist()
    spy = [G["spy"].get(int(d)) for d in days]
    f = FX.compute(T, O, H, L, C, V, {"SPY": spy} if all(x is not None for x in spy[-260:]) else None)
    keys = list(f.keys())
    rows = []
    for i in idx.tolist():
        x = {k: f[k][i] for k in keys}
        a = x.get("atr")
        if not a:
            continue
        x["price"] = C[i]
        x["move_atr"] = (C[i] - C[i - 1]) / a
        x["regime"] = G["reg"].get(int(days[i]))
        lv = FX.levels(H, L, C, f["atr"], end=i)
        p = EN3.plan_v3(x, lv, None)
        ts = EN3.technical_score(x)
        es = EN3.entry_score(x, p, None, None, lv)
        comp = {k["key"]: k["value"] for k in es["components"]}
        old = EN.indicator_score(x, price_only=True)["value"]
        tr_new = sim(O, H, L, C, i, p["stop"] if p else None, a)
        tr_old = sim(O, H, L, C, i, math.floor((O[i + 1] * (1 + SLIP) - a) * 100) / 100, a)
        r = {"sym": sym, "day": int(days[i]), "rsi": x.get("rsi"), "state": x.get("state"), "tech": ts["value"],
             "old_ind": old, "stop": p["stop"] is not None if p else False, "rr": p.get("rr") if p else None,
             "blocks": bool(p and p.get("res_blocks")), "move_atr": x["move_atr"], "knife": EN3.falling_knife(x),
             "pullback": EN3.pullback_in_uptrend(x)}
        for k in ("location", "rr", "confirm", "volmom", "risk"):
            r["e_" + k] = comp.get(k)
        for k in ts["components"]:
            r["t_" + k["key"]] = k["value"]
        if tr_new:
            r.update(R=tr_new[0], kind=tr_new[1], dd=tr_new[2], ret=tr_new[3])
        if tr_old:
            r.update(R_old=tr_old[0], kind_old=tr_old[1], dd_old=tr_old[2], ret_old=tr_old[3])
        rows.append(r)
    return rows


def combine_entry(df, hist_col, W=None):
    """engine3.entry_score's arithmetic on columns (coverage shrink, the same caps)."""
    W = W or EW
    cols = {"location": "e_location", "rr": "e_rr", "confirm": "e_confirm", "volmom": "e_volmom", "risk": "e_risk"}
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for k, col in list(cols.items()) + ([("history", hist_col)] if hist_col else []):
        x = df[col].to_numpy(dtype=float)
        ok = np.isfinite(x)
        num += np.where(ok, W[k] * np.nan_to_num(x), 0)
        den += np.where(ok, W[k], 0)
    cov = den / sum(W.values())
    val = np.where(cov >= 0.6, 5 + (num / np.maximum(den, 1e-9) - 5) * cov, np.nan)
    val = np.where(~df.stop.to_numpy() & (val > 4.5), 4.5, val)
    val = np.where((df.move_atr.to_numpy() > 2) & (val > 4.0), 4.0, val)
    return np.round(val, 1)


def combine_tech(df, W):
    """Technical score re-weighted from its stored family values (sensitivity test)."""
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for k in W:
        x = df["t_" + k].to_numpy(dtype=float)
        ok = np.isfinite(x)
        num += np.where(ok, W[k] * np.nan_to_num(x), 0)
        den += np.where(ok, W[k], 0)
    cov = den / sum(W.values())
    return np.where(cov >= 0.6, 5 + (num / np.maximum(den, 1e-9) - 5) * cov, np.nan)


def bucket_of(r):
    return next(lab for lo, hi, lab in EN.RSI_BINS if lo <= r < hi) if r is not None and np.isfinite(r) else None


def hist_value(row, k=500, span=0.30):
    shr = row["edge"] * row["n_eff"] / (row["n_eff"] + k)
    v = max(0.5, min(9.5, 5 + 4.5 * shr / span))
    lo, hi = row.get("edge_ci") or [None, None]
    if not (lo is not None and (lo > 0 or hi < 0)):
        v = min(v, 6.4)
    return v


def rsi_table(panel, base_all, val_only=False):
    tf = pd.Timestamp(TEST_FROM)
    val, test = panel[panel.date < tf], panel[panel.date >= tf]
    rows = {}
    for b in B4.LABELS:
        d = panel[panel.bucket == b]
        if len(d) < 30:
            rows[b] = {"n": int(len(d))}
            continue
        dv, dt = val[val.bucket == b], test[test.bucket == b]
        stops = d[d.kind.isin([2, 5])]
        R = d.R.to_numpy()
        pos, neg = R[R > 0].sum(), -R[R < 0].sum()
        eq = d.sort_values("exit_date").R.cumsum().to_numpy()
        rows[b] = {"n": int(len(d)), "n_eff": B4.n_eff(d[["sym", "date"]], HZ),
                   "exp_R": round(float(R.mean()), 3), "edge": round(float(R.mean() - base_all.R.mean()), 3),
                   "edge_ci": B4.edge_ci(d, base_all),
                   "val_edge": round(float(dv.R.mean() - val.R.mean()), 3) if len(dv) >= 30 else None,
                   "test_edge": round(float(dt.R.mean() - test.R.mean()), 3) if len(dt) >= 30 and not val_only else None,
                   "val_n": int(len(dv)), "test_n": int(len(dt)),
                   "profit": round(float((R > 0).mean() * 100), 1), "win": round(float((R > 0).mean() * 100), 1),
                   "target": round(float((d.kind == 1).mean() * 100), 1), "stop": round(float(len(stops) / len(d) * 100), 1),
                   "avg_win_R": round(float(R[R > 0].mean()), 2) if (R > 0).any() else None,
                   "avg_loss_R": round(float(R[R <= 0].mean()), 2) if (R <= 0).any() else None,
                   "pf": round(float(pos / neg), 2) if neg > 0 else None,
                   "max_dd_R": round(float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()), 1),
                   "gap_loss": round(float((stops.R < -1.25).mean() * 100), 1) if len(stops) else None,
                   "worst_R": round(float(R.min()), 2)}
    return rows


def score_eval(df, col, base, cuts=(0, 3, 4.5, 5.5, 6.5, 8, 10.01)):
    """Monthly IC (rank of score vs raw R) and buckets with edge vs every sampled day of the same period."""
    d = df[np.isfinite(df[col])]
    out = {"ic": B4.monthly_ic(d, col), "n": int(len(d)), "buckets": []}
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        g = d[(d[col] >= lo) & (d[col] < hi)]
        if len(g) < 30:
            out["buckets"].append({"range": "%g-%g" % (lo, min(hi, 10)), "n": int(len(g))})
            continue
        out["buckets"].append({"range": "%g-%g" % (lo, min(hi, 10)), "n": int(len(g)), "n_eff": B4.n_eff(g[["sym", "date"]], HZ),
                               "exp": round(float(g.R.mean()), 3), "win": round(float((g.R > 0).mean() * 100), 1),
                               "edge": round(float(g.R.mean() - base.R.mean()), 3), "edge_ci": B4.edge_ci(g, base)})
    top = d[d[col] >= 6.5]
    out["top"] = B4.metrics(top, base, HZ) if len(top) >= 30 else {"n": int(len(top))}
    if "r_hist" in out["top"]:
        out["top"].pop("r_hist")
    return out


def periods(df):
    tf = pd.Timestamp(TEST_FROM)
    return {"val": df[df.date < tf], "test": df[df.date >= tf]}


def main():
    D.DEMO = "--demo" in sys.argv
    os.makedirs(B6.BARS, exist_ok=True)
    t0 = time.time()
    syms = B6.universe()
    if D.DEMO:
        syms = syms[:60]
    print("universe %d" % len(syms), flush=True)
    import concurrent.futures as cf
    paths = {}
    with cf.ThreadPoolExecutor(max_workers=1 if D.DEMO else 4) as ex:
        for k, (s, p) in enumerate(zip(syms, ex.map(B6.fetch, syms))):
            if p:
                paths[s] = p
            if k % 1000 == 0:
                print("  fetched %d / %d (%.0fs)" % (k, len(syms), time.time() - t0), flush=True)
    print("bars for %d stocks (%.0fs)" % (len(paths), time.time() - t0), flush=True)
    _, spy = B4.load("SPY")
    _, qqq = B4.load("QQQ")
    try:
        _, vix = B4.load("^VIX")
    except Exception:
        vix = None
    st = [r[0] for r in spy]
    sc = [float(r[4]) for r in spy]
    qa = FX.align(st, [r[0] for r in qqq], [float(r[4]) for r in qqq])
    va = FX.align(st, [r[0] for r in vix], [float(r[4]) for r in vix]) if vix else None
    _, lab = MK.regime_series(sc, qa, va, EN.CFG.get("regime"))
    G["spy"] = {int(a // 86400): b for a, b in zip(st, sc)}
    G["reg"] = {int(a // 86400): b for a, b in zip(st, lab)}
    rows = []
    items = sorted(paths.items())
    ctx = mp.get_context("fork")
    with ctx.Pool(1 if D.DEMO else max(2, os.cpu_count() or 2)) as pool:
        for k, r in enumerate(pool.imap_unordered(work, items, chunksize=8)):
            if r:
                rows.extend(r)
            if k % 500 == 0:
                print("  scored %d / %d stocks, %d rows (%.0fs)" % (k, len(items), len(rows), time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    del rows
    df["date"] = pd.to_datetime(df.day * 86400, unit="s")
    df["month"] = df.date.dt.to_period("M")
    for col in ("tech", "old_ind", "rr", "e_location", "e_rr", "e_confirm", "e_volmom", "e_risk", "rsi", "R", "R_old"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["bucket"] = df.rsi.map(bucket_of)
    print("rows %d, trades (new) %d, trades (old) %d (%.0fs)" % (len(df), df.R.notna().sum(), df.R_old.notna().sum(), time.time() - t0), flush=True)
    new = df[df.R.notna()].copy()
    new["exit_date"] = new.date + pd.to_timedelta(new.dd * 1.45, unit="D")
    old = df[df.R_old.notna()].drop(columns=["R", "kind", "dd", "ret"], errors="ignore").copy()
    old = old.rename(columns={"R_old": "R", "kind_old": "kind", "ret_old": "ret", "dd_old": "days"})
    old["exit_date"] = old.date + pd.to_timedelta(old.days * 1.45, unit="D")
    new["days"] = new.dd
    tf = pd.Timestamp(TEST_FROM)
    res = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
           "stocks": int(df.sym.nunique()), "rows": int(len(df)), "trades_new": int(len(new)), "trades_old": int(len(old)),
           "from": str(df.date.min().date()), "to": str(df.date.max().date()), "test_from": TEST_FROM,
           "no_stop_share": round(float((~df.stop).mean() * 100), 1),
           "not_point_in_time": ["מועדי דוחות", "השוואה לסקטור"]}
    # 1. the production history table (whole period) and a validation-only one for the out-of-sample test
    tab = rsi_table(new, new)
    meta = {"version": "v3", "generated": res["generated"], "stocks": int(new.sym.nunique()), "from": res["from"], "to": res["to"],
            "test_from": TEST_FROM, "base_R": round(float(new.R.mean()), 3), "base_profit": round(float((new.R > 0).mean() * 100), 1),
            "base_target": round(float((new.kind == 1).mean() * 100), 1), "n": int(len(new)), "buckets": tab,
            "universe": "%d מניות מכל שוק ארה\"ב (סחירות ביום האות)" % new.sym.nunique(),
            "trade": {"stop": "structural (plan_v3)", "tgt": 3.0, "hz": HZ, "cost": COST, "slippage_side": SLIP, "entry": "next open"},
            "trade_he": "כניסה בפתיחה הבאה, סטופ מבני, יעד 3R, 10 ימים",
            "excluded": "crypto, ETFs (not in the research universe)"}
    with open(os.path.join(OUT, "score_table_v3.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    for b, r in tab.items():
        if "edge" in r:
            print("bucket %-6s n %7d n_eff %6d edge %+.3f %s val %s test %s" % (b, r["n"], r["n_eff"], r["edge"], r["edge_ci"], r["val_edge"], r["test_edge"]), flush=True)
    vnew = new[new.date < tf]
    vtab = rsi_table(vnew, vnew, val_only=True)
    hv = {b: hist_value(r) for b, r in vtab.items() if r.get("n_eff", 0) >= 100}
    new["e_history_val"] = new.bucket.map(hv)
    new["entry_nohist"] = combine_entry(new, None)
    new["entry_full"] = combine_entry(new, "e_history_val")
    res["history_table"] = {b: {k: r.get(k) for k in ("n", "n_eff", "exp_R", "edge", "edge_ci", "val_edge", "test_edge", "win", "avg_win_R", "avg_loss_R", "pf", "max_dd_R")}
                            for b, r in tab.items()}
    # 2. does each score predict R? (new trade for v3 scores; the old score also on the old trade, as it was used)
    ev = {}
    P, Po = periods(new), periods(old)
    for name, col, frame, Pf in (("tech_v3", "tech", new, P), ("entry_v3_no_history", "entry_nohist", new, P),
                                 ("entry_v3_full", "entry_full", new, P), ("old_indicator_v2", "old_ind", new, P),
                                 ("old_indicator_v2_on_old_trade", "old_ind", old, Po)):
        ev[name] = {}
        for part, d in Pf.items():
            if name == "entry_v3_full" and part == "val":
                continue  # its history part was fitted on validation: only the test period is honest
            ev[name][part] = score_eval(d, col, d)
            print("%-30s %-4s IC %s top %s" % (name, part, ev[name][part]["ic"], {k: ev[name][part]["top"].get(k) for k in ("n", "expectancy", "edge", "edge_ci")}), flush=True)
    res["scores"] = ev
    # 3. the decision rule (without events and without the history requirement, which needs the table)
    rule = lambda d, col: d[d.stop & (d.rr >= 2) & ~d.blocks & (d[col] >= 6.5) & ~d.knife & (d.move_atr <= 2)]
    res["rule"] = {}
    for part, d in P.items():
        col = "entry_nohist" if part == "val" else "entry_full"
        g = rule(d, col)
        m = B4.metrics(g, d, HZ) if len(g) >= 30 else {"n": int(len(g))}
        m.pop("r_hist", None)
        res["rule"][part] = m
        gh = g[g.bucket.map(lambda b: bool(vtab.get(b, {}).get("edge_ci") and vtab[b]["edge_ci"][0] is not None and vtab[b]["edge_ci"][0] > 0))]
        res["rule"][part + "_with_hist_ok"] = {"n": int(len(gh)), "exp": round(float(gh.R.mean()), 3) if len(gh) else None}
    # 4. falling knife vs pullback in an uptrend, inside the oversold zone (RSI < 35)
    res["knife"] = {}
    mr = new[new.rsi < 35]
    for name, g in (("סכין נופלת", mr[mr.knife]), ("תיקון במגמה עולה", mr[mr.pullback]), ("אחר", mr[~mr.knife & ~mr.pullback])):
        for part, d in periods(g).items():
            base = P[part]
            res["knife"]["%s|%s" % (name, part)] = {"n": int(len(d)), "exp": round(float(d.R.mean()), 3) if len(d) else None,
                                                    "edge": round(float(d.R.mean() - base.R.mean()), 3) if len(d) else None,
                                                    "edge_ci": B4.edge_ci(d, base) if len(d) >= 30 else None}
    # 5. old trade (1 ATR stop) vs new trade (structural stop) on the same days
    both = df[df.R.notna() & df.R_old.notna()]
    res["old_vs_new"] = {}
    for part, d in periods(both).items():
        for nm, col in (("new", "R"), ("old", "R_old")):
            R = d[col].to_numpy()
            pos, neg = R[R > 0].sum(), -R[R < 0].sum()
            res["old_vs_new"]["%s|%s" % (nm, part)] = {"n": int(len(R)), "exp": round(float(R.mean()), 3), "win": round(float((R > 0).mean() * 100), 1),
                                                      "pf": round(float(pos / neg), 2) if neg else None}
    # 6. sensitivity: random weights around the spec's, and the decision thresholds
    rng = np.random.default_rng(3)
    test = P["test"]
    smp = test.sample(min(len(test), 150000), random_state=1) if len(test) else test
    sens = {"entry": [], "tech": []}
    if len(smp) > 500:
        for k in TW:
            smp["t_" + k] = pd.to_numeric(smp["t_" + k], errors="coerce").astype(float)
        base_t = combine_tech(smp, TW)
        for _ in range(40):
            w = rng.dirichlet(np.array(list(TW.values()), float) / 100 * 40) * 100
            e = combine_tech(smp, dict(zip(TW.keys(), w)))
            ok = np.isfinite(e) & np.isfinite(base_t)
            rho = pd.Series(e[ok]).rank().corr(pd.Series(base_t[ok]).rank())
            ic = B4.monthly_ic(smp.assign(_e=e), "_e")["ic"]
            sens["tech"].append({"rho": round(float(rho), 3), "flip_pct": round(float(np.mean((e[ok] >= 6.5) != (base_t[ok] >= 6.5)) * 100), 2), "ic": ic})
        sens["tech_summary"] = {k: [round(float(np.nanpercentile([x[k] for x in sens["tech"] if x[k] is not None], q)), 4) for q in (5, 50, 95)]
                                for k in ("rho", "flip_pct", "ic")}
        sens["tech"] = sens["tech"][:10]
        base_e = combine_entry(smp, "e_history_val")
        for _ in range(40):
            w = rng.dirichlet(np.array(list(EW.values()), float) / 100 * 40) * 100
            W = dict(zip(EW.keys(), w))
            e = combine_entry(smp, "e_history_val", W)
            ok = np.isfinite(e) & np.isfinite(base_e)
            rho = pd.Series(e[ok]).rank().corr(pd.Series(base_e[ok]).rank())
            flip = float(np.mean((e[ok] >= 6.5) != (base_e[ok] >= 6.5)) * 100)
            ic = B4.monthly_ic(smp.assign(_e=e), "_e")["ic"]
            sens["entry"].append({"rho": round(float(rho), 3), "flip_pct": round(flip, 2), "ic": ic})
        sens["entry_summary"] = {k: [round(float(np.nanpercentile([x[k] for x in sens["entry"] if x[k] is not None], q)), 4) for q in (5, 50, 95)]
                                 for k in ("rho", "flip_pct", "ic")}
        sens["entry"] = sens["entry"][:10]
    th = []
    for e_min in (6.0, 6.5, 7.0):
        for rr_min in (1.5, 2.0, 2.5):
            g = test[test.stop & (test.rr >= rr_min) & ~test.blocks & (test.entry_full >= e_min) & ~test.knife & (test.move_atr <= 2)]
            th.append({"entry_min": e_min, "rr_min": rr_min, "n": int(len(g)),
                       "exp": round(float(g.R.mean()), 3) if len(g) else None,
                       "edge": round(float(g.R.mean() - test.R.mean()), 3) if len(g) else None,
                       "edge_ci": B4.edge_ci(g, test) if len(g) >= 30 else None})
    sens["thresholds_test"] = th
    res["sensitivity"] = sens
    res["runtime_s"] = round(time.time() - t0)
    with open(os.path.join(OUT, "scores_v3.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    md = report(res)
    with open(os.path.join(OUT, "REPORT7.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


def report(res):
    f = B4.fmt_ci
    L = ["# Research v3: do the two scores predict? (whole US market)", "",
         "%d stocks, %d sampled days, %s to %s, test from %s. New trade: %d trades (no structural stop on %.1f%% of days). Runtime %ds." % (
             res["stocks"], res["rows"], res["from"], res["to"], res["test_from"], res["trades_new"], res["no_stop_share"], res["runtime_s"]), "",
         "## Scores: monthly IC (rank of score vs R) and the top bucket (>= 6.5) vs every sampled day", "",
         "| score | part | IC | t | top n | top exp R | top edge [CI] |", "|---|---|---|---|---|---|---|"]
    for name, parts in res["scores"].items():
        for part, e in parts.items():
            ic, top = e["ic"], e["top"]
            L.append("| %s | %s | %s | %s | %s | %s | %s %s |" % (name, part, ic.get("ic"), ic.get("t"), top.get("n"), top.get("expectancy"),
                                                             top.get("edge"), f(top.get("edge_ci")) if top.get("edge_ci") else ""))
    L += ["", "## Buckets (test period)", ""]
    for name, parts in res["scores"].items():
        e = parts.get("test")
        if not e:
            continue
        L.append("**%s**: " % name + "; ".join("%s n %s exp %s edge %s %s" % (b["range"], b["n"], b.get("exp"), b.get("edge"), f(b["edge_ci"]) if b.get("edge_ci") else "")
                                             for b in e["buckets"]))
    L += ["", "## Decision rule (stop, R/R >= 2, no near resistance, entry >= 6.5, no falling knife, not chased)", ""]
    for part, m in res["rule"].items():
        L.append("- %s: %s" % (part, {k: m.get(k) for k in ("n", "n_eff", "win", "expectancy", "edge", "edge_ci", "pf", "max_dd_R", "exp")}))
    L += ["", "## Falling knife vs pullback in an uptrend (RSI < 35)", ""]
    for k, v in res["knife"].items():
        L.append("- %s: %s" % (k, v))
    L += ["", "## Old trade (1 ATR stop) vs new trade (structural stop), same days", ""]
    for k, v in res["old_vs_new"].items():
        L.append("- %s: %s" % (k, v))
    s = res["sensitivity"]
    L += ["", "## Sensitivity", "", "- technical weights, 40 random draws (5th/50th/95th pct of rank corr with the spec weights, %% changing side of 6.5, test IC): %s" % s.get("tech_summary"),
          "- entry weights, 40 random draws around the spec (5th/50th/95th pct): %s" % s.get("entry_summary"),
          "- thresholds (test): " + "; ".join("entry>=%g rr>=%g n %d edge %s %s" % (x["entry_min"], x["rr_min"], x["n"], x["edge"], f(x["edge_ci"]) if x["edge_ci"] else "")
                                             for x in s["thresholds_test"])]
    L += ["", "## History table (production, whole period)", "",
          "| RSI | n | n_eff | win% | avg win R | avg loss R | PF | exp R | edge [CI] | val | test |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in res["history_table"].items():
        if r.get("edge") is None:
            continue
        L.append("| %s | %d | %d | %s | %s | %s | %s | %+.3f | %+.3f %s | %s | %s |" % (b, r["n"], r["n_eff"], r["win"], r["avg_win_R"], r["avg_loss_R"], r["pf"],
                                                                                r["exp_R"], r["edge"], f(r["edge_ci"]), r["val_edge"], r["test_edge"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
