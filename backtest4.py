#!/usr/bin/env python3
"""Stage-1 research for Trader Dash v2 (pre-registered; see the plan in the README).

    python backtest4.py          # real data (GitHub Actions)
    python backtest4.py --demo   # synthetic data, to test the code

What it does, in order. Nothing here is tuned on the final test period.
 1. Data: 10 years of daily bars (split-adjusted Yahoo closes) for the watchlist (no crypto,
    no leveraged/inverse ETFs) + the earlier research list; SPY, QQQ, VIX and the 11 sector ETFs.
    Features come from features.py, the SAME code the site uses, on closed bars only.
 2. Outcome: the site's model trade on EVERY day (next open, stop 1 ATR, target 3R, exit after 10
    sessions), with 0.10 % round-trip cost and 0.05 % slippage on the entry and on stop exits.
    If a bar touches both stop and target, the stop counts (daily bars cannot tell the order).
 3. Historical-score models, walk-forward by calendar year (train on everything before the year,
    15-day embargo, predict the year):
       M0  the current RSI-bucket table (shrunk n/(n+500))           <- baseline
       M1  RSI bucket x market regime (shrunk toward M0)
       M2  ridge regression on all pre-registered features (alpha fixed at 100)
       M3  gradient boosting (depth 3, 200 trees, min leaf 500; fixed, not tuned)
    Validation = walk-forward predictions before 2025-04-03. Test = from 2025-04-03, opened once.
    Adoption rule (fixed in advance): a model replaces M0 only if, on validation, its top-decile
    edge beats M0's with a month-block-bootstrap 95 % CI above 0 AND its monthly rank IC is higher.
 4. Indicator score v2 (price families only: fundamentals are not point-in-time): does it predict R?
    Correlation between its components (double counting).
 5. Strategy families with fixed rules: Mean Reversion and Trend Following, 4 exit/entry variants each,
    vs an every-day placebo with the same exits; split by regime; full metric set.
 6. Writes research/model_v2.json and research/REPORT4.md.
"""
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402  (universe lists)
import dashboard as D  # noqa: E402
import engine as EN  # noqa: E402
import features as FX  # noqa: E402
import market as MK  # noqa: E402

OUT = os.path.join(HERE, "research")
TEST_FROM = "2025-04-03"
EMBARGO_DAYS = 15
COST, SLIP = 0.001, 0.0005
BINS = [0, 20, 25, 30, 35, 40, 50, 60, 70, 101]
LABELS = ["<20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-60", "60-70", ">70"]
FEATS = ["rsi", "cci", "macd_h", "d_sma20", "d_sma50", "d_sma150", "d_sma200", "slope50", "adx", "di", "atr_pct",
         "atr_pctile", "rv20", "rs_spy_20", "rs_spy_60", "rs_spy_120", "rs_qqq_60", "rs_sector_60", "rvol", "updown",
         "obv20", "room_atr", "ext_atr", "state", "consol", "clv", "body", "lwick", "gap", "ret20", "regime_n",
         "sector_trend", "vix"]
FEATS_HE = {"rsi": "RSI", "cci": "CCI", "macd_h": "MACD מול אות", "d_sma20": "מרחק מ-SMA20", "d_sma50": "מרחק מ-SMA50",
            "d_sma150": "מרחק מ-SMA150", "d_sma200": "מרחק מ-SMA200", "slope50": "שיפוע SMA50", "adx": "ADX",
            "di": "DI+ פחות DI-", "atr_pct": "ATR%", "atr_pctile": "אחוזון ATR", "rv20": "תנודתיות ממומשת",
            "rs_spy_20": "עוצמה יחסית SPY 20", "rs_spy_60": "עוצמה יחסית SPY 60", "rs_spy_120": "עוצמה יחסית SPY 120",
            "rs_qqq_60": "עוצמה יחסית QQQ 60", "rs_sector_60": "עוצמה יחסית לסקטור 60", "rvol": "נפח יחסי",
            "updown": "נפח עלייה/ירידה", "obv20": "OBV", "room_atr": "מרחק להתנגדות", "ext_atr": "מתיחה מעל SMA20",
            "state": "מבנה שיאים/שפלים", "consol": "דשדוש", "clv": "מיקום הסגירה בנר", "body": "גוף הנר",
            "lwick": "פתיל תחתון", "gap": "פער פתיחה", "ret20": "תשואה 20 יום", "regime_n": "מצב שוק",
            "sector_trend": "מגמת סקטור", "vix": "VIX"}


# ---------------------------------------------------------------- data

def load(sym, rng="10y"):
    if D.DEMO:  # synthetic 10 years (random walk with mild mean reversion), only to test the code
        import random
        rnd = random.Random(sym)
        p, t0, rows = rnd.uniform(20, 300), int(time.time()) - 2600 * 86400, []
        for i in range(2600):
            o = p * (1 + rnd.gauss(0, 0.006))
            c = o * (1 + rnd.gauss(0.0004, 0.018))
            rows.append((t0 + i * 86400, o, max(o, c) * (1 + abs(rnd.gauss(0, 0.008))),
                         min(o, c) * (1 - abs(rnd.gauss(0, 0.008))), c, rnd.uniform(1e6, 5e6)))
            p = c
        meta = {"instrumentType": "ETF" if sym.startswith("X") or sym in ("SPY", "QQQ") else "EQUITY"}
        if sym == "^VIX":
            rows = [(r[0], 18, 19, 17, 15 + 10 * abs(math.sin(i / 90)), 0) for i, r in enumerate(rows)]
    else:
        doc = D.yahoo("/v8/finance/chart/%s?range=%s&interval=1d" % (D.ysym(sym), rng))
        res = doc["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        meta = res.get("meta", {})
        rows = [(t, o, h, l, c, v or 0) for t, o, h, l, c, v in
                zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"]) if None not in (o, h, l, c)]
    # one bar per calendar day, last one wins
    seen = {}
    for r in rows:
        seen[int(r[0] // 86400)] = r
    rows = [seen[k] for k in sorted(seen)]
    return meta, rows


def sectors(syms):
    path = os.path.join(OUT, "sectors.json")
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
    except Exception:
        m = {}
    for s in syms:
        if s in m:
            continue
        if D.DEMO:
            m[s] = list(MK.SECTOR_ETF)[hash(s) % 11]
            continue
        try:
            m[s] = D.fundamentals(s).get("sector")
        except Exception:
            m[s] = None
        time.sleep(0.3)
    if not D.DEMO:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=0)
    return m


# ---------------------------------------------------------------- trade simulation

def simulate(o, h, l, c, atr, swing, idx, v):
    """One trade per signal bar i in idx. Entry next open (or after a retest). Returns arrays R, kind, days, ret."""
    n = len(c)
    R = np.full(len(idx), np.nan)
    kind = np.zeros(len(idx), dtype=np.int8)      # 1 target, 2 stop, 3 time, 4 trailing, 5 ambiguous(stop)
    days = np.zeros(len(idx), dtype=np.int16)
    ret = np.full(len(idx), np.nan)
    hz = v["hz"]
    for q, i in enumerate(idx):
        a = atr[i]
        if a is None or not a > 0:
            continue
        e_i = i + 1
        if v.get("entry") == "retest":
            lv = max(c[max(0, i - 20):i]) if i >= 20 else None
            e_i = None
            if lv:
                for j in range(i + 1, min(n - 1, i + 6)):
                    if l[j] <= lv * 1.005 and c[j] > lv:
                        e_i = j + 1
                        break
            if e_i is None:
                continue
            a = atr[e_i - 1] or a
        if e_i + hz > n:
            continue
        entry = o[e_i] * (1 + SLIP)
        if v["stop"] == "swing":
            sw = swing[e_i - 1]
            stop = sw - 0.1 * a if sw else None
            if stop is None or not (0.5 * a <= entry - stop <= 2.5 * a):
                stop = entry - 1.0 * a
        else:
            stop = entry - v["k"] * a
        risk = entry - stop
        if risk <= 0:
            continue
        tgt = entry + v["tgt"] * risk if v.get("tgt") else None
        trail = v.get("trail")
        cur_stop, hi = stop, entry
        ex, kd, d = None, 3, hz
        for j in range(e_i, e_i + hz):
            sh = l[j] <= cur_stop
            th = tgt is not None and h[j] >= tgt
            if sh:
                px = min(o[j], cur_stop) if j > e_i else cur_stop
                ex, kd, d = px * (1 - SLIP), (5 if th else (4 if cur_stop > stop else 2)), j - e_i + 1
                break
            if th:
                ex, kd, d = (max(o[j], tgt) if j > e_i else tgt), 1, j - e_i + 1
                break
            if trail:
                hi = max(hi, h[j])
                cur_stop = max(cur_stop, hi - trail * (atr[j] or a))
        if ex is None:
            ex = c[e_i + hz - 1]
        R[q] = (ex - entry) / risk - COST * entry / risk
        kind[q], days[q], ret[q] = kd, d, ex / entry - 1
    return R, kind, days, ret


MODEL_TRADE = {"stop": "atr", "k": 1.0, "tgt": 3.0, "hz": 10}
VARIANTS = {
    "MR": {"MR-a ATR1, 3R, 10d (המודל באתר)": MODEL_TRADE,
           "MR-b סטופ לפי שפל, 3R, 10d": {"stop": "swing", "tgt": 3.0, "hz": 10},
           "MR-c ATR1, 3R, יציאה בזמן 5d": {"stop": "atr", "k": 1.0, "tgt": 3.0, "hz": 5},
           "MR-d ATR1.5, סטופ נגרר 3ATR, 20d": {"stop": "atr", "k": 1.5, "tgt": None, "trail": 3.0, "hz": 20}},
    "TF": {"TF-a ATR2, 3R, 20d": {"stop": "atr", "k": 2.0, "tgt": 3.0, "hz": 20},
           "TF-b סטופ לפי שפל, 2R, 20d": {"stop": "swing", "tgt": 2.0, "hz": 20},
           "TF-c ATR2, סטופ נגרר 3ATR, 40d": {"stop": "atr", "k": 2.0, "tgt": None, "trail": 3.0, "hz": 40},
           "TF-d כניסה אחרי Retest, ATR2, 3R, 20d": {"stop": "atr", "k": 2.0, "tgt": 3.0, "hz": 20, "entry": "retest"}},
}


# ---------------------------------------------------------------- statistics

def month_boot(per_month, n=1000, seed=7):
    """95 % CI of a mean of monthly values (months resampled with replacement)."""
    x = np.asarray([v for v in per_month if np.isfinite(v)])
    if len(x) < 6:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def ratio_ci(tr, n=600, seed=5):
    """CI of the pooled mean R (sum / count), months resampled with replacement."""
    g = tr.groupby("month")["R"].agg(["sum", "count"]).to_numpy()
    if len(g) < 6:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        x = g[rng.integers(0, len(g), len(g))].sum(0)
        if x[1] > 0:
            vals.append(x[0] / x[1])
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def edge_ci(sig, base, n=600, seed=11):
    """CI of (mean R of signals - mean R of all days), month-block bootstrap."""
    months = sorted(set(base.month))
    gs = sig.groupby("month")["R"].agg(["sum", "count"]).reindex(months).fillna(0).to_numpy()
    gb = base.groupby("month")["R"].agg(["sum", "count"]).reindex(months).fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        p = rng.integers(0, len(months), len(months))
        a, b = gs[p].sum(0), gb[p].sum(0)
        if a[1] > 0 and b[1] > 0:
            vals.append(a[0] / a[1] - b[0] / b[1])
    if len(vals) < 50:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def n_eff(tr, hz):
    """Trades left after dropping those that overlap an earlier counted trade in the same stock."""
    k = 0
    for _, g in tr.sort_values("date").groupby("sym"):
        last = None
        for d in g.date:
            if last is None or (d - last).days > hz * 1.4:
                k += 1
                last = d
    return k


def metrics(tr, base=None, hz=10):
    if len(tr) == 0:
        return {"n": 0}
    R = tr.R.to_numpy()
    pos, neg = R[R > 0].sum(), -R[R < 0].sum()
    eq = tr.sort_values("exit_date").R.cumsum().to_numpy()
    dd = float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()) if len(eq) else 0.0
    mon = tr.groupby(tr.exit_date.dt.to_period("M")).R.sum()
    sd = mon.std()
    dn = mon[mon < 0]
    out = {"n": int(len(tr)), "n_eff": n_eff(tr, hz), "win": round(float((R > 0).mean() * 100), 1),
           "target": round(float((tr.kind == 1).mean() * 100), 1),
           "stop": round(float(tr.kind.isin([2, 5]).mean() * 100), 1),
           "ambiguous": round(float((tr.kind == 5).mean() * 100), 1),
           "expectancy": round(float(R.mean()), 3), "median_R": round(float(np.median(R)), 3),
           "pf": round(float(pos / neg), 2) if neg > 0 else None,
           "ret_mean": round(float(tr.ret.mean() * 100), 2), "ret_median": round(float(tr.ret.median() * 100), 2),
           "days": round(float(tr.days.mean()), 1), "max_dd_R": round(dd, 1),
           "sharpe_m": round(float(mon.mean() / sd * math.sqrt(12)), 2) if sd and sd > 0 else None,
           "sortino_m": round(float(mon.mean() / math.sqrt((dn ** 2).mean()) * math.sqrt(12)), 2) if len(dn) else None,
           "exp_ci": ratio_ci(tr),
           "r_hist": np.histogram(np.clip(R, -2, 4), bins=[-2, -1, -0.5, 0, 0.5, 1, 2, 3, 4])[0].tolist()}
    if base is not None and len(base):
        out["base_R"] = round(float(base.R.mean()), 3)
        out["edge"] = round(float(R.mean() - base.R.mean()), 3)
        out["edge_ci"] = edge_ci(tr, base)
    return out


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    m = len(p)
    for rank in range(m, 0, -1):
        i = o[rank - 1]
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj


def tdist_p(t, df):
    # two-sided p from a normal approximation (df is large: months)
    return math.erfc(abs(t) / math.sqrt(2))


def monthly_ic(df, col, ycol="R"):
    def ic(g):
        if len(g) < 30 or g[col].nunique() < 3:
            return np.nan
        # rank of the predictor vs the RAW R (not its rank): R is bimodal (-1 / +3), and ranking it
        # makes groups with the same mean R look different.
        return g[col].rank().corr(g[ycol])
    s = df.groupby("month").apply(ic).dropna()
    if len(s) < 6:
        return {"ic": None, "t": None, "p": None, "months": int(len(s))}
    t = s.mean() / (s.std() / math.sqrt(len(s))) if s.std() > 0 else 0.0
    return {"ic": round(float(s.mean()), 4), "t": round(float(t), 2), "p": round(tdist_p(t, len(s) - 1), 4), "months": int(len(s))}


# ---------------------------------------------------------------- build the panel

def build_panel(syms, sect):
    bench = {}
    for s in ["SPY", "QQQ", "^VIX"] + sorted(set(MK.SECTOR_ETF.values())):
        try:
            _, rows = load(s)
            bench[s] = ([r[0] for r in rows], [float(r[4]) for r in rows])
        except Exception as e:
            print("bench FAIL %s %s" % (s, e), flush=True)
        if not D.DEMO:
            time.sleep(0.3)
    st, sc = bench["SPY"]
    qa = FX.align(st, *bench["QQQ"]) if "QQQ" in bench else None
    va = FX.align(st, *bench["^VIX"]) if "^VIX" in bench else None
    pts, lab = MK.regime_series(sc, qa, va, EN.CFG.get("regime"))
    day = lambda x: int(x // 86400)
    reg_by_day = {day(t): l_ for t, l_ in zip(st, lab)}
    vix_by_day = {day(t): v_ for t, v_ in zip(st, va or [None] * len(st))}
    etf_tr = {e: dict(zip([day(t) for t in bench[e][0]], MK.etf_trend(bench[e][1]))) for e in MK.SECTOR_ETF.values() if e in bench}
    parts, trades = [], {}
    sig_rows = {"MR": [], "TF": []}
    for s in syms:
        try:
            meta, rows = load(s)
            if len(rows) < 450:
                raise RuntimeError("short history (%d)" % len(rows))
            at = MK.asset_type(s, meta)
            if at["type"] in ("leveraged_etf", "inverse_etf", "crypto"):
                print("skip %-6s %s" % (s, at["type"]), flush=True)
                continue
            T = [r[0] for r in rows]
            O, H, L, C, V = ([float(r[i]) for r in rows] for i in range(1, 6))
            etf = MK.SECTOR_ETF.get(sect.get(s) or "")
            b = {"SPY": FX.align(T, st, sc)}
            if "QQQ" in bench:
                b["QQQ"] = FX.align(T, *bench["QQQ"])
            if etf and etf in bench:
                b["SECTOR"] = FX.align(T, *bench[etf])
            f = FX.compute(T, O, H, L, C, V, b)
            n = len(C)
            df = pd.DataFrame({k: f[k] for k in f if isinstance(f[k], list) and len(f[k]) == n})
            df["date"] = pd.to_datetime(pd.Series(T), unit="s").dt.normalize()
            df["sym"], df["price"] = s, C
            df["chg1"] = pd.Series(C).pct_change() * 100
            df["di"] = df.pdi - df.ndi
            df["regime"] = [reg_by_day.get(day(t)) for t in T]
            df["regime_n"] = df.regime.map({"on": 1, "neutral": 0, "off": -1})
            df["vix"] = [vix_by_day.get(day(t)) for t in T]
            df["sector_etf"] = etf
            df["sector_trend"] = [etf_tr.get(etf, {}).get(day(t)) for t in T] if etf else None
            for col in ("rs_sector_20", "rs_sector_60", "rs_sector_120"):
                if col not in df:
                    df[col] = np.nan
            for col in ("state", "obv20", "consol", "bos_up", "failed_bo"):
                df[col] = pd.to_numeric(df[col].map(lambda x: np.nan if x is None else float(x)), errors="coerce")
            # outcome of the model trade on every day
            idx = np.arange(n)
            R, kd, dy, rt = simulate(O, H, L, C, f["atr"], f["swing_low"], idx, MODEL_TRADE)
            df["R"], df["kind"], df["days"], df["ret"] = R, kd, dy, rt
            # price-only indicator score v2 (same function as the site)
            recs = df.replace({np.nan: None}).to_dict("records")
            iv, fam = [], {k: [] for k in ("trend", "momentum", "volume", "market", "entry")}
            for r_ in recs:
                r_["obv20"] = None if r_["obv20"] is None else int(r_["obv20"])
                r_["state"] = None if r_["state"] is None else int(r_["state"])
                r_["sector_trend"] = None if r_.get("sector_trend") is None else int(r_["sector_trend"])
                r_["rs_spy_20"], r_["rs_spy_60"], r_["rs_spy_120"] = r_.get("rs_spy_20"), r_.get("rs_spy_60"), r_.get("rs_spy_120")
                sco = EN.indicator_score(r_, price_only=True)
                iv.append(sco["value"])
                for c_ in sco["components"]:
                    if c_["key"] in fam:
                        fam[c_["key"]].append(c_["value"])
            df["ind_v2"] = pd.to_numeric(pd.Series(iv), errors="coerce")
            for k, vals in fam.items():
                df["fam_" + k] = pd.to_numeric(pd.Series(vals), errors="coerce")
            # strategy signals (fixed rules)
            mr = df.index[(df.rsi < 30)].to_numpy()
            hi20 = pd.Series(C).shift(1).rolling(20).max()
            tf = df.index[(df.price > df.sma50) & (df.sma50 > df.sma200) & (df.rs_spy_60 > 0) & (df.adx > 20) & (df.price > hi20)].to_numpy()
            for fam_, ix in (("MR", mr), ("TF", tf)):
                for vname, v in VARIANTS[fam_].items():
                    Rv, kv, dv, rv = simulate(O, H, L, C, f["atr"], f["swing_low"], ix, v)
                    Rb, kb, db, rb = simulate(O, H, L, C, f["atr"], f["swing_low"], idx, dict(v, entry=None))
                    base = pd.DataFrame({"date": df.date, "R": Rb, "regime": df.regime})
                    sig = pd.DataFrame({"date": df.date.iloc[ix].to_numpy(), "R": Rv, "kind": kv, "days": dv, "ret": rv,
                                        "regime": df.regime.iloc[ix].to_numpy(), "sector": etf})
                    sig["exit_date"] = sig.date + pd.to_timedelta(sig.days * 1.45, unit="D")
                    sig["sym"] = s
                    trades.setdefault(vname, {"sig": [], "base": []})
                    trades[vname]["sig"].append(sig.dropna(subset=["R"]))
                    trades[vname]["base"].append(base.dropna(subset=["R"]))
            df = df.iloc[252:]
            parts.append(df.drop(columns=[c_ for c_ in ("res", "swing_low", "last_ph", "pdi", "ndi") if c_ in df]))
            print("ok   %-6s %5d rows  %-4s MR %d TF %d" % (s, len(df), etf or "-", len(mr), len(tf)), flush=True)
        except Exception as e:
            print("FAIL %-6s %s" % (s, str(e)[:120]), flush=True)
        if not D.DEMO:
            time.sleep(0.3)
    panel = pd.concat(parts, ignore_index=True).dropna(subset=["R", "rsi"])
    panel["month"] = panel.date.dt.to_period("M")
    panel["bucket"] = pd.cut(panel.rsi, BINS, labels=LABELS, right=False).astype(str)
    for k in trades:
        trades[k] = {"sig": pd.concat(trades[k]["sig"], ignore_index=True), "base": pd.concat(trades[k]["base"], ignore_index=True)}
        for x in trades[k].values():
            x["month"] = x.date.dt.to_period("M")
    return panel, trades


# ---------------------------------------------------------------- historical-score models

def table_pred(train, test, key, shrink=500, prior=None):
    base = train.R.mean()
    g = train.groupby(key).R.agg(["mean", "count"])
    e = (g["mean"] - base) * g["count"] / (g["count"] + shrink)
    if prior is not None:  # shrink toward the prior prediction instead of toward 0
        pass
    return test[key].map(e).astype(float).fillna(0.0).to_numpy(), e


def fit_predict(train, test, which):
    if which == "M0":
        p, _ = table_pred(train, test, "bucket")
        return p
    if which == "M1":
        p0, e0 = table_pred(train, test, "bucket")
        key = train.bucket + "|" + train.regime.fillna("na")
        base = train.R.mean()
        g = train.assign(k=key, e0=train.bucket.map(e0)).groupby("k").agg(m=("R", "mean"), n=("R", "count"), e0=("e0", "first"))
        e = g.e0 + ((g.m - base) - g.e0) * g.n / (g.n + 500)
        tk = test.bucket + "|" + test.regime.fillna("na")
        return tk.map(e).astype(float).fillna(pd.Series(p0, index=test.index)).to_numpy()
    X, Xt = train[FEATS].astype(float), test[FEATS].astype(float)
    if which == "M2":
        from sklearn.linear_model import Ridge
        med = X.median()
        mu, sd = X.fillna(med).mean(), X.fillna(med).std().replace(0, 1)
        Z, Zt = ((X.fillna(med) - mu) / sd).clip(-5, 5), ((Xt.fillna(med) - mu) / sd).clip(-5, 5)
        dm = pd.get_dummies(train.bucket).reindex(columns=LABELS, fill_value=0).astype(float)
        dmt = pd.get_dummies(test.bucket).reindex(columns=LABELS, fill_value=0).astype(float)
        y = train.R.clip(-2, 4) - train.R.mean()
        m = Ridge(alpha=100.0).fit(np.hstack([Z, dm]), y)
        fit_predict.last = {"model": m, "med": med, "mu": mu, "sd": sd}
        return m.predict(np.hstack([Zt, dmt]))
    if which == "M3":
        from sklearn.ensemble import HistGradientBoostingRegressor
        y = train.R.clip(-2, 4) - train.R.mean()
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=500,
                                          l2_regularization=1.0, random_state=0).fit(X, y)
        fit_predict.last = {"model": m}
        return m.predict(Xt)
    raise ValueError(which)


def walk_forward(panel, models=("M0", "M1", "M2", "M3")):
    years = sorted(panel.date.dt.year.unique())
    first = years[0] + 2 if len(years) > 4 else years[0] + 1
    tf = pd.Timestamp(TEST_FROM)
    windows = []
    for y in range(first, years[-1] + 1):
        lo, hi = pd.Timestamp("%d-01-01" % y), pd.Timestamp("%d-01-01" % (y + 1))
        if lo < tf < hi:
            windows.append((lo, tf, "val"))
            windows.append((tf, hi, "test"))
        else:
            windows.append((lo, hi, "test" if lo >= tf else "val"))
    preds = []
    for lo, hi, part in windows:
        te = panel[(panel.date >= lo) & (panel.date < hi)]
        cut = lo - pd.Timedelta(days=EMBARGO_DAYS)
        tr = panel[panel.date < cut]
        if len(te) == 0 or len(tr) < 5000:
            continue
        out = te[["date", "month", "sym", "R", "bucket", "regime", "rsi", "ind_v2"]].copy()
        out["part"] = part
        for m in models:
            out[m] = fit_predict(tr, te, m)
        preds.append(out)
        print("wf  %s..%s %-4s train %d test %d" % (lo.date(), hi.date(), part, len(tr), len(te)), flush=True)
    return pd.concat(preds, ignore_index=True)


def top_decile(df, col, seed=3):
    """Per month: mean R of the top 10 % predictions minus the month's mean R (ties broken at random,
    so a table model with few distinct values still selects exactly 10 %)."""
    rng = np.random.default_rng(seed)
    d = df.assign(_j=rng.random(len(df))).sort_values([col, "_j"])
    d["_r"] = d.groupby("month").cumcount() / d.groupby("month")[col].transform("count")
    top = d[d._r >= 0.9]
    return top.groupby("month").R.mean() - d.groupby("month").R.mean()


def evaluate(pred, models):
    res = {}
    for part in ("val", "test"):
        d = pred[pred.part == part]
        if len(d) == 0:
            continue
        r = {}
        base_top = top_decile(d, "M0")
        for m in models:
            per = top_decile(d, m)
            diff = (per - base_top.reindex(per.index)).dropna()
            score = 5 + 4.5 * d[m] / EN.CFG.get("historical_score", {}).get("span_R", 0.30)
            sel7, sel56 = d[score >= 7], d[score >= 5.6]
            r[m] = {"ic": monthly_ic(d, m), "top10_edge": round(float(per.mean()), 3), "top10_ci": month_boot(per.to_numpy()),
                    "vs_M0": round(float(diff.mean()), 3) if m != "M0" else 0.0,
                    "vs_M0_ci": month_boot(diff.to_numpy()) if m != "M0" else [0, 0],
                    "score7": {"n": int(len(sel7)), "edge": round(float(sel7.R.mean() - d.R.mean()), 3) if len(sel7) else None},
                    "score56": {"n": int(len(sel56)), "edge": round(float(sel56.R.mean() - d.R.mean()), 3) if len(sel56) else None}}
        res[part] = r
    return res


def adopt(ev):
    v = ev.get("val", {})
    ok = []
    for m, r in v.items():
        if m == "M0":
            continue
        lo = (r.get("vs_M0_ci") or [None])[0]
        if lo is not None and lo > 0 and (r["ic"]["ic"] or -1) > (v["M0"]["ic"]["ic"] or -1):
            ok.append((r["vs_M0"], m))
    return max(ok)[1] if ok else "M0"


# ---------------------------------------------------------------- feature and indicator checks

def feature_ics(panel):
    d = panel[panel.date < TEST_FROM]
    rows = []
    for f in FEATS + ["ind_v2", "fam_trend", "fam_momentum", "fam_volume", "fam_market", "fam_entry"]:
        a = monthly_ic(d, f)
        b = monthly_ic(d[d.rsi < 30], f)
        rows.append({"feature": f, "he": FEATS_HE.get(f, f), "all": a, "oversold": b})
    for key in ("all", "oversold"):
        ps = [r[key]["p"] if r[key]["p"] is not None else 1.0 for r in rows]
        for r, q in zip(rows, bh(ps)):
            r[key]["q"] = round(float(q), 4)
    return rows


def correlations(panel):
    cols = ["rsi", "cci", "macd_h", "d_sma20", "d_sma50", "d_sma200", "slope50", "adx", "di", "rs_spy_60",
            "updown", "rvol", "ext_atr", "fam_trend", "fam_momentum", "fam_volume", "fam_market", "fam_entry"]
    c = panel[cols].astype(float).sample(min(len(panel), 60000), random_state=1).corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((round(float(c.loc[a, b]), 2), a, b))
    pairs.sort(key=lambda x: -abs(x[0]))
    return {"top_pairs": pairs[:15], "families": c.loc[[x for x in cols if x.startswith("fam_")], [x for x in cols if x.startswith("fam_")]].round(2).to_dict()}


def strategies(trades):
    out = {}
    for name, d in trades.items():
        sig, base = d["sig"], d["base"]
        hz = [v for fam in VARIANTS.values() for k, v in fam.items() if k == name][0]["hz"]
        r = {}
        for part, lo, hi in (("all", None, None), ("val", None, TEST_FROM), ("test", TEST_FROM, None)):
            s, b = sig, base
            if lo:
                s, b = s[s.date >= lo], b[b.date >= lo]
            if hi:
                s, b = s[s.date < hi], b[b.date < hi]
            r[part] = metrics(s, b, hz)
        r["by_regime"] = {}
        for g in ("on", "neutral", "off"):
            s, b = sig[sig.regime == g], base[base.regime == g]
            r["by_regime"][g] = {"n": int(len(s)), "edge": round(float(s.R.mean() - b.R.mean()), 3) if len(s) >= 30 else None,
                                 "edge_ci": edge_ci(s, b) if len(s) >= 30 else [None, None]}
        r["by_sector"] = {k: {"n": int(len(g)), "exp": round(float(g.R.mean()), 3)} for k, g in sig.groupby("sector") if len(g) >= 30}
        out[name] = r
        print("strat %-40s n %d exp %s edge %s" % (name, r["all"]["n"], r["all"].get("expectancy"), r["all"].get("edge")), flush=True)
    return out


def score_table_v2(panel):
    """The table the site's historical score reads: per RSI bucket, over the FULL period, with the
    effective sample (overlapping trades in one stock counted once), a month-block CI of the edge,
    and the validation / test edges separately so the site can tell whether the sign held."""
    base = panel.R.mean()
    tf = pd.Timestamp(TEST_FROM)
    val, test = panel[panel.date < tf], panel[panel.date >= tf]
    rows = {}
    for b in LABELS:
        d = panel[panel.bucket == b]
        if len(d) < 30:
            rows[b] = {"n": int(len(d))}
            continue
        dv, dt = val[val.bucket == b], test[test.bucket == b]
        stops = d[d.kind.isin([2, 5])]
        rows[b] = {"n": int(len(d)), "n_eff": n_eff(d[["sym", "date"]], 10),
                   "exp_R": round(float(d.R.mean()), 3), "edge": round(float(d.R.mean() - base), 3), "edge_ci": edge_ci(d, panel),
                   "val_edge": round(float(dv.R.mean() - val.R.mean()), 3) if len(dv) >= 30 else None,
                   "test_edge": round(float(dt.R.mean() - test.R.mean()), 3) if len(dt) >= 30 else None,
                   "val_n": int(len(dv)), "test_n": int(len(dt)),
                   "profit": round(float((d.R > 0).mean() * 100), 1), "target": round(float((d.kind == 1).mean() * 100), 1),
                   "stop": round(float(len(stops) / len(d) * 100), 1),
                   "gap_loss": round(float((stops.R < -1.25).mean() * 100), 1) if len(stops) else None,
                   "worst_R": round(float(d.R.min()), 2)}
        print("bucket %-6s n %6d n_eff %5d edge %+.3f %s" % (b, rows[b]["n"], rows[b]["n_eff"], rows[b]["edge"], rows[b]["edge_ci"]), flush=True)
    return {"version": "v2", "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "stocks": int(panel.sym.nunique()), "from": str(panel.date.min().date()), "to": str(panel.date.max().date()),
            "test_from": TEST_FROM, "trade": dict(MODEL_TRADE, cost=COST, slippage_side=SLIP, entry="next open"),
            "base_R": round(float(base), 3), "base_profit": round(float((panel.R > 0).mean() * 100), 1),
            "base_target": round(float((panel.kind == 1).mean() * 100), 1), "n": int(len(panel)), "buckets": rows,
            "excluded": "crypto, leveraged and inverse ETFs (not in the research universe)"}


def sensitivity(panel):
    """RSI<30 model trade edge across stop/target/horizon (all on the same signals), validation period."""
    return None  # computed in backtest2.py (32 variants); not repeated here


# ---------------------------------------------------------------- export the adopted model

def export(panel, which):
    tr = panel[panel.date < pd.Timestamp(TEST_FROM) - pd.Timedelta(days=EMBARGO_DAYS)]
    if which == "M0":
        _, e = table_pred(tr, tr, "bucket")
        return {"type": "bucket", "edges": {k: round(float(v), 4) for k, v in e.items()}}
    if which == "M1":
        base = tr.R.mean()
        _, e0 = table_pred(tr, tr, "bucket")
        g = tr.assign(k=tr.bucket + "|" + tr.regime.fillna("na")).groupby("k").R.agg(["mean", "count"])
        e = {k: float(e0.get(k.split("|")[0], 0) + ((r["mean"] - base) - e0.get(k.split("|")[0], 0)) * r["count"] / (r["count"] + 500))
             for k, r in g.iterrows()}
        return {"type": "bucket_regime", "edges": {k: round(v, 4) for k, v in e.items()}, "bucket_edges": {k: round(float(v), 4) for k, v in e0.items()}}
    if which == "M2":
        fit_predict(tr, tr.head(1), "M2")
        L = fit_predict.last
        coef = L["model"].coef_
        return {"type": "ridge", "features": FEATS, "median": L["med"].round(6).to_dict(), "mean": L["mu"].round(6).to_dict(),
                "sd": L["sd"].round(6).to_dict(), "coef": [round(float(x), 6) for x in coef[:len(FEATS)]],
                "bucket_coef": dict(zip(LABELS, [round(float(x), 6) for x in coef[len(FEATS):]])),
                "intercept": round(float(L["model"].intercept_), 6), "clip": 5}
    if which == "M3":
        fit_predict(tr, tr.head(1), "M3")
        m = fit_predict.last["model"]
        trees = []
        for it in m._predictors:
            nodes = it[0].nodes
            trees.append([[int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]), int(n["right"]),
                           float(n["value"]), bool(n["is_leaf"]), bool(n["missing_go_to_left"])] for n in nodes])
        return {"type": "gbm", "features": FEATS, "baseline": float(m._baseline_prediction.ravel()[0]), "trees": trees}


# ---------------------------------------------------------------- report

def fmt_ci(ci):
    return "[%+.3f, %+.3f]" % tuple(ci) if ci and ci[0] is not None else "—"


def report(res):
    L = ["# Research v2 (stage 1)", "",
         "%d stocks, %s to %s. Survivorship: the list is today's watchlist + large caps chosen today, so firms that "
         "failed or were delisted are missing; results lean optimistic in absolute terms (edges vs the every-day "
         "baseline of the same stocks are less affected)." % (res["stocks"], res["from"], res["to"]), "",
         "Model trade: next open, stop 1 ATR, target 3R, 10 sessions, cost 0.10 %% + slippage 0.05 %% per side. "
         "Same-bar stop and target counted as stop (%.1f %% of all trades)." % res["ambiguous_pct"], "",
         "## Historical score: models (walk-forward, %s)" % res["wf_note"], ""]
    for part in ("val", "test"):
        ev = res["eval"].get(part)
        if not ev:
            continue
        L += ["### %s" % ("Validation (used for the decision)" if part == "val" else "Test from %s (opened once, not used to choose)" % TEST_FROM), "",
              "| model | monthly IC | t | top-10% edge [CI] | vs M0 [CI] | score>=7: n, edge | score>=5.6: n, edge |", "|---|---|---|---|---|---|---|"]
        for m, r in ev.items():
            L.append("| %s | %s | %s | %+.3f %s | %+.3f %s | %d, %s | %d, %s |" % (
                m, r["ic"]["ic"], r["ic"]["t"], r["top10_edge"], fmt_ci(r["top10_ci"]), r["vs_M0"], fmt_ci(r["vs_M0_ci"]),
                r["score7"]["n"], r["score7"]["edge"], r["score56"]["n"], r["score56"]["edge"]))
        L.append("")
    L += ["**Adopted: %s** (%s)" % (res["adopted"], res["adopt_reason"]), "",
          "## Features: monthly IC (feature rank vs raw R) (validation period, BH-adjusted q)", "",
          "| feature | IC all | q | IC within RSI<30 | q |", "|---|---|---|---|---|"]
    for r in res["features"]:
        L.append("| %s | %s | %s | %s | %s |" % (r["feature"], r["all"]["ic"], r["all"].get("q"), r["oversold"]["ic"], r["oversold"].get("q")))
    L += ["", "## Indicator components: strongest Spearman correlations", ""]
    for c, a, b in res["corr"]["top_pairs"]:
        L.append("- %s / %s: %+.2f" % (a, b, c))
    L += ["", "## Strategy families", "",
          "| variant | part | n | n_eff | win% | target% | stop% | exp R [CI] | median R | PF | edge vs every day [CI] | Sharpe(m) | Sortino(m) | MaxDD R | days |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["strategies"].items():
        for part in ("all", "val", "test"):
            m = r[part]
            if not m.get("n"):
                continue
            L.append("| %s | %s | %d | %d | %.1f | %.1f | %.1f | %+.3f %s | %+.3f | %s | %+.3f %s | %s | %s | %s | %s |" % (
                name, part, m["n"], m["n_eff"], m["win"], m["target"], m["stop"], m["expectancy"], fmt_ci(m["exp_ci"]),
                m["median_R"], m["pf"], m.get("edge", 0), fmt_ci(m.get("edge_ci")), m["sharpe_m"], m["sortino_m"], m["max_dd_R"], m["days"]))
    L += ["", "### By market regime (all periods)", "", "| variant | Risk-On | Neutral | Risk-Off |", "|---|---|---|---|"]
    for name, r in res["strategies"].items():
        cells = []
        for g in ("on", "neutral", "off"):
            x = r["by_regime"][g]
            cells.append("n %d, %s %s" % (x["n"], "%+.3f" % x["edge"] if x["edge"] is not None else "—", fmt_ci(x["edge_ci"])))
        L.append("| %s | %s |" % (name, " | ".join(cells)))
    L += ["", "Sharpe/Sortino: monthly sums of R (1R per trade, every signal taken), annualised x sqrt(12). "
          "MaxDD: deepest fall of the cumulative-R curve ordered by exit date. These assume every signal is taken "
          "with equal risk; they are not account returns."]
    return "\n".join(L) + "\n"


def main():
    D.DEMO = "--demo" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    wl = [s for s in D.read_watchlist()["symbols"] if s not in B.SKIP]
    syms = list(dict.fromkeys(wl + B.EXTRA))[: (25 if D.DEMO else None)]
    sect = sectors(syms)
    panel, trades = build_panel(syms, sect)
    print("panel %d rows, %d stocks" % (len(panel), panel.sym.nunique()), flush=True)
    models = ("M0", "M1", "M2", "M3")
    pred = walk_forward(panel, models)
    ev = evaluate(pred, models)
    chosen = adopt(ev)
    reason = ("passed the pre-registered rule on validation" if chosen != "M0" else
              "no candidate beat M0 on validation with a CI above 0 and a higher IC; the current model stays")
    res = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
           "version": "v2-stage1", "stocks": int(panel.sym.nunique()), "rows": int(len(panel)),
           "from": str(panel.date.min().date()), "to": str(panel.date.max().date()), "test_from": TEST_FROM,
           "trade": dict(MODEL_TRADE, cost=COST, slippage_side=SLIP, entry="next open"),
           "ambiguous_pct": round(float((panel.kind == 5).mean() * 100), 2),
           "wf_note": "yearly refit, %d-day embargo" % EMBARGO_DAYS, "eval": ev, "adopted": chosen, "adopt_reason": reason,
           "model": export(panel, chosen), "features": feature_ics(panel), "corr": correlations(panel),
           "strategies": strategies(trades), "rsi_table_10y": {}}
    for part, d in (("all", panel), ("val", panel[panel.date < TEST_FROM]), ("test", panel[panel.date >= TEST_FROM])):
        base = d.R.mean()
        res["rsi_table_10y"][part] = {b: {"n": int(len(g)), "edge": round(float(g.R.mean() - base), 3)} for b, g in d.groupby("bucket")}
    st = score_table_v2(panel)
    if not D.DEMO:
        with open(os.path.join(OUT, "score_table_v2.json"), "w", encoding="utf-8") as fh:
            json.dump(st, fh, ensure_ascii=False, indent=1)
    res["score_table_v2"] = st
    with open(os.path.join(OUT, "model_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    md = report(res)
    with open(os.path.join(OUT, "REPORT4.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
