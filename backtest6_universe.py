#!/usr/bin/env python3
"""O'Neil / CAN SLIM on the WHOLE US market, not a hand-picked list.

    python backtest6_universe.py          # real data (GitHub Actions; SEC_USER_AGENT for C and A)
    python backtest6_universe.py --demo   # synthetic universe, to test the code

Same book rules and the same four pre-registered variants as backtest5_oneil.py; only the universe changes:
- every common stock listed today on Nasdaq / NYSE / NYSE American / Cboe (Nasdaq Trader symbol directories):
  no ETFs, warrants, units, rights, preferreds, notes or test issues;
- a stock can only signal on a day when it was LIQUID THEN: close >= $10 and 50-day average dollar volume
  >= $10M (O'Neil's liquid, institution-grade stocks), measured with data up to that day only;
- RS Rating = percentile of the IBD-style 12-month strength among ALL stocks in the file that day;
- C and A from SEC filings as first reported (net income growth, split-proof), fetched only for stocks
  that had an RS-80 breakout (the only ones where C/A can change a result);
- the "every day" baseline uses every 5th liquid day of every stock, with the same exits.
Survivorship: companies delisted before today are missing. Much smaller than in a hand-picked list of
today's large caps, but not zero; it still flatters long-only momentum.
"""
import concurrent.futures as cf
import datetime
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest4 as B4  # noqa: E402
import backtest5_oneil as B5  # noqa: E402
import dashboard as D  # noqa: E402
import oneil as ON  # noqa: E402
import secpit  # noqa: E402

OUT = os.path.join(HERE, "research")
BARS = os.path.join(HERE, "cache", "universe")
TEST_FROM = B4.TEST_FROM
MIN_PRICE, MIN_DOLLAR_VOL = 10.0, 10e6
BAD_NAME = ("warrant", "unit", " right", "rights", "preferred", "notes due", "debenture", "depositary share representing",
            "% ", "trust preferred", "acquisition corp", "subordinated")


def universe():
    if D.DEMO:
        return ["S%03d" % i for i in range(120)]
    try:
        return _nasdaq_dir()
    except Exception as e:
        print("Nasdaq Trader directory unavailable (%s); using SEC's exchange list" % str(e)[:80], flush=True)
    ua = os.environ.get("SEC_USER_AGENT")
    doc = B4._sec_get("https://www.sec.gov/files/company_tickers_exchange.json", ua)
    fi = doc["fields"]
    out = set()
    for row in doc["data"]:
        r = dict(zip(fi, row))
        tk, nm = (r.get("ticker") or "").upper(), (r.get("name") or "").lower()
        if r.get("exchange") not in ("Nasdaq", "NYSE", "CBOE") or not tk or any(b in nm for b in BAD_NAME):
            continue
        if "-" in tk and tk.split("-")[-1] in ("WT", "WS", "U", "UN", "R", "RT") or tk.endswith(("W", "WS")) and len(tk) == 5:
            continue
        out.add(tk)
    return sorted(out)


def _nasdaq_dir():
    out = set()
    for url, sym_col, etf_col, test_col in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", 0, 6, 3),
                                            ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", 0, 4, 6)):
        txt = D.get(url, timeout=30)
        for line in txt.splitlines()[1:]:
            f = line.split("|")
            if len(f) < 7 or line.startswith("File Creation"):
                continue
            sym, name = f[sym_col].strip(), f[1].lower()
            if f[etf_col] == "Y" or f[test_col] == "Y" or not sym or "$" in sym:
                continue
            if any(b in name for b in BAD_NAME):
                continue
            out.add(sym.replace(".", "-"))
    return sorted(out)


def fetch(sym):
    """10 years of daily bars -> cached .npz. Returns path or None."""
    path = os.path.join(BARS, sym + ".npz")
    if os.path.exists(path):
        return path
    if D.DEMO:
        meta, rows = B4.load(sym)
    else:
        for attempt in range(4):
            try:
                doc = D.yahoo("/v8/finance/chart/%s?range=10y&interval=1d" % sym)
                break
            except Exception as e:
                if attempt == 3:
                    return None
                time.sleep(3 * (attempt + 1))
        res = ((doc.get("chart") or {}).get("result") or [None])[0]
        if not res or not res.get("timestamp"):
            return None
        q = res["indicators"]["quote"][0]
        rows = [(t, o, h, l, c, v or 0) for t, o, h, l, c, v in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"])
                if None not in (o, h, l, c)]
        seen = {}
        for r in rows:
            seen[int(r[0] // 86400)] = r
        rows = [seen[k] for k in sorted(seen)]
    if len(rows) < 320:
        return None
    a = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(path, t=a[:, 0].astype(np.int64), o=a[:, 1], h=a[:, 2], l=a[:, 3], c=a[:, 4], v=a[:, 5])
    return path


def rs_raw_np(c):
    s = pd.Series(c)
    r = lambda n: s / s.shift(n) - 1
    return (0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)).to_numpy(dtype=np.float32)


def main():
    D.DEMO = "--demo" in sys.argv
    os.makedirs(BARS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    syms = universe()
    print("universe: %d symbols" % len(syms), flush=True)
    paths = {}
    with cf.ThreadPoolExecutor(max_workers=1 if D.DEMO else 4) as ex:
        for k, (s, p) in enumerate(zip(syms, ex.map(fetch, syms))):
            if p:
                paths[s] = p
            if k % 500 == 0:
                print("  fetched %d / %d (%d usable, %.0fs)" % (k, len(syms), len(paths), time.time() - t0), flush=True)
    print("bars for %d stocks (%.0fs)" % (len(paths), time.time() - t0), flush=True)
    # market direction from SPY / QQQ (same as the site)
    _, spy = B4.load("SPY")
    _, qqq = B4.load("QQQ")
    st, sc, sv = [r[0] for r in spy], [float(r[4]) for r in spy], [float(r[5] or 0) for r in spy]
    qd = {int(r[0] // 86400): (float(r[4]), float(r[5] or 0)) for r in qqq}
    al = [qd.get(int(x // 86400)) for x in st]
    okq = all(al)
    mst = ON.market_direction(sc, sv, [x[0] for x in al] if okq else None, [x[1] for x in al] if okq else None)
    m_by_day = {int(t // 86400): (m or {}).get("state") for t, m in zip(st, mst)}
    # pass 1: RS raw for everyone -> cross-sectional percentile by day
    cols = {}
    for s, p in paths.items():
        z = np.load(p)
        cols[s] = pd.Series(rs_raw_np(z["c"]), index=(z["t"] // 86400))
    rs = pd.DataFrame(cols)
    rs_pct = rs.rank(axis=1, pct=True) * 100
    del rs, cols
    print("RS ranks: %d days x %d stocks (%.0fs)" % (rs_pct.shape[0], rs_pct.shape[1], time.time() - t0), flush=True)
    # pass 2: breakouts and trades
    sig_rows, base_rows, need_sec = [], [], set()
    for k, (s, p) in enumerate(paths.items()):
        z = np.load(p)
        t, o, h, l, c, v = z["t"], z["o"], z["h"], z["l"], z["c"], z["v"]
        n = len(c)
        dv = pd.Series(c * v).rolling(50).mean().shift(1).to_numpy()
        av = pd.Series(v).rolling(50).mean().shift(1).to_numpy()
        hi = pd.Series(h).rolling(325, min_periods=60).max().shift(1).to_numpy()
        liquid = (c >= MIN_PRICE) & (dv >= MIN_DOLLAR_VOL)
        idx = np.arange(n)
        cand = np.flatnonzero(liquid & (idx >= 300) & (idx < n - 1) & (c >= 0.85 * np.nan_to_num(hi)) &
                              (v >= ON.VOL_BREAKOUT * np.nan_to_num(av, nan=np.inf)) & (np.r_[False, c[1:] > c[:-1]]))
        days = t // 86400
        if len(cand):
            H, L, C, V, O = h.tolist(), l.tolist(), c.tolist(), v.tolist(), o.tolist()
            s200 = None
            got = []
            for i in cand:
                m = m_by_day.get(int(days[i - 1]))
                b = ON.detect_base(H, L, C, V, int(i) - 1, m in ("correction", "rally"))
                bo = ON.breakout_at(H, L, C, V, int(i), b)
                if not (bo and bo["ok"]):
                    continue
                if s200 is None:
                    import features as FX
                    s200 = FX.sma(C, 200)
                d = int(days[i])
                r = rs_pct.at[d, s] if d in rs_pct.index else np.nan
                got.append({"i": int(i), "type": b["type"], "stage": ON.base_stage(C, H, b["start"], s200),
                            "rs": None if (r is None or (isinstance(r, float) and math.isnan(r))) else float(r),
                            "m": m_by_day.get(d), "vol": bo["vol_ratio"], "price": C[i]})
            if got:
                R, kd, dy, rt = B5.simulate_oneil(O, H, L, C, [g["i"] for g in got])
                for g, r_, k_, d_, x_ in zip(got, R, kd, dy, rt):
                    if np.isfinite(r_):
                        sig_rows.append(dict(g, sym=s, date=pd.Timestamp(int(t[g["i"]]), unit="s").normalize(),
                                             R=float(r_), kind=int(k_), days=int(d_), ret=float(x_)))
                        if g["rs"] is not None and g["rs"] >= 80:
                            need_sec.add(s)
        # baseline: every 5th liquid day, same exits
        base_idx = [int(i) for i in np.flatnonzero(liquid & (idx >= 300) & (idx < n - 1))[::5]]
        if base_idx:
            Rb, *_ = B5.simulate_oneil(o.tolist(), h.tolist(), l.tolist(), c.tolist(), base_idx)
            keep = np.isfinite(Rb)
            base_rows.append(pd.DataFrame({"date": pd.to_datetime(t[base_idx][keep], unit="s").normalize(), "R": Rb[keep],
                                           "m": [m_by_day.get(int(days[i])) for i, kk in zip(base_idx, keep) if kk]}))
        if k % 500 == 0:
            print("  scanned %d / %d stocks, %d breakouts (%.0fs)" % (k, len(paths), len(sig_rows), time.time() - t0), flush=True)
    sig = pd.DataFrame(sig_rows)
    base = pd.concat(base_rows, ignore_index=True)
    base["month"] = base.date.dt.to_period("M")
    print("breakouts %d, RS-80 stocks needing SEC %d (%.0fs)" % (len(sig), len(need_sec), time.time() - t0), flush=True)
    # SEC growth (C and A), point in time, only where it can matter
    pit = {}
    ua = os.environ.get("SEC_USER_AGENT")
    if ua and not D.DEMO and need_sec:
        import facts as FA
        tick = B4._sec_get("https://www.sec.gov/files/company_tickers.json", ua)
        cik = {v_["ticker"].upper(): str(v_["cik_str"]).zfill(10) for v_ in tick.values()}
        for s in sorted(need_sec):
            kk = cik.get(s.upper())
            if not kk:
                continue
            try:
                pit[s] = secpit.growth_events(B4._sec_get(FA.FACTS_URL % kk, ua))
            except Exception:
                pass
    print("SEC growth for %d stocks (%.0fs)" % (len(pit), time.time() - t0), flush=True)
    if len(sig):
        def ca_of(row):
            evs = pit.get(row["sym"])
            if not evs:
                return None
            ev = secpit.known_at(evs, row["date"].strftime("%Y-%m-%d"))
            if not ev or (row["date"].date() - datetime.date.fromisoformat(ev["end"])).days > 200:
                return None
            ey, ry, ty = ev.get("eps_yoy"), ev.get("rev_yoy"), ev.get("eps_ttm_yoy")
            rp = [x for x in ev.get("rev_yoy_prev", []) if x is not None]
            accel = ry is not None and len(rp) == 2 and ry > rp[-1] > rp[-2]
            return bool(ey is not None and ey >= 0.25 and ry is not None and (ry >= 0.25 or accel) and ty is not None and ty >= 0.25)
        sig["ca"] = sig.apply(ca_of, axis=1)
        sig["month"] = sig.date.dt.to_period("M")
        sig["exit_date"] = sig.date + pd.to_timedelta(sig.days * 1.45, unit="D")
    res = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
           "universe": len(syms), "stocks_with_bars": len(paths), "stocks_with_breakouts": int(sig.sym.nunique()) if len(sig) else 0,
           "sec_stocks": len(pit), "from": str(base.date.min().date()), "to": str(base.date.max().date()), "test_from": TEST_FROM,
           "filters": {"min_price": MIN_PRICE, "min_dollar_volume_50d": MIN_DOLLAR_VOL}, "base_all": round(float(base.R.mean()), 3),
           "variants": {}, "by_type": {}, "by_stage": {}, "by_year_on2": {}, "runtime_s": round(time.time() - t0)}
    if len(sig):
        V = {"ON1 פריצה מבסיס (טכני)": sig, "ON2 + RS 80 ומעלה": sig[sig.rs >= 80],
             "ON3 + שוק במגמה עולה (M)": sig[(sig.rs >= 80) & (sig.m == "uptrend")],
             "ON4 CAN SLIM מלא (+C, A)": sig[(sig.rs >= 80) & (sig.m == "uptrend") & (sig.ca == True)],  # noqa: E712
             "בדיקה: ON1 + M בלבד": sig[sig.m == "uptrend"], "בדיקה: ON2 + C/A בלבד": sig[(sig.rs >= 80) & (sig.ca == True)]}  # noqa: E712
        for name, d in V.items():
            r = {}
            for part, lo, hi_ in (("all", None, None), ("val", None, TEST_FROM), ("test", TEST_FROM, None)):
                x, b_ = d, base
                if lo:
                    x, b_ = x[x.date >= lo], b_[b_.date >= lo]
                if hi_:
                    x, b_ = x[x.date < hi_], b_[b_.date < hi_]
                r[part] = B4.metrics(x, b_, 40) if len(x) else {"n": 0}
                if len(x):
                    r[part]["avg_win_pct"] = round(float(x.ret[x.ret > 0].mean() * 100), 1) if (x.ret > 0).any() else None
                    r[part]["avg_loss_pct"] = round(float(x.ret[x.ret <= 0].mean() * 100), 1) if (x.ret <= 0).any() else None
            res["variants"][name] = r
            print("var %-28s n %d exp %s edge %s %s" % (name, r["all"].get("n", 0), r["all"].get("expectancy"), r["all"].get("edge"),
                                                     r["all"].get("edge_ci")), flush=True)
        for k_, g in sig.groupby("type"):
            res["by_type"][ON.TYPE_HE.get(k_, k_)] = {"n": int(len(g)), "exp": round(float(g.R.mean()), 3)}
        for k_, g in sig.groupby(sig.stage.clip(upper=4)):
            res["by_stage"]["%d%s" % (k_, "+" if k_ == 4 else "")] = {"n": int(len(g)), "exp": round(float(g.R.mean()), 3)}
        on2 = sig[sig.rs >= 80]
        for y, g in on2.groupby(on2.date.dt.year):
            b_ = base[base.date.dt.year == y]
            res["by_year_on2"][str(y)] = {"n": int(len(g)), "exp": round(float(g.R.mean()), 3),
                                          "edge": round(float(g.R.mean() - b_.R.mean()), 3) if len(b_) else None}
    with open(os.path.join(OUT, "universe.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    md = report(res)
    with open(os.path.join(OUT, "REPORT6.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


def report(res):
    f = B4.fmt_ci
    L = ["# O'Neil / CAN SLIM on the whole US market", "",
         "%d listed common stocks, %d with 10-year bars, %d had at least one breakout; %s to %s. Liquid on the day only "
         "(close >= $%g, 50-day dollar volume >= $%gM). SEC growth for %d stocks. Book rules, nothing fitted. "
         "Every liquid day (1 in 5) with the same exits: %+.3fR. Runtime %ds." % (
             res["universe"], res["stocks_with_bars"], res["stocks_with_breakouts"], res["from"], res["to"],
             res["filters"]["min_price"], res["filters"]["min_dollar_volume_50d"] / 1e6, res["sec_stocks"], res["base_all"], res["runtime_s"]), "",
         "| variant | part | n | n_eff | win% | avg win % | avg loss % | exp R [CI] | edge vs every day [CI] | PF | MaxDD R | days |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["variants"].items():
        for part in ("all", "val", "test"):
            m = r.get(part) or {}
            if not m.get("n"):
                L.append("| %s | %s | 0 | | | | | | | | | |" % (name, part))
                continue
            L.append("| %s | %s | %d | %d | %.1f | %s | %s | %+.3f %s | %+.3f %s | %s | %s | %s |" % (
                name, part, m["n"], m["n_eff"], m["win"], m.get("avg_win_pct"), m.get("avg_loss_pct"), m["expectancy"],
                f(m["exp_ci"]), m.get("edge", 0), f(m.get("edge_ci")), m["pf"], m["max_dd_R"], m["days"]))
    L += ["", "By base type (ON1): " + ", ".join("%s n %d exp %+.3f" % (k, v["n"], v["exp"]) for k, v in res["by_type"].items()),
          "By base stage (ON1): " + ", ".join("%s n %d exp %+.3f" % (k, v["n"], v["exp"]) for k, v in res["by_stage"].items()),
          "ON2 by year: " + ", ".join("%s n %d exp %+.2f edge %s" % (k, v["n"], v["exp"], v["edge"]) for k, v in res["by_year_on2"].items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
