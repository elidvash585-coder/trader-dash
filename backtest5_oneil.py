#!/usr/bin/env python3
"""Does O'Neil's method (How to Make Money in Stocks) hold up on our data? Pre-registered test.

    python backtest5_oneil.py          # real data (GitHub Actions, SEC_USER_AGENT set for C and A)
    python backtest5_oneil.py --demo   # synthetic data, to test the code

Nothing is tuned: every threshold is the book's (oneil.py). Four variants, fixed before looking:
  ON1  breakout from a proper base: close above the pivot, volume >= 1.4 x the 50-day average, within 5 %
  ON2  ON1 + RS Rating >= 80 (percentile of the IBD-style 12-month strength among the stocks tested that day)
  ON3  ON2 + market in a confirmed uptrend (M: distribution days / follow-through day on SPY and QQQ)
  ON4  ON3 + C and A from SEC filings as first reported: latest quarter EPS >= +25 % with sales >= +25 % or
       accelerating, and trailing-12-month EPS >= +25 %                                    (= full CAN SLIM)
plus two single-filter checks: ON1 + M only, ON1 + C/A only.
Trade (the book's rules): buy the next open; stop 8 % under the entry (a gap through it fills at the open);
take the profit at +20 %, EXCEPT when +20 % comes within 15 sessions (3 weeks) -> hold 40 sessions (8 weeks)
from entry; otherwise exit at the close of session 65 (13 weeks). 0.10 % costs + 0.05 % slippage per side.
R = return / 8 %. Edge = mean R of the signals minus mean R of entering on every day with the same exits.
Survivorship: the stocks are today's watchlist and today's large caps. Buying strength in today's winners is
exactly where that bias is largest, so absolute returns here are optimistic; the edge vs every day of the
SAME stocks is the fairer number, and even it leans optimistic.
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
import backtest as B  # noqa: E402
import backtest4 as B4  # noqa: E402
import dashboard as D  # noqa: E402
import features as FX  # noqa: E402
import market as MK  # noqa: E402
import oneil as ON  # noqa: E402
import secpit  # noqa: E402

OUT = os.path.join(HERE, "research")
TEST_FROM = B4.TEST_FROM
COST, SLIP = B4.COST, B4.SLIP


def simulate_oneil(o, h, l, c, idx, stop=ON.STOP, take=ON.PROFIT[0], hold_rule=True, max_days=65):
    """The book's exits for entries at the next open after each signal bar. Returns arrays."""
    n = len(c)
    R = np.full(len(idx), np.nan)
    kind = np.zeros(len(idx), dtype=np.int8)       # 1 profit, 2 stop, 3 time, 4 8-week hold, 5 same-bar (stop)
    days = np.zeros(len(idx), dtype=np.int16)
    ret = np.full(len(idx), np.nan)
    for q, i in enumerate(idx):
        e_i = i + 1
        if e_i + 2 > n:
            continue
        entry = o[e_i] * (1 + SLIP)
        sp, tp = entry * (1 - stop), entry * (1 + take)
        hold_until = None
        ex = kd = None
        last = min(n - 1, e_i + max_days - 1)
        for j in range(e_i, last + 1):
            d = j - e_i + 1
            if l[j] <= sp:
                px = min(o[j], sp) if j > e_i else sp
                ex, kd = px * (1 - SLIP), (5 if h[j] >= tp else 2)
                break
            if h[j] >= tp and hold_until is None:
                if hold_rule and d <= 15:
                    hold_until = e_i + 39          # 8 weeks from the entry
                else:
                    ex, kd = max(o[j], tp) if j > e_i else tp, 1
                    break
            if hold_until is not None and j >= hold_until:
                ex, kd = c[j], 4
                break
        if ex is None:
            if last - e_i + 1 < max_days and hold_until is None:
                continue                           # not enough bars left to finish the trade
            ex, kd, j = c[last], (4 if hold_until else 3), last
        R[q] = (ex / entry - 1) / stop - COST / stop
        kind[q], days[q], ret[q] = kd, j - e_i + 1, ex / entry - 1
    return R, kind, days, ret


def main():
    D.DEMO = "--demo" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    wl = [s for s in D.read_watchlist()["symbols"] if s not in B.SKIP]
    syms = list(dict.fromkeys(wl + B.EXTRA))[: (25 if D.DEMO else None)]
    # index data and O'Neil's market direction on SPY (+ QQQ distribution days)
    _, spy = B4.load("SPY")
    _, qqq = B4.load("QQQ")
    st, sc, sv = [r[0] for r in spy], [float(r[4]) for r in spy], [float(r[5] or 0) for r in spy]
    qd = {int(r[0] // 86400): (float(r[4]), float(r[5] or 0)) for r in qqq}
    al = [qd.get(int(x // 86400)) for x in st]
    ok = all(al)
    mstates = ON.market_direction(sc, sv, [x[0] for x in al] if ok else None, [x[1] for x in al] if ok else None)
    m_by_day = {int(t // 86400): (m or {}).get("state") for t, m in zip(st, mstates)}
    states = [(m or {}).get("state") for m in mstates[260:]]
    m_share = {ON.M_HE[k]: round(states.count(k) / len(states) * 100, 1) for k in ON.M_HE}   # sanity check
    ftds = sum(1 for i in range(1, len(mstates)) if mstates[i] and mstates[i - 1] and mstates[i]["ftd"] != mstates[i - 1]["ftd"])
    # SEC point-in-time growth
    pit = {}
    ua = os.environ.get("SEC_USER_AGENT")
    if ua and not D.DEMO:
        import facts as FA
        tick = B4._sec_get("https://www.sec.gov/files/company_tickers.json", ua)
        cik = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in tick.values()}
        for s in syms:
            k = cik.get(s.upper().replace(".", "-"))
            if not k:
                continue
            try:
                pit[s] = secpit.growth_events(B4._sec_get(FA.FACTS_URL % k, ua))
            except Exception as e:
                print("sec FAIL %s %s" % (s, str(e)[:80]), flush=True)
    # pass 1: bars and RS raw for every stock
    data = {}
    for s in syms:
        try:
            meta, rows = B4.load(s)
            if len(rows) < 500:
                continue
            if MK.asset_type(s, meta)["type"] in ("leveraged_etf", "inverse_etf", "crypto"):
                continue
            t = [r[0] for r in rows]
            o, h, l, c, v = ([float(r[i] or 0) for r in rows] for i in range(1, 6))
            rs = [ON.rs_raw(c, i) for i in range(len(c))]
            data[s] = (t, o, h, l, c, v, rs)
        except Exception as e:
            print("FAIL %-6s %s" % (s, str(e)[:100]), flush=True)
        if not D.DEMO:
            time.sleep(0.25)
    # RS percentile by day across the stocks tested (cross-section)
    rs_tab = pd.DataFrame({s: pd.Series(d[6], index=[int(x // 86400) for x in d[0]]) for s, d in data.items()})
    rs_pct = rs_tab.rank(axis=1, pct=True) * 100
    last_day = rs_tab.dropna(how="all").index.max()
    ref = [round(float(x), 4) for x in rs_tab.loc[last_day].dropna().tolist()]
    with open(os.path.join(OUT, "rs_reference.json"), "w", encoding="utf-8") as fh:
        json.dump({"date": datetime.datetime.utcfromtimestamp(last_day * 86400).strftime("%Y-%m-%d"), "n": len(ref),
                   "raw": ref, "formula": "0.4*r63 + 0.2*r126 + 0.2*r189 + 0.2*r252"}, fh)
    # pass 2: signals and trades
    sig_rows, base_rows = [], []
    for s, (t, o, h, l, c, v, rs) in data.items():
        days = [int(x // 86400) for x in t]
        evs = pit.get(s) or []
        ev_ptr, ev_cur = 0, None
        hi_roll = pd.Series(h).rolling(325, min_periods=60).max().shift(1).to_numpy()
        avg50 = pd.Series(v).rolling(50).mean().shift(1).to_numpy()
        s200 = FX.sma(c, 200)
        cand = []
        for i in range(300, len(c) - 1):
            if not (hi_roll[i] and c[i] >= 0.85 * hi_roll[i] and avg50[i] and v[i] >= ON.VOL_BREAKOUT * avg50[i] and c[i] > c[i - 1]):
                continue
            mst = m_by_day.get(days[i - 1])
            b = ON.detect_base(h, l, c, v, i - 1, mst in ("correction", "rally"))
            bo = ON.breakout_at(h, l, c, v, i, b)
            if not (bo and bo["ok"]):
                continue
            iso = datetime.datetime.utcfromtimestamp(t[i]).strftime("%Y-%m-%d")
            while ev_ptr < len(evs) and evs[ev_ptr]["known"] <= iso:
                ev_cur = evs[ev_ptr]
                ev_ptr += 1
            ey, ry, ty = (ev_cur or {}).get("eps_yoy"), (ev_cur or {}).get("rev_yoy"), (ev_cur or {}).get("eps_ttm_yoy")
            rp = [x for x in (ev_cur or {}).get("rev_yoy_prev", []) if x is not None]
            accel = ry is not None and len(rp) == 2 and ry > rp[-1] > rp[-2]
            fresh = ev_cur is not None and (datetime.date.fromisoformat(iso) - datetime.date.fromisoformat(ev_cur["end"])).days <= 200
            ca = None if not fresh else bool(ey is not None and ey >= 0.25 and ry is not None and (ry >= 0.25 or accel)
                                                and ty is not None and ty >= 0.25)
            cand.append({"i": i, "type": b["type"], "stage": ON.base_stage(c, h, b["start"], s200),
                         "rs": float(rs_pct.at[days[i], s]) if days[i] in rs_pct.index and not math.isnan(rs_pct.at[days[i], s]) else None,
                         "m": m_by_day.get(days[i]), "ca": ca, "vol": bo["vol_ratio"], "depth": b["depth"]})
        idx = [x["i"] for x in cand]
        if idx:
            R, k, dy, rt = simulate_oneil(o, h, l, c, idx)
            R25, *_ = simulate_oneil(o, h, l, c, idx, take=ON.PROFIT[1])
            for x, r_, k_, d_, rr_, r25 in zip(cand, R, k, dy, rt, R25):
                if np.isfinite(r_):
                    sig_rows.append(dict(x, sym=s, date=pd.Timestamp(t[x["i"]], unit="s").normalize(), R=r_, kind=int(k_),
                                         days=int(d_), ret=rr_, R25=r25))
        allidx = list(range(300, len(c) - 1))
        Rb, kb, db, rb = simulate_oneil(o, h, l, c, allidx)
        base_rows.append(pd.DataFrame({"date": pd.to_datetime([t[i] for i in allidx], unit="s").normalize(), "R": Rb,
                                       "m": [m_by_day.get(days[i]) for i in allidx]}).dropna(subset=["R"]))
        print("ok   %-6s signals %d" % (s, len(cand)), flush=True)
    sig = pd.DataFrame(sig_rows)
    base = pd.concat(base_rows, ignore_index=True)
    base["month"] = base.date.dt.to_period("M")
    res = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "demo": D.DEMO,
           "stocks": len(data), "from": str(base.date.min().date()), "to": str(base.date.max().date()), "test_from": TEST_FROM,
           "sec_stocks": len(pit), "variants": {}, "by_type": {}, "by_stage": {}, "by_market": {},
           "market_share": m_share, "follow_through_days": ftds}
    if len(sig):
        sig["month"] = sig.date.dt.to_period("M")
        sig["exit_date"] = sig.date + pd.to_timedelta(sig.days * 1.45, unit="D")
        V = {"ON1 פריצה מבסיס (טכני)": sig,
             "ON2 + RS 80 ומעלה": sig[sig.rs >= 80],
             "ON3 + שוק במגמה עולה (M)": sig[(sig.rs >= 80) & (sig.m == "uptrend")],
             "ON4 CAN SLIM מלא (+C, A)": sig[(sig.rs >= 80) & (sig.m == "uptrend") & (sig.ca == True)],  # noqa: E712
             "בדיקה: ON1 + M בלבד": sig[sig.m == "uptrend"],
             "בדיקה: ON1 + C/A בלבד": sig[sig.ca == True]}  # noqa: E712
        for name, d in V.items():
            r = {}
            for part, lo, hi in (("all", None, None), ("val", None, TEST_FROM), ("test", TEST_FROM, None)):
                x, b_ = d, base
                if lo:
                    x, b_ = x[x.date >= lo], b_[b_.date >= lo]
                if hi:
                    x, b_ = x[x.date < hi], b_[b_.date < hi]
                r[part] = B4.metrics(x, b_, 40) if len(x) else {"n": 0}
                if len(x):
                    r[part]["R25"] = round(float(x.R25.mean()), 3)
                    r[part]["avg_win_pct"] = round(float(x.ret[x.ret > 0].mean() * 100), 1) if (x.ret > 0).any() else None
                    r[part]["avg_loss_pct"] = round(float(x.ret[x.ret <= 0].mean() * 100), 1) if (x.ret <= 0).any() else None
            res["variants"][name] = r
            print("var %-28s n %d exp %s edge %s" % (name, r["all"].get("n", 0), r["all"].get("expectancy"), r["all"].get("edge")), flush=True)
        for k, g in sig.groupby("type"):
            res["by_type"][ON.TYPE_HE.get(k, k)] = {"n": int(len(g)), "exp": round(float(g.R.mean()), 3),
                                                     "win": round(float((g.R > 0).mean() * 100), 1)}
        sig["stage_b"] = sig.stage.clip(upper=4)
        for k, g in sig.groupby("stage_b"):
            res["by_stage"]["%d%s" % (k, "+" if k == 4 else "")] = {"n": int(len(g)), "exp": round(float(g.R.mean()), 3)}
        for k, g in sig.groupby(sig.m.fillna("na")):
            res["by_market"][ON.M_HE.get(k, k)] = {"n": int(len(g)), "exp": round(float(g.R.mean()), 3)}
    res["base_all"] = round(float(base.R.mean()), 3)
    res["base_by_market"] = {ON.M_HE.get(k, k): round(float(g.R.mean()), 3) for k, g in base.groupby(base.m.fillna("na"))}
    with open(os.path.join(OUT, "oneil.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    md = report(res)
    with open(os.path.join(OUT, "REPORT5.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


def report(res):
    f = B4.fmt_ci
    L = ["# O'Neil / CAN SLIM backtest", "",
         "%d stocks, %s to %s; SEC growth data for %d. Book rules only, nothing fitted. Buy next open, stop -8 %%, "
         "profit +20 %% (8-week hold if +20 %% within 3 weeks), max 13 weeks, costs included. R = return / 8 %%." % (
             res["stocks"], res["from"], res["to"], res["sec_stocks"]),
         "Every day with the same exits: mean %+.3fR. Survivorship bias favours this method in particular (see the header of the script)." % res["base_all"], "",
         "| variant | part | n | n_eff | win% | avg win % | avg loss % | exp R [CI] | edge vs every day [CI] | PF | MaxDD R | days | exp R at +25 % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["variants"].items():
        for part in ("all", "val", "test"):
            m = r.get(part) or {}
            if not m.get("n"):
                L.append("| %s | %s | 0 | | | | | | | | | | |" % (name, part))
                continue
            L.append("| %s | %s | %d | %d | %.1f | %s | %s | %+.3f %s | %+.3f %s | %s | %s | %s | %s |" % (
                name, part, m["n"], m["n_eff"], m["win"], m.get("avg_win_pct"), m.get("avg_loss_pct"), m["expectancy"],
                f(m["exp_ci"]), m.get("edge", 0), f(m.get("edge_ci")), m["pf"], m["max_dd_R"], m["days"], m.get("R25")))
    L += ["", "By base type (ON1): " + ", ".join("%s n %d exp %+.3f" % (k, v["n"], v["exp"]) for k, v in res["by_type"].items()),
          "By base stage (ON1): " + ", ".join("%s n %d exp %+.3f" % (k, v["n"], v["exp"]) for k, v in res["by_stage"].items()),
          "By market direction (ON1): " + ", ".join("%s n %d exp %+.3f" % (k, v["n"], v["exp"]) for k, v in res["by_market"].items()),
          "Every day by market direction: " + ", ".join("%s %+.3f" % (k, v) for k, v in res["base_by_market"].items()),
          "Time in each market state (sanity check of M): " + ", ".join("%s %.0f%%" % (k, v) for k, v in res.get("market_share", {}).items()) +
          "; follow-through days: %s" % res.get("follow_through_days")]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
