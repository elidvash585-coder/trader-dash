#!/usr/bin/env python3
"""Opportunities by sector: the strongest liquid US stocks, and where each one stands for an entry now.

    python scanner.py          # real data (GitHub Actions, twice a day: after the close and before the open)
    python scanner.py --demo   # synthetic data, to test the code

Two separate questions, never mixed into one number:
  1. Stock quality (verifiable, not a prediction): every common stock listed on Nasdaq / NYSE / NYSE American /
     Cboe that is liquid (close >= $10, 50-day dollar volume >= $10M), ranked by our internal RS (IBD-style formula,
     percentile against ALL listed stocks with a year of bars, NOT IBD's RS Rating). It passes when:
       internal RS >= 80, close above the 50- and 200-day SMA, and within 25 % of its 52-week high.
  2. Entry now: the same engine as the watchlist (engine3 decision, entry and technical scores, structural plan)
     and O'Neil's base stage, on closed daily bars.
Research (REPORT6/7, whole market): high technical / entry scores and O'Neil breakouts did NOT beat a random entry.
This list shows leaders and their state; it does not say "buy".

Writes scan/scan.json (read by the site) and research/rs_reference_universe.json (the site's RS reference).
"""
import concurrent.futures as cf
import datetime
import json
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
import market as MK  # noqa: E402
import oneil as ON  # noqa: E402
import secpit  # noqa: E402

OUT_DIR = os.path.join(HERE, "scan")
MIN_RS, MAX_FROM_HIGH, MAX_CANDIDATES = 80, 25.0, 400
SECTOR_ETF = dict(MK.SECTOR_ETF)


def bars(sym):
    try:
        _, rows = B4.load(sym, "2y")
    except Exception:
        for k in range(2):
            time.sleep(2 * (k + 1))
            try:
                _, rows = B4.load(sym, "2y")
                break
            except Exception:
                rows = None
        if not rows:
            return sym, None
    if len(rows) < 60:
        return sym, None
    a = np.asarray([r[:6] for r in rows], dtype=float)
    return sym, a


def metrics(sym, a, spy_by_day):
    t, c, h, v = a[:, 0], a[:, 4], a[:, 2], a[:, 5]
    n = len(c)
    out = {"sym": sym, "n": n, "price": float(c[-1]), "t": int(t[-1])}
    dv = float(np.mean(c[-51:-1] * v[-51:-1])) if n > 51 else 0.0
    out["dollar_vol"] = dv
    out["liquid"] = bool(c[-1] >= B6.MIN_PRICE and dv >= B6.MIN_DOLLAR_VOL)
    out["rs_raw"] = float(ON.rs_raw(c.tolist(), n - 1)) if n > 252 else None
    out["sma50"] = float(c[-50:].mean()) if n >= 50 else None
    out["sma200"] = float(c[-200:].mean()) if n >= 200 else None
    out["sma200_up"] = bool(n >= 220 and c[-200:].mean() > c[-220:-20].mean())
    hi = float(h[-252:].max())
    out["from_high"] = round((1 - c[-1] / hi) * 100, 1)
    days = (t // 86400).astype(int)
    exc = {}
    for lab, k in (("3m", 63), ("6m", 126), ("12m", 252)):
        s1, s0 = spy_by_day.get(int(days[-1])), spy_by_day.get(int(days[-1 - k])) if n > k else None
        if s0 and s1 and n > k and c[-1 - k]:
            exc[lab] = round((c[-1] / c[-1 - k] - s1 / s0) * 100, 1)
    out["excess"] = exc
    out["chg1"] = round((c[-1] / c[-2] - 1) * 100, 2) if n > 1 else None
    out["spark"] = [round(float(x), 2) for x in c[-60:]]
    return out


def sec_growth(syms, ua):
    """Revenue / earnings growth as first reported (secpit), known today. {sym: event}."""
    out = {}
    if not ua or D.DEMO:
        return out
    try:
        tick = B4._sec_get("https://www.sec.gov/files/company_tickers.json", ua)
    except Exception:
        return out
    cik = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in tick.values()}
    import facts as FA
    today = datetime.date.today().isoformat()
    for s in syms:
        k = cik.get(s.upper().replace("-", "."), cik.get(s.upper()))
        if not k:
            continue
        try:
            ev = secpit.known_at(secpit.growth_events(B4._sec_get(FA.FACTS_URL % k, ua)), today)
            if ev:
                out[s] = ev
        except Exception:
            continue
    return out


def why_of(dec):
    """The most useful single reason: a hard stop first, then what is missing other than the history
    requirement (which no RSI zone meets today, so it would be the same line on every stock)."""
    if dec.get("hard"):
        return dec["hard"][0]
    other = [x for x in dec.get("missing", []) if not x.startswith("יתרון היסטורי")]
    return (other or dec.get("missing") or [""])[0]


def entry_view(sym, m, growth):
    """The watchlist engine on one stock (closed bars), compacted for the scanner."""
    base = D.tech_part(sym)
    tech = base["tech"]
    try:
        fund = D.demo_fund(sym, tech["price"]) if D.DEMO else D.fundamentals(sym)
    except Exception:
        fund = None
    cs = base["candles"]
    sector = D.sector_part(fund, ([x[0] for x in cs], [x[4] for x in cs])) if fund else None
    ind = D.ind_score(tech, None, fund, None, sector=sector)
    ent, plan, dec = D.engine_block(tech, fund, None, base.get("ext"), ind, None, sym=sym)
    sec = {"growth": growth} if growth else None
    try:
        on = D.oneil_block(sym, cs, fund, sec)
    except Exception:
        on = {}
    f = fund or {}
    ne = None
    if f.get("next_earnings"):
        try:
            ne = (datetime.date.fromisoformat(f["next_earnings"]) - datetime.date.today()).days
        except Exception:
            ne = None
    g = growth or {}
    row = dict(m)
    for k in ("rs_raw", "n", "t", "sma50", "sma200"):
        row.pop(k, None)
    row["price"] = round(row["price"], 2)
    row["dollar_vol_m"] = round(row.pop("dollar_vol") / 1e6, 1)
    row.update(name=f.get("name"), sector=f.get("sector") or "לא ידוע", industry=f.get("industry"),
               sector_etf=SECTOR_ETF.get(f.get("sector") or ""), market_cap=f.get("market_cap"),
               rev_yoy=round(g["rev_yoy"] * 100, 1) if g.get("rev_yoy") is not None else (round(f["rev_growth"] * 100, 1) if f.get("rev_growth") is not None else None),
               rev_src="SEC" if g.get("rev_yoy") is not None else ("Yahoo" if f.get("rev_growth") is not None else None),
               eps_yoy=round(g["eps_yoy"] * 100, 1) if g.get("eps_yoy") is not None else None,
               profitable=g.get("eps_pos"), earnings_days=ne, inst=f.get("inst_pct"),
               decision={"code": dec["code"], "verdict": dec["verdict"], "why": why_of(dec)} if dec else None,
               entry=ent.get("value") if ent else None, tech=ind.get("value") if ind else None,
               reliability=((ent or {}).get("reliability") or {}).get("label"),
               plan={k: plan.get(k) for k in ("entry", "stop", "stop_pct", "t1", "t2", "rr")} if dec and dec["code"] == "go" and plan else None,
               on={"status": on.get("status"), "status_he": on.get("status_he"), "pivot": on.get("pivot"),
                   "canslim": "".join(i["letter"] for i in on.get("canslim", []) if i["ok"] is True)} if on else None,
               ext={k: (base.get("ext") or {}).get(k) for k in ("session", "price", "change")} if (base.get("ext") or {}).get("active") else None)
    return row


def main():
    D.DEMO = "--demo" in sys.argv
    D.STATIC_BUILD = True
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    syms = B6.universe()
    if D.DEMO:
        syms = syms[:60]
    _, spy = B4.load("SPY", "2y")
    spy_by_day = {int(r[0] // 86400): float(r[4]) for r in spy}
    allm = []
    with cf.ThreadPoolExecutor(max_workers=1 if D.DEMO else 6) as ex:
        for k, (s, a) in enumerate(ex.map(bars, syms)):
            if a is not None:
                # drop a bar that is still trading (pre-market runs see yesterday's close as the last bar anyway)
                allm.append(metrics(s, a, spy_by_day))
            if k % 1000 == 0:
                print("  bars %d / %d (%.0fs)" % (k, len(syms), time.time() - t0), flush=True)
    df = pd.DataFrame(allm)
    ref = df.rs_raw.dropna()
    df["rs"] = (df.rs_raw.rank(pct=True) * 100).round().clip(1, 99)
    print("universe %d, bars %d, with RS %d, liquid %d (%.0fs)" % (len(syms), len(df), len(ref), int(df.liquid.sum()), time.time() - t0), flush=True)
    os.makedirs(os.path.join(HERE, "research"), exist_ok=True)
    asof = datetime.datetime.fromtimestamp(int(df.t.max()), datetime.timezone.utc).strftime("%Y-%m-%d")
    rsref = {"date": asof, "n": int(len(ref)), "raw": [round(float(x), 4) for x in ref.tolist()], "scope": "universe",
             "formula": "0.4*r63 + 0.2*r126 + 0.2*r189 + 0.2*r252 (IBD-style; NOT IBD's RS Rating)", "source": "scanner.py"}
    if not D.DEMO:
        with open(os.path.join(HERE, "research", "rs_reference_universe.json"), "w", encoding="utf-8") as fh:
            json.dump(rsref, fh)
    rsref["scope"] = "universe"
    D.RS_REF = rsref
    q = df[df.liquid & (df.rs >= MIN_RS) & (df.price > df.sma50) & ((df.sma200.isna()) | (df.price > df.sma200)) & (df.from_high <= MAX_FROM_HIGH)]
    q = q.sort_values("rs", ascending=False).head(MAX_CANDIDATES)
    print("pass the quality screen: %d (%.0fs)" % (len(q), time.time() - t0), flush=True)
    growth = sec_growth(q.sym.tolist(), os.environ.get("SEC_USER_AGENT"))
    print("SEC growth for %d (%.0fs)" % (len(growth), time.time() - t0), flush=True)
    D.market_ctx()
    rows, fails = [], 0
    recs = q.to_dict("records")
    with cf.ThreadPoolExecutor(max_workers=1 if D.DEMO else 4) as ex:
        futs = {ex.submit(entry_view, r["sym"], r, growth.get(r["sym"])): r["sym"] for r in recs}
        for f in cf.as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as e:
                fails += 1
                print("  fail %s: %s" % (futs[f], str(e)[:100]), flush=True)
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.bool_,)):
                r[k] = bool(v)
    secs = {}
    for r in rows:
        s = secs.setdefault(r["sector"], {"name": r["sector"], "he": MK.SECTOR_HE.get(r.get("sector_etf") or "", r["sector"]),
                                          "etf": r.get("sector_etf"), "n": 0, "go": 0, "wait": 0, "no": 0})
        s["n"] += 1
        code = (r.get("decision") or {}).get("code")
        s["no" if code in ("no", "reduce") else code if code in ("go", "wait") else "wait"] += 1
    for s in secs.values():
        if s["etf"]:
            try:
                rows_ = D.etf_rows(s["etf"])
                s["etf_trend"] = MK.etf_trend([float(x[4]) for x in rows_])[-1]
            except Exception:
                s["etf_trend"] = None
    m = D.market_ctx() or {}
    now = datetime.datetime.now(datetime.timezone.utc)
    ny = now.astimezone(D.DQ.NY)
    session = "pre_market" if ny.hour < 9 or (ny.hour == 9 and ny.minute < 30) else "after_close" if ny.hour >= 16 else "session"
    out = {"generated": now.strftime("%Y-%m-%d %H:%M UTC"), "as_of": asof, "session": session, "demo": D.DEMO,
           "universe": len(syms), "with_bars": int(len(df)), "with_rs": int(len(ref)), "liquid": int(df.liquid.sum()),
           "passed": int(len(q)), "shown": len(rows), "failed": fails, "runtime_s": round(time.time() - t0),
           "criteria": {"min_rs": MIN_RS, "max_from_high": MAX_FROM_HIGH, "min_price": B6.MIN_PRICE,
                        "min_dollar_volume": B6.MIN_DOLLAR_VOL, "above_sma50": True, "above_sma200": True},
           "market": {"regime": (m.get("regime") or {}).get("label"), "oneil": (m.get("oneil_m") or {}).get("label"),
                      "oneil_state": (m.get("oneil_m") or {}).get("state")},
           "sectors": sorted(secs.values(), key=lambda s: -s["n"]),
           "rows": sorted(rows, key=lambda r: -r["rs"])}
    with open(os.path.join(OUT_DIR, "scan.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"), default=str)
    print("scan: %d shown in %d sectors, %d go / %d wait / %d no, %ds" % (
        len(rows), len(secs), sum(s["go"] for s in secs.values()), sum(s["wait"] for s in secs.values()),
        sum(s["no"] for s in secs.values()), out["runtime_s"]), flush=True)


if __name__ == "__main__":
    main()
