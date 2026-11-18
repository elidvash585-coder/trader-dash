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
SESSION_OPEN = False


NASDAQ_SECTOR = {"Technology": "Technology", "Health Care": "Healthcare", "Finance": "Financial Services",
                 "Consumer Discretionary": "Consumer Cyclical", "Industrials": "Industrials", "Energy": "Energy",
                 "Real Estate": "Real Estate", "Utilities": "Utilities", "Basic Materials": "Basic Materials",
                 "Telecommunications": "Communication Services", "Consumer Staples": "Consumer Defensive"}


def sic_sector(sic):
    """SEC SIC code -> the sector names the site uses (Yahoo's). Coarse, only a fallback."""
    try:
        x = int(sic)
    except (TypeError, ValueError):
        return None
    if 2833 <= x <= 2836 or 3841 <= x <= 3845 or 3851 <= x <= 3851 or 8000 <= x <= 8099 or x == 5047 or x == 5122:
        return "Healthcare"
    if 3570 <= x <= 3579 or 3661 <= x <= 3679 or 3820 <= x <= 3829 or 7370 <= x <= 7379:
        return "Technology"
    if x in (6798,) or 6500 <= x <= 6553:
        return "Real Estate"
    if 6000 <= x <= 6799:
        return "Financial Services"
    if 1300 <= x <= 1399 or 2900 <= x <= 2999 or x in (4610, 4612, 4613, 4922, 4923):
        return "Energy"
    if 4900 <= x <= 4991:
        return "Utilities"
    if 4800 <= x <= 4899 or 2710 <= x <= 2741 or 7810 <= x <= 7841:
        return "Communication Services"
    if 1000 <= x <= 1499 or 2600 <= x <= 2679 or 2800 <= x <= 2829 or 2860 <= x <= 2899 or 3300 <= x <= 3399:
        return "Basic Materials"
    if 2000 <= x <= 2199 or 2840 <= x <= 2844 or 5400 <= x <= 5499 or 5140 <= x <= 5149:
        return "Consumer Defensive"
    if 2300 <= x <= 2399 or 3710 <= x <= 3716 or 5000 <= x <= 5999 or 7000 <= x <= 7099 or 7900 <= x <= 7999 or 2500 <= x <= 2599:
        return "Consumer Cyclical"
    if 1500 <= x <= 1799 or 3400 <= x <= 3899 or 4000 <= x <= 4799 or 8700 <= x <= 8799 or 7300 <= x <= 7389:
        return "Industrials"
    return None


def sectors_for(syms, ua):
    """{sym: {"sector", "industry", "name", "src"}}: Nasdaq's public stock screener (one call, all US listings),
    then SEC SIC codes for whatever is still missing. Never guessed: unknown stays unknown."""
    out = {}
    if D.DEMO:
        return out
    try:
        doc = json.loads(D.get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true", timeout=40))
        for r in (doc.get("data") or {}).get("rows") or []:
            sec = NASDAQ_SECTOR.get((r.get("sector") or "").strip())
            if sec:
                out[r["symbol"].strip().replace("/", "-").replace("^", "-")] = {
                    "sector": sec, "industry": (r.get("industry") or "").strip() or None, "name": r.get("name"), "src": "Nasdaq"}
    except Exception as e:
        print("Nasdaq screener unavailable: %s" % str(e)[:80], flush=True)
    miss = [s for s in syms if s not in out]
    if miss and ua:
        try:
            tick = B4._sec_get("https://www.sec.gov/files/company_tickers.json", ua)
            cik = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in tick.values()}
            for s in miss:
                k = cik.get(s.upper().replace("-", "."), cik.get(s.upper()))
                if not k:
                    continue
                try:
                    sub = B4._sec_get("https://data.sec.gov/submissions/CIK%s.json" % k, ua)
                    sec = sic_sector(sub.get("sic"))
                    if sec:
                        out[s] = {"sector": sec, "industry": sub.get("sicDescription"), "name": sub.get("name"), "src": "SEC SIC"}
                except Exception:
                    continue
        except Exception:
            pass
    return {s: out[s] for s in syms if s in out}


def fund_of(sym):
    """Yahoo fundamentals with a retry: the crumb service throttles bursts."""
    if D.DEMO:
        return D.demo_fund(sym, 100.0)
    for k in range(3):
        try:
            return D.fundamentals(sym)
        except Exception:
            D._crumb["fail"] = 0  # let the next try fetch a fresh crumb instead of waiting 2 minutes
            time.sleep(4 * (k + 1))
    return None


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
    if SESSION_OPEN and rows and datetime.datetime.fromtimestamp(rows[-1][0], D.DQ.NY).date() == datetime.datetime.now(D.DQ.NY).date():
        rows = rows[:-1]  # today's bar is still trading: closed bars only, like the rest of the site
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


def entry_view(sym, m, growth, secinfo=None):
    """The watchlist engine on one stock (closed bars), compacted for the scanner."""
    base = D.tech_part(sym)
    tech = base["tech"]
    fund = fund_of(sym)
    si = secinfo or {}
    if si.get("sector") and not (fund or {}).get("sector"):
        fund = dict(fund or {}, sector=si["sector"], industry=si.get("industry"), name=(fund or {}).get("name") or si.get("name"))
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
               sector_src="Yahoo" if (fund or {}).get("sector") and not si.get("sector") else si.get("src"), fund_ok=bool(fund and fund.get("next_earnings") is not None or (fund or {}).get("rev_growth") is not None),
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
    global SESSION_OPEN
    D.DEMO = "--demo" in sys.argv
    D.STATIC_BUILD = True
    SESSION_OPEN = D.us_session_open() and not D.DEMO
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
    secinfo = sectors_for(q.sym.tolist(), os.environ.get("SEC_USER_AGENT"))
    print("sectors for %d of %d (%.0fs)" % (len(secinfo), len(q), time.time() - t0), flush=True)
    D.market_ctx()
    rows, fails = [], 0
    recs = q.to_dict("records")
    with cf.ThreadPoolExecutor(max_workers=1 if D.DEMO else 2) as ex:
        futs = {ex.submit(entry_view, r["sym"], r, growth.get(r["sym"]), secinfo.get(r["sym"])): r["sym"] for r in recs}
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
           "fund_ok": sum(1 for r in rows if r.get("fund_ok")), "closed_bars_only": True,
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
