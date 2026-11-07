"""Portfolio risk for Trader Dash (stage 3). Standard library only.

Works from a LOCAL holdings file (holdings.json next to the app; never part of the public site):
    {"cash": 2000, "positions": [{"symbol": "NVDA", "qty": 10, "stop": 160.5, "cost": 120}, ...]}
"stop" and "cost" are optional. Without holdings nothing here runs: the watchlist is not a portfolio.

What it reports, and what each number assumes:
- exposure: market value, weights, sector weights, EFFECTIVE exposure (a 2x daily ETF counts twice,
  an inverse one negatively), concentration (largest weight, effective number of positions = 1/HHI).
- risk to stops: qty x (price - stop). A stop is NOT a maximum loss: gaps can jump it.
- correlation: daily returns over the last 120 sessions; pairs at 0.7+ behave like one bet.
- stress: (a) SPY -10 % / -20 % shocks through each position's 1-year beta, (b) the worst 10-session
  window in the joint history, replayed with today's positions.
- Monte Carlo: 3,000 paths of 10 and 20 sessions, drawn as 5-session blocks of REAL joint daily
  returns (keeps correlation and short-term dependence). Each position compounds daily, so a leveraged
  daily ETF shows its path effect (volatility drag). Results are model outcomes, not a forecast.
Leveraged ETFs with short history are extended with leverage x the underlying's daily return
(daily reset), when the underlying can be read from the fund name.
"""
import json
import math
import os
import random
import re

import features as FX
import market as MK

HERE = os.path.dirname(os.path.abspath(__file__))
HOLDINGS = os.path.join(HERE, "holdings.json")


def read_holdings(path=HOLDINGS):
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        return None
    pos = []
    for p in doc.get("positions") or []:
        try:
            q = float(p["qty"])
        except (KeyError, TypeError, ValueError):
            continue
        if q:
            pos.append({"symbol": str(p["symbol"]).upper().strip(), "qty": q,
                        "stop": float(p["stop"]) if p.get("stop") not in (None, "") else None,
                        "cost": float(p["cost"]) if p.get("cost") not in (None, "") else None})
    return {"cash": float(doc.get("cash") or 0), "positions": pos}


def _rets(rows):
    """{day: daily simple return} from (t,o,h,l,c,v) rows."""
    out = {}
    for i in range(1, len(rows)):
        a, b = rows[i - 1][4], rows[i][4]
        if a and b:
            out[int(rows[i][0] // 86400)] = b / a - 1
    return out


def underlying_of(name):
    m = re.search(r"\b(?:Long|Short)\s+([A-Z]{1,5})\b", name or "")
    return m.group(1) if m else None


def _corr(a, b):
    n = len(a)
    if n < 20:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if not sa or not sb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def _beta(r, m):
    n = len(r)
    if n < 60:
        return None
    mr, mm = sum(r) / n, sum(m) / n
    vm = sum((y - mm) ** 2 for y in m)
    return sum((x - mr) * (y - mm) for x, y in zip(r, m)) / vm if vm else None


def portfolio(hold, fetch, fund_of=None, mc_paths=3000, seed=11):
    """hold: read_holdings(). fetch(sym) -> (meta, rows) daily bars. fund_of(sym) -> fundamentals or None."""
    if not hold or not hold["positions"]:
        return None
    errors, pos = [], []
    series = {}
    spy_meta, spy_rows = fetch("SPY")
    spy = _rets(spy_rows)
    for p in hold["positions"]:
        try:
            meta, rows = fetch(p["symbol"])
        except Exception as e:
            errors.append("%s: %s" % (p["symbol"], str(e)[:80]))
            continue
        name = meta.get("longName") or meta.get("shortName") or ""
        at = MK.asset_type(p["symbol"], meta, name)
        r = _rets(rows)
        synth = 0
        if at["type"] in ("leveraged_etf", "inverse_etf") and len(r) < 300:
            u = underlying_of(name)
            if u:
                try:
                    _, ur = fetch(u)
                    for d, x in _rets(ur).items():
                        if d not in r:
                            r[d] = max(-0.99, at["leverage"] * x)
                            synth += 1
                except Exception:
                    pass
        series[p["symbol"]] = r
        price = rows[-1][4]
        f = (fund_of(p["symbol"]) if fund_of else None) or {}
        sector = f.get("sector") or {"crypto": "קריפטו", "etf": "תעודות סל", "leveraged_etf": "ממונפים",
                                     "inverse_etf": "ממונפים"}.get(at["type"], "לא ידוע")
        days = sorted(set(r) & set(spy))[-250:]
        beta = _beta([r[d] for d in days], [spy[d] for d in days])
        last = sorted(r)[-250:]
        vol = math.sqrt(sum(r[d] ** 2 for d in last) / len(last)) * math.sqrt(252) * 100 if last else None
        value = p["qty"] * price
        pos.append({"symbol": p["symbol"], "qty": p["qty"], "price": round(price, 2), "value": round(value, 2),
                    "asset": at, "sector": sector, "beta": round(beta, 2) if beta is not None else None,
                    "vol": round(vol, 1) if vol else None, "stop": p["stop"], "cost": p["cost"],
                    "pnl": round((price - p["cost"]) * p["qty"], 2) if p["cost"] else None,
                    "risk_to_stop": round(max(0.0, (price - p["stop"]) * p["qty"]), 2) if p["stop"] and p["qty"] > 0 else None,
                    "eff": round(value * at["leverage"], 2), "synthetic_days": synth})
    if not pos:
        return {"errors": errors}
    gross = sum(abs(x["value"]) for x in pos)
    equity = sum(x["value"] for x in pos) + hold["cash"]
    for x in pos:
        x["weight"] = round(x["value"] / equity * 100, 2) if equity else None
    w2 = sum((abs(x["value"]) / gross) ** 2 for x in pos) if gross else 1
    sectors = {}
    for x in pos:
        sectors[x["sector"]] = sectors.get(x["sector"], 0) + x["value"]
    eff = sum(x["eff"] for x in pos)
    stops = [x for x in pos if x["risk_to_stop"] is not None]
    out = {"equity": round(equity, 2), "cash": hold["cash"], "invested": round(gross, 2), "positions": pos,
           "effective_exposure_pct": round(eff / equity * 100, 1) if equity else None,
           "largest": max(pos, key=lambda x: abs(x["value"]))["symbol"],
           "largest_pct": round(max(abs(x["value"]) for x in pos) / equity * 100, 1) if equity else None,
           "effective_n": round(1 / w2, 1) if w2 else None,
           "sectors": sorted(([k, round(v / equity * 100, 1)] for k, v in sectors.items()), key=lambda z: -z[1]),
           "risk_to_stops": round(sum(x["risk_to_stop"] for x in stops), 2),
           "risk_to_stops_pct": round(sum(x["risk_to_stop"] for x in stops) / equity * 100, 2) if equity else None,
           "without_stop": [x["symbol"] for x in pos if x["risk_to_stop"] is None],
           "leveraged": [x["symbol"] for x in pos if x["asset"]["type"] in ("leveraged_etf", "inverse_etf")],
           "errors": errors}
    # correlation (last 120 common sessions)
    syms = [x["symbol"] for x in pos]
    pairs = []
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            days = sorted(set(series[a]) & set(series[b]))[-120:]
            c = _corr([series[a][d] for d in days], [series[b][d] for d in days])
            if c is not None:
                pairs.append([a, b, round(c, 2)])
    pairs.sort(key=lambda z: -z[2])
    out["corr_high"] = [p for p in pairs if p[2] >= 0.7][:10]
    out["corr_avg"] = round(sum(p[2] for p in pairs) / len(pairs), 2) if pairs else None
    # stress (a): beta shocks
    out["stress"] = []
    for shock in (-0.10, -0.20):
        loss = sum(x["value"] * (x["beta"] if x["beta"] is not None else 1.0) * shock for x in pos)
        out["stress"].append({"name": "SPY %d%%" % (shock * 100), "pnl": round(loss, 2),
                              "pct": round(loss / equity * 100, 1) if equity else None})
    # joint history (days where every position has a return; positions keep today's quantities)
    common = sorted(set.intersection(*[set(series[s]) for s in syms]))[-500:]
    out["history_days"] = len(common)
    if len(common) >= 60:
        vals = [x["value"] for x in pos]
        worst, worst_at = 0.0, None
        for i in range(len(common) - 10):
            v = list(vals)
            for d in common[i:i + 10]:
                v = [vi * (1 + series[s][d]) for vi, s in zip(v, syms)]
            pnl = sum(v) - sum(vals)
            if pnl < worst:
                worst, worst_at = pnl, common[i]
        out["stress"].append({"name": "החלון הגרוע ב-10 ימים בהיסטוריה", "pnl": round(worst, 2),
                              "pct": round(worst / equity * 100, 1) if equity else None,
                              "from_day": worst_at})
        out["mc"] = monte_carlo(vals, syms, series, common, hold["cash"], mc_paths, seed)
    return out


def monte_carlo(vals, syms, series, days, cash, paths=3000, seed=11, block=5):
    rnd = random.Random(seed)
    n = len(days)
    eq0 = sum(vals) + cash
    res = {}
    for hz in (10, 20):
        finals, dds = [], []
        for _ in range(paths):
            v = list(vals)
            peak, dd, t = eq0, 0.0, 0
            while t < hz:
                s = rnd.randrange(0, n - block)
                for d in days[s:s + block]:
                    if t >= hz:
                        break
                    v = [vi * (1 + series[sy][d]) for vi, sy in zip(v, syms)]
                    eq = sum(v) + cash
                    peak = max(peak, eq)
                    dd = max(dd, (peak - eq) / peak if peak else 0)
                    t += 1
            finals.append((sum(v) + cash) / eq0 - 1)
            dds.append(dd)
        finals.sort()
        dds.sort()
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        tail = finals[: max(1, int(0.05 * len(finals)))]
        res[str(hz)] = {"median": round(q(finals, 0.5) * 100, 2), "p05": round(q(finals, 0.05) * 100, 2),
                        "p95": round(q(finals, 0.95) * 100, 2), "es05": round(sum(tail) / len(tail) * 100, 2),
                        "p_loss5": round(sum(1 for x in finals if x <= -0.05) / paths * 100, 1),
                        "p_loss10": round(sum(1 for x in finals if x <= -0.10) / paths * 100, 1),
                        "dd_median": round(q(dds, 0.5) * 100, 2), "dd_p95": round(q(dds, 0.95) * 100, 2)}
    res["paths"], res["block"], res["from_days"] = paths, block, n
    return res


def leveraged_path_demo(daily, lev, days=20):
    """Cumulative return of a daily-reset leveraged product vs lev x the base return over the same path."""
    base, prod = 1.0, 1.0
    for r in daily[:days]:
        base *= 1 + r
        prod *= 1 + lev * r
    return prod - 1, lev * (base - 1)


def watchlist_peers(closes, top=3, min_corr=0.6, look=120):
    """closes: {sym: [(t, close), ...]}. For each symbol, its most correlated others (daily returns, last
    `look` common sessions). Watchlist overlap only: not a portfolio and not holdings."""
    rets = {}
    for s, rows in closes.items():
        rets[s] = {int(rows[i][0] // 86400): rows[i][1] / rows[i - 1][1] - 1 for i in range(1, len(rows)) if rows[i - 1][1]}
    out = {}
    syms = list(rets)
    for a in syms:
        lst = []
        for b in syms:
            if a == b:
                continue
            d = sorted(set(rets[a]) & set(rets[b]))[-look:]
            c = _corr([rets[a][x] for x in d], [rets[b][x] for x in d])
            if c is not None and c >= min_corr:
                lst.append([b, round(c, 2)])
        out[a] = sorted(lst, key=lambda z: -z[1])[:top]
    return out
