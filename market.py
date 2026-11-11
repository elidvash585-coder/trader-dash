"""Market context: regime (Risk-On / Neutral / Risk-Off), sector ETF and asset type.

Standard library only. The regime rule is fixed in model_config.json and is the same in the
research (backtest4.py) and on the site. Breadth is NOT used: there is no reliable free source,
and it is not invented. VIX is used when Yahoo returns it, and skipped (not guessed) when not.
"""
import re

import features as F

SECTOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Energy": "XLE",
              "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI",
              "Basic Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC"}
SECTOR_HE = {"XLK": "טכנולוגיה", "XLF": "פיננסים", "XLV": "בריאות", "XLE": "אנרגיה", "XLY": "צריכה מחזורית",
             "XLP": "צריכה בסיסית", "XLI": "תעשייה", "XLB": "חומרי גלם", "XLU": "תשתיות", "XLRE": "נדל\"ן",
             "XLC": "תקשורת"}
REGIME_HE = {"on": "Risk-On", "neutral": "Neutral", "off": "Risk-Off"}


def _trend_pts(c, s50, s200, slope50, i):
    if s200[i] is None or s50[i] is None or slope50[i] is None:
        return None
    return (1 if c[i] > s200[i] else -1) + (0.5 if c[i] > s50[i] else -0.5) + (0.5 if slope50[i] > 0 else -0.5)


def regime_series(spy_c, qqq_c=None, vix_c=None, cfg=None):
    """Per bar: points and label, from SPY (and QQQ, VIX when given, aligned to SPY's bars).
    points = sum over SPY/QQQ of [+-1 vs SMA200, +-0.5 vs SMA50, +-0.5 SMA50 slope]
             + VIX term (<18: +1, >25: -1, >32: -2) + SPY 20d realized-vol percentile > 80: -1."""
    cfg = cfg or {}
    on, off = cfg.get("risk_on_min", 3.0), cfg.get("risk_off_max", -1.5)
    N = len(spy_c)
    parts = []
    for c in (spy_c, qqq_c):
        if not c:
            continue
        s50, s200 = F.sma(c, 50), F.sma(c, 200)
        sl = [(s50[i] / s50[i - 20] - 1) if i >= 20 and s50[i] and s50[i - 20] else None for i in range(len(c))]
        parts.append((c, s50, s200, sl))
    rvp = F.pct_rank(F.realized_vol(spy_c), 252)
    pts, lab = [None] * N, [None] * N
    for i in range(N):
        p = [_trend_pts(*x, i) for x in parts]
        if any(x is None for x in p):
            continue
        s = sum(p) * (2.0 / len(p))  # scaled as if both indices were present
        if vix_c and vix_c[i]:
            s += 1 if vix_c[i] < 18 else -2 if vix_c[i] > 32 else -1 if vix_c[i] > 25 else 0
        if rvp[i] is not None and rvp[i] > 80:
            s -= 1
        pts[i] = s
        lab[i] = "on" if s >= on else "off" if s <= off else "neutral"
    return pts, lab


def etf_trend(c):
    """+1 above SMA50 and SMA200, -1 below both, 0 mixed (None without enough history)."""
    s50, s200 = F.sma(c, 50), F.sma(c, 200)
    out = [None] * len(c)
    for i in range(len(c)):
        if s50[i] and s200[i]:
            out[i] = 1 if c[i] > s50[i] and c[i] > s200[i] else -1 if c[i] < s50[i] and c[i] < s200[i] else 0
    return out


def asset_type(sym, meta=None, name=None):
    """stock / etf / leveraged_etf / inverse_etf / crypto, and the stated daily leverage."""
    it = ((meta or {}).get("instrumentType") or "").upper()
    nm = (name or (meta or {}).get("longName") or (meta or {}).get("shortName") or "")
    if it == "CRYPTOCURRENCY" or sym.endswith("-USD"):
        return {"type": "crypto", "leverage": 1}
    lev = re.search(r"(-?\d(?:\.\d)?)\s*[xX]\b", nm)
    inverse = bool(re.search(r"\b(inverse|short|bear)\b", nm, re.I))
    if it == "ETF" or re.search(r"\b(ETF|Fund|Trust)\b", nm):
        if lev or re.search(r"\b(ultra|leveraged|daily)\b", nm, re.I):
            k = abs(float(lev.group(1))) if lev else 2.0
            return {"type": "inverse_etf" if inverse else "leveraged_etf", "leverage": -k if inverse else k}
        return {"type": "inverse_etf" if inverse else "etf", "leverage": -1 if inverse else 1}
    return {"type": "stock", "leverage": 1}


def context(fetch, cfg=None):
    """Live context. fetch(sym) -> (meta, rows[(t,o,h,l,c,v)]). Errors leave parts out."""
    out = {"spy": None, "qqq": None, "vix": None, "regime": None, "errors": []}
    data = {}
    vols = {}
    for s in ("SPY", "QQQ", "^VIX"):
        try:
            _, rows = fetch(s)
            data[s] = ([r[0] for r in rows], [float(r[4]) for r in rows])
            vols[s] = [float(r[5] or 0) for r in rows]
        except Exception as e:
            out["errors"].append("%s: %s" % (s, str(e)[:60]))
    if "SPY" not in data:
        return out
    t, spy = data["SPY"]
    qqq = F.align(t, *data["QQQ"]) if "QQQ" in data else None
    vix = F.align(t, *data["^VIX"]) if "^VIX" in data else None
    pts, lab = regime_series(spy, qqq, vix, cfg)
    s50, s200 = F.sma(spy, 50), F.sma(spy, 200)
    out.update(spy={"t": t, "c": spy, "v": vols.get("SPY"), "above50": spy[-1] > (s50[-1] or 0), "above200": spy[-1] > (s200[-1] or 0)},
               qqq={"t": data["QQQ"][0], "c": data["QQQ"][1], "v": vols.get("QQQ")} if "QQQ" in data else None,
               vix=round(vix[-1], 2) if vix and vix[-1] else None,
               regime={"code": lab[-1], "label": REGIME_HE.get(lab[-1]), "points": round(pts[-1], 2) if pts[-1] is not None else None,
                       "uses_vix": bool(vix and vix[-1]), "uses_qqq": qqq is not None})
    return out
