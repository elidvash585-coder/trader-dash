"""Indicator score v2 (0-10): six families, weights from model_config.json.

Rules that keep it honest:
- Correlated indicators are grouped into ONE cluster that counts once (RSI+CCI+MACD = "oscillators",
  SMA20/50/150/200 + slope = "MA stack"). A family is the mean of its clusters.
- Missing data is never 0 and never positive: the family is left out and the weights are
  renormalised over the families that have data. Below min_coverage of the weight -> N/A.
- This score is NOT a probability. Whether it predicts anything is measured in backtest4.py
  (price-based families only: fundamentals are not available point-in-time from Yahoo).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY_HE = {"trend": "מגמה ומבנה", "momentum": "מומנטום ועוצמה יחסית", "volume": "נפחים",
             "fundamentals": "פונדמנטלי", "market": "שוק וסקטור", "entry": "איכות כניסה וסיכון"}


def load_cfg():
    try:
        with open(os.path.join(HERE, "model_config.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


CFG = load_cfg()


def cl(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- family scorers
# Each takes a dict of last values and returns (value or None, [why...]).

def fam_trend(x):
    clusters, why = [], []
    keys = ("price", "sma20", "sma50", "sma150", "sma200", "slope50")
    if all(x.get(k) is not None for k in keys):
        p = x["price"]
        pts = sum([p > x["sma20"], p > x["sma50"], p > x["sma150"], p > x["sma200"], x["sma50"] > x["sma200"], x["slope50"] > 0])
        clusters.append(pts / 6 * 10)
        why.append("ממוצעים: %d מתוך 6 חיוביים" % pts)
    if x.get("adx") is not None and x.get("pdi") is not None:
        a = x["adx"]
        if a < 20:
            clusters.append(5.0)
            why.append("ADX %.0f: אין מגמה ברורה" % a)
        else:
            d = 1 if x["pdi"] > x["ndi"] else -1
            clusters.append(5 + d * min(a - 20, 20) / 20 * 4)
            why.append("ADX %.0f, כיוון %s" % (a, "עולה" if d > 0 else "יורד"))
    if x.get("state") is not None:
        v = {2: 8.0, 0: 5.0, -2: 2.0}[x["state"]]
        if x.get("failed_bo"):
            v -= 1.5
        clusters.append(cl(v))
        why.append({2: "שיאים ושפלים עולים", 0: "מבנה מעורב", -2: "שיאים ושפלים יורדים"}[x["state"]] +
                   (", פריצה שנכשלה" if x.get("failed_bo") else ""))
    return _mean(clusters), why


def fam_momentum(x):
    clusters, why = [], []
    osc = []
    if x.get("rsi") is not None:
        osc.append(cl(5 + (x["rsi"] - 50) / 5))
    if x.get("cci") is not None:
        osc.append(cl(5 + x["cci"] / 40))
    if x.get("macd_h") is not None:
        osc.append(6.5 if x["macd_h"] > 0 else 3.5)
    if osc:
        clusters.append(sum(osc) / len(osc))
        why.append("מתנדים: RSI %s, CCI %s" % (_f(x.get("rsi")), _f(x.get("cci"))))
    rs = []
    for b in ("spy", "sector", "qqq"):
        vals = [x.get("rs_%s_%d" % (b, n)) for n in (20, 60, 120)]
        if all(v is None for v in vals):
            continue
        part = _mean([cl(5 + vals[0] / 2) if vals[0] is not None else None,
                      cl(5 + vals[1] / 4) if vals[1] is not None else None,
                      cl(5 + vals[2] / 6) if vals[2] is not None else None])
        rs.append(part)
        if b == "spy" and vals[1] is not None:
            trend = "" if vals[0] is None else (", מתחזקת לאחרונה" if vals[0] / 20 > vals[1] / 60 else ", נחלשת לאחרונה")
            why.append("מול SPY: %+.1f%% ב-60 יום%s" % (vals[1], trend))
    if rs:
        clusters.append(sum(rs) / len(rs))
    return _mean(clusters), why


def fam_volume(x):
    clusters, why = [], []
    acc = []
    if x.get("updown") is not None:
        acc.append(cl(5 + (x["updown"] - 1) * 5))
    if x.get("obv20") is not None:
        acc.append({1: 7.0, 0: 5.0, -1: 3.0}[x["obv20"]])
    if acc:
        clusters.append(sum(acc) / len(acc))
        why.append("נפח עלייה/ירידה %s, OBV %s" % (_f(x.get("updown"), 2), {1: "עולה", 0: "שטוח", -1: "יורד"}.get(x.get("obv20"), "—")))
    if x.get("rvol") is not None and x.get("chg1") is not None:
        r, up = x["rvol"], x["chg1"] >= 0
        v = 5.0 if r < 1.2 else (7.0 if up else 3.0)
        clusters.append(v)
        if r >= 1.2:
            why.append("נפח x%.1f ביום %s" % (r, "עלייה" if up else "ירידה"))
    return _mean(clusters), why


def fam_fundamentals(f):
    if not f:
        return None, []
    clusters, why = [], []
    g = _mean([_grow(f.get("rev_growth"), 0.20), _grow(f.get("earn_growth"), 0.30)])
    if g is not None:
        clusters.append(g)
        why.append("צמיחת הכנסות %s" % _p(f.get("rev_growth")))
    rev, fcf = f.get("revenue"), f.get("fcf")
    prof = _mean([_lvl(f.get("op_margin"), 0.0, 0.20), _lvl(fcf / rev if rev and fcf is not None else None, 0.0, 0.15),
                  _lvl(f.get("roe"), 0.0, 0.20)])
    if prof is not None:
        clusters.append(prof)
        why.append("שולי תפעול %s" % _p(f.get("op_margin")))
    val = []
    peg, pe, fpe = f.get("peg"), f.get("pe"), f.get("fwd_pe")
    if peg is not None and 0 < peg < 10:  # a PEG is meaningless for losses or no growth
        val.append(8 if peg < 1 else 6 if peg < 2 else 4.5 if peg < 3 else 3)
    if pe and fpe and pe > 0 and fpe > 0:
        val.append(6.5 if fpe < pe * 0.85 else 5 if fpe <= pe * 1.05 else 3.5)
    if val:
        clusters.append(sum(val) / len(val))
        why.append("תמחור: PEG %s, מכפיל עתידי %s" % (_f(peg, 1), _f(fpe, 1)))
    sur = [h.get("surprise") for h in (f.get("eps_history") or []) if h.get("surprise") is not None]
    if sur:
        s = sum(sur[-4:]) / len(sur[-4:])
        clusters.append(cl(5 + s * 20))
        why.append("הפתעות רווח ממוצעות %+.0f%%" % (s * 100))
    an = f.get("analyst") or {}
    return _mean(clusters), why


def fam_market(x):
    clusters, why = [], []
    rg = x.get("regime")
    if rg:
        clusters.append({"on": 7.0, "neutral": 5.0, "off": 3.0}[rg])
        why.append("שוק %s" % {"on": "Risk-On", "neutral": "Neutral", "off": "Risk-Off"}[rg])
    st = x.get("sector_trend")
    if st is not None:
        clusters.append({1: 7.0, 0: 5.0, -1: 3.0}[st])
        why.append("סקטור %s: %s" % (x.get("sector_etf") or "", {1: "מגמה עולה", 0: "מעורב", -1: "מגמה יורדת"}[st]))
    return _mean(clusters), why


def fam_entry(x, tm=None):
    clusters, why = [], []
    if x.get("atr"):
        room = x.get("room_atr")
        if room is None:
            clusters.append(8.0)
            why.append("אין התנגדות מאושרת קרובה")
        else:
            clusters.append(8.0 if room >= 3 else 7.0 if room >= 2 else 5.0 if room >= 1 else 2.5)
            why.append("התנגדות במרחק %.1f ATR" % room)
    e = x.get("ext_atr")
    if e is not None:
        clusters.append(2.5 if e > 3 else 4.5 if e > 2 else 6.5 if e >= -1 else 5.0)
        if e > 2:
            why.append("מתוח %.1f ATR מעל ממוצע 20" % e)
    if x.get("atr_pctile") is not None and x["atr_pctile"] > 90:
        clusters.append(3.5)
        why.append("תנודתיות חריגה (אחוזון %.0f)" % x["atr_pctile"])
    if tm and tm.get("share") is not None:
        clusters.append(cl(5 + (tm["share"] - tm["breakeven"]) / 3))
        why.append("Time Machine: %.0f%% מול איזון %.0f%%" % (tm["share"], tm["breakeven"]))
    return _mean(clusters), why


# ---------------------------------------------------------------- the score

def indicator_score(x, fund=None, tm=None, cfg=None, price_only=False):
    """x: last values (features + price, regime, sector_trend). Returns the score dict."""
    cfg = (cfg or CFG).get("indicator_score", {})
    W = cfg.get("weights", {"trend": 25, "momentum": 20, "volume": 15, "fundamentals": 15, "market": 15, "entry": 10})
    fams = {"trend": fam_trend(x), "momentum": fam_momentum(x), "volume": fam_volume(x),
            "fundamentals": (None, []) if price_only else fam_fundamentals(fund),
            "market": fam_market(x), "entry": fam_entry(x, None if price_only else tm)}
    if price_only:
        W = {k: w for k, w in W.items() if k != "fundamentals"}
    tot = sum(W.values())
    have = {k: v for k, v in fams.items() if k in W and v[0] is not None}
    cov = sum(W[k] for k in have) / tot if tot else 0
    comps = [{"key": k, "name": FAMILY_HE[k], "weight": W[k], "value": round(fams[k][0], 1) if fams[k][0] is not None else None,
              "why": "; ".join(fams[k][1]) if fams[k][0] is not None else "אין מספיק נתונים"} for k in W]
    if cov < cfg.get("min_coverage", 0.6):
        return {"value": None, "label": "N/A", "coverage": round(cov, 2), "components": comps, "n": len(have),
                "missing": [FAMILY_HE[k] for k in W if k not in have], "version": "v2"}
    val = sum(W[k] * have[k][0] for k in have) / sum(W[k] for k in have)
    val = round(val, 1)
    label = next(l for t, l in cfg.get("labels", [[6.5, "חיובי"], [5.5, "חיובי קל"], [4.5, "ניטרלי"], [3.5, "שלילי קל"], [0, "שלילי"]]) if val >= t)
    return {"value": val, "label": label, "coverage": round(cov, 2), "components": comps, "n": len(have),
            "missing": [FAMILY_HE[k] for k in W if k not in have], "version": "v2"}


# ---------------------------------------------------------------- helpers

def _f(x, d=0):
    return "—" if x is None else ("%." + str(d) + "f") % x


def _p(x):
    return "—" if x is None else "%+.0f%%" % (x * 100)


def _grow(g, strong):
    if g is None:
        return None
    return cl(5 + g / strong * 3)


def _lvl(m, zero, good):
    if m is None:
        return None
    if m <= zero:
        return cl(4 + m / max(good, 1e-9) * 4, 1.5, 4)  # losses lower the cluster but never to 0
    return cl(5 + (m - zero) / (good - zero) * 3, 5, 9)
