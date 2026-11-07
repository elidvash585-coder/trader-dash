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
import math
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
        v = pts / 6 * 10
        # ADX (trend STRENGTH, not direction) only scales how far the MA picture moves the score from 5.
        # Research v2: +DI/-DI direction was 0.94 correlated with RSI, so it is not counted again.
        if x.get("adx") is not None:
            v = 5 + (v - 5) * min(1.0, x["adx"] / 25.0)
            why.append("ממוצעים: %d מתוך 6 חיוביים, ADX %.0f (%s)" % (pts, x["adx"], "מגמה ברורה" if x["adx"] >= 20 else "מגמה חלשה"))
        else:
            why.append("ממוצעים: %d מתוך 6 חיוביים" % pts)
        clusters.append(v)
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
    # research v2: oscillators are 0.75 correlated with the trend family -> half weight inside the family
    if len(clusters) == 2:
        return (0.5 * clusters[0] + clusters[1]) / 1.5, why
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
    if e is not None and e > 3:  # only the extreme: the full range is 0.95 correlated with RSI
        clusters.append(3.0)
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


# ================================================================ historical score v2
RSI_BINS = [(0, 20, "<20"), (20, 25, "20-25"), (25, 30, "25-30"), (30, 35, "30-35"), (35, 40, "35-40"),
            (40, 50, "40-50"), (50, 60, "50-60"), (60, 70, "60-70"), (70, 101, ">70")]
NOT_COVERED = {"crypto": "קריפטו", "leveraged_etf": "תעודת סל ממונפת", "inverse_etf": "תעודת סל הפוכה"}


def load_table():
    try:
        with open(os.path.join(HERE, "research", "score_table_v2.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


TABLE = load_table()


def _bucket_row(tab, lab, min_n_eff):
    rows = tab["buckets"]
    r = rows.get(lab)
    if lab == "<20" and (not r or r.get("n_eff", 0) < min_n_eff):  # too few alone: pool with 20-25
        a, b = rows.get("<20") or {}, rows.get("20-25") or {}
        if "edge" in b:
            n = a.get("n", 0) + b["n"]
            w = lambda k: ((a.get(k) or 0) * a.get("n", 0) + b[k] * b["n"]) / n
            r = dict(b, n=n, n_eff=a.get("n_eff", 0) + b["n_eff"], edge=round(w("edge"), 3), exp_R=round(w("exp_R"), 3),
                     profit=round(w("profit"), 1), target=round(w("target"), 1), label="<25")
    return r


def hist_score(rsi, asset=None, blockers=None, tab=None, cfg=None):
    """0-10 from the 10-year research table. 5 = like entering on a random day.
    score = 5 + 4.5 * shrunk_edge / span_R, shrunk_edge = edge * n_eff / (n_eff + k);
    capped at 6.4 when the CI of the edge includes 0, at 5.5 when validation and test disagree in sign,
    at 5.0 by blockers (earnings inside the trade, fresh material filing, pre-market under the stop).
    N/A for assets outside the research universe or a group with too few independent trades."""
    tab = tab or TABLE
    c = (cfg or CFG).get("historical_score", {})
    k, span, min_n = c.get("shrink_k", 500), c.get("span_R", 0.30), c.get("min_n_eff", 100)
    base = {"value": None, "label": "N/A", "code": "na", "reasons": [], "version": "v2"}
    if not tab:
        base["reasons"].append(("warn", "טבלת המחקר לא זמינה."))
        return base
    at = (asset or {}).get("type")
    if at in NOT_COVERED:
        base["reasons"].append(("warn", "%s לא נכללה במחקר (רק מניות ותעודות סל רגילות), לכן אין ציון היסטורי." % NOT_COVERED[at]))
        return base
    if rsi is None:
        base["reasons"].append(("warn", "אין RSI."))
        return base
    lab = next(l for lo, hi, l in RSI_BINS if lo <= rsi < hi)
    r = _bucket_row(tab, lab, min_n)
    if not r or "edge" not in r or r.get("n_eff", 0) < min_n:
        base["reasons"].append(("warn", "מעט מדי עסקאות בלתי תלויות בטווח הזה (פחות מ-%d)." % min_n))
        return base
    shr = r["edge"] * r["n_eff"] / (r["n_eff"] + k)
    val = max(0.5, min(9.5, 5 + 4.5 * shr / span))
    lo, hi = r.get("edge_ci") or [None, None]
    sig = lo is not None and (lo > 0 or hi < 0)
    caps = []
    if not sig and val > c.get("cap_if_ci_has_zero", 6.4):
        val = c.get("cap_if_ci_has_zero", 6.4)
        caps.append("רווח הסמך כולל אפס")
    ve, te = r.get("val_edge"), r.get("test_edge")
    stable = ve is not None and te is not None and (ve > 0) == (te > 0) == (r["edge"] > 0)
    if ve is not None and te is not None and not stable and val > c.get("cap_if_recent_opposite", 5.5):
        val = c.get("cap_if_recent_opposite", 5.5)
        caps.append("הכיוון לא נשמר בין התקופות")
    bl = [b for b in (blockers or []) if b]
    if bl and val > c.get("cap_if_blocked", 5.0):
        val = c.get("cap_if_blocked", 5.0)
    val = round(val, 1)
    reasons = [("up" if r["edge"] > 0 else "down" if r["edge"] < 0 else "acc",
                "RSI %.0f (טווח %s). באותה עסקה בעבר: %.0f%% ברווח, %.0f%% הגיעו ליעד, ממוצע %+.2fR (יום אקראי: %+.2fR)." % (
                    rsi, r.get("label", lab), r["profit"], r["target"], r["exp_R"], tab["base_R"])),
               ("acc", "יתרון מול יום אקראי %+.2fR, רווח סמך %s%s. %s עסקאות (%s בלתי תלויות)." % (
                   r["edge"], "[%+.2f, %+.2f]" % (lo, hi) if lo is not None else "—", " (מובהק)" if sig else " (לא מובהק)",
                   format(r["n"], ","), format(r["n_eff"], ",")))]
    if ve is not None and te is not None:
        reasons.append(("up" if stable else "warn", "מחוץ למדגם: עד %s %+.2fR, מאז %+.2fR%s." % (
            tab["test_from"], ve, te, " (אותו כיוון)" if stable else " (לא יציב)")))
    if caps:
        reasons.append(("warn", "הציון הוגבל: " + ", ".join(caps) + "."))
    if bl:
        reasons.append(("warn", ("הציון הוגבל ל-5: " if val == c.get("cap_if_blocked", 5.0) else "שים לב: ") + "; ".join(bl) + "."))
    label = ("שווה לשקול כניסה" if val >= 7 else "חיובי קל" if val >= 5.6 else
             "ניטרלי, כמו כניסה אקראית" if val >= 4.4 else "לא כדאי עכשיו")
    return {"value": val, "label": label, "code": "go" if val >= 7 else "wait" if val >= 5.6 else "no", "version": "v2",
            "bucket": r.get("label", lab), "p_profit": r["profit"], "p_target": r["target"], "exp_R": r["exp_R"],
            "edge": r["edge"], "edge_ci": [lo, hi], "significant": sig, "stable": stable, "n": r["n"], "n_eff": r["n_eff"],
            "gap_loss": r.get("gap_loss"), "reasons": reasons, "capped": bool(caps or bl),
            "meta": {"stocks": tab["stocks"], "from": tab["from"], "to": tab["to"], "test_from": tab["test_from"],
                     "cost": "0.10% + החלקה 0.05% לכל צד"}}


# ================================================================ trade plan and decision

def plan_v2(x, cfg=None):
    """The tested trade (MR-a: next open, stop 1 ATR, target 3R, 10 sessions) + a first target at the nearest
    confirmed resistance when it sits between 1R and 3R. The stop is never moved to improve the R/R."""
    p, a = x.get("price"), x.get("atr")
    if not p or not a:
        return None
    entry = round(p, 2)
    stop = math.floor((entry - a) * 100) / 100
    risk = entry - stop
    if risk <= 0:
        return None
    t2 = round(entry + 3 * risk, 2)
    res = x.get("res")
    t1 = round(res, 2) if res and entry + risk <= res < t2 else None
    blocked_by_res = bool(res and res < entry + risk)
    rr = ((t1 or t2) - entry) / risk
    return {"entry": entry, "entry_zone": [round(entry - 0.25 * a, 2), round(entry + 0.1 * a, 2)], "stop": stop,
            "stop_pct": round(risk / entry * 100, 2), "stop_atr": 1.0, "t1": t1, "t2": t2, "rr_t1": round(rr, 2),
            "res": round(res, 2) if res else None, "res_r": round((res - entry) / risk, 2) if res else None,
            "res_blocks": blocked_by_res, "swing_low": round(x["swing_low"], 2) if x.get("swing_low") else None,
            "horizon": 10, "invalidation": "סגירה מתחת ל-%.2f, או 10 ימי מסחר בלי יעד" % stop}


def strategy_of(x, fund=None, asset=None):
    """Fixed order, decided before looking at any result: MR, then TF, then Quality Growth."""
    if (asset or {}).get("type") in NOT_COVERED:
        return {"key": None, "name": "מחוץ ליקום המחקר", "evidence": "לא נבדק לנכס מהסוג הזה"}
    rsi = x.get("rsi")
    if rsi is not None and rsi < 35:
        return {"key": "MR", "name": "Mean Reversion (מכירת יתר)", "evidence": "נבדק: יתרון חיובי, לא מובהק ברוב הגרסאות"}
    p = x.get("price")
    ma = [p > x[k] for k in ("sma20", "sma50", "sma150", "sma200") if x.get(k)] if p else []
    if ma and all(ma) and (x.get("rs_spy_60") or 0) > 0:
        return {"key": "TF", "name": "Trend Following (מגמה)", "evidence": "נבדק: ללא יתרון, פריצות היו שליליות מול יום אקראי"}
    f = fund or {}
    if (f.get("rev_growth") or 0) > 0.10 and (f.get("op_margin") or 0) > 0:
        return {"key": "QG", "name": "Quality Growth (צמיחה איכותית)", "evidence": "עוד לא נבדק לאחור (שלב 3)"}
    return {"key": None, "name": "אין מצב מוגדר", "evidence": ""}


def decide(x, hist, ind, plan, events=None, asset=None, fund=None, cfg=None):
    """One verbal decision from fixed rules (no average of the two scores).
    events: {"earnings_days": int|None, "earnings_known": bool, "sec_material": str|None, "sec_known": bool,
             "premarket_below_stop": bool|None}"""
    c = (cfg or CFG).get("decision", {})
    ev = events or {}
    hv = (hist or {}).get("value")
    iv = (ind or {}).get("value")
    st = strategy_of(x, fund, asset)
    items = []

    def item(label, ok, detail):
        items.append({"label": label, "ok": ok, "detail": detail})
    rsi = x.get("rsi")
    item("מצב מכירת יתר שנבדק (RSI מתחת ל-35)", None if rsi is None else rsi < 35,
         "RSI %.0f" % rsi if rsi is not None else "אין נתון")
    item("יתרון היסטורי (ציון 1 לפחות %.1f)" % c.get("min_hist", 5.6), None if hv is None else hv >= c.get("min_hist", 5.6),
         "ציון %.1f" % hv if hv is not None else "אין ציון היסטורי לנכס הזה")
    if plan:
        ok = not plan["res_blocks"] and plan["rr_t1"] >= c.get("min_rr", 2.0)
        item("מקום עד ההתנגדות (סיכוי/סיכון לפחות %.0f)" % c.get("min_rr", 2.0), ok,
             "התנגדות %.2f (%s)" % (plan["res"], "%.1fR" % plan["res_r"]) if plan["res"] else "אין התנגדות מאושרת מעל המחיר")
    else:
        item("מקום עד ההתנגדות", None, "אין נתונים לחישוב")
    if ev.get("earnings_known"):
        d = ev.get("earnings_days")
        inside = d is not None and 0 <= d <= 14
        item("אין דוח כספי בתוך תקופת העסקה", not inside, "דוח בעוד %d ימים" % d if d is not None and d >= 0 else "לא צפוי דוח קרוב")
    else:
        item("אין דוח כספי בתוך תקופת העסקה", None, "אין מועד דוח ידוע")
    if ev.get("sec_known"):
        item("אין דיווח מהותי טרי ל-SEC", not ev.get("sec_material"), ev.get("sec_material") or "אין ב-7 הימים האחרונים")
    else:
        item("אין דיווח מהותי טרי ל-SEC", None, "אין נתוני SEC לנכס הזה")
    if ev.get("premarket_below_stop") is not None:
        item("המחיר מחוץ לשעות המסחר לא מתחת לסטופ", not ev["premarket_below_stop"], "")
    covered = (asset or {}).get("type") not in NOT_COVERED
    item("הנכס בתוך יקום המחקר והנתונים תקינים", covered,
         "מניה או תעודת סל רגילה" if covered else NOT_COVERED.get((asset or {}).get("type"), ""))
    items = items[:7]

    pros, cons = [], []
    if hv is not None and hv >= c.get("min_hist", 5.6):
        pros.append("בעבר, מצב RSI כזה נתן יתרון של %+.2fR מול יום אקראי" % hist["edge"])
    elif hv is not None and hv < 4.4:
        cons.append("בעבר, מצב RSI כזה היה חלש מיום אקראי (%+.2fR)" % hist["edge"])
    if iv is not None and iv >= 6.5:
        pros.append("התנאים הטכניים והפונדמנטליים חיוביים (ציון 2: %.1f), אבל זה לא נמצא מנבא במחקר" % iv)
    elif iv is not None and iv < 4.5:
        cons.append("התנאים הנוכחיים חלשים (ציון 2: %.1f)" % iv)
    if x.get("regime") == "off":
        cons.append("השוק ב-Risk-Off (במחקר זה לא פגע במכירת יתר, אבל פגע בפריצות)")
    if st["key"] == "TF":
        cons.append("מצב מגמה: במחקר, כניסה בפריצה לא נתנה יתרון")
    failed = [i for i in items if i["ok"] is False]
    unknown = [i for i in items if i["ok"] is None]

    weak = (iv is not None and iv <= 3.5) or (x.get("state") == -2 and x.get("bos_dn"))
    bad_event = any(i["ok"] is False for i in items if "SEC" in i["label"])
    if hv is not None and not failed:
        verdict, code = "כניסה אפשרית", "go"
        missing = None
        change = "ביטול: %s" % plan["invalidation"] if plan else ""
    elif hv is not None and hv >= c.get("min_hist", 5.6):
        verdict, code = "המתנה", "wait"
        missing = failed[0]["label"] + (" (" + failed[0]["detail"] + ")" if failed[0]["detail"] else "")
        change = "כניסה תהיה אפשרית כשהתנאי הזה יתקיים והציון ההיסטורי יישאר מעל %.1f" % c.get("min_hist", 5.6)
    elif weak or bad_event:
        verdict, code = "אם אתה מחזיק: בדוק חשיפה", "exposure"
        missing = "התנאים הנוכחיים חלשים" if weak else "דיווח מהותי טרי"
        change = "שיפור במבנה (שפל עולה מאושר) או ירידה למכירת יתר עמוקה"
    elif iv is not None and iv >= 6.5:
        verdict, code = "המתנה", "wait"
        missing = "אין יתרון היסטורי מוכח בכניסה עכשיו (ציון 1 %s)" % ("%.1f" % hv if hv is not None else "N/A")
        change = "ירידה לאזור מכירת יתר (RSI מתחת ל-35) בלי שבירת המגמה"
    else:
        verdict, code = "מעקב", "watch"
        missing = "אין יתרון היסטורי מוכח (ציון 1 %s)" % ("%.1f" % hv if hv is not None else "N/A")
        change = "ירידה לאזור מכירת יתר (RSI מתחת ל-35)"
    for i in failed:
        txt = i["label"] + (": " + i["detail"] if i["detail"] else "")
        if not (missing and missing.startswith(i["label"])) and not i["label"].startswith("יתרון היסטורי"):
            cons.append(txt)
    return {"verdict": verdict, "code": code, "items": items, "pros": pros[:3], "cons": cons[:3], "missing": missing,
            "change": change, "unknown": len(unknown), "strategy": st, "version": "v2"}
