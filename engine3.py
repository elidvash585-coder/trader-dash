"""Trader Dash v3: the two scores, the trade plan and the decision. Standard library only.

  Technical score (0-10): the state of the stock now. Weights from the spec, kept in model_config.json:
      trend & moving averages 25, highs/lows structure 20, momentum 20, volume 15,
      relative strength & market/sector 10, volatility & stability 10.
  Entry score (0-10): the quality of THIS trade now, not of the stock:
      entry location vs support 25, reward/risk to realistic targets 25, breakout/retest confirmation 20,
      volume & momentum confirmation 10, historical expectancy in this situation 15, event & volatility risk 5.

Rules that keep it conservative and checkable:
- Weights are a starting point. research/scores_v3.json reports, out of sample, whether each score predicts;
  the site shows that verdict next to the score. Nothing here claims a score predicts until that test says so.
- RSI, CCI and MACD move together, so momentum is ONE family (their median), not three votes.
- Missing data never helps: the weighted mean of what is known is pulled toward 5 by the share that is
  missing (value = 5 + (mean - 5) * coverage), and below 60 % coverage there is no score.
- No structural stop -> no reward/risk is shown and the entry score is capped at 4.5.
- A stock that already ran away from the entry zone (today, or on the last closed bar) is capped, so a
  big up day can never by itself produce a high entry score.
- "Possible entry" needs ALL of: a statistically significant historical edge that kept its sign out of
  sample, a structural stop, reward/risk >= 2 to a realistic target, the price still in the entry zone,
  entry score >= 6.5, no falling-knife structure, no known event inside the trade, verified data.
"""
import json
import math
import os

import engine as EN

CFG = EN.CFG
TECH_W = {"trend": 25, "structure": 20, "momentum": 20, "volume": 15, "rs": 10, "vol": 10}
ENTRY_W = {"location": 25, "rr": 25, "confirm": 20, "volmom": 10, "history": 15, "risk": 5}
TECH_HE = {"trend": "מגמה וממוצעים נעים", "structure": "מבנה שיאים ושפלים", "momentum": "מומנטום (RSI, CCI, MACD)",
           "volume": "מחזור ביחס לממוצע", "rs": "חוזק יחסי מול SPY והסקטור", "vol": "תנודתיות ויציבות"}
ENTRY_HE = {"location": "מיקום הכניסה מול התמיכה", "rr": "סיכוי/סיכון ליעד ריאלי", "confirm": "אישור פריצה או ריטסט",
            "volmom": "אישור מחזור ומומנטום", "history": "תוחלת היסטורית במצב דומה", "risk": "סיכון אירועים ותנודתיות"}
VERDICTS = {"go": "כניסה אפשרית", "wait": "המתנה לאישור", "no": "לא להיכנס", "reduce": "בחינת הקטנה או יציאה"}


def _load_evidence():
    try:
        with open(os.path.join(EN.HERE, "research", "scores_v3.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


EVID = _load_evidence()


def evidence(name, val_name=None):
    """Out-of-sample verdict for a score from research/scores_v3.json. Predictive only if the monthly IC is
    positive with t >= 2 in BOTH validation and test, and the 6.5+ bucket beats a random day with a CI above 0."""
    r = (EVID or {}).get("scores", {})
    t = (r.get(name) or {}).get("test")
    v = (r.get(val_name or name) or {}).get("val")
    if not t:
        return None
    ic, top = t.get("ic") or {}, t.get("top") or {}
    vic = (v or {}).get("ic") or {}
    ci = top.get("edge_ci")
    pred = bool(ic.get("ic") and ic["ic"] > 0 and (ic.get("t") or 0) >= 2 and vic.get("ic") and vic["ic"] > 0 and (vic.get("t") or 0) >= 2
                and ci and ci[0] is not None and ci[0] > 0)
    return {"predictive": pred, "ic": ic.get("ic"), "t": ic.get("t"), "val_ic": vic.get("ic"), "val_t": vic.get("t"),
            "top_edge": top.get("edge"), "top_ci": ci, "top_n": top.get("n"), "stocks": EVID.get("stocks"),
            "test_from": EVID.get("test_from"), "generated": EVID.get("generated")}


def cl(v, lo=0.0, hi=10.0):
    return max(lo, min(hi, v))


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _combine(fams, W, min_cov=0.6):
    tot = float(sum(W.values()))
    have = {k: v for k, v in fams.items() if k in W and v[0] is not None}
    cov = sum(W[k] for k in have) / tot if tot else 0.0
    if cov < min_cov or not have:
        return None, cov
    m = sum(W[k] * have[k][0] for k in have) / sum(W[k] for k in have)
    return round(5 + (m - 5) * cov, 1), cov


def _weights(name, default):
    w = (CFG.get(name) or {}).get("weights")
    return dict(w) if w else dict(default)


# ================================================================ technical score

def t_trend(x):
    p = x.get("price")
    if not p:
        return None, "אין מחיר"
    pts, n = 0, 0
    for k in ("sma20", "sma50", "sma150", "sma200"):
        if x.get(k):
            n += 1
            pts += p > x[k]
    if x.get("sma50") and x.get("sma200"):
        n += 1
        pts += x["sma50"] > x["sma200"]
    if x.get("slope50") is not None:
        n += 1
        pts += x["slope50"] > 0
    if n < 3:
        return None, "אין מספיק היסטוריה לממוצעים"
    v = pts / n * 10
    adx = x.get("adx")
    if adx is not None:  # ADX is strength, not direction: it only scales how far the picture moves from 5
        v = 5 + (v - 5) * min(1.0, adx / 25.0)
    why = "%d מתוך %d תנאי ממוצעים (SMA) חיוביים" % (pts, n) + (", ADX %.0f" % adx if adx is not None else "")
    if n < 6:
        why += " (חסר ממוצע 200: היסטוריה קצרה)"
    return cl(v), why


def t_structure(x):
    st = x.get("state")
    if st is None:
        return None, "אין מספיק שיאים ושפלים מאושרים"
    v = {2: 8.0, 0: 5.0, -2: 2.0}[st]
    why = {2: "שיאים ושפלים עולים", 0: "מבנה מעורב", -2: "שיאים ושפלים יורדים"}[st]
    if x.get("bos_up"):
        v += 1
        why += ", פריצת שיא מקומי"
    if x.get("bos_dn"):
        v -= 1
        why += ", שבירת שפל מקומי"
    if x.get("failed_bo"):
        v -= 1.5
        why += ", פריצה שנכשלה"
    return cl(v), why


def t_momentum(x):
    parts, why = [], []
    r = x.get("rsi")
    if r is not None:
        parts.append(cl(5 + (r - 50) / 20 * 3.5) if r <= 75 else cl(8.5 - (r - 75) / 10 * 2))
        why.append("RSI %.0f" % r)
    c = x.get("cci")
    if c is not None:
        parts.append(cl(5 + c / 100 * 2.5))
        why.append("CCI %.0f" % c)
    m = x.get("macd_h")
    if m is not None:
        parts.append(cl(5 + m * 8))
        why.append("MACD %s" % ("מעל הסיגנל" if m > 0 else "מתחת לסיגנל"))
    if not parts:
        return None, "אין נתוני מומנטום"
    return _median(parts), ", ".join(why) + " (נספרים כמקור אחד: הם תלויים זה בזה)"


def t_volume(x):
    parts, why = [], []
    ud = x.get("updown")
    if ud is not None:
        parts.append(cl(5 + (ud - 1) * 3))
        why.append("מחזור בימי עלייה/ירידה %.1f" % ud)
    o = x.get("obv20")
    if o is not None:
        parts.append({1: 6.5, 0: 5.0, -1: 3.5}[o])
        why.append("OBV %s" % {1: "עולה", 0: "שטוח", -1: "יורד"}[o])
    rv = x.get("rvol")
    if rv is not None and x.get("clv") is not None and rv >= 1.5:
        up = x["clv"] >= 0.6
        parts.append(7.0 if up else 3.0)
        why.append("מחזור חריג x%.1f ב%s" % (rv, "סגירה חזקה" if up else "סגירה חלשה"))
    if not parts:
        return None, "אין נתוני מחזור"
    return sum(parts) / len(parts), ", ".join(why)


def t_rs(x):
    parts, why = [], []
    r = x.get("rs_spy_60")
    if r is not None:
        parts.append(cl(5 + r / 5))
        why.append("מול SPY %+.0f%% ב-60 יום" % r)
    s = x.get("rs_sector_60")
    if s is not None:
        parts.append(cl(5 + s / 5))
        why.append("מול הסקטור %+.0f%%" % s)
    rg = x.get("regime")
    if rg:
        parts.append({"on": 7.0, "neutral": 5.0, "off": 3.0}[rg])
        why.append("שוק %s" % {"on": "Risk-On", "neutral": "ניטרלי", "off": "Risk-Off"}[rg])
    st = x.get("sector_trend")
    if st is not None:
        parts.append({1: 7.0, 0: 5.0, -1: 3.0}[st])
        why.append("סקטור %s" % {1: "במגמה עולה", 0: "מעורב", -1: "במגמה יורדת"}[st])
    if not parts:
        return None, "אין נתוני שוק וסקטור"
    return sum(parts) / len(parts), ", ".join(why)


def t_vol(x):
    p = x.get("atr_pctile")
    if p is None:
        return None, "אין מספיק היסטוריה לתנודתיות"
    v = cl(8 - p / 100 * 6)
    why = "תנודתיות באחוזון %.0f של השנה" % p
    g, ap = x.get("gap"), x.get("atr_pct")
    if g is not None and ap and abs(g) > 2 * ap:
        v = min(v, 3.0)
        why += ", גאפ חריג %+.1f%%" % g
    return v, why


def technical_score(x, cfg=None):
    W = _weights("technical_score", TECH_W)
    fams = {"trend": t_trend(x), "structure": t_structure(x), "momentum": t_momentum(x), "volume": t_volume(x),
            "rs": t_rs(x), "vol": t_vol(x)}
    val, cov = _combine(fams, W)
    comps = [{"key": k, "name": TECH_HE[k], "weight": W[k], "value": None if fams[k][0] is None else round(fams[k][0], 1),
              "why": fams[k][1]} for k in W]
    label = "N/A" if val is None else next(l for t, l in [[6.5, "חיובי"], [5.5, "חיובי קל"], [4.5, "ניטרלי"], [3.5, "שלילי קל"], [0, "שלילי"]] if val >= t)
    return {"value": val, "label": label, "coverage": round(cov, 2), "components": comps, "version": "v3",
            "missing": [TECH_HE[k] for k in W if fams[k][0] is None], "evidence": evidence("tech_v3")}


# ================================================================ trade plan

def plan_v3(x, lv=None, live=None, cfg=None):
    """Entry, structural stop, up to two targets. Levels from CLOSED bars; the live price only decides
    whether the entry is still where it was (in the zone) and is the entry reference during the session.
    The stop sits 0.1 ATR under the nearest confirmed support that is 0.5-3 ATR below the entry.
    T1 = nearest resistance between 1R and 3R (shown, not the tested exit); T2 = 3R (the tested exit, 10 sessions)."""
    a, close = x.get("atr"), x.get("price")
    if not a or not close:
        return None
    lv = lv or {}
    lp = (live or {}).get("price")
    ref = round(lp if (live or {}).get("session_open") and lp else close, 2)
    zone = [round(close - 0.5 * a, 2), round(close + 0.5 * a, 2)]
    dev = (ref - close) / a
    sup = [s["price"] for s in lv.get("support", [])]
    if x.get("swing_low") and all(abs(x["swing_low"] - s) > 0.25 * a for s in sup):
        sup.append(x["swing_low"])
    sup = sorted([s for s in sup if s < ref], reverse=True)
    stop, basis = None, None
    for s in sup:
        st = math.floor((s - 0.1 * a) * 100) / 100
        d = (ref - st) / a
        if 0.5 <= d <= 3.0:
            stop, basis = st, s
            break
    p = {"entry": ref, "close": round(close, 2), "zone": zone, "dev_atr": round(dev, 2), "in_zone": zone[0] <= ref <= zone[1],
         "atr": round(a, 2), "stop": stop, "stop_basis": round(basis, 2) if basis else None, "horizon": 10,
         "t1": None, "t2": None, "rr": None, "res_blocks": False, "notes": []}
    if stop is None:
        p["notes"].append("אין תמיכה מבנית במרחק 0.5-3 ATR מתחת למחיר: אין סטופ מבני, ולכן לא מוצג יחס סיכוי/סיכון")
        return p
    risk = ref - stop
    t2 = round(ref + 3 * risk, 2)
    # targets and blocks use only levels that matter: touched twice or more, or support turned resistance.
    # A single minor swing high is drawn on the chart but does not cap the trade.
    res = sorted(r["price"] for r in lv.get("resistance", []) if r["price"] > ref and (r["touches"] >= 2 or r.get("flip")))
    t1 = next((r for r in res if ref + risk <= r < t2), None)
    blocks = bool(res and res[0] < ref + risk)
    target = res[0] if blocks else (t1 or t2)
    p.update(stop_pct=round(risk / ref * 100, 2), stop_atr=round(risk / a, 2), t1=round(t1, 2) if t1 else None, t2=t2,
             t1_pct=round((t1 / ref - 1) * 100, 2) if t1 else None, t2_pct=round((t2 / ref - 1) * 100, 2),
             rr=round((target - ref) / risk, 2), res_blocks=blocks, res_near=round(res[0], 2) if res else None,
             invalidation="סגירה מתחת ל-%.2f (שבירת התמיכה %.2f), או 10 ימי מסחר בלי יעד" % (stop, basis))
    if blocks:
        p["notes"].append("התנגדות ב-%.2f, פחות מ-1R מעל הכניסה" % res[0])
    if not res:
        p["notes"].append("אין התנגדות משמעותית (שתי נגיעות ומעלה) מעל המחיר: היעד הוא 3R שנבדק, לא רמה במחיר")
    if risk < 0.75 * a:
        p["notes"].append("הסטופ במרחק %.1f ATR בלבד: תנודה רגילה יכולה להפעיל אותו" % (risk / a))
    return p


# ================================================================ entry score

def e_location(x, p, lv):
    if not p:
        return None, "אין נתונים"
    a, ref = p["atr"], p["entry"]
    sup = [s["price"] for s in (lv or {}).get("support", []) if s["price"] < ref] or ([x["swing_low"]] if x.get("swing_low") and x["swing_low"] < ref else [])
    if sup:
        d = (ref - max(sup)) / a
        v = 6.0 if d < 0.3 else 8.0 if d <= 1.5 else 5.5 if d <= 2.5 else 3.5 if d <= 4 else 2.0
        why = "%.1f ATR מעל התמיכה %.2f" % (d, max(sup))
    else:
        v, why = 4.0, "אין תמיכה מאושרת מתחת למחיר"
    e = x.get("ext_atr")
    if e is not None and e > 3:
        v = min(v, 3.0)
        why += ", מתוח %.1f ATR מעל ממוצע 20" % e
    ma = x.get("move_atr")
    if ma is not None and ma > 2:
        v = min(v, 3.5)
        why += ", זינוק של %.1f ATR בנר האחרון" % ma
    if p["dev_atr"] > 1.0:
        v = min(v, 2.5)
        why += ", המחיר כבר התרחק %.1f ATR מהסגירה" % p["dev_atr"]
    return v, why


def e_rr(p):
    if not p:
        return None, "אין נתונים"
    if p["stop"] is None:
        return None, "אין סטופ מבני: לא מחושב"
    if p["res_blocks"]:
        return 1.0, "התנגדות %.2f לפני 1R" % p["res_near"]
    r = p["rr"]
    v = 9.0 if r >= 3 else 7.5 if r >= 2 else 5.0 if r >= 1.5 else 3.0 if r >= 1 else 1.0
    return v, "%.1f:1 עד %s" % (r, "התנגדות %.2f" % p["t1"] if p.get("t1") else "יעד 3R")


def e_confirm(x, p, lv):
    st = x.get("state")
    if st is None:
        return None, "אין מבנה מאושר"
    if x.get("failed_bo"):
        return 2.0, "פריצה שנכשלה"
    if st == -2:
        return (1.5, "שיאים ושפלים יורדים ושבירת שפל") if x.get("bos_dn") else (2.5, "שיאים ושפלים יורדים")
    a = (p or {}).get("atr") or x.get("atr")
    ref = (p or {}).get("entry") or x.get("price")
    flip = [s for s in (lv or {}).get("support", []) if s.get("flip") and a and 0 <= ref - s["price"] <= 0.75 * a]
    if flip and st >= 0:
        return 8.0, "ריטסט: חזרה לרמה %.2f שהייתה התנגדות והפכה לתמיכה" % flip[0]["price"]
    if x.get("bos_up"):
        return 7.0, "פריצת שיא מקומי, עוד בלי ריטסט"
    return (6.5, "שיאים ושפלים עולים") if st == 2 else (4.5, "מבנה מעורב, אין אישור")


def e_volmom(x):
    parts, why = [], []
    ud = x.get("updown")
    if ud is not None:
        parts.append(cl(5 + (ud - 1) * 3))
    m = x.get("macd_h")
    if m is not None:
        parts.append(6.5 if m > 0 else 3.5)
        why.append("MACD %s" % ("חיובי" if m > 0 else "שלילי"))
    rv, c = x.get("rvol"), x.get("clv")
    if rv is not None and c is not None:
        parts.append(7.0 if rv >= 1.3 and c >= 0.6 else 3.5 if rv >= 1.3 and c < 0.4 else 5.0)
        why.append("מחזור x%.1f" % rv)
    if not parts:
        return None, "אין נתונים"
    return sum(parts) / len(parts), ", ".join(why)


def e_history(hist):
    if not hist or hist.get("value") is None:
        return None, "אין טבלת מחקר מתאימה לנכס הזה"
    return hist["value"], "יתרון %+.2fR מול יום אקראי, %s" % (hist["edge"], "מובהק" if hist.get("significant") else "לא מובהק")


def e_risk(x, ev):
    parts, why = [], []
    ev = ev or {}
    if ev.get("earnings_known"):
        d = ev.get("earnings_days")
        inside = d is not None and 0 <= d <= 14
        parts.append(1.5 if inside else 7.0)
        why.append("דוח בעוד %d ימים" % d if inside else "אין דוח בתקופת העסקה")
    p = x.get("atr_pctile")
    if p is not None:
        parts.append(3.0 if p > 90 else 6.5 if p < 70 else 5.0)
        why.append("תנודתיות באחוזון %.0f" % p)
    g, ap = x.get("gap"), x.get("atr_pct")
    if g is not None and ap and abs(g) > 2 * ap:
        parts.append(3.0)
        why.append("גאפ %+.1f%%" % g)
    if not parts:
        return None, "אין נתונים"
    return min(parts), ", ".join(why)


def entry_score(x, plan, hist=None, ev=None, lv=None, cfg=None):
    W = _weights("entry_score", ENTRY_W)
    fams = {"location": e_location(x, plan, lv), "rr": e_rr(plan), "confirm": e_confirm(x, plan, lv),
            "volmom": e_volmom(x), "history": e_history(hist), "risk": e_risk(x, ev)}
    val, cov = _combine(fams, W)
    caps = []
    if val is not None and plan and plan.get("stop") is None and val > 4.5:
        val, caps = 4.5, caps + ["אין סטופ מבני"]
    if val is not None and plan and (plan["dev_atr"] > 1.0 or (x.get("move_atr") or 0) > 2) and val > 4.0:
        val, caps = 4.0, caps + ["המחיר התרחק מנקודת הכניסה"]
    comps = [{"key": k, "name": ENTRY_HE[k], "weight": W[k], "value": None if fams[k][0] is None else round(fams[k][0], 1),
              "why": fams[k][1]} for k in W]
    label = entry_label(val)
    return {"value": val, "label": label, "coverage": round(cov, 2), "components": comps, "caps": caps, "version": "v3",
            "missing": [ENTRY_HE[k] for k in W if fams[k][0] is None], "evidence": evidence("entry_v3_full", "entry_v3_no_history")}


def entry_label(val):
    return "N/A" if val is None else ("תנאי כניסה טובים" if val >= 7 else "תנאי כניסה סבירים" if val >= 5.5 else
                                      "תנאי כניסה חלשים" if val >= 4 else "לא מתאים לכניסה")


# ================================================================ decision

def falling_knife(x):
    p = x.get("price")
    return bool(x.get("state") == -2 and p and x.get("sma50") and p < x["sma50"] and (not x.get("sma200") or p < x["sma200"]))


def pullback_in_uptrend(x):
    p = x.get("price")
    return bool(p and x.get("sma200") and x.get("sma50") and p > x["sma200"] and x["sma50"] > x["sma200"]
                and (x.get("rsi") or 50) < 45 and x.get("state") != -2)


def hist_ok(hist):
    """A statistically significant edge that also kept its sign out of sample."""
    if not hist or hist.get("value") is None:
        return False
    lo = (hist.get("edge_ci") or [None])[0]
    return bool(lo is not None and lo > 0 and hist.get("stable") and (hist.get("test_edge") or 0) > 0)


def decide_v3(x, tech, entry, plan, hist=None, ev=None, blockers=None, asset=None, held=False, data_level="ok", cfg=None):
    c = (CFG.get("decision_v3") or {})
    min_rr, min_entry = c.get("min_rr", 2.0), c.get("min_entry", 6.5)
    ev = ev or {}
    blockers = [b for b in (blockers or []) if b]
    ev_ = entry.get("value") if entry else None
    tv = tech.get("value") if tech else None
    knife = falling_knife(x)
    items = []

    def item(key, label, ok, detail=""):
        items.append({"key": key, "label": label, "ok": ok, "detail": detail})
    covered = (asset or {}).get("type") not in EN.NOT_COVERED
    item("hist", "יתרון היסטורי מובהק שנשמר מחוץ למדגם", hist_ok(hist) if covered else None,
         ("יתרון %+.2fR, רווח סמך [%+.2f, %+.2f]" % (hist["edge"], hist["edge_ci"][0], hist["edge_ci"][1])
          if hist and hist.get("value") is not None and hist.get("edge_ci") and hist["edge_ci"][0] is not None else
          "הנכס מחוץ ליקום המחקר" if not covered else "אין נתון"))
    item("stop", "סטופ מבני סביר", None if not plan else plan.get("stop") is not None,
         "%.2f, %.1f%% מתחת (%.1f ATR)" % (plan["stop"], plan["stop_pct"], plan["stop_atr"]) if plan and plan.get("stop") else "אין תמיכה מתאימה")
    rr_ok = None if not plan or plan.get("stop") is None else (not plan["res_blocks"] and plan["rr"] >= min_rr)
    item("rr", "סיכוי/סיכון לפחות %.0f:1 ליעד ריאלי" % min_rr, rr_ok,
         "%.1f:1" % plan["rr"] if plan and plan.get("rr") is not None else "לא מחושב בלי סטופ")
    item("zone", "המחיר עדיין באזור הכניסה", None if not plan else plan["in_zone"] and (x.get("move_atr") or 0) <= 2,
         "%.2f, אזור %.2f-%.2f" % (plan["entry"], plan["zone"][0], plan["zone"][1]) if plan else "")
    item("entry", "ציון כניסה לפחות %.1f" % min_entry, None if ev_ is None else ev_ >= min_entry, "%.1f" % ev_ if ev_ is not None else "N/A")
    item("knife", "לא 'סכין נופלת' (שיאים ושפלים יורדים מתחת לממוצעים)", None if x.get("state") is None else not knife, "")
    if ev.get("earnings_known") or not covered:
        d = ev.get("earnings_days")
        inside = d is not None and 0 <= d <= 14
        item("events", "אין דוח או אירוע מהותי בתוך העסקה", not inside and not blockers,
             "; ".join(blockers) if blockers else ("דוח בעוד %d ימים" % d if inside else "לא צפוי בתקופה"))
    else:
        item("events", "אין דוח או אירוע מהותי בתוך העסקה", False if blockers else None, "; ".join(blockers) if blockers else "מועד הדוח לא ידוע")
    item("data", "הנתונים מאומתים", data_level != "bad", {"ok": "מאומת", "partial": "חלקי", "bad": "לא מאומת"}.get(data_level, ""))
    byk = {i["key"]: i for i in items}

    hard = []
    if knife:
        hard.append("שיאים ושפלים יורדים והמחיר מתחת לממוצעים: אין סימן לסיום הירידה")
    if plan and (plan["dev_atr"] > 1.0 or (x.get("move_atr") or 0) > 2):
        hard.append("המחיר כבר התרחק מנקודת הכניסה: הסיכון גדל והפוטנציאל קטן")
    if ev_ is not None and ev_ < 4:
        hard.append("ציון כניסה נמוך (%.1f)" % ev_)
    if hist and hist.get("edge_ci") and hist["edge_ci"][1] is not None and hist["edge_ci"][1] < 0:
        hard.append("במחקר, המצב הזה היה גרוע מכניסה אקראית באופן מובהק")
    if byk["events"]["ok"] is False:
        hard.append(byk["events"]["detail"] or "אירוע בתוך העסקה")

    if held and (knife or (tv is not None and tv <= 3.5) or (x.get("bos_dn") and x.get("state") == -2) or
                 any("מהותי" in b for b in blockers)):
        code = "reduce"
    elif data_level == "bad":
        code = "wait"
    elif all(i["ok"] is True for i in items):
        code = "go"
    elif hard:
        code = "no"
    else:
        code = "wait"

    pros = [i["label"] + (": " + i["detail"] if i["detail"] else "") for i in items if i["ok"] is True and i["key"] in ("hist", "stop", "rr", "zone", "entry")]
    if pullback_in_uptrend(x):
        pros.append("תיקון בתוך מגמה עולה (מעל ממוצע 200, ממוצע 50 מעל 200)")
    if tv is not None and tv >= 6.5:
        pros.append("ציון טכני %.1f" % tv)
    missing = [i["label"] + (" (" + i["detail"] + ")" if i["detail"] else "") for i in items if i["ok"] is not True]
    change = []
    if code == "reduce":
        change.append("בדוק את גודל הפוזיציה מול הסטופ שלך. המצב ישתפר בשפל עולה מאושר או חזרה מעל ממוצע 50")
    if knife:
        change.append("שפל עולה מאושר או סגירה מעל ממוצע 50")
    if byk["zone"]["ok"] is False and plan:
        change.append("חזרה לאזור %.2f-%.2f" % (plan["zone"][0], plan["zone"][1]))
    if plan and plan.get("res_blocks"):
        change.append("פריצה של %.2f ואז ריטסט מלמעלה" % plan["res_near"])
    if byk["stop"]["ok"] is False:
        change.append("היווצרות שפל מאושר שיאפשר סטופ מבני")
    if byk["hist"]["ok"] is not True and covered:
        change.append("כניסה תוצג רק כשהמחקר יראה יתרון מובהק ויציב למצב הזה. כרגע אין")
    if code == "go" and plan:
        change.append("ביטול: " + plan["invalidation"])
    dec = {"code": code, "verdict": VERDICTS[code], "items": items, "pros": pros[:3], "hard": hard[:3],
           "missing": missing[:3], "change": change[:3], "knife": knife, "version": "v3"}
    if code == "no" and entry and entry.get("value") is not None and entry["value"] > 4.9:
        # the score and the decision must not disagree: a hard "no" caps the entry score below 5
        entry["value"] = 4.9
        entry["caps"] = entry.get("caps", []) + [hard[0]]
        entry["label"] = entry_label(4.9)
    bad = consistency(dec, entry, plan)
    if bad:
        dec.update(code="wait", verdict=VERDICTS["wait"], contradictions=bad)
    return dec


def consistency(dec, entry, plan):
    """Contradictions between the decision, the scores and the data. Empty list = consistent."""
    out = []
    ev_ = (entry or {}).get("value")
    if dec["code"] == "go":
        if ev_ is None or ev_ < (CFG.get("decision_v3") or {}).get("min_entry", 6.5):
            out.append("כניסה עם ציון כניסה נמוך")
        if not plan or plan.get("stop") is None or plan.get("rr") is None:
            out.append("כניסה בלי סטופ או יחס סיכוי/סיכון")
        if plan and not plan.get("in_zone"):
            out.append("כניסה כשהמחיר מחוץ לאזור")
        if dec.get("knife"):
            out.append("כניסה בסכין נופלת")
    if dec["code"] == "no" and ev_ is not None and ev_ > 4.9:
        out.append("'לא להיכנס' עם ציון כניסה גבוה")
    return out


def reliability(data, tech, entry, hist, asset=None):
    """Analysis reliability: verified / partial / not verified. A display flag, not a third score."""
    lvl = (data or {}).get("level", "partial")
    why = [t for k, t in (data or {}).get("issues", [])]
    if tech is None or tech.get("value") is None or entry is None or entry.get("value") is None:
        lvl = "bad"
        why.append("אין מספיק נתונים לאחד הציונים")
    else:
        if tech.get("coverage", 1) < 1 or entry.get("coverage", 1) < 1:
            lvl = "partial" if lvl == "ok" else lvl
            why.append("חסר: " + ", ".join((tech.get("missing") or []) + (entry.get("missing") or [])))
    if (asset or {}).get("type") in EN.NOT_COVERED:
        lvl = "partial" if lvl == "ok" else lvl
        why.append("הנכס לא נכלל במחקר ההיסטורי")
    return {"level": lvl, "label": {"ok": "מאומת", "partial": "חלקי", "bad": "לא מאומת"}[lvl], "why": why[:4]}
