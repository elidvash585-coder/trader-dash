"""William J. O'Neil's method (How to Make Money in Stocks; CAN SLIM) for Trader Dash. Standard library only.

The rules are coded as the book states them, on DAILY bars (O'Neil reads weekly charts; weeks = 5 sessions):

Bases (the most recent one, ending at bar e; nothing after e is used):
- cup with handle: prior uptrend >= 30 %; cup at least 7 weeks; depth 12-33 % (up to 40 % allowed when the
  market corrected); U-shaped bottom, not a V; right side back near the old high; handle at least 1 week,
  in the upper half of the base, depth <= 12 %, ideally on drying volume. Pivot = handle high + $0.10.
- cup without handle: same cup; pivot = left-side high + $0.10.
- double bottom: W shape, at least 7 weeks, the second low undercuts the first (by up to 5 %);
  pivot = middle peak + $0.10.
- flat base: at least 5 weeks, depth <= 15 %, after a prior advance of >= 20 %; pivot = base high + $0.10.
- high tight flag: price up 100 %+ in at most 8 weeks, then a 3-5 week flag that gives back <= 25 %;
  pivot = flag high + $0.10.
Buy: a close above the pivot on volume at least 40 % above the 50-day average. Buy zone: pivot to +5 %
("extended" above that: the book says do not chase). Late-stage bases (3rd, 4th) fail more often.
Sell: cut every loss at 7-8 % below the buy price; take most profits at 20-25 %, EXCEPT a stock that rises
20 % within 3 weeks of the breakout is held at least 8 weeks; other sell signals: a failed breakout
(back under the pivot), a break of the 50-day line on heavy volume, a climax run.
CAN SLIM: C current quarterly EPS +25 % or more (sales +25 % or accelerating); A annual EPS growth 25 %+;
N new high from a proper base; S supply and demand (volume, accumulation); L leader, RS Rating 80+;
I institutional sponsorship; M market direction (confirmed uptrend: no new buys in a correction).

What cannot be measured from free data is marked as such, never guessed: new products or management (N),
the trend in the number of funds owning the stock (I), share buybacks (S).
"""
import math

import features as FX

PIVOT_ADD = 0.10
BUY_ZONE = 0.05
VOL_BREAKOUT = 1.40
STOP = 0.08
PROFIT = (0.20, 0.25)
TYPE_HE = {"cup_handle": "ספל עם ידית", "cup": "ספל בלי ידית", "double_bottom": "תחתית כפולה (W)",
           "flat": "בסיס שטוח", "htf": "דגל גבוה וצר (High Tight Flag)"}


def _argmax(a, s, e):
    i, m = s, a[s]
    for k in range(s + 1, e + 1):
        if a[k] > m:
            i, m = k, a[k]
    return i


def _argmin(a, s, e):
    i, m = s, a[s]
    for k in range(s + 1, e + 1):
        if a[k] < m:
            i, m = k, a[k]
    return i


def _avg(a, s, e):
    xs = a[max(0, s):e + 1]
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- bases

def detect_base(h, l, c, v, e, market_corrected=False):
    """The base that ends at bar e (inclusive), or None. Only bars 0..e are read."""
    if e < 60:
        return None
    # high tight flag: a 3-5 week flag after a 100 %+ run in <= 8 weeks
    if e >= 70:
        f = _argmax(h, e - 25, e - 15)
        if h[f] >= max(h[f:e + 1]):
            lo = min(l[f - 40:f + 1])
            depth = 1 - min(l[f:e + 1]) / h[f]
            if lo > 0 and h[f] / lo - 1 >= 1.0 and depth <= 0.25:
                return {"type": "htf", "start": f, "end": e, "pivot": round(h[f] + PIVOT_ADD, 2), "high": h[f],
                        "low": min(l[f:e + 1]), "depth": round(depth * 100, 1), "weeks": round((e - f) / 5, 1),
                        "prior_gain": round((h[f] / lo - 1) * 100, 0)}
    s = max(0, e - 324)
    p = _argmax(h, s, e)
    blen = e - p
    if blen < 24:
        return None                     # no base of 5 weeks or more: still running, or just broke out
    b = _argmin(l, p, e)
    hp, lb = h[p], l[b]
    depth = 1 - lb / hp
    base_lo = min(l[max(0, p - 130):p + 1]) if p >= 20 else None
    prior = hp / base_lo - 1 if base_lo else 0.0
    max_depth = 0.40 if market_corrected else 0.33
    common = {"start": p, "end": e, "high": hp, "low": lb, "depth": round(depth * 100, 1),
              "weeks": round(blen / 5, 1), "prior_gain": round(prior * 100, 0)}
    avg50 = _avg(v, p - 50, p - 1) or _avg(v, e - 50, e)
    # cup (with or without handle)
    if blen >= 34 and 0.12 <= depth <= max_depth and prior >= 0.30 and b - p >= 10:
        zone = lb + (hp - lb) * 0.15          # "time at the bottom": a U spends weeks there, a V does not
        bottom_days = sum(1 for k in range(p, e + 1) if c[k] <= zone)
        r = _argmax(h, b, e)
        upper = lb + (hp - lb) * 0.5
        if bottom_days >= 10 and h[r] >= max(upper, hp * 0.85):
            hl = e - r
            if 5 <= hl <= 30:
                hlow = min(l[r:e + 1])
                hdepth = 1 - hlow / h[r]
                if hdepth <= 0.12 and hlow >= upper:
                    dry = _avg(v, r, e) < avg50 if avg50 else None
                    return dict(common, type="cup_handle", pivot=round(h[r] + PIVOT_ADD, 2),
                                handle={"start": r, "high": h[r], "low": hlow, "depth": round(hdepth * 100, 1),
                                        "weeks": round(hl / 5, 1), "volume_dry": dry}, u_shape=True)
            elif hl < 5 and h[r] >= hp * 0.95:
                return dict(common, type="cup", pivot=round(hp + PIVOT_ADD, 2), u_shape=True)
    # double bottom
    if blen >= 34 and 0.12 <= depth <= max_depth and prior >= 0.30:
        best = None
        for m in range(p + 8, e - 4):
            b1 = _argmin(l, p, m)
            b2 = _argmin(l, m, e)
            if m - b1 < 5 or b2 - m < 5:
                continue
            if not (l[b1] * 0.95 <= l[b2] <= l[b1]):
                continue
            if h[m] < max(h[b1:b2 + 1]) or h[m] < l[b1] + 0.4 * (hp - l[b1]):
                continue
            if best is None or h[m] > h[best[0]]:
                best = (m, b1, b2)
        if best:
            m, b1, b2 = best
            return dict(common, type="double_bottom", pivot=round(h[m] + PIVOT_ADD, 2), middle=m, lows=[b1, b2])
    # flat base
    if depth <= 0.15 and prior >= 0.20:
        return dict(common, type="flat", pivot=round(hp + PIVOT_ADD, 2))
    return None


def base_stage(c, h, e, sma200=None):
    """Approximate base count since the stock last crossed above its 200-day line: a base is counted each
    time price falls >= 12 % from a high and later makes a new high. 1-2 = early, 3+ = late stage."""
    s200 = sma200 or FX.sma(c, 200)
    s0 = 0
    for k in range(e, 200, -1):
        if s200[k] and c[k] < s200[k]:
            s0 = k
            break
    hi, in_base, n = h[s0], False, 0
    for k in range(s0 + 1, e + 1):
        if not in_base and c[k] <= hi * 0.88:
            in_base = True
        if h[k] > hi:
            if in_base:
                n += 1
                in_base = False
            hi = h[k]
    return n + 1


def breakout_at(h, l, c, v, t, base):
    """Is bar t a breakout from `base` (which ended at t-1)? Close above the pivot, volume >= 1.4 x the
    50-day average, still within the 5 % buy zone."""
    if not base:
        return None
    avg = _avg(v, t - 50, t - 1)
    vr = v[t] / avg if avg else None
    ok = c[t] > base["pivot"] and vr is not None and vr >= VOL_BREAKOUT and c[t] <= base["pivot"] * (1 + BUY_ZONE)
    return {"ok": ok, "vol_ratio": round(vr, 2) if vr else None, "close": c[t]}


# ---------------------------------------------------------------- RS, accumulation, market direction

def rs_raw(c, i):
    """IBD-style 12-month relative strength, weighting the latest quarter double."""
    if i < 252 or not c[i - 252]:
        return None
    r = lambda n: c[i] / c[i - n] - 1
    return 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)


def rs_line(c, bench, i, look=252):
    """Stock / SPY ratio: is it at (or within 2 % of) its 52-week high? New highs in the RS line before or
    at the breakout are a sign of a leader."""
    if not bench or i < 60:
        return None
    ratio = [c[k] / bench[k] if bench[k] else None for k in range(max(0, i - look), i + 1)]
    ratio = [x for x in ratio if x]
    if not ratio:
        return None
    return {"at_high": ratio[-1] >= max(ratio) * 0.98, "vs_high": round((ratio[-1] / max(ratio) - 1) * 100, 1)}


def acc_dis(c, v, i, n=65):
    """Accumulation/distribution over 13 weeks: up-day volume / down-day volume -> A..E."""
    up = sum(v[k] for k in range(max(1, i - n + 1), i + 1) if c[k] > c[k - 1])
    dn = sum(v[k] for k in range(max(1, i - n + 1), i + 1) if c[k] < c[k - 1])
    if not dn:
        return None
    r = up / dn
    return {"ratio": round(r, 2), "grade": "A" if r >= 1.5 else "B" if r >= 1.2 else "C" if r >= 0.9 else "D" if r >= 0.7 else "E"}


def distribution_days(c, v, i, n=25, since=None):
    """Index down >= 0.2 % on higher volume than the day before, in the last n sessions; a day stops counting
    once the index closes 5 % above that day's close (IBD convention). `since`: a follow-through day starts a
    new uptrend, so only days after it count."""
    out = []
    lo = max(1, i - n + 1) if since is None else max(1, i - n + 1, since + 1)
    for k in range(lo, i + 1):
        if c[k] < c[k - 1] * 0.998 and v[k] > v[k - 1]:
            if max(c[k:i + 1]) < c[k] * 1.05:
                out.append(k)
    return out


def market_direction(c, v, c2=None, v2=None):
    """O'Neil's M, as a state machine over the index history (SPY; QQQ adds distribution days).
    states: uptrend (confirmed), pressure (uptrend with 4+ distribution days), correction, rally (attempt).
    Correction: the index below its 50-day line AND either 8 % under its 50-day high or 6+ distribution days
    (distribution without price damage only puts the uptrend "under pressure", as IBD does).
    Follow-through day: day 4 or later of a rally attempt, SPY or QQQ up >= 1.25 % on higher volume than the
    day before -> confirmed uptrend (a follow-through on any one major index counts). Undercutting the rally low restarts the attempt. Returns a list of states."""
    s50 = FX.sma(c, 50)
    st, low, day1, ftd = "uptrend", None, None, None
    out = [None] * len(c)
    for i in range(1, len(c)):
        d = len(distribution_days(c, v, i, since=ftd))
        if c2 is not None and v2 is not None and len(c2) == len(c):
            d = max(d, len(distribution_days(c2, v2, i, since=ftd)))
        if st in ("uptrend", "pressure"):
            # after a follow-through day, a new correction is measured from highs made SINCE that day; the
            # follow-through fails outright if the index undercuts the rally low
            start = max(0, i - 50) if ftd is None else max(i - 50, ftd)
            hi50 = max(c[start:i + 1])
            if ftd is not None and low is not None and c[i] < low:
                st, low, day1 = "correction", c[i], None
                out[i] = {"state": st, "dist": d, "rally_day": None, "ftd": ftd}
                continue
            # a correction needs price damage: 8 % under the 50-day high below the 50-day line, or a heavy
            # distribution cluster (6+) WITH the index under its 50-day line. Distribution alone = "pressure".
            below50 = bool(s50[i] and c[i] < s50[i])
            if below50 and (c[i] <= hi50 * 0.92 or d >= 6):
                st, low, day1 = "correction", c[i], None
            else:
                st = "pressure" if d >= 4 else "uptrend"
        else:
            if c[i] < low:
                low, day1, st = c[i], None, "correction"
            elif day1 is None and c[i] > c[i - 1]:
                day1, st = i, "rally"
            elif day1 is not None:
                k = i - day1 + 1
                spy_ftd = c[i] / c[i - 1] - 1 >= 0.0125 and v[i] > v[i - 1]
                qqq_ftd = (c2 is not None and v2 is not None and len(c2) == len(c) and c2[i - 1] and
                           c2[i] / c2[i - 1] - 1 >= 0.0125 and v2[i] > v2[i - 1])
                if k >= 4 and (spy_ftd or qqq_ftd):   # the book: a follow-through on ANY major index
                    st, ftd = "uptrend", i
        out[i] = {"state": st, "dist": d, "rally_day": (i - day1 + 1) if st == "rally" and day1 else None, "ftd": ftd}
    return out


M_HE = {"uptrend": "מגמה עולה מאושרת", "pressure": "מגמה עולה תחת לחץ", "correction": "שוק בתיקון",
        "rally": "ניסיון התאוששות"}


# ---------------------------------------------------------------- CAN SLIM

def canslim(ev=None, fund=None, rs_pct=None, rsl=None, near_high=None, acc=None, mstate=None, breakout=None):
    """Seven letters, each True / False / None (None = cannot be measured from the data we have)."""
    f = fund or {}
    items = []

    def add(letter, name, ok, detail):
        items.append({"letter": letter, "label": "%s: %s" % (letter, name), "ok": ok, "detail": detail})
    ey = (ev or {}).get("eps_yoy")
    ry = (ev or {}).get("rev_yoy")
    rprev = [x for x in (ev or {}).get("rev_yoy_prev", []) if x is not None]
    accel = ry is not None and len(rprev) == 2 and ry > rprev[-1] > rprev[-2]
    if ey is not None:
        ok = ey >= 0.25 and (ry is not None and (ry >= 0.25 or accel))
        add("C", "רווח רבעוני אחרון +25% ומעלה", ok, "EPS %+.0f%%, מכירות %s%s (רבעון %s)" % (
            ey * 100, "%+.0f%%" % (ry * 100) if ry is not None else "—", ", מאיצות" if accel else "", ev.get("end", "")))
    else:
        add("C", "רווח רבעוני אחרון +25% ומעלה", None, "אין נתוני רווח למניה (SEC)")
    ty = (ev or {}).get("eps_ttm_yoy")
    hist = [x for x in (ev or {}).get("eps_ttm_hist", []) if x is not None]
    rising = len(hist) >= 3 and all(b > a for a, b in zip(hist, hist[1:]))
    roe = f.get("roe")
    if ty is not None:
        ok = ty >= 0.25 and (rising or len(hist) < 3) and (roe is None or roe >= 0.17)
        add("A", "צמיחה שנתית 25%+ ו-ROE 17%+", ok, "רווח 12 חודשים %+.0f%%%s%s" % (
            ty * 100, ", עולה 3 שנים" if rising else "", ", ROE %.0f%%" % (roe * 100) if roe is not None else ""))
    else:
        add("A", "צמיחה שנתית 25%+ ו-ROE 17%+", None, "אין מספיק היסטוריית רווח")
    if near_high is None:
        add("N", "שיא חדש מבסיס תקין", None, "אין מספיק היסטוריה")
    else:
        add("N", "שיא חדש מבסיס תקין", near_high <= 15 and bool(breakout),
            "%.0f%% מתחת לשיא השנתי%s. מוצר או הנהלה חדשים: לא ניתן לבדוק אוטומטית" % (near_high, "" if breakout else ", אין פריצה מבסיס"))
    if acc:
        add("S", "ביקוש: נפח ואיסוף", acc["grade"] in ("A", "B"), "איסוף/פיזור 13 שבועות: %s (יחס %.2f)%s" % (
            acc["grade"], acc["ratio"], ", נפח פריצה x%.1f" % breakout["vol_ratio"] if breakout and breakout.get("vol_ratio") else ""))
    else:
        add("S", "ביקוש: נפח ואיסוף", None, "אין נתוני נפח")
    if rs_pct is not None:
        add("L", "מניה מובילה, RS 80+", rs_pct >= 80, "RS %d%s" % (rs_pct, ", קו RS בשיא" if rsl and rsl["at_high"] else ""))
    else:
        add("L", "מניה מובילה, RS 80+", None, "אין מספיק היסטוריה או מניות להשוואה")
    inst = f.get("inst_pct")
    if inst is not None:
        add("I", "בעלות מוסדית", 0.20 <= inst <= 0.95, "מוסדיים מחזיקים %.0f%%. מגמת מספר הקרנות: לא זמינה" % (inst * 100))
    else:
        add("I", "בעלות מוסדית", None, "אין נתון")
    if mstate:
        add("M", "כיוון השוק", mstate["state"] == "uptrend", M_HE[mstate["state"]] + (
            ", %d ימי פיזור" % mstate["dist"] if mstate.get("dist") else "") + (
            ", יום %d לניסיון" % mstate["rally_day"] if mstate.get("rally_day") else ""))
    else:
        add("M", "כיוון השוק", None, "אין נתוני מדד")
    return items


# ---------------------------------------------------------------- the live view

def analyze(t, o, h, l, c, v, spy=None, ev=None, fund=None, rs_pct=None, mstate=None, closed=True):
    """Everything the site shows for one stock, from bars 0..len-1 (pass closed bars only)."""
    e = len(c) - 1
    if e < 260:
        return {"status": "short", "status_he": "פחות משנה של היסטוריה: אין ניתוח O'Neil"}
    s50, s200 = FX.sma(c, 50), FX.sma(c, 200)
    corrected = bool(mstate and mstate["state"] in ("correction", "rally"))
    # a breakout in the last 10 sessions?
    brk = None
    for k in range(e - 10, e + 1):
        b = detect_base(h, l, c, v, k - 1, corrected)
        bo = breakout_at(h, l, c, v, k, b)
        if bo and c[k] > b["pivot"] and bo["vol_ratio"] and bo["vol_ratio"] >= VOL_BREAKOUT:
            brk = {"base": b, "day": k, "t": t[k], "vol_ratio": bo["vol_ratio"]}
    base = brk["base"] if brk else detect_base(h, l, c, v, e, corrected)
    hi52 = max(h[e - 251:e + 1])
    near = (1 - c[e] / hi52) * 100
    acc = acc_dis(c, v, e)
    rsl = rs_line(c, spy, e) if spy else None
    out = {"price": c[e], "near_high": round(near, 1), "acc": acc, "rs_line": rsl, "rs_pct": rs_pct,
           "above200": bool(s200[e] and c[e] > s200[e]), "market": mstate}
    if base:
        piv = base["pivot"]
        stage = base_stage(c, h, base["start"], s200)
        dist = (c[e] / piv - 1) * 100
        out["base"] = dict(base, type_he=TYPE_HE[base["type"]], stage=stage,
                           t_start=t[base["start"]], t_end=t[base["end"]])
        out["pivot"], out["buy_zone"] = piv, [piv, round(piv * (1 + BUY_ZONE), 2)]
        out["stop"] = round(piv * (1 - STOP), 2)
        out["targets"] = [round(piv * (1 + x), 2) for x in PROFIT]
        out["dist_pivot"] = round(dist, 1)
        if brk:
            days = e - brk["day"]
            gain = (c[e] / piv - 1)
            out["breakout"] = {"days_ago": days, "t": brk["t"], "vol_ratio": brk["vol_ratio"]}
            if c[e] < piv * (1 - STOP):
                st, txt = "stopped", "ירד 8% מתחת לנקודת הקנייה: לפי הספר, למכור"
            elif c[e] < piv:
                st, txt = "failed", "חזר מתחת ל-pivot: פריצה שנכשלת. לפי הספר לא לקנות, ומי שמחזיק עוקב אחרי ה-7-8%"
            elif gain > BUY_ZONE:
                st, txt = "extended", "%.1f%% מעל ה-pivot: מחוץ לאזור הקנייה (עד 5%%). לפי הספר לא רודפים" % (gain * 100)
            else:
                st, txt = "buy_zone", "פריצה לפני %d ימים בנפח x%.1f, המחיר באזור הקנייה" % (days, brk["vol_ratio"])
        else:
            if dist >= -3:
                st, txt = "near_pivot", "בסיס מוכן, %.1f%% מתחת ל-pivot: לחכות לפריצה בנפח x1.4 ומעלה" % -dist
            else:
                st, txt = "forming", "בסיס בבנייה, %.1f%% מתחת ל-pivot" % -dist
        if stage >= 3:
            txt += ". בסיס בשלב %d: מאוחר, נכשל יותר לפי הספר" % stage
    else:
        if c[e] >= hi52 * 0.97:
            st, txt = "no_base_high", "קרוב לשיא בלי בסיס תקין: מתוח, לחכות לבסיס חדש"
        elif not out["above200"]:
            st, txt = "downtrend", "מתחת לממוצע 200 ובלי בסיס: לא מועמדת לפי הספר"
        else:
            st, txt = "no_base", "אין בסיס תקין כרגע"
    out["status"], out["status_he"] = st, txt
    # sell signals (for someone who already holds the stock)
    sells = []
    avg50v = _avg(v, e - 50, e - 1)
    if s50[e] and c[e] < s50[e] and c[e - 1] >= (s50[e - 1] or 0) and avg50v and v[e] >= 1.4 * avg50v:
        sells.append("שבירת ממוצע 50 בנפח גבוה")
    if brk:
        run = max(c[brk["day"]:e + 1]) / out["pivot"] - 1
        if run >= 0.20 and (e - brk["day"]) <= 15:
            sells.append("עלתה 20% תוך 3 שבועות מהפריצה: לפי כלל 8 השבועות, להחזיק לפחות 8 שבועות מהפריצה")
        elif run >= 0.20:
            sells.append("באזור מימוש 20-25% מעל נקודת הקנייה")
    if e > 1 and c[e - 1]:
        ups = [c[k] / c[k - 1] - 1 for k in range(max(1, e - 120), e + 1)]
        if ups and ups[-1] == max(ups) and ups[-1] > 0.05 and o[e] > h[e - 1] and near < 1:
            sells.append("ייתכן ריצת שיא (Climax): העלייה היומית הגדולה במהלך, עם פער פתיחה")
    out["sell"] = sells
    out["canslim"] = canslim(ev, fund, rs_pct, rsl, near, acc, mstate, brk)
    out["closed"] = closed
    return out


def rs_percentile(raw, reference):
    """Percentile (1-99) of `raw` within the reference list of raw RS values."""
    ref = [x for x in reference or [] if x is not None]
    if raw is None or len(ref) < 10:
        return None
    below = sum(1 for x in ref if x < raw)
    return max(1, min(99, int(round(below / len(ref) * 100))))
