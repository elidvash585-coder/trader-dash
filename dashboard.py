#!/usr/bin/env python3
"""Trader Dashboard: a local stock-analysis app for your TradingView watchlist.

    python dashboard.py            # starts http://127.0.0.1:8765 and opens your browser
    python dashboard.py --demo     # synthetic data, no internet needed (for testing the screen)

Standard library only. Data: Yahoo Finance (chart, quoteSummary, headline RSS), free and
unofficial, can lag. Everything here is analysis tooling, not financial advice.
"""

import argparse
import datetime
import http.cookiejar
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as FX  # noqa: E402  (v2 engine, standard library only)
import market as MK  # noqa: E402
import engine as EN  # noqa: E402
import risk as RK  # noqa: E402
import events as EV  # noqa: E402
import oneil as ON  # noqa: E402
import engine3 as EN3  # noqa: E402  (v3: technical score, entry score, plan, decision)
import dataqa as DQ  # noqa: E402  (market-data checks)
try:
    import extras  # time machine, SEC, Congress (day-trader-skill)
except Exception:
    extras = None
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PORT = 8765
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
WATCHLIST = os.path.join(HERE, "watchlist.json")
CONFIG = os.path.join(HERE, "app_config.json")
CACHE_TTL = 600
DEMO = False

# ---------------------------------------------------------------- http with Yahoo cookie + crumb

_jar = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))
_crumb = {"value": None, "at": 0, "fail": 0}
_lock = threading.Lock()


def get(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
    with _opener.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def crumb():
    with _lock:
        if _crumb["value"] and time.time() - _crumb["at"] < 3600:
            return _crumb["value"]
        if time.time() - _crumb["fail"] < 120:
            return None
        for warm in ("https://fc.yahoo.com", "https://finance.yahoo.com/quote/AAPL/"):
            try:
                get(warm, 6)
            except Exception:
                pass  # fc.yahoo.com answers 404 but still sets the cookie
            try:
                c = get("https://query2.finance.yahoo.com/v1/test/getcrumb", 6).strip()
                if c and "<" not in c and len(c) < 40:
                    _crumb.update(value=c, at=time.time())
                    return c
            except Exception:
                continue
        _crumb["fail"] = time.time()
        return None


def yahoo(path):
    last = None
    for host in ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"):
        try:
            return json.loads(get(host + path))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403) and "quoteSummary" in path:
                _crumb["value"] = None
                break
            if e.code == 404:
                break
            continue
        except Exception as e:
            last = e
    raise RuntimeError("Yahoo: %s" % last)


def ysym(s):
    return s.upper().replace(".", "-")


# ---------------------------------------------------------------- indicators

def sma(a, n):
    out, s = [None] * len(a), 0.0
    for i, x in enumerate(a):
        s += x
        if i >= n:
            s -= a[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(a, n):
    out, k, e = [None] * len(a), 2.0 / (n + 1), None
    for i, x in enumerate(a):
        if i == n - 1:
            e = sum(a[:n]) / n
        elif i >= n:
            e = x * k + e * (1 - k)
        out[i] = e
    return out


def rsi(c, n=14):
    out = [None] * len(c)
    if len(c) <= n:
        return out
    g = [max(0, c[i] - c[i - 1]) for i in range(1, len(c))]
    l = [max(0, c[i - 1] - c[i]) for i in range(1, len(c))]
    ag, al = sum(g[:n]) / n, sum(l[:n]) / n
    out[n] = 100 - 100 / (1 + ag / al) if al else 100.0
    for i in range(n, len(g)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al) if al else 100.0
    return out


def cci(h, l, c, n=20):
    tp = [(h[i] + l[i] + c[i]) / 3 for i in range(len(c))]
    out = [None] * len(c)
    for i in range(n - 1, len(c)):
        w = tp[i - n + 1:i + 1]
        m = sum(w) / n
        md = sum(abs(x - m) for x in w) / n
        out[i] = (tp[i] - m) / (0.015 * md) if md else 0.0
    return out


def atr_series(h, l, c, n=14):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    out, a = [None] * len(c), None
    for i, x in enumerate(tr):
        if i == n - 1:
            a = sum(tr[:n]) / n
        elif i >= n:
            a = (a * (n - 1) + x) / n
        out[i] = a
    return out


def macd(c):
    e12, e26 = ema(c, 12), ema(c, 26)
    line = [a - b if a is not None and b is not None else None for a, b in zip(e12, e26)]
    start = next(i for i, x in enumerate(line) if x is not None) if any(x is not None for x in line) else len(c)
    sig_part = ema(line[start:], 9) if len(c) - start >= 9 else [None] * (len(c) - start)
    sig = [None] * start + sig_part
    hist = [a - b if a is not None and b is not None else None for a, b in zip(line, sig)]
    return line, sig, hist


def bollinger(c, n=20, k=2):
    m = sma(c, n)
    up, lo = [None] * len(c), [None] * len(c)
    for i in range(n - 1, len(c)):
        w = c[i - n + 1:i + 1]
        sd = (sum((x - m[i]) ** 2 for x in w) / n) ** 0.5
        up[i], lo[i] = m[i] + k * sd, m[i] - k * sd
    return m, up, lo


def levels(h, l, c, look=120, win=5):
    """Support / resistance from swing highs and lows, clustered within 1.5 %."""
    n = len(c)
    pts = []
    for i in range(max(win, n - look), n - win):
        if h[i] == max(h[i - win:i + win + 1]):
            pts.append(h[i])
        if l[i] == min(l[i - win:i + win + 1]):
            pts.append(l[i])
    pts.sort()
    clusters = []
    for p in pts:
        if clusters and abs(p / clusters[-1][-1] - 1) < 0.015:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    price = c[-1]
    lv = [(sum(g) / len(g), len(g)) for g in clusters]
    sup = sorted([x for x in lv if x[0] < price], key=lambda x: -x[0])[:3]
    res = sorted([x for x in lv if x[0] > price], key=lambda x: x[0])[:3]
    return ([{"price": round(p, 2), "touches": t} for p, t in sup],
            [{"price": round(p, 2), "touches": t} for p, t in res])


def r2(x, d=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, d)


def last(a):
    for x in reversed(a):
        if x is not None:
            return x
    return None


# ---------------------------------------------------------------- channel + volume

def channel(t, c, look=60):
    """Linear-regression trend channel over the last `look` closes (+-2 standard deviations)."""
    n = min(look, len(c))
    y = c[-n:]
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(y) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (yy - my) for x, yy in zip(xs, y)) / sxx
    icpt = my - slope * mx
    fit = [icpt + slope * x for x in xs]
    resid = [yy - f for yy, f in zip(y, fit)]
    sd = (sum(r * r for r in resid) / n) ** 0.5
    ss_tot = sum((yy - my) ** 2 for yy in y) or 1
    r2v = 1 - sum(r * r for r in resid) / ss_tot
    move = slope * (n - 1) / fit[0] * 100 if fit[0] else 0
    kind = "עולה" if move > 5 else "יורדת" if move < -5 else "צידית"
    up_now, lo_now = fit[-1] + 2 * sd, fit[-1] - 2 * sd
    pos = (c[-1] - lo_now) / (up_now - lo_now) if up_now > lo_now else 0.5
    where = "ליד הגבול העליון" if pos > 0.8 else "ליד הגבול התחתון" if pos < 0.2 else "באמצע התעלה"
    return {"kind": kind, "move_pct": round(move, 1), "per_day_pct": round(slope / c[-1] * 100, 3), "r2": round(r2v, 2),
            "upper": round(up_now, 2), "lower": round(lo_now, 2), "mid": round(fit[-1], 2), "pos": round(pos, 2),
            "where": where, "days": n,
            "lines": {"upper": [[t[-n], round(fit[0] + 2 * sd, 2)], [t[-1], round(up_now, 2)]],
                      "mid": [[t[-n], round(fit[0], 2)], [t[-1], round(fit[-1], 2)]],
                      "lower": [[t[-n], round(fit[0] - 2 * sd, 2)], [t[-1], round(lo_now, 2)]]}}


def volume_analysis(t, o, h, l, c, v):
    n = len(c)
    avg20 = sum(v[-20:]) / 20
    avg50 = sum(v[-50:]) / min(50, n)
    prev20 = sum(v[-21:-1]) / 20
    up_v = sum(v[i] for i in range(n - 20, n) if c[i] >= c[i - 1])
    dn_v = sum(v[i] for i in range(n - 20, n) if c[i] < c[i - 1])
    obv, run = [], 0
    for i in range(n):
        if i:
            run += v[i] if c[i] > c[i - 1] else -v[i] if c[i] < c[i - 1] else 0
        obv.append(run)
    obv_chg = obv[-1] - obv[-21] if n > 21 else 0
    price_chg = c[-1] / c[-21] - 1 if n > 21 else 0
    if obv_chg > 0 and price_chg >= 0:
        obv_txt = "OBV עולה עם המחיר: הקונים מובילים (אישור)"
    elif obv_chg > 0 > price_chg:
        obv_txt = "OBV עולה בזמן שהמחיר יורד: איסוף שקט (סימן חיובי)"
    elif obv_chg < 0 < price_chg:
        obv_txt = "OBV יורד בזמן שהמחיר עולה: העלייה בלי נפח (אזהרה)"
    else:
        obv_txt = "OBV יורד עם המחיר: המוכרים מובילים"
    # volume profile, last 120 sessions
    k = min(120, n)
    lo_p, hi_p = min(l[-k:]), max(h[-k:])
    bins = 24
    step = (hi_p - lo_p) / bins or 1
    prof = [0.0] * bins
    for i in range(n - k, n):
        tp = (h[i] + l[i] + c[i]) / 3
        b = min(bins - 1, int((tp - lo_p) / step))
        prof[b] += v[i]
    poc_i = max(range(bins), key=lambda i: prof[i])
    poc = lo_p + (poc_i + 0.5) * step
    # value area: 70 % of volume around the POC
    total, acc, lo_i, hi_i = sum(prof), prof[poc_i], poc_i, poc_i
    while acc < 0.7 * total and (lo_i > 0 or hi_i < bins - 1):
        down = prof[lo_i - 1] if lo_i > 0 else -1
        upv = prof[hi_i + 1] if hi_i < bins - 1 else -1
        if upv >= down:
            hi_i += 1
            acc += upv
        else:
            lo_i -= 1
            acc += down
    big = sorted(range(max(1, n - 60), n), key=lambda i: -v[i])[:5]
    big_days = [{"date": datetime.datetime.utcfromtimestamp(t[i]).strftime("%Y-%m-%d"), "vol_x": round(v[i] / avg50, 1) if avg50 else None,
                 "change": round((c[i] / c[i - 1] - 1) * 100, 2)} for i in sorted(big, reverse=True)]
    ratio = up_v / dn_v if dn_v else None
    trend_txt = "הנפח עולה (10 ימים מעל ממוצע 50)" if sum(v[-10:]) / 10 > avg50 * 1.1 else \
        "הנפח יורד (שקט יחסי)" if sum(v[-10:]) / 10 < avg50 * 0.9 else "הנפח יציב"
    return {"last": int(v[-1]), "avg20": int(avg20), "avg50": int(avg50), "rvol": round(v[-1] / prev20, 2) if prev20 else None,
            "dollar_avg20": round(avg20 * c[-1]), "up_down_ratio": round(ratio, 2) if ratio else None,
            "obv_text": obv_txt, "obv_up": obv_chg > 0, "trend_text": trend_txt,
            "poc": round(poc, 2), "va_low": round(lo_p + lo_i * step, 2), "va_high": round(lo_p + (hi_i + 1) * step, 2),
            "profile": [{"price": round(lo_p + (i + 0.5) * step, 2), "vol": int(prof[i]), "in_va": lo_i <= i <= hi_i} for i in range(bins)],
            "big_days": big_days,
            "obv_series": [[t[i], obv[i]] for i in range(max(0, n - 260), n)]}


def decision(tech, vol, fund, plan, tm=None, sec=None):
    """Rule-based entry checklist. Not advice: it only says which rules pass."""
    ch = tech["channel"]
    items = []

    def item(label, ok, detail):
        items.append({"label": label, "ok": ok, "detail": detail})
    p = tech["price"]
    s50, s200 = tech["sma50"], tech["sma200"]
    item("מגמה ראשית", (s50 is not None and p > s50 and (s200 is None or s50 > s200)),
         "מחיר %.2f, ממוצע 50 %s, ממוצע 200 %s" % (p, s50 or "-", s200 or "-"))
    item("תעלת מגמה", ch["kind"] == "עולה" or (ch["kind"] == "צידית" and ch["pos"] < 0.35),
         "תעלה %s (%+.1f%% ב-%d ימים), המחיר %s" % (ch["kind"], ch["move_pct"], ch["days"], ch["where"]))
    item("מומנטום (CCI + MACD)", (tech["cci"] or 0) > 0 and (tech["macd"] or 0) > (tech["macd_signal"] or 0),
         "CCI %s, MACD %s מול אות %s" % (tech["cci"], tech["macd"], tech["macd_signal"]))
    item("לא קנייה ביתר", (tech["rsi"] or 50) < 70, "RSI %s" % tech["rsi"])
    item("נפח מאשר", bool(vol["up_down_ratio"] and vol["up_down_ratio"] > 1 and vol["obv_up"]),
         "נפח בימי עלייה פי %s מימי ירידה, %s" % (vol["up_down_ratio"], vol["obv_text"].split(":")[0]))
    room = None
    if plan:
        res = [r["price"] for r in tech["resistance"] if r["price"] > plan["trigger"]]
        nearest = res[0] if res else None
        room = (nearest - plan["trigger"]) / plan["risk_per_share"] if nearest and plan["risk_per_share"] else None
        item("מקום עד ההתנגדות", room is None or room >= 1.5,
             "אין התנגדות קרובה" if room is None else "ההתנגדות ב-%.2f, %.1fR מהכניסה" % (nearest, room))
    if fund:
        ne = fund.get("next_earnings")
        days = None
        try:
            days = (datetime.date.fromisoformat(ne) - datetime.date.today()).days if ne else None
        except Exception:
            pass
        item("אין דוח בשבוע הקרוב", days is None or days < 0 or days > 7,
             "דוח הבא %s" % ne if ne else "אין תאריך דוח ידוע")
        a = fund.get("analyst") or {}
        if a.get("mean"):
            up = (a["mean"] / p - 1) * 100
            item("אנליסטים", up > 5 and (a.get("score") or 3) <= 2.6, "יעד ממוצע %.2f (%+.0f%%), ציון %s" % (a["mean"], up, r2(a.get("score"))))
    if tm and tm.get("share") is not None:
        item("סטטיסטיקה (Time Machine)", True if tm["lean"] > 0 else False if tm["lean"] < 0 else None,
             "ב-%s מתוך 5,000 הדמיות היעד הגיע לפני הסטופ (%s%% מההכרעות, איזון %s%%)" % (
                 "%.0f%%" % tm["p_target"], tm["share"], tm["breakeven"]))
    if sec:
        recent = [f for f in sec.get("filings", []) if f.get("important")
                  and (datetime.date.today() - datetime.date.fromisoformat(f["date"])).days <= 7]
        if recent:
            item("אין דיווח מהותי טרי ל-SEC", False, "%s ב-%s: %s" % (recent[0]["form"], recent[0]["date"], ", ".join(recent[0]["items_he"])))
    known = [x for x in items if x["ok"] is not None]
    passed = sum(1 for x in known if x["ok"])
    ratio = passed / len(known) if known else 0
    rsi = tech.get("rsi")
    if rsi is not None and rsi < MODEL["rsi_max"]:
        verdict, code = "אות פעיל: מכירת יתר", "go"
        action = ("RSI %.1f, מתחת ל-30. המודל קונה בפתיחה הבאה (בערך %.2f), סטופ %.2f, יעד %.2f, "
                  "ויוצא אחרי %d ימי מסחר אם לא הגיע לאחד מהם." % (rsi, plan["entry"], plan["stop"], plan["target"], MODEL["horizon"])) if plan \
            else "RSI %.1f, מתחת ל-30." % rsi
    elif rsi is not None and rsi < MODEL["rsi_watch"]:
        verdict, code = "קרוב לאות", "wait"
        action = "RSI %.1f. האות נדלק מתחת ל-30, כלומר עוד ירידה קטנה." % rsi
    else:
        verdict, code = "אין אות", "no"
        action = "RSI %s. המודל נכנס רק במכירת יתר (RSI מתחת ל-30)." % (("%.1f" % rsi) if rsi is not None else "-")
    warn = [x["detail"] for x in items if x["label"] in ("אין דוח בשבוע הקרוב", "אין דיווח מהותי טרי ל-SEC") and x["ok"] is False]
    if code == "go" and warn:
        action += " שים לב: " + "; ".join(warn) + ". הבדיקה לאחור לא סיננה דוחות, ופער בדוח יכול לעבור את הסטופ."
    return {"verdict": verdict, "code": code, "passed": passed, "total": len(known), "score": round(ratio * 100),
            "items": items, "action": action, "rsi": rsi, "model": MODEL, "evidence": VALIDATED}


# ---------------------------------------------------------------- analysis

def load_config():
    cfg = {"budget": 5000, "max_loss_per_trade_usd": 50, "max_position_pct": 75, "cash_buffer_pct": 10, "target_r": 2.0}
    try:
        with open(CONFIG, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except Exception:
        pass
    return cfg


def technicals(t, o, h, l, c, v):
    s20, s50, s200 = sma(c, 20), sma(c, 50), sma(c, 200)
    e9, e21 = ema(c, 9), ema(c, 21)
    rs, cc, at = rsi(c), cci(h, l, c), atr_series(h, l, c)
    ml, ms, mh = macd(c)
    bm, bu, bl = bollinger(c)
    avgv = sum(v[-21:-1]) / 20 if len(v) > 21 else None
    rvol = v[-1] / avgv if avgv else None
    sup, res = levels(h, l, c)
    price = c[-1]
    hi52, lo52 = max(h[-252:]), min(l[-252:])
    chg = lambda k: (c[-1] / c[-1 - k] - 1) * 100 if len(c) > k else None

    signals = []  # (text, +1 bullish / -1 bearish / 0 neutral)

    def add(cond, text, w):
        if cond:
            signals.append({"text": text, "w": w})
    L = lambda a: last(a)
    add(L(s50) and price > L(s50), "המחיר מעל ממוצע 50", 1)
    add(L(s50) and price < L(s50), "המחיר מתחת לממוצע 50", -1)
    add(L(s200) and price > L(s200), "המחיר מעל ממוצע 200 (מגמה ראשית עולה)", 1)
    add(L(s200) and price < L(s200), "המחיר מתחת לממוצע 200 (מגמה ראשית יורדת)", -1)
    add(L(s50) and L(s200) and L(s50) > L(s200), "ממוצע 50 מעל 200 (Golden alignment)", 1)
    add(L(e9) and L(e21) and L(e9) > L(e21), "EMA 9 מעל EMA 21 (מומנטום קצר עולה)", 1)
    add(L(e9) and L(e21) and L(e9) < L(e21), "EMA 9 מתחת ל-EMA 21", -1)
    r = L(rs)
    add(r is not None and r > 70, "RSI מעל 70: קנייה ביתר", -1)
    add(r is not None and r < 30, "RSI מתחת ל-30: מכירת יתר", 1)
    k = L(cc)
    add(k is not None and k > 100, "CCI מעל 100: מומנטום חזק", 1)
    add(k is not None and k < -100, "CCI מתחת ל-100-: חולשה / מכירת יתר", -1)
    if len(cc) > 2 and cc[-2] is not None and cc[-1] is not None and cc[-2] < -100 <= cc[-1]:
        add(True, "CCI חצה מעל 100- (אות היפוך)", 1)
    add(L(mh) is not None and L(mh) > 0, "MACD מעל קו האות", 1)
    add(L(mh) is not None and L(mh) < 0, "MACD מתחת לקו האות", -1)
    if len(mh) > 2 and mh[-2] is not None and mh[-1] is not None:
        add(mh[-2] < 0 <= mh[-1], "MACD חצה למעלה עכשיו", 1)
        add(mh[-2] > 0 >= mh[-1], "MACD חצה למטה עכשיו", -1)
    add(rvol is not None and rvol >= 1.5, "נפח חריג (x%.1f מהממוצע)" % (rvol or 0), 0)
    add(price > max(c[-21:-1]) if len(c) > 21 else False, "פריצת שיא סגירה של 20 יום", 1)
    add(L(bu) and price > L(bu), "סגירה מעל רצועת בולינגר העליונה", 0)
    add(L(bl) and price < L(bl), "סגירה מתחת לרצועת בולינגר התחתונה", 0)
    if len(c) > 16:
        run = c[-6] / c[-16] - 1
        rng5 = (max(c[-5:]) - min(c[-5:])) / price
        add(run > 0.10 and rng5 < 0.05, "דגל שורי (עלייה של %.0f%% ואז דשדוש צר)" % (run * 100), 1)
    score = sum(s["w"] for s in signals)
    nmax = sum(1 for s in signals if s["w"] != 0) or 1
    tech_score = round(50 + 50 * score / nmax)

    if L(s50) and L(s200):
        trend = "עולה" if price > L(s50) > L(s200) else "יורדת" if price < L(s50) < L(s200) else "מעורבת"
    else:
        trend = "לא ידוע"

    def series(a, d=2):
        return [[t[i], round(a[i], d)] for i in range(len(a)) if a[i] is not None]
    return {
        "price": r2(price), "change_1d": r2(chg(1)), "change_5d": r2(chg(5)), "change_1m": r2(chg(21)),
        "change_3m": r2(chg(63)), "change_1y": r2(chg(252)) if len(c) > 252 else r2((c[-1] / c[0] - 1) * 100),
        "sma20": r2(L(s20)), "sma50": r2(L(s50)), "sma200": r2(L(s200)), "ema9": r2(L(e9)), "ema21": r2(L(e21)),
        "rsi": r2(r, 1), "cci": r2(k, 1), "atr": r2(L(at)), "atr_pct": r2(L(at) / price * 100 if L(at) else None),
        "macd": r2(L(ml), 3), "macd_signal": r2(L(ms), 3), "rvol": r2(rvol), "hi52": r2(hi52), "lo52": r2(lo52),
        "from_hi52": r2((price / hi52 - 1) * 100), "support": sup, "resistance": res, "trend": trend,
        "signals": signals, "tech_score": tech_score, "channel": channel(t, c),
        "prev_high": r2(h[-1]), "high20": r2(max(h[-21:-1])) if len(h) > 21 else r2(max(h)),
        "series": {"sma20": series(s20), "sma50": series(s50), "sma200": series(s200), "bb_up": series(bu), "bb_lo": series(bl),
                   "rsi": series(rs, 1), "cci": series(cc, 1), "macd": series(ml, 3), "macd_signal": series(ms, 3),
                   "macd_hist": series(mh, 3)},
    }


# The model the backtest supports (research/REPORT2.md, run 2026-11-06 on 89 stocks, 2022-11 to 2026-11):
# RSI(14) under 30 at the close -> buy the next open, stop 1 ATR under the entry, target 3R, exit after
# 10 sessions. It was the only setup whose edge over buying on any day was positive in train, validation
# and test, and in all 32 stop/target/horizon variants on the test period. Breakouts: 0 of 32.
MODEL = {"setup": "oversold", "rsi_max": 30.0, "rsi_watch": 35.0, "stop_atr": 1.0, "target_r": 3.0, "horizon": 10}
VALIDATED = {
    "source": "research/REPORT2.md", "run": "2026-11-06", "stocks": 89, "period": "2022-11-08 .. 2026-11-05",
    "test_from": "2025-04-03", "test_n": 1123, "test_win": 44.0, "test_R": 0.227, "test_R_ci": [-0.041, 0.502],
    "test_edge": 0.146, "test_edge_ci": [-0.096, 0.388], "train_edge": 0.069, "val_edge": 0.429,
    "variants_positive": "32/32", "breakout_variants_positive": "0/32", "pullback_variants_positive": "16/32",
    "checklist_auc": 0.496,
    "setups": [["פריצת השיא של אתמול (התוכנית הקודמת)", -0.010], ["פריצת שיא סגירה של 20 יום", -0.011],
               ["חציה מעל ממוצע 20", -0.006], ["תיקון במגמה עולה", 0.019], ["מכירת יתר (RSI<30)", 0.146]],
}
try:
    with open(os.path.join(HERE, "research", "strategies.json"), encoding="utf-8") as _fh:
        _st = json.load(_fh)
    _c = _st.get("chosen") or {}
    if _c.get("setup") == MODEL["setup"]:
        VALIDATED.update(test_n=_c["test_n"], test_win=_c["test_win"], test_R=_c["test_R"], test_R_ci=_c["test_R_ci"],
                         test_edge=_c["test_edge"], test_edge_ci=_c["test_edge_ci"], run=_st.get("generated", "")[:10])
except Exception:
    pass


SCORE_TABLE = None
try:
    with open(os.path.join(HERE, "research", "score_table.json"), encoding="utf-8") as _fh:
        SCORE_TABLE = json.load(_fh)
except Exception:
    SCORE_TABLE = None
RSI_BINS = [(0, 20, "<20"), (20, 25, "20-25"), (25, 30, "25-30"), (30, 35, "30-35"), (35, 40, "35-40"),
            (40, 50, "40-50"), (50, 60, "50-60"), (60, 70, "60-70"), (70, 101, ">70")]
SHRINK_K = 500  # trades: a group with n trades keeps n / (n + 500) of its measured edge


def score(tech, fund=None, sec=None, ext=None, plan=None):
    """0-10 score of the model trade, from history: what the same trade earned when RSI was in the
    same range (89 stocks, 5 years), shrunk toward 'no edge' for small groups. 5 = like a random day."""
    rsi = tech.get("rsi")
    if rsi is None or not SCORE_TABLE:
        return None
    t_all, t_test = SCORE_TABLE["all"], SCORE_TABLE["test"]
    lab = next(l for lo, hi, l in RSI_BINS if lo <= rsi < hi)
    def pick(rows, lab):
        r = next((x for x in rows if x["rsi"] == lab and "edge" in x), None)
        if lab == "<20":  # too few trades alone: pool with the neighbouring 20-25 group
            r2_ = next((x for x in rows if x["rsi"] == "20-25" and "edge" in x), None)
            if r and r2_:
                n = r["n"] + r2_["n"]
                w = lambda k: (r[k] * r["n"] + r2_[k] * r2_["n"]) / n
                r = {"rsi": "<25", "n": n, "win": round(w("win"), 1), "profit": round(w("profit"), 1), "R": round(w("R"), 3),
                     "edge": round(w("edge"), 3), "edge_ci": r2_["edge_ci"]}
        return r
    row, trow = pick(t_all["rows"], lab), pick(t_test["rows"], lab)
    if not row:
        return None
    shr = lambda r: r["edge"] * r["n"] / (r["n"] + SHRINK_K)
    span = max(abs(shr(r)) for r in t_all["rows"] if "edge" in r) or 1
    raw = 5 + 4.5 * shr(row) / span
    val = round(max(0.0, min(10.0, raw)), 1)
    reasons = [("up" if row["edge"] > 0 else "down" if row["edge"] < 0 else "acc",
                "RSI %.0f (טווח %s). בהיסטוריה, באותה עסקה כשה-RSI היה בטווח הזה: %.0f%% הסתיימו ברווח, %.0f%% הגיעו ליעד, "
                "ממוצע %+.2fR לעסקה (כניסה ביום אקראי: %+.2fR). %s עסקאות." % (
                    rsi, row["rsi"], row["profit"], row["win"], row["R"], t_all["base_R"], format(row["n"], ",")))]
    lo, hi = row["edge_ci"]
    reasons.append(("acc", "היתרון מול יום אקראי: %+.2fR, טווח סביר %+.2f עד %+.2f%s." % (
        row["edge"], lo, hi, " (מובהק)" if lo > 0 or hi < 0 else " (כולל אפס, לא מובהק)")))
    if trow:
        same = (trow["edge"] > 0) == (row["edge"] > 0)
        reasons.append(("up" if same else "warn", "בתקופה האחרונה (מ-%s) לבד: %+.2fR מול יום אקראי, %s." % (
            SCORE_TABLE["test_from"], trow["edge"], "באותו כיוון" if same else "בכיוון ההפוך, כך שהממצא פחות יציב")))
    caps = []
    if fund and fund.get("next_earnings"):
        try:
            days = (datetime.date.fromisoformat(fund["next_earnings"]) - datetime.date.today()).days
            if 0 <= days <= 10:
                caps.append("דוח כספי בעוד %d ימים, בתוך 10 ימי העסקה" % days)
        except Exception:
            pass
    if sec:
        rec = [f for f in sec.get("filings", []) if f.get("important")
               and (datetime.date.today() - datetime.date.fromisoformat(f["date"])).days <= 7]
        if rec:
            caps.append("דיווח מהותי טרי ל-SEC (%s)" % ", ".join(rec[0]["items_he"]))
    if ext and ext.get("active") and plan and ext.get("price") is not None and ext["price"] <= plan["stop"]:
        caps.append("המחיר ב-pre-market כבר מתחת לסטופ המתוכנן")
    if caps and val > 5:
        val = 5.0
        reasons.append(("warn", "הציון הוגבל ל-5: " + "; ".join(caps) + ". (כלל זהירות שלי, לא נבדק לאחור.)"))
    elif caps:
        reasons.append(("warn", "שים לב: " + "; ".join(caps) + "."))
    label = ("שווה לשקול כניסה" if val >= 7 else "חיובי קל" if val >= 5.6 else
             "ניטרלי, כמו כניסה אקראית" if val >= 4.4 else "לא כדאי עכשיו")
    code = "go" if val >= 7 else "wait" if val >= 5.6 else "no"
    return {"value": val, "label": label, "code": code, "bucket": row["rsi"], "p_profit": row["profit"], "p_target": row["win"],
            "exp_R": row["R"], "edge": row["edge"], "n": row["n"], "reasons": reasons, "capped": bool(caps and val == 5.0)}


def _cl(x):
    return max(0.0, min(10.0, x))


def ind_score_v1(tech, plan=None, fund=None, tm=None, congress=None):
    """OLD second score (v1, kept for comparison): every other indicator, equal weights, 0-10 each, averaged. NOT validated:
    in the backtest these did not improve the forecast. Shown so the user can weigh them himself."""
    comp = []

    def add(key, name, val, why):
        if val is not None:
            comp.append({"key": key, "name": name, "value": round(_cl(val), 1), "why": why})
    p, s50, s200 = tech["price"], tech["sma50"], tech["sma200"]
    if s50 and s200:
        k = sum([p > s50, s50 > s200, p > s200])
        add("trend", "מגמה (ממוצעים)", k / 3 * 10, "%d מתוך 3: מחיר מעל 50, 50 מעל 200, מחיר מעל 200" % k)
    ch = tech.get("channel")
    if ch:
        base = {"עולה": 7.0, "צידית": 5.0, "יורדת": 2.5}[ch["kind"]]
        add("channel", "תעלת מגמה", base + (0.5 - ch["pos"]) * 3, "תעלה %s, %s" % (ch["kind"], ch["where"]))
    if tech.get("cci") is not None and tech.get("macd") is not None:
        v = (_cl(5 + tech["cci"] / 40) + (7 if tech["macd"] > tech["macd_signal"] else 3)) / 2
        add("momentum", "מומנטום (CCI + MACD)", v, "CCI %.0f, MACD %s קו האות" % (tech["cci"], "מעל" if tech["macd"] > tech["macd_signal"] else "מתחת"))
    vol = tech.get("volume") or {}
    if vol.get("up_down_ratio") is not None:
        v = (_cl(5 + (vol["up_down_ratio"] - 1) * 10) + (7 if vol.get("obv_up") else 3)) / 2
        add("volume", "נפחים", v, "נפח עלייה/ירידה %.2f, OBV %s" % (vol["up_down_ratio"], "עולה" if vol.get("obv_up") else "יורד"))
    if plan:
        res = [r["price"] for r in tech["resistance"] if r["price"] > plan["entry"]]
        room = (res[0] - plan["entry"]) / plan["risk_per_share"] if res and plan["risk_per_share"] else None
        sup = [x["price"] for x in tech["support"]]
        near_sup = bool(sup and (p - sup[0]) / p < 0.03)
        v = (8 if room is None or room >= 2 else 6 if room >= 1 else 3) + (1 if near_sup else 0)
        add("structure", "מבנה (תמיכה/התנגדות)", v, ("אין התנגדות קרובה" if room is None else "התנגדות במרחק %.1fR" % room) +
            (", קרוב לתמיכה" if near_sup else ""))
    a = (fund or {}).get("analyst") or {}
    if a.get("mean") and p:
        up = (a["mean"] / p - 1) * 100
        rec = a.get("score")
        v = _cl(5 + up / 5) if rec is None else (_cl(5 + up / 5) + _cl((3 - rec) * 5)) / 2
        add("analysts", "אנליסטים", v, "יעד %+.0f%%%s" % (up, ", ציון המלצה %.1f (1 = קנייה חזקה)" % rec if rec else ""))
    if fund and fund.get("rev_growth") is not None:
        g, nm, fcf = fund["rev_growth"], fund.get("net_margin"), fund.get("fcf")
        parts = [8 if g > 0.15 else 6 if g > 0 else 3]
        if nm is not None:
            parts.append(8 if nm > 0.15 else 6 if nm > 0 else 3)
        if fcf is not None:
            parts.append(7 if fcf > 0 else 3)
        add("fundamentals", "פונדמנטלי", sum(parts) / len(parts), "צמיחת הכנסות %+.0f%%%s" % (
            g * 100, ", שולי נקי %.0f%%" % (nm * 100) if nm is not None else ""))
    if tm and tm.get("share") is not None:
        add("timemachine", "Time Machine", 5 + (tm["share"] - tm["breakeven"]) / 3,
            "יעד לפני סטופ ב-%.0f%% מההכרעות (איזון %.0f%%)" % (tm["share"], tm["breakeven"]))
    if congress:
        buys = sum(1 for t in congress if "purchase" in (t.get("type") or ""))
        sells = sum(1 for t in congress if "sale" in (t.get("type") or ""))
        add("congress", "חברי קונגרס", 5 + (buys - sells), "%d קניות ו-%d מכירות ב-90 יום" % (buys, sells))
    if not comp:
        return None
    val = round(sum(c["value"] for c in comp) / len(comp), 1)
    return {"value": val, "components": comp, "n": len(comp),
            "label": "חיובי" if val >= 6.5 else "חיובי קל" if val >= 5.5 else "ניטרלי" if val >= 4.5 else "שלילי קל" if val >= 3.5 else "שלילי"}


_mkt = {"t": 0, "ctx": None}
_etf_cache = {}


def market_ctx():
    """SPY / QQQ / VIX context, cached 30 minutes (5 during the session), plus O'Neil's market direction."""
    ttl = 300 if us_session_open() else 1800
    if _mkt["ctx"] is None or time.time() - _mkt["t"] > ttl:
        ctx = MK.context(demo_chart if DEMO else chart, EN.CFG.get("regime"))
        try:
            ctx["oneil_m"] = oneil_market(ctx)
        except Exception as e:
            ctx["oneil_m"] = None
            print("market direction failed: %s" % e)
        _mkt["ctx"] = ctx
        _mkt["t"] = time.time()
    return _mkt["ctx"]


def us_session_open():
    try:
        import zoneinfo
        n = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        return False
    m = n.hour * 60 + n.minute
    return n.weekday() < 5 and 570 <= m < 960


def _closed(t, *arrays):
    """Drop today's bar while the US session is still open (it is not a closed daily bar yet)."""
    if t and us_session_open():
        import zoneinfo
        today = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York")).date()
        last = datetime.datetime.fromtimestamp(t[-1], zoneinfo.ZoneInfo("America/New_York")).date()
        if last == today:
            return [t[:-1]] + [a[:-1] for a in arrays]
    return [t] + list(arrays)


def oneil_market(ctx):
    """O'Neil's M from SPY (QQQ adds distribution days), on closed bars."""
    spy, qqq = ctx.get("spy"), ctx.get("qqq")
    if not spy or not spy.get("v"):
        return None
    t, c, v = _closed(spy["t"], spy["c"], spy["v"])
    c2 = v2 = None
    if qqq and qqq.get("v"):
        qd = {int(a // 86400): (b, w) for a, b, w in zip(qqq["t"], qqq["c"], qqq["v"])}
        al = [qd.get(int(x // 86400)) for x in t]
        if all(al):
            c2, v2 = [x[0] for x in al], [x[1] for x in al]
    st = ON.market_direction(c, v, c2, v2)
    last = dict(st[-1])
    if last.get("ftd") is not None:
        last["ftd_date"] = datetime.datetime.utcfromtimestamp(t[last["ftd"]]).strftime("%Y-%m-%d")
    last["label"] = ON.M_HE[last["state"]]
    last["as_of"] = datetime.datetime.utcfromtimestamp(t[-1]).strftime("%Y-%m-%d")
    return last


RS_REF = None
try:
    with open(os.path.join(HERE, "research", "rs_reference.json"), encoding="utf-8") as _fh:
        RS_REF = json.load(_fh)
except Exception:
    RS_REF = None


ON_RES = None
try:  # backtest5_oneil.py results, summarised for the O'Neil section
    with open(os.path.join(HERE, "research", "oneil.json"), encoding="utf-8") as _fh:
        _r = json.load(_fh)
    _pick = lambda pre: next(((k, v["all"]) for k, v in _r.get("variants", {}).items() if k.startswith(pre)), (None, {}))
    ON_RES = {"stocks": _r.get("stocks"), "from": _r.get("from", "")[:4], "to": _r.get("to", "")[:4]}
    for _k in ("ON1", "ON2", "ON3", "ON4"):
        _n, _a = _pick(_k)
        ON_RES[_k.lower()] = {"n": _a.get("n", 0), "exp": _a.get("expectancy"), "edge": _a.get("edge"), "ci": _a.get("edge_ci")}
except Exception:
    ON_RES = None


def oneil_block(sym, candles, fund=None, sec=None):
    """O'Neil analysis on closed bars."""
    t = [x[0] for x in candles]
    o, h, l, c, v = ([float(x[i] or 0) for x in candles] for i in range(1, 6))
    t, o, h, l, c, v = _closed(t, o, h, l, c, v)
    m = market_ctx() or {}
    spy = FX.align(t, m["spy"]["t"], m["spy"]["c"]) if m.get("spy") else None
    raw_ = ON.rs_raw(c, len(c) - 1)
    ref = (RS_REF or {}).get("raw")
    if not ref:  # no research reference yet: rank among the watchlist bars already loaded
        ref = [ON.rs_raw([float(x[4]) for x in v_[1]["candles"]], len(v_[1]["candles"]) - 1) for v_ in list(_cache.values())]
    rs_pct = ON.rs_percentile(raw_, ref)
    out = ON.analyze(t, o, h, l, c, v, spy, (sec or {}).get("growth"), fund, rs_pct, m.get("oneil_m"))
    out["rs_ref"] = "%d מניות גדולות (מחקר %s)" % (len(RS_REF["raw"]), RS_REF.get("date", "")) if RS_REF else "רשימת המעקב בלבד"
    out["research"] = ON_RES
    return out


def etf_rows(sym):
    hit = _etf_cache.get(sym)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    _, rows = demo_chart(sym) if DEMO else chart(sym)
    _etf_cache[sym] = (time.time(), rows)
    return rows


def fx_block(t, o, h, l, c, v, partial):
    """v2 features on CLOSED bars only (a bar still trading is left out), plus chart overlays."""
    if partial and len(c) > 60:
        t, o, h, l, c, v = t[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]
    bench = {}
    try:
        m = market_ctx()
        if m.get("spy"):
            bench["SPY"] = FX.align(t, m["spy"]["t"], m["spy"]["c"])
        if m.get("qqq"):
            bench["QQQ"] = FX.align(t, m["qqq"]["t"], m["qqq"]["c"])
    except Exception:
        m = {}
    f = FX.compute(t, o, h, l, c, v, bench)
    keys = ["atr", "sma20", "sma50", "sma150", "sma200", "slope50", "adx", "pdi", "ndi", "rsi", "cci", "macd_h",
            "atr_pct", "atr_pctile", "rv20", "state", "bos_up", "bos_dn", "failed_bo", "res", "swing_low", "last_ph",
            "consol", "ext_atr", "room_atr", "rvol", "updown", "obv20", "clv", "gap", "ret20", "ret60", "ret120",
            "d_sma20", "d_sma50", "d_sma150", "d_sma200"] + [k for k in f if k.startswith("rs_")]
    x = {k: FX.last(f, k) for k in keys}
    x["price"] = c[-1]
    x["chg1"] = (c[-1] / c[-2] - 1) * 100 if len(c) > 1 else None
    x["move_atr"] = (c[-1] - c[-2]) / x["atr"] if len(c) > 1 and x.get("atr") else None
    x["regime"] = ((m or {}).get("regime") or {}).get("code")
    x["bar_partial_excluded"] = bool(partial)
    # chart overlays (off by default in the UI): anchored VWAPs and an approximate volume profile
    ph, pl = FX.pivots(h, l)
    lows = [j for j in range(max(0, len(c) - 130), len(c) - FX.K_PIVOT) if pl[j]]
    anchor_low = min(lows, key=lambda j: l[j]) if lows else None
    anchor_hi = max(range(max(0, len(c) - 252), len(c)), key=lambda j: h[j])
    ov = {}
    for name, a in (("avwap_low", anchor_low), ("avwap_high", anchor_hi)):
        if a is not None:
            s_ = FX.avwap(h, l, c, v, a)
            ov[name] = {"anchor": t[a], "series": [[t[i], round(s_[i], 2)] for i in range(a, len(c))]}
    ov["vprofile"] = FX.volume_profile(h, l, v)
    lv = FX.levels(h, l, c, f["atr"])
    poc = (ov["vprofile"] or {}).get("poc") if isinstance(ov["vprofile"], dict) else None
    for side in ("support", "resistance"):
        for z in lv[side]:
            z["volume_node"] = bool(poc and x.get("atr") and abs(z["price"] - poc) <= 0.5 * x["atr"])
    ov["levels"] = lv
    return x, ov


def ind_score(tech, plan=None, fund=None, tm=None, congress=None, sector=None):
    """Indicator score v2 (engine.py). Congress trades are no longer scored (no evidence they predict);
    they stay as information. Falls back to v1 only if the v2 features are missing."""
    x = dict(tech.get("fx") or {})
    if not x:
        return ind_score_v1(tech, plan, fund, tm, congress)
    if sector:
        x.update(sector)
    return EN3.technical_score(x)


def sector_part(fund, t_last_bars):
    """Sector ETF trend and relative strength vs the sector, when the sector is known."""
    etf = MK.SECTOR_ETF.get((fund or {}).get("sector") or "")
    if not etf:
        return None
    try:
        rows = etf_rows(etf)
        et, ec = [r[0] for r in rows], [float(r[4]) for r in rows]
        tr = MK.etf_trend(ec)[-1]
        t, c = t_last_bars
        b = FX.align(t, et, ec)
        out = {"sector_etf": etf, "sector_he": MK.SECTOR_HE.get(etf), "sector_trend": tr}
        for n in (20, 60, 120):
            out["rs_sector_%d" % n] = FX.last({"s": FX.rel_strength(c, b, n)}, "s")
        return out
    except Exception:
        return None


_pf = {"t": 0, "v": None}


def portfolio_view(force=False):
    """Local only: holdings.json next to the app. The public site never has it."""
    hold = RK.read_holdings()
    if hold is None:
        return {"none": True, "path": RK.HOLDINGS}
    if force or _pf["v"] is None or time.time() - _pf["t"] > 600:
        fetch = demo_chart if DEMO else etf_rows_meta
        _pf["v"] = RK.portfolio(hold, fetch, lambda s: (_fund_cache.get(s) or (0, None))[1])
        _pf["t"] = time.time()
    return _pf["v"]


def etf_rows_meta(sym):
    hit = _etf_cache.get("m:" + sym)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    v = chart(sym)
    _etf_cache["m:" + sym] = (time.time(), v)
    return v


def peers_for(sym):
    """Most correlated symbols in the watchlist (from bars already loaded). Overlap, not holdings."""
    closes = {s: [(c[0], c[4]) for c in v[1]["candles"][-200:]] for s, v in list(_cache.items())}
    if sym not in closes or len(closes) < 3:
        return None
    return RK.watchlist_peers(closes).get(sym)


def events_for(fund, sec, ext, plan2):
    """Facts only (earnings date, SEC filings, extended-hours price). Unknown stays unknown."""
    ev = {"earnings_known": False, "earnings_days": None, "sec_known": sec is not None, "sec_material": None,
          "premarket_below_stop": None}
    blockers = []
    if fund and fund.get("next_earnings"):
        try:
            d = (datetime.date.fromisoformat(fund["next_earnings"]) - datetime.date.today()).days
            ev["earnings_known"], ev["earnings_days"] = True, d
            if 0 <= d <= 14:
                blockers.append("דוח כספי בעוד %d ימים, בתוך תקופת העסקה" % d)
        except Exception:
            pass
    if sec:
        rec = [f for f in sec.get("filings", []) if f.get("important")
               and (datetime.date.today() - datetime.date.fromisoformat(f["date"])).days <= 7]
        if rec:
            ev["sec_material"] = "דיווח %s (%s)" % (rec[0]["date"], ", ".join(rec[0]["items_he"]))
            blockers.append("דיווח מהותי טרי ל-SEC")
    if ext and ext.get("active") and plan2 and plan2.get("stop") and ext.get("price") is not None:
        ev["premarket_below_stop"] = ext["price"] <= plan2["stop"]
        if ev["premarket_below_stop"]:
            blockers.append("המחיר מחוץ לשעות המסחר כבר מתחת לסטופ")
    return ev, blockers


def held_symbols():
    """Symbols in the local holdings file (never on the public site)."""
    if STATIC_BUILD:
        return set()
    try:
        return {p["symbol"] for p in (RK.read_holdings() or [])}
    except Exception:
        return set()


STATIC_BUILD = False


def engine_block(tech, fund, sec, ext, ind, event=None, sym=None):
    """v3: entry score, the plan and the decision, all from closed bars (the live price only says whether the
    entry is still in its zone). Returns (entry score with the history and reliability inside, plan, decision)."""
    x = dict(tech.get("fx") or {})
    if not x:
        return None, None, None
    lv = tech.get("levels")
    plan = EN3.plan_v3(x, lv, tech.get("live"))
    ev, blockers = events_for(fund, sec, ext, plan)
    if event and event.get("level") == "critical":
        blockers.append("אירוע מהותי ב-3 ימי המסחר האחרונים (%s)" % event.get("verify"))
    hist = EN.hist_score(x.get("rsi"), tech.get("asset"), blockers)
    ent = EN3.entry_score(x, plan, hist, ev, lv)
    if event and event.get("level") == "critical" and ent.get("value") is not None and ent["value"] > 5.0:
        ent["value"], ent["caps"] = 5.0, ent["caps"] + ["אירוע מהותי ב-3 ימי המסחר האחרונים"]
    data = tech.get("data") or {}
    dec = EN3.decide_v3(x, ind, ent, plan, hist, ev, blockers, tech.get("asset"), held=bool(sym and sym in held_symbols()),
                        data_level=data.get("level", "partial"))
    ent["hist"] = hist
    ent["reliability"] = EN3.reliability(data, ind, ent, hist, tech.get("asset"))
    ent["code"] = dec["code"]
    if plan and plan.get("stop"):
        cfg = load_config()
        risk = plan["entry"] - plan["stop"]
        cash = cfg["budget"] * (1 - cfg["cash_buffer_pct"] / 100.0)
        cap = min(cash, cfg["budget"] * cfg["max_position_pct"] / 100.0)
        sh = max(0, min(int(cfg["max_loss_per_trade_usd"] // risk), int(cap // plan["entry"])))
        plan["size"] = {"budget": cfg["budget"], "max_loss_cfg": cfg["max_loss_per_trade_usd"], "shares": sh,
                        "cost": round(sh * plan["entry"], 2), "risk_usd": round(sh * risk, 2),
                        "risk_per_1000": round(1000 / plan["entry"] * risk, 2)}
    return ent, plan, dec


_halts = {"t": 0, "v": {}}


def halts():
    """Nasdaq trade halts (public RSS), cached 5 minutes. Empty when unreachable (never guessed)."""
    if DEMO:
        return {}
    if time.time() - _halts["t"] > 300:
        try:
            _halts["v"] = EV.parse_halts(get(EV.HALTS_URL, timeout=10))
        except Exception:
            pass
        _halts["t"] = time.time()
    return _halts["v"]


def event_for(sym, candles, sec=None, news_=None, fund=None):
    m = market_ctx() or {}
    spy = (m["spy"]["t"], m["spy"]["c"]) if m.get("spy") else None
    try:
        return EV.assess(sym, candles, spy, sec, halts(), news_, fund)
    except Exception as e:
        print("event check failed for %s: %s" % (sym, e))
        return None


def trade_plan(tech, cfg):
    """The backtested plan: enter at the next open (estimated by the last price), stop 1 ATR, target 3R, 10 days."""
    price, a = tech["price"], tech["atr"]
    if not a or not price:
        return None
    entry = round(price, 2)
    stop = math.floor((entry - MODEL["stop_atr"] * a) * 100) / 100
    risk = entry - stop
    if risk <= 0:
        return None
    target = round(entry + MODEL["target_r"] * risk, 2)
    cash = cfg["budget"] * (1 - cfg["cash_buffer_pct"] / 100.0)
    cap = min(cash, cfg["budget"] * cfg["max_position_pct"] / 100.0)
    shares = max(0, min(int(cfg["max_loss_per_trade_usd"] // risk), int(cap // entry)))
    rsi = tech.get("rsi")
    return {"trigger": entry, "entry": entry, "stop": stop, "target": target, "risk_per_share": round(risk, 2),
            "rr": MODEL["target_r"], "horizon": MODEL["horizon"], "shares": shares, "cost": round(shares * entry, 2),
            "max_loss": round(shares * risk, 2), "potential": round(shares * (target - entry), 2),
            "basis": "מודל מכירת יתר: כניסה בפתיחה הבאה", "active": rsi is not None and rsi < MODEL["rsi_max"],
            "resistance_in_way": None, "budget": cfg["budget"], "max_loss_cfg": cfg["max_loss_per_trade_usd"]}


def raw(d, k):
    x = (d or {}).get(k)
    if isinstance(x, dict):
        return x.get("raw")
    return x


def fundamentals(sym):
    c = crumb()
    mods = ",".join(["price", "summaryDetail", "financialData", "defaultKeyStatistics", "recommendationTrend",
                     "earningsTrend", "calendarEvents", "earnings", "upgradeDowngradeHistory", "assetProfile",
                     "earningsHistory"])
    q = "/v10/finance/quoteSummary/%s?modules=%s" % (ysym(sym), mods) + ("&crumb=%s" % urllib.parse.quote(c) if c else "")
    try:
        doc = yahoo(q)
    except RuntimeError:
        c = crumb()  # the crumb went stale: fetch a fresh one and try once more
        if not c:
            raise
        doc = yahoo("/v10/finance/quoteSummary/%s?modules=%s&crumb=%s" % (ysym(sym), mods, urllib.parse.quote(c)))
    res = ((doc.get("quoteSummary") or {}).get("result") or [None])[0]
    if not res:
        raise RuntimeError("no fundamentals")
    pr, sd, fd, ks = res.get("price", {}), res.get("summaryDetail", {}), res.get("financialData", {}), res.get("defaultKeyStatistics", {})
    prof = res.get("assetProfile", {})
    out = {
        "name": pr.get("longName") or pr.get("shortName") or sym, "exchange": pr.get("exchangeName"),
        "currency": pr.get("currency"), "sector": prof.get("sector"), "industry": prof.get("industry"),
        "summary": (prof.get("longBusinessSummary") or "")[:600], "employees": prof.get("fullTimeEmployees"),
        "website": prof.get("website"),
        "market_cap": raw(pr, "marketCap"), "pe": raw(sd, "trailingPE"), "fwd_pe": raw(sd, "forwardPE") or raw(ks, "forwardPE"),
        "peg": raw(ks, "pegRatio"), "ps": raw(sd, "priceToSalesTrailing12Months"), "pb": raw(ks, "priceToBook"),
        "beta": raw(sd, "beta"), "div_yield": raw(sd, "dividendYield"), "short_pct": raw(ks, "shortPercentOfFloat"),
        "inst_pct": raw(ks, "heldPercentInstitutions"),
        "eps_ttm": raw(ks, "trailingEps"), "eps_fwd": raw(ks, "forwardEps"),
        "revenue": raw(fd, "totalRevenue"), "rev_growth": raw(fd, "revenueGrowth"), "earn_growth": raw(fd, "earningsGrowth"),
        "gross_margin": raw(fd, "grossMargins"), "op_margin": raw(fd, "operatingMargins"), "net_margin": raw(fd, "profitMargins"),
        "roe": raw(fd, "returnOnEquity"), "debt_to_equity": raw(fd, "debtToEquity"), "cash": raw(fd, "totalCash"),
        "debt": raw(fd, "totalDebt"), "fcf": raw(fd, "freeCashflow"), "current_ratio": raw(fd, "currentRatio"),
        "analyst": {"mean": raw(fd, "targetMeanPrice"), "high": raw(fd, "targetHighPrice"), "low": raw(fd, "targetLowPrice"),
                    "median": raw(fd, "targetMedianPrice"), "n": raw(fd, "numberOfAnalystOpinions"),
                    "key": fd.get("recommendationKey"), "score": raw(fd, "recommendationMean")},
    }
    rt = (res.get("recommendationTrend") or {}).get("trend") or []
    out["rec_trend"] = [{"period": x.get("period"), "strongBuy": x.get("strongBuy", 0), "buy": x.get("buy", 0),
                         "hold": x.get("hold", 0), "sell": x.get("sell", 0), "strongSell": x.get("strongSell", 0)} for x in rt]
    ud = (res.get("upgradeDowngradeHistory") or {}).get("history") or []
    out["rating_changes"] = [{"date": datetime.datetime.utcfromtimestamp(x["epochGradeDate"]).strftime("%Y-%m-%d"),
                              "firm": x.get("firm"), "to": x.get("toGrade"), "from": x.get("fromGrade"), "action": x.get("action")}
                             for x in ud[:8] if x.get("epochGradeDate")]
    et = (res.get("earningsTrend") or {}).get("trend") or []
    out["estimates"] = [{"period": x.get("period"), "end": x.get("endDate"), "growth": raw(x, "growth"),
                         "eps_avg": raw(x.get("earningsEstimate"), "avg"), "eps_low": raw(x.get("earningsEstimate"), "low"),
                         "eps_high": raw(x.get("earningsEstimate"), "high"), "eps_year_ago": raw(x.get("earningsEstimate"), "yearAgoEps"),
                         "rev_avg": raw(x.get("revenueEstimate"), "avg"), "rev_growth": raw(x.get("revenueEstimate"), "growth"),
                         "n": raw(x.get("earningsEstimate"), "numberOfAnalysts")} for x in et]
    ce = (res.get("calendarEvents") or {}).get("earnings") or {}
    ed = ce.get("earningsDate") or []
    out["next_earnings"] = datetime.datetime.utcfromtimestamp(ed[0]["raw"]).strftime("%Y-%m-%d") if ed and isinstance(ed[0], dict) else None
    q = ((res.get("earnings") or {}).get("financialsChart") or {}).get("quarterly") or []
    out["quarterly"] = [{"q": x.get("date"), "revenue": raw(x, "revenue"), "earnings": raw(x, "earnings")} for x in q]
    eh = (res.get("earningsHistory") or {}).get("history") or []
    out["eps_history"] = [{"q": (x.get("quarter") or {}).get("fmt"), "actual": raw(x, "epsActual"), "estimate": raw(x, "epsEstimate"),
                           "surprise": raw(x, "surprisePercent")} for x in eh]
    return out


NEWS_TAGS = [("earnings", r"earnings|results|quarter|guidance|EPS|revenue"), ("analyst", r"upgrade|downgrade|price target|analyst|rating"),
             ("deal", r"acqui|merger|deal|buyout|partnership"), ("legal", r"lawsuit|sue|probe|investigat|SEC |DOJ|antitrust"),
             ("capital", r"offering|buyback|dividend|split"), ("people", r"CEO|CFO|resign|appoint")]


def news(sym):
    xml = get("https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US" % ysym(sym))
    root = ET.fromstring(xml)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        when = it.findtext("pubDate")
        try:
            ts = datetime.datetime.strptime(when[:25], "%a, %d %b %Y %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts = when
        out.append({"title": title, "link": it.findtext("link"), "time": ts,
                    "tags": [k for k, rx in NEWS_TAGS if re.search(rx, title, re.I)]})
    return out[:12]


def chart(sym):
    doc = yahoo("/v8/finance/chart/%s?range=2y&interval=1d&includePrePost=false" % ysym(sym))
    res = doc["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = [(t, o, h, l, c, v) for t, o, h, l, c, v in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"])
            if None not in (o, h, l, c)]
    return res.get("meta", {}), rows


def extended(sym, avg_vol=None):
    """Pre-market / after-hours snapshot from today's 5-minute bars (Yahoo includePrePost)."""
    doc = yahoo("/v8/finance/chart/%s?range=1d&interval=5m&includePrePost=true" % ysym(sym))
    res = doc["chart"]["result"][0]
    meta = res.get("meta", {})
    if meta.get("instrumentType") in ("CRYPTOCURRENCY",):
        return None
    tp = meta.get("currentTradingPeriod") or {}
    q = res["indicators"]["quote"][0]
    ts = res.get("timestamp") or []
    now = time.time()
    pre, reg, post = tp.get("pre") or {}, tp.get("regular") or {}, tp.get("post") or {}
    if pre.get("start") and pre["start"] <= now < pre.get("end", 0):
        session, lo_t, hi_t = "pre", pre["start"], pre["end"]
    elif reg.get("start") and reg["start"] <= now < reg.get("end", 0):
        session, lo_t, hi_t = "regular", reg["start"], reg["end"]
    elif post.get("start") and post["start"] <= now < post.get("end", 0):
        session, lo_t, hi_t = "post", post["start"], post["end"]
    else:
        # market closed (night / weekend): show the latest after-hours of the last session, if any
        session, lo_t, hi_t = "post", post.get("start", 0), post.get("end", 0)
    bars = [(ts[i], q["close"][i], q["high"][i], q["low"][i], q["volume"][i] or 0)
            for i in range(len(ts)) if lo_t <= ts[i] < hi_t and q["close"][i] is not None]
    if session == "regular" or not bars:
        return {"session": session, "active": False}
    prev = meta.get("regularMarketPrice")  # until the open this is still the last regular close
    price = bars[-1][1]
    vol = sum(b[4] for b in bars)
    chg = (price / prev - 1) * 100 if prev else None
    return {"session": session, "active": True, "label": "Pre-market" if session == "pre" else "After-hours",
            "price": r2(price), "ref": r2(prev), "change": r2(chg), "high": r2(max(b[2] or b[1] for b in bars)),
            "low": r2(min(b[3] or b[1] for b in bars)), "volume": int(vol),
            "vol_pct": r2(vol / avg_vol * 100, 1) if avg_vol else None, "gap": chg is not None and abs(chg) >= 3,
            "time": datetime.datetime.utcfromtimestamp(bars[-1][0]).strftime("%H:%M UTC"),
            "series": [[b[0], r2(b[1])] for b in bars]}


def ext_alerts(ext, plan):
    out = []
    if not ext or not ext.get("active"):
        return out
    word = "ב-pre-market" if ext["session"] == "pre" else "במסחר המאוחר"
    if ext.get("gap"):
        out.append({"level": "warn", "text": "פער %s: המחיר %s %+.1f%% מהסגירה (%.2f)." % (
            "למעלה" if ext["change"] > 0 else "למטה", word, ext["change"], ext["price"])})
    if plan:
        if plan.get("active"):
            out.append({"level": "info", "text": "הכניסה במודל היא בפתיחה, כך שהמחיר %s (%.2f) הוא בערך מחיר הכניסה. הסטופ והיעד יחושבו מחדש מהמחיר בפתיחה." % (
                word, ext["price"])})
        if ext["price"] <= plan["stop"]:
            out.append({"level": "bad", "text": "המחיר %s (%.2f) כבר מתחת לסטופ %.2f: התוכנית לא רלוונטית לפתיחה." % (
                word, ext["price"], plan["stop"])})
    return out


def verdict(tech, fund):
    """A plain-language summary built from rules. Not advice."""
    parts, pts = [], 0
    ts = tech["tech_score"]
    pts += (ts - 50) / 25
    parts.append("טכנית: מגמה %s, ציון %d/100." % (tech["trend"], ts))
    a = (fund or {}).get("analyst") or {}
    if a.get("mean") and tech["price"]:
        up = (a["mean"] / tech["price"] - 1) * 100
        pts += max(-1, min(1, up / 20))
        parts.append("אנליסטים: יעד ממוצע %.2f (%+.0f%%), המלצה %s." % (a["mean"], up, REC_HE.get(a.get("key"), a.get("key") or "-")))
    if fund and fund.get("rev_growth") is not None:
        g = fund["rev_growth"] * 100
        pts += 0.5 if g > 10 else -0.5 if g < 0 else 0
        parts.append("צמיחת הכנסות %+.0f%% בשנה." % g)
    if fund and fund.get("next_earnings"):
        try:
            days = (datetime.date.fromisoformat(fund["next_earnings"]) - datetime.date.today()).days
            if 0 <= days <= 14:
                parts.append("שים לב: דוח כספי בעוד %d ימים, המניה עלולה לפתוח בפער." % days)
        except Exception:
            pass
    label = "חיובי" if pts >= 1 else "חיובי קל" if pts >= 0.3 else "שלילי" if pts <= -1 else "שלילי קל" if pts <= -0.3 else "ניטרלי"
    return {"label": label, "points": round(pts, 2), "text": " ".join(parts)}


REC_HE = {"strong_buy": "קנייה חזקה", "buy": "קנייה", "hold": "החזק", "underperform": "תת-ביצוע", "sell": "מכירה", "none": "-"}

_cache = {}       # sym -> (time, tech part)
_full = {}        # sym -> (time, full analysis)
_fund_cache = {}  # sym -> (time, fund)
_pool = ThreadPoolExecutor(max_workers=10)


def tech_part(sym, force=False):
    hit = _cache.get(sym)
    if hit and not force and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    meta, rows = demo_chart(sym) if DEMO else chart(sym)
    rows, dq_issues = DQ.clean_bars(rows)
    if len(rows) < 60:
        raise RuntimeError("not enough price history for %s (%d sessions)" % (sym, len(rows)))
    t = [r[0] for r in rows]
    o, h, l, c, v = ([float(r[i] or 0) for r in rows] for i in range(1, 6))
    tech = technicals(t, o, h, l, c, v)
    tech["volume"] = volume_analysis(t, o, h, l, c, v)
    reg = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    partial = bool(reg.get("start") and reg["start"] <= time.time() < reg.get("end", 0) and t[-1] >= reg["start"])
    tech["bar_partial"] = partial
    try:
        tech["fx"], tech["overlays"] = fx_block(t, o, h, l, c, v, partial)
    except Exception as e:
        tech["fx"], tech["overlays"] = None, None
        print("fx failed for %s: %s" % (sym, e))
    sess = us_session_open()
    data = DQ.check(rows, meta, session_open=sess, bar_partial=partial)
    data["issues"] = dq_issues + data["issues"]
    if any(k == "bad" for k, _ in data["issues"]):
        data["level"] = "bad"
    elif data["issues"] and data["level"] == "ok":
        data["level"] = "partial"
    data["label"] = DQ.LEVELS[data["level"]]
    tech["data"] = data
    tech["live"] = {"price": meta.get("regularMarketPrice"), "session_open": bool(sess and partial)}
    fx = tech.get("fx")
    if fx:
        # one definition everywhere: oscillators and ATR from CLOSED bars (the list, the card and the plan agree)
        for k in ("rsi", "cci", "atr", "rvol"):
            if fx.get(k) is not None:
                tech[k] = round(fx[k], 2)
        lv = (tech.get("overlays") or {}).get("levels")
        if lv:
            tech["levels"] = lv
            tech["support"] = [{"price": x["price"], "touches": x["touches"]} for x in lv["support"]]
            tech["resistance"] = [{"price": x["price"], "touches": x["touches"]} for x in lv["resistance"]]
    try:
        tech["asset"] = MK.asset_type(sym, meta)
    except Exception:
        tech["asset"] = None
    try:
        ext = demo_ext(sym, c[-1], tech["volume"]["avg20"]) if DEMO else extended(sym, tech["volume"]["avg20"])
    except Exception:
        ext = None
    out = {"ext": ext, "symbol": sym, "as_of": datetime.datetime.utcfromtimestamp(t[-1]).strftime("%Y-%m-%d"),
           "fetched": datetime.datetime.now().strftime("%H:%M"), "market_price": meta.get("regularMarketPrice"),
           "candles": [[t[i], r2(o[i]), r2(h[i]), r2(l[i]), r2(c[i]), int(v[i])] for i in range(len(t))], "tech": tech}
    _cache[sym] = (time.time(), out)
    return out


def get_fund(sym, force=False):
    hit = _fund_cache.get(sym)
    if hit and not force and time.time() - hit[0] < 3 * CACHE_TTL:
        return hit[1]
    f = demo_fund(sym, tech_part(sym)["tech"]["price"]) if DEMO else fundamentals(sym)
    _fund_cache[sym] = (time.time(), f)
    return f


def analyze(sym, force=False):
    sym = sym.upper().strip()
    hit = _full.get(sym)
    if hit and not force and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    ff = _pool.submit(get_fund, sym, force)
    fn = _pool.submit(lambda: demo_news(sym) if DEMO else news(sym))
    base = tech_part(sym, force)
    out = dict(base, errors=[])
    out["plan"] = trade_plan(base["tech"], load_config())
    try:
        out["fund"] = ff.result(timeout=25)
    except Exception as e:
        out["fund"] = None
        out["errors"].append("נתונים פונדמנטליים ואנליסטים לא זמינים כרגע (%s)" % (str(e)[:80] or "timeout"))
    try:
        out["news"] = fn.result(timeout=15)
    except Exception as e:
        out["news"] = []
        out["errors"].append("חדשות לא זמינות (%s)" % (str(e)[:80] or "timeout"))
    out["verdict"] = verdict(base["tech"], out["fund"])
    out["tm"] = get_tm(sym, base, out["plan"])
    out["sec"] = None
    ua = os.environ.get("SEC_USER_AGENT") or load_config().get("sec_user_agent")
    if extras and ua and not DEMO:
        try:
            out["sec"] = extras.sec_block(sym, ua)
        except Exception as e:
            out["errors"].append("נתוני SEC לא זמינים (%s)" % str(e)[:80])
    out["decision"] = decision(base["tech"], base["tech"]["volume"], out["fund"], out["plan"], out["tm"], out["sec"])
    out["decision"]["ext_alerts"] = ext_alerts(base.get("ext"), out["plan"])
    out["score"] = score(base["tech"], out["fund"], out["sec"], base.get("ext"), out["plan"])
    cs = base["candles"]
    out["sector"] = sector_part(out["fund"], ([x[0] for x in cs], [x[4] for x in cs]))
    out["market"] = {k: (market_ctx() or {}).get(k) for k in ("regime", "vix")}
    out["ind_score"] = ind_score(base["tech"], out["plan"], out["fund"], out["tm"], sector=out["sector"])
    try:
        out["peers"] = peers_for(sym)
    except Exception:
        out["peers"] = None
    out["mc"] = (EN.TABLE or {}).get("mc")
    if out["score"]:
        out["decision"]["code"] = out["score"]["code"]
    out["event"] = event_for(sym, base["candles"], out["sec"], out["news"], out["fund"])
    try:
        out["oneil"] = oneil_block(sym, base["candles"], out["fund"], out["sec"])
    except Exception as e:
        out["oneil"] = None
        out["errors"].append("ניתוח O'Neil נכשל (%s)" % str(e)[:80])
    out["market"]["oneil"] = (market_ctx() or {}).get("oneil_m")
    hist, plan2, dec = engine_block(base["tech"], out["fund"], out["sec"], base.get("ext"), out["ind_score"], out["event"], sym=sym)
    if hist is not None:
        out["score_v1"], out["score"], out["plan2"] = out["score"], hist, plan2
        dec["ext_alerts"] = ext_alerts(base.get("ext"), v1_plan(plan2))
        out["decision_v1"], out["decision"] = out["decision"], dec
        # one plan on the page: the v1-shaped plan (chart line, Time Machine) is now the v3 plan
        out["plan"] = v1_plan(plan2)
        out["tm"] = get_tm(sym, base, out["plan"]) if out["plan"] else None
    _full[sym] = (time.time(), out)
    return out


_tm_cache = {}


def v1_plan(p):
    """The v3 plan in the older shape the Time Machine and the alerts read. None without a structural stop."""
    if not p or not p.get("stop"):
        return None
    sz = p.get("size") or {}
    risk = p["entry"] - p["stop"]
    return {"trigger": p["entry"], "entry": p["entry"], "stop": p["stop"], "target": p["t2"], "risk_per_share": round(risk, 2),
            "rr": 3, "horizon": p["horizon"], "shares": sz.get("shares", 0), "cost": sz.get("cost", 0),
            "max_loss": sz.get("risk_usd", 0), "potential": round(sz.get("shares", 0) * (p["t2"] - p["entry"]), 2),
            "basis": "תוכנית v3: סטופ מבני, יעד 3R, 10 ימים", "active": True, "resistance_in_way": p.get("res_near") if p.get("res_blocks") else None,
            "budget": sz.get("budget"), "max_loss_cfg": sz.get("max_loss_cfg")}


def get_tm(sym, base, plan):
    if not extras or not plan:
        return None
    key = (sym, plan["trigger"], plan["stop"], plan["target"], base["as_of"])
    if key not in _tm_cache:
        try:
            _tm_cache[key] = extras.time_machine(base["candles"], plan, horizon=plan.get("horizon", 20))
        except Exception:
            _tm_cache[key] = None
    return _tm_cache[key]


def quick(sym, force=False):
    """One overview card: price data only, so it is fast. Analyst upside if already known."""
    sym = sym.upper().strip()
    try:
        a = tech_part(sym, force)
        t = a["tech"]
        f = (_fund_cache.get(sym) or (0, {}))[1] or {}
        an = f.get("analyst") or {}
        label = verdict(t, f if f else None)["label"]
        pl = trade_plan(t, load_config())
        tm = get_tm(sym, a, pl)
        full = (_full.get(sym) or (0, {}))[1]
        dec = decision(t, t["volume"], f if f else None, pl, tm, full.get("sec"))
        sc = score(t, f if f else None, full.get("sec"), a.get("ext"), pl)
        isc = (full.get("ind_score") if full else None) or ind_score(t, pl, f if f else None, tm)
        if sc:
            dec["code"] = sc["code"]
        if t.get("fx"):
            evq = full.get("event") if full else event_for(sym, a["candles"])
            hv, _, d2 = engine_block(t, f if f else None, full.get("sec"), a.get("ext"), isc, evq, sym=sym)
            sc = hv
            dec = {"verdict": d2["verdict"], "code": d2["code"], "score": None}
        return {"symbol": sym, "price": t["price"], "change_1d": t["change_1d"], "change_1m": t["change_1m"],
                "rsi": t["rsi"], "cci": t["cci"], "rvol": t["rvol"], "trend": t["trend"], "tech_score": t["tech_score"],
                "verdict": label, "target_up": r2((an["mean"] / t["price"] - 1) * 100, 1) if an.get("mean") else None,
                "spark": [x[4] for x in a["candles"][-60:]], "decision": dec["verdict"], "decision_code": dec["code"],
                "decision_score": dec["score"], "channel": t["channel"]["kind"], "name": f.get("name"),
                "tm": {"p_target": tm["p_target"], "lean": tm["lean"]} if tm else None,
                "score": sc["value"] if sc else None, "score_label": sc["label"] if sc else None, "act": dec["verdict"], "act_code": d2["code"] if t.get("fx") else None,
                "event": (evq or {}).get("level") if t.get("fx") else None,
                "on": ((full.get("oneil") if full else None) or {}).get("status"),
                "p_profit": (sc.get("p_profit") if "p_profit" in sc else (sc.get("hist") or {}).get("p_profit")) if sc else None,
                "ind": isc["value"] if isc else None, "rel": ((sc or {}).get("reliability") or {}).get("level"),
                "ext": {k: (a.get("ext") or {}).get(k) for k in ("session", "active", "price", "change", "gap", "volume", "vol_pct")}
                if (a.get("ext") or {}).get("active") else None}
    except Exception as e:
        return {"symbol": sym, "error": str(e)[:120]}


# ---------------------------------------------------------------- demo data (no internet)

def demo_chart(sym):
    rnd = random.Random(sym)
    p = rnd.uniform(20, 400)
    drift = rnd.uniform(-0.0006, 0.0015)
    t0 = int(time.time()) - 700 * 86400
    rows, d = [], 0
    for i in range(700):
        ts = t0 + i * 86400
        if datetime.datetime.utcfromtimestamp(ts).weekday() >= 5:
            continue
        o = p * (1 + rnd.gauss(0, 0.004))
        c = o * (1 + rnd.gauss(drift, 0.018))
        h = max(o, c) * (1 + abs(rnd.gauss(0, 0.007)))
        l = min(o, c) * (1 - abs(rnd.gauss(0, 0.007)))
        rows.append((ts, o, h, l, c, int(rnd.uniform(2e6, 9e6) * (1.8 if rnd.random() < 0.05 else 1))))
        p = c
    return {"regularMarketPrice": rows[-1][4]}, rows[-504:]


def demo_fund(sym, price):
    rnd = random.Random(sym + "f")
    tgt = price * rnd.uniform(0.85, 1.35)
    qs = ["3Q2025", "4Q2025", "1Q2026", "2Q2026"]
    return {"name": sym + " Corporation", "exchange": "NasdaqGS", "currency": "USD", "sector": "Technology",
            "industry": "Semiconductors", "summary": "Demo company used to preview the dashboard layout without internet access.",
            "market_cap": rnd.uniform(5e9, 2e12), "pe": rnd.uniform(12, 60), "fwd_pe": rnd.uniform(10, 45), "peg": rnd.uniform(0.6, 3),
            "ps": rnd.uniform(1, 20), "pb": rnd.uniform(1, 30), "beta": rnd.uniform(0.6, 2.2), "div_yield": rnd.choice([None, 0.012]),
            "short_pct": rnd.uniform(0.005, 0.12), "eps_ttm": rnd.uniform(1, 12), "eps_fwd": rnd.uniform(1, 14),
            "revenue": rnd.uniform(1e9, 3e11), "rev_growth": rnd.uniform(-0.1, 0.5), "earn_growth": rnd.uniform(-0.3, 0.8),
            "gross_margin": rnd.uniform(0.3, 0.75), "op_margin": rnd.uniform(0.05, 0.45), "net_margin": rnd.uniform(0.02, 0.35),
            "roe": rnd.uniform(0.05, 0.6), "debt_to_equity": rnd.uniform(10, 200), "cash": rnd.uniform(1e9, 6e10),
            "debt": rnd.uniform(1e9, 1e11), "fcf": rnd.uniform(-1e9, 5e10), "current_ratio": rnd.uniform(0.8, 3),
            "analyst": {"mean": tgt, "high": tgt * 1.3, "low": tgt * 0.7, "median": tgt, "n": rnd.randint(8, 50),
                        "key": rnd.choice(["strong_buy", "buy", "hold"]), "score": rnd.uniform(1.4, 3)},
            "rec_trend": [{"period": p, "strongBuy": rnd.randint(2, 12), "buy": rnd.randint(5, 20), "hold": rnd.randint(2, 12),
                           "sell": rnd.randint(0, 3), "strongSell": rnd.randint(0, 1)} for p in ("0m", "-1m", "-2m", "-3m")],
            "rating_changes": [{"date": "2026-10-%02d" % (28 - i * 4), "firm": f, "to": "Buy", "from": "Hold", "action": "up"}
                               for i, f in enumerate(["Morgan Stanley", "Goldman Sachs", "Jefferies"])],
            "estimates": [{"period": p, "eps_avg": rnd.uniform(1, 4), "eps_low": 1, "eps_high": 5, "eps_year_ago": rnd.uniform(1, 3),
                           "rev_avg": rnd.uniform(1e9, 8e10), "rev_growth": rnd.uniform(-0.05, 0.4), "growth": rnd.uniform(-0.1, 0.5),
                           "n": rnd.randint(5, 40)} for p in ("0q", "+1q", "0y", "+1y")],
            "next_earnings": (datetime.date.today() + datetime.timedelta(days=rnd.randint(3, 80))).isoformat(),
            "quarterly": [{"q": q, "revenue": rnd.uniform(5e9, 3e10) * (1 + i * 0.05), "earnings": rnd.uniform(5e8, 6e9)} for i, q in enumerate(qs)],
            "eps_history": [{"q": q, "actual": rnd.uniform(1, 3), "estimate": rnd.uniform(1, 3), "surprise": rnd.uniform(-0.1, 0.2)} for q in qs]}


def demo_ext(sym, close, avg):
    rnd = random.Random(sym + "x")
    chg = rnd.gauss(0, 2.2)
    t0 = int(time.time()) - 3 * 3600
    series = [[t0 + i * 300, round(close * (1 + chg / 100 * i / 30 + rnd.gauss(0, .002)), 2)] for i in range(31)]
    vol = int(avg * rnd.uniform(0.01, 0.15))
    return {"session": "pre", "active": True, "label": "Pre-market", "price": series[-1][1], "ref": round(close, 2),
            "change": round((series[-1][1] / close - 1) * 100, 2), "high": max(x[1] for x in series), "low": min(x[1] for x in series),
            "volume": vol, "vol_pct": round(vol / avg * 100, 1), "gap": abs(series[-1][1] / close - 1) >= .03,
            "time": "12:35 UTC", "series": series}


def demo_news(sym):
    return [{"title": "%s beats quarterly earnings estimates, raises guidance" % sym, "link": "#", "time": "2026-11-05 16:05", "tags": ["earnings"]},
            {"title": "Analyst raises %s price target after strong demand" % sym, "link": "#", "time": "2026-11-04 09:12", "tags": ["analyst"]},
            {"title": "%s announces new partnership in AI infrastructure" % sym, "link": "#", "time": "2026-11-03 13:40", "tags": ["deal"]}]


# ---------------------------------------------------------------- watchlist + server

def read_watchlist():
    try:
        with open(WATCHLIST, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {"symbols": d}
    except Exception:
        return {"symbols": ["AAPL", "TSLA", "NVDA"], "source": "default"}


def write_watchlist(doc):
    with open(WATCHLIST, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        force = q.get("refresh", ["0"])[0] == "1"
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "dashboard.html"), "rb") as fh:
                    return self.send(200, fh.read(), "text/html; charset=utf-8")
            if u.path == "/lwc.js":
                with open(os.path.join(HERE, "lwc.js"), "rb") as fh:
                    return self.send(200, fh.read(), "application/javascript; charset=utf-8")
            if u.path == "/api/watchlist":
                return self.send(200, read_watchlist())
            if u.path == "/api/quick":
                return self.send(200, quick(q["symbol"][0], force))
            if u.path == "/api/meta":
                return self.send(200, {"watchlist": read_watchlist(), "demo": DEMO})
            if u.path == "/api/analyze":
                return self.send(200, analyze(q["symbol"][0], force))
            if u.path == "/api/config":
                return self.send(200, load_config())
            if u.path == "/api/portfolio":
                return self.send(200, portfolio_view(force))
            return self.send(404, {"error": "not found"})
        except Exception as e:
            return self.send(500, {"error": str(e)})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/watchlist":
            syms = []
            for s in body.get("symbols", []):
                s = re.sub(r"[^A-Z0-9.\-]", "", str(s).upper().split(":")[-1])
                if s and s not in syms:
                    syms.append(s)
            doc = read_watchlist()
            doc["symbols"] = syms
            write_watchlist(doc)
            return self.send(200, doc)
        if self.path == "/api/config":
            cfg = load_config()
            for k in ("budget", "max_loss_per_trade_usd", "target_r"):
                if k in body:
                    cfg[k] = float(body[k])
            with open(CONFIG, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            _cache.clear()
            return self.send(200, cfg)
        return self.send(404, {"error": "not found"})


def main():
    global DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    DEMO = a.demo
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = "http://127.0.0.1:%d/" % a.port
    print("Trader Dashboard running at %s  (close this window to stop)" % url)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
