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


def decision(tech, vol, fund, plan):
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
    known = [x for x in items if x["ok"] is not None]
    passed = sum(1 for x in known if x["ok"])
    ratio = passed / len(known) if known else 0
    trend_ok = items[0]["ok"]
    room_bad = any(x["label"] in ("מקום עד ההתנגדות", "אין דוח בשבוע הקרוב") and x["ok"] is False for x in items)
    if ratio >= 0.75 and trend_ok and not room_bad:
        verdict, code = "כניסה אפשרית", "go"
    elif ratio >= 0.5:
        verdict, code = "להמתין", "wait"
    else:
        verdict, code = "לא להיכנס", "no"
    fails = [x["label"] for x in items if x["ok"] is False]
    if code == "go" and plan:
        action = "רק אם המחיר עובר את %.2f. סטופ %.2f, יעד %.2f." % (plan["trigger"], plan["stop"], plan["target"])
    elif code == "wait":
        WAIT = {"מגמה ראשית": "המחיר יחזור מעל ממוצע 50", "תעלת מגמה": "התעלה תתהפך לעולה או שהמחיר ירד לתחתית שלה",
                "מומנטום (CCI + MACD)": "CCI יעלה מעל 0 ו-MACD יחצה מעל קו האות", "לא קנייה ביתר": "RSI ירד מתחת ל-70",
                "נפח מאשר": "הנפח בימי עלייה יגבר על ימי הירידה", "מקום עד ההתנגדות": "המחיר יפרוץ את ההתנגדות הקרובה",
                "אין דוח בשבוע הקרוב": "הדוח הכספי יעבור", "אנליסטים": "יעד האנליסטים ישתפר"}
        action = "לחכות עד ש" + ", ".join(WAIT.get(f, f) for f in fails[:3]) + "." if fails else "לחכות לאישור נוסף."
    else:
        action = "רוב הכללים לא מתקיימים: " + ", ".join(fails[:4]) + "."
    return {"verdict": verdict, "code": code, "passed": passed, "total": len(known), "score": round(ratio * 100),
            "items": items, "action": action}


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


def trade_plan(tech, cfg):
    price, a = tech["price"], tech["atr"]
    if not a:
        return None
    trigger = round(max(tech["prev_high"], price) + 0.01, 2)
    breakout = tech["high20"] and tech["high20"] > price and (tech["high20"] - price) < 1.5 * a
    if breakout:
        trigger = round(tech["high20"] + 0.01, 2)
    stop = math.floor((trigger - 1.0 * a) * 100) / 100
    sup = [s["price"] for s in tech["support"] if s["price"] < trigger]
    if sup and trigger - a * 1.6 < sup[0] < stop:
        stop = math.floor((sup[0] - 0.05 * a) * 100) / 100  # tuck the stop under nearby support
    risk = trigger - stop
    target = round(trigger + cfg["target_r"] * risk, 2)
    cash = cfg["budget"] * (1 - cfg["cash_buffer_pct"] / 100.0)
    cap = min(cash, cfg["budget"] * cfg["max_position_pct"] / 100.0)
    shares = max(0, min(int(cfg["max_loss_per_trade_usd"] // risk), int(cap // trigger)))
    res = [r["price"] for r in tech["resistance"]]
    blocker = next((x for x in res if trigger < x < target), None)
    return {"trigger": trigger, "stop": stop, "target": target, "risk_per_share": round(risk, 2),
            "rr": cfg["target_r"], "shares": shares, "cost": round(shares * trigger, 2),
            "max_loss": round(shares * risk, 2), "potential": round(shares * (target - trigger), 2),
            "basis": "פריצה מעל שיא 20 יום" if breakout else "כניסה מעל השיא של אתמול",
            "resistance_in_way": blocker, "budget": cfg["budget"], "max_loss_cfg": cfg["max_loss_per_trade_usd"]}


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
    if len(rows) < 60:
        raise RuntimeError("not enough price history for %s" % sym)
    t = [r[0] for r in rows]
    o, h, l, c, v = ([float(r[i] or 0) for r in rows] for i in range(1, 6))
    tech = technicals(t, o, h, l, c, v)
    tech["volume"] = volume_analysis(t, o, h, l, c, v)
    out = {"symbol": sym, "as_of": datetime.datetime.utcfromtimestamp(t[-1]).strftime("%Y-%m-%d"),
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
    out["decision"] = decision(base["tech"], base["tech"]["volume"], out["fund"], out["plan"])
    _full[sym] = (time.time(), out)
    return out


def quick(sym, force=False):
    """One overview card: price data only, so it is fast. Analyst upside if already known."""
    sym = sym.upper().strip()
    try:
        a = tech_part(sym, force)
        t = a["tech"]
        f = (_fund_cache.get(sym) or (0, {}))[1] or {}
        an = f.get("analyst") or {}
        label = verdict(t, f if f else None)["label"]
        dec = decision(t, t["volume"], f if f else None, trade_plan(t, load_config()))
        return {"symbol": sym, "price": t["price"], "change_1d": t["change_1d"], "change_1m": t["change_1m"],
                "rsi": t["rsi"], "cci": t["cci"], "rvol": t["rvol"], "trend": t["trend"], "tech_score": t["tech_score"],
                "verdict": label, "target_up": r2((an["mean"] / t["price"] - 1) * 100, 1) if an.get("mean") else None,
                "spark": [x[4] for x in a["candles"][-60:]], "decision": dec["verdict"], "decision_code": dec["code"],
                "decision_score": dec["score"], "channel": t["channel"]["kind"], "name": f.get("name")}
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
