"""Market-data checks for Trader Dash v3. Standard library only.

Every analysis carries a `data` block saying where each number comes from, when it was updated and
whether it can be trusted. Nothing is filled in by estimate: a bad bar is dropped and reported, a
missing value stays missing, and the analysis is marked "partial" or "not verified".

Source: Yahoo v8 chart, daily bars. Open/high/low/close and volume are adjusted for SPLITS by Yahoo;
dividends are NOT adjusted (the research uses the same bars, so the two agree).
"""
import datetime

try:
    import zoneinfo
    NY = zoneinfo.ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY = datetime.timezone(datetime.timedelta(hours=-5))

LEVELS = {"ok": "מאומת", "partial": "חלקי", "bad": "לא מאומת"}


def clean_bars(rows):
    """rows [(t, o, h, l, c, v)] -> (clean rows, issues). One bar per New York calendar day (the last one wins),
    bars with non-positive prices or an impossible high/low are dropped, not repaired."""
    issues = []
    by_day = {}
    dups = 0
    for r in rows:
        d = datetime.datetime.fromtimestamp(r[0], NY).date()
        if d in by_day:
            dups += 1
        by_day[d] = r
    if dups:
        issues.append(("warn", "%d נרות כפולים לאותו יום הוסרו" % dups))
    out, bad = [], 0
    for d in sorted(by_day):
        t, o, h, l, c, v = by_day[d]
        if None in (o, h, l, c) or min(o, h, l, c) <= 0:
            bad += 1
            continue
        eps = 1e-6 * c
        if l > min(o, c) + eps or h < max(o, c) - eps or h < l:
            bad += 1
            continue
        out.append((t, o, h, l, c, v or 0))
    if bad:
        issues.append(("bad" if bad > 3 else "warn", "%d נרות עם מחיר לא תקין (גבוה/נמוך לא מתאימים לפתיחה/סגירה) הוסרו" % bad))
    return out, issues


def weekdays_between(d0, d1):
    """Weekdays strictly after d0 up to and including d1."""
    n, d = 0, d0
    while d < d1:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def check(rows, meta=None, now=None, session_open=False, bar_partial=False):
    """Quality report for one symbol's daily bars (already cleaned)."""
    meta = meta or {}
    now = now or datetime.datetime.now(datetime.timezone.utc)
    issues = []
    n = len(rows)
    if n == 0:
        return {"level": "bad", "label": LEVELS["bad"], "issues": [("bad", "אין נתוני מחיר")], "bars": 0}
    last_t = rows[-1][0]
    last_d = datetime.datetime.fromtimestamp(last_t, NY).date()
    today = now.astimezone(NY).date()
    lag = weekdays_between(last_d, today)
    closed_last = rows[-2][0] if bar_partial and n > 1 else last_t
    closed_d = datetime.datetime.fromtimestamp(closed_last, NY).date()
    if lag > 3:
        issues.append(("bad", "הנר האחרון מ-%s: הנתונים ישנים (%d ימי מסחר)" % (last_d.isoformat(), lag)))
    elif lag > 1 and not session_open:
        issues.append(("warn", "הנר האחרון מ-%s (ייתכן יום חג)" % last_d.isoformat()))
    if n < 60:
        issues.append(("bad", "רק %d ימי מסחר: מעט מדי לניתוח" % n))
    elif n < 200:
        issues.append(("warn", "רק %d ימי מסחר: אין ממוצע 200, וחלק מהמדדים חלקיים" % n))
    elif n < 252:
        issues.append(("warn", "פחות משנת מסחר (%d ימים)" % n))
    jumps = [i for i in range(1, n) if rows[i - 1][4] > 0 and abs(rows[i][4] / rows[i - 1][4] - 1) > 0.45]
    if jumps:
        d = datetime.datetime.fromtimestamp(rows[jumps[-1]][0], NY).date().isoformat()
        issues.append(("warn", "%d תנועות יומיות חריגות (מעל 45%%), האחרונה ב-%s: לבדוק אם זה אירוע אמיתי או פיצול שלא תואם" % (len(jumps), d)))
    zero = sum(1 for r in rows[-20:] if not r[5])
    if zero >= 3:
        issues.append(("warn", "%d ימים בלי מחזור ב-20 הימים האחרונים" % zero))
    live_t = meta.get("regularMarketTime")
    live_age = None
    if live_t:
        live_age = (now.timestamp() - live_t) / 60
        if session_open and live_age > 30:
            issues.append(("warn", "המחיר החי לא התעדכן %d דקות" % live_age))
    level = "bad" if any(k == "bad" for k, _ in issues) else "partial" if issues else "ok"
    return {"level": level, "label": LEVELS[level], "issues": issues, "bars": n,
            "first_bar": datetime.datetime.fromtimestamp(rows[0][0], NY).date().isoformat(),
            "last_bar": last_d.isoformat(), "last_closed_bar": closed_d.isoformat(), "bar_partial": bool(bar_partial),
            "live_price": meta.get("regularMarketPrice"),
            "live_time": datetime.datetime.fromtimestamp(live_t, NY).strftime("%Y-%m-%d %H:%M") + " ניו יורק" if live_t else None,
            "live_age_min": round(live_age) if live_age is not None else None,
            "fetched_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
            "source": "Yahoo, נרות יומיים. מתואם לפיצולים, לא לדיבידנדים",
            "indicators_on": "נרות סגורים בלבד" + (" (הנר של היום עוד פתוח ולא נכלל)" if bar_partial else "")}
