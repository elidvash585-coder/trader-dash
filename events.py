"""Major-event layer for Trader Dash. Standard library only.

Three layers, from most to least reliable:
 1. VERIFIED events: SEC 8-K items filed in the last days (CEO/CFO change, restatement, cyber incident,
    bankruptcy, delisting notice, results) and Nasdaq trading halts (public RSS feed).
 2. MARKET REACTION: the stock's move net of the market (beta x SPY), in units of its own recent
    residual volatility (z), with relative volume. The only objective measure of how big news is.
 3. HEADLINES: several headlines on the same serious subject in 24 hours. One headline alone is
    shown as "unverified" and never raises the level by itself.

Severity:
  critical  verified material event, active/recent halt, or an extreme move (|z| >= 4 with RVOL >= 3),
            or a large move (|z| >= 3) together with 2+ serious headlines
  high      results filed, |z| >= 2.5 with RVOL >= 2, 2+ serious headlines, fresh downgrade
  info      anything else (stays in the news card, no banner)
A critical event inside the last 3 sessions suspends entry (decision -> wait) and caps the
historical score at 5: the research does not describe those situations. That 3-session pause is a
caution rule; backtest4.py measures whether it matters (section "after extreme moves").
Nothing here is a sentiment score, and nothing here says sell.
"""
import datetime
import math
import re
import xml.etree.ElementTree as ET

SEC_CRITICAL = {"1.03": "פשיטת רגל", "4.02": "הדוחות הקודמים לא אמינים (הצגה מחדש)", "1.05": "אירוע סייבר מהותי",
                "3.01": "הודעה על מחיקה מהמסחר", "5.02": "שינוי בהנהלה הבכירה (מנכ\"ל / סמנכ\"ל כספים / דירקטורים)",
                "2.06": "ירידת ערך מהותית"}
SEC_HIGH = {"2.02": "פרסום תוצאות כספיות", "1.01": "הסכם מהותי", "2.01": "רכישה או מכירה של נכסים",
            "8.01": "אירוע אחר שהחברה דיווחה עליו"}
SERIOUS = re.compile(r"\b(arrest\w*|charged|indict\w*|fraud|investigat\w*|probe|subpoena|lawsuit|sued|resign\w*|steps? down|"
                     r"fired|ousted|oust|bankrupt\w*|chapter 11|halt\w*|recall\w*|restat\w*|espionage|spy|spying|sanction\w*|"
                     r"delist\w*|guidance cut|cuts? guidance|warns?|plunge\w*|crash\w*|scandal|raid\w*|DOJ|SEC charges|short seller)\b", re.I)
HALT_REASON = {"T1": "עצירה: חדשות בדרך", "T2": "עצירה: חדשות פורסמו", "T3": "חידוש מסחר אחרי חדשות",
               "LUDP": "עצירת תנודתיות (Limit Up/Limit Down)", "LUDS": "עצירת תנודתיות", "H10": "השעיה על ידי ה-SEC",
               "H4": "אי עמידה בדרישות רישום", "H9": "אי עמידה בדרישות רישום", "T5": "עצירה בניירות מסוג מסוים",
               "T12": "עצירה: בקשת מידע נוסף", "M": "עצירת תנודתיות", "MWC1": "עצירת שוק רוחבית"}


# ---------------------------------------------------------------- market reaction

def _day(t):
    return int(t // 86400)


def abnormal(candles, spy_t, spy_c, beta_n=120, vol_n=60, look=3):
    """Last bar's return net of beta x SPY, in units of the residual daily volatility of the previous
    vol_n sessions. candles: [[t,o,h,l,c,v], ...]. Uses the last bar even if it is still trading
    (an event is live information; the volume ratio is then on a partial day, so it errs low)."""
    if not candles or len(candles) < beta_n + 3 or not spy_c:
        return None
    m = {}
    for i in range(1, len(spy_c)):
        if spy_c[i - 1]:
            m[_day(spy_t[i])] = spy_c[i] / spy_c[i - 1] - 1
    rows = []
    for i in range(1, len(candles)):
        p0, p1 = candles[i - 1][4], candles[i][4]
        d = _day(candles[i][0])
        if p0 and d in m:
            rows.append((d, p1 / p0 - 1, m[d], i))
    if len(rows) < beta_n + look + 1:
        return None
    hist = rows[-beta_n - look:-look]           # beta and volatility from BEFORE the bars being judged
    xs, ys = [x[2] for x in hist], [x[1] for x in hist]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx else 1.0
    res = [y - beta * x for x, y in zip(xs[-vol_n:], ys[-vol_n:])]
    mr = sum(res) / len(res)
    sd = math.sqrt(sum((r - mr) ** 2 for r in res) / (len(res) - 1)) if len(res) > 2 else None
    best = None
    for d, r, mk, i in rows[-look:]:            # the most extreme of the last `look` sessions
        resid = r - beta * mk
        vols = [c[5] for c in candles[max(0, i - 20):i] if c[5]]
        rvol = candles[i][5] / (sum(vols) / len(vols)) if vols and candles[i][5] else None
        gap = (candles[i][1] / candles[i - 1][4] - 1) if candles[i - 1][4] else None
        z = resid / sd if sd else None
        cand = {"date": datetime.datetime.utcfromtimestamp(candles[i][0]).strftime("%Y-%m-%d"),
                "move": round(r * 100, 2), "market": round(mk * 100, 2), "beta": round(beta, 2),
                "resid": round(resid * 100, 2), "z": round(z, 1) if z is not None else None,
                "rvol": round(rvol, 1) if rvol else None, "gap": round(gap * 100, 2) if gap is not None else None,
                "last": i == len(candles) - 1}
        if best is None or abs(z or 0) > abs(best["z"] or 0):
            best = cand
    return best


# ---------------------------------------------------------------- Nasdaq halts

HALTS_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"


def parse_halts(xml_text):
    """{SYMBOL: [halt, ...]} from the Nasdaq Trader trade-halts RSS (all US-listed securities)."""
    out = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for it in root.iter("item"):
        f = {}
        for ch in it:
            f[ch.tag.split("}")[-1]] = (ch.text or "").strip()
        sym = (f.get("IssueSymbol") or "").upper()
        if not sym:
            continue
        code = f.get("ReasonCode", "")
        out.setdefault(sym, []).append({"date": f.get("HaltDate"), "time": f.get("HaltTime"), "code": code,
                                        "reason": HALT_REASON.get(code, code), "resume_date": f.get("ResumptionDate") or None,
                                        "resume_time": f.get("ResumptionTradeTime") or None, "name": f.get("IssueName")})
    return out


# ---------------------------------------------------------------- headlines

def headline_cluster(news, now=None, hours=24):
    """Serious headlines in the last `hours` hours. Duplicates (same title) are counted once."""
    now = now or datetime.datetime.utcnow()
    seen, hits = set(), []
    for n in news or []:
        try:
            t = datetime.datetime.strptime(n.get("time") or "", "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if (now - t).total_seconds() > hours * 3600:
            continue
        title = (n.get("title") or "").strip()
        key = re.sub(r"\W+", " ", title.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        if SERIOUS.search(title):
            hits.append({"title": title, "time": n.get("time"), "link": n.get("link")})
    return hits


# ---------------------------------------------------------------- the assessment

def assess(sym, candles=None, spy=None, sec=None, halts=None, news=None, fund=None, today=None, now=None):
    """Returns None when there is nothing above 'info', else the event block for the banner."""
    today = today or datetime.date.today()
    facts, sources, level = [], [], 0          # 2 critical, 1 high
    verified = False
    # 1a. SEC filings (last 3 days for critical, 5 for high)
    for f in (sec or {}).get("filings", []) if sec else []:
        try:
            age = (today - datetime.date.fromisoformat(f["date"])).days
        except Exception:
            continue
        items = f.get("items") or []
        crit = [SEC_CRITICAL[i] for i in items if i in SEC_CRITICAL]
        high = [SEC_HIGH[i] for i in items if i in SEC_HIGH]
        if crit and age <= 3:
            level = max(level, 2)
            verified = True
            facts.append("דיווח %s ל-SEC מ-%s: %s" % (f.get("form", "8-K"), f["date"], ", ".join(crit)))
            sources.append({"type": "SEC", "time": f["date"], "text": ", ".join(crit), "url": f.get("url")})
        elif high and age <= 2:
            level = max(level, 1)
            verified = True
            facts.append("דיווח ל-SEC מ-%s: %s" % (f["date"], ", ".join(high)))
            sources.append({"type": "SEC", "time": f["date"], "text": ", ".join(high), "url": f.get("url")})
    # 1b. trading halts (feed covers the current and recent sessions)
    for h in (halts or {}).get(sym.upper(), []):
        level = max(level, 2)
        verified = True
        st = "המסחר חודש %s %s" % (h["resume_date"] or "", h["resume_time"] or "") if h.get("resume_time") else "המסחר עדיין עצור"
        facts.append("עצירת מסחר ב-Nasdaq %s %s: %s. %s" % (h.get("date") or "", h.get("time") or "", h["reason"], st))
        sources.append({"type": "halt", "time": "%s %s" % (h.get("date") or "", h.get("time") or ""), "text": h["reason"],
                        "url": "https://www.nasdaqtrader.com/trader.aspx?id=tradehalts"})
    # 2. market reaction
    rx = abnormal(candles, *(spy or (None, None))) if candles and spy else None
    z = abs(rx["z"]) if rx and rx.get("z") is not None else 0
    rv = (rx or {}).get("rvol") or 0
    recent = rx and (today - datetime.date.fromisoformat(rx["date"])).days <= 4
    if recent and z >= 4 and rv >= 3:
        level = max(level, 2)
    elif recent and z >= 2.5 and rv >= 2:
        level = max(level, 1)
    # 3. headlines
    hl = headline_cluster(news, now)
    if len(hl) >= 2 and recent and z >= 3:
        level = max(level, 2)
    elif len(hl) >= 2:
        level = max(level, 1)
    for h in hl[:3]:
        sources.append({"type": "headline", "time": h["time"], "text": h["title"], "url": h["link"]})
    # fresh downgrade from a rating firm
    for c in (fund or {}).get("rating_changes", [])[:5]:
        try:
            if c.get("action") == "down" and (today - datetime.date.fromisoformat(c["date"])).days <= 2:
                level = max(level, 1)
                facts.append("הורדת דירוג: %s, מ-%s ל-%s (%s)" % (c.get("firm"), c.get("from"), c.get("to"), c["date"]))
        except Exception:
            continue
    if level == 0:
        return None
    if verified:
        verify = "מאומת"
    elif len(hl) >= 2:
        verify = "מדווח בכמה כותרות"
    elif hl:
        verify = "כותרת בודדת, לא מאומת"
    else:
        verify = "תנועה חריגה בלי הסבר בחדשות"
    if rx and recent and z >= 2.5:
        facts.append("המניה %+.1f%% (השוק %+.1f%%): %+.1f%% מעבר למה שהשוק מסביר, %.1f סטיות תקן, נפח x%s" % (
            rx["move"], rx["market"], rx["resid"], rx["z"], rx["rvol"] if rx["rvol"] else "—"))
    if hl and not verified:
        facts.append("כותרות: " + " | ".join(h["title"] for h in hl[:2]))
    crit = level == 2
    return {"level": "critical" if crit else "high", "verify": verify, "facts": facts[:5], "reaction": rx if recent else None,
            "sources": sources[:6], "headlines": len(hl),
            "effects": (["כניסה מושהית ל-3 ימי מסחר: המחקר לא מתאר מצבים כאלה",
                         "הציון ההיסטורי מוגבל ל-5 ולא תקף כרגע",
                         "פער מחיר יכול לעבור את הסטופ"] if crit else
                        ["שים לב לתנודתיות; ההחלטה לא שונתה"])}
