"""Extra analysis for Trader Dash, built on day-trader-skill (MIT, (c) 2026 Jake Schincariol):

- time_machine(): a swing version of trader-timemachine. Replays the trade plan 5,000 times
  from the stock's own daily bars, and re-runs the same rule on every past session.
- sec_block():    trader-analyst. Reported quarters from SEC EDGAR XBRL + recent 8-K items.
- congress():     trader-pelosi. House PTR filings in the last 90 days, matched to the watchlist.

Standard library only (pdftotext is used for the PTR PDFs when installed).
"""
import datetime
import json
import os
import random
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


# ---------------------------------------------------------------- time machine (swing)

def time_machine(candles, plan, horizon=20, replays=5000, seed=19, lookback=500):
    """candles: [[t,o,h,l,c,v], ...] oldest first. plan: trigger/stop/target.
    Each replay: enter at the trigger, then walk up to `horizon` sessions drawing random past
    days as ratios to the previous close. The stop is checked before the target in a bar (and a
    gap through the stop counts as the stop), so the odds are on the conservative side."""
    if not plan or len(candles) < 80:
        return None
    entry, stop, target = plan["trigger"], plan["stop"], plan["target"]
    risk = entry - stop
    if risk <= 0:
        return None
    rr = (target - entry) / risk
    bars = candles[-lookback:]
    pool = []
    for i in range(1, len(bars)):
        pc = bars[i - 1][4]
        if pc:
            pool.append((bars[i][1] / pc, bars[i][2] / pc, bars[i][3] / pc, bars[i][4] / pc))
    rnd = random.Random(seed)
    hit_t = hit_s = 0
    days_t, end_r = [], []
    for _ in range(replays):
        px = entry
        outcome = None
        for d in range(1, horizon + 1):
            o, h, l, c = rnd.choice(pool)
            o, h, l, c = px * o, px * h, px * l, px * c
            if o <= stop or l <= stop:
                outcome = "stop"
                break
            if h >= target:
                outcome = "target"
                days_t.append(d)
                break
            px = c
        if outcome == "target":
            hit_t += 1
            end_r.append(rr)
        elif outcome == "stop":
            hit_s += 1
            end_r.append(-1.0)
        else:
            end_r.append((px - entry) / risk)
    n = float(replays)
    p_t, p_s = hit_t / n, hit_s / n
    # history: every past day that broke the previous day's high, same stop distance in ATRs
    atr_mult = risk / max(1e-9, _atr(candles))
    hist = {"trades": 0, "target": 0, "stop": 0, "open": 0}
    c = candles
    atrs = _atr_series(c)
    for i in range(30, len(c) - 1):
        trig = c[i - 1][2] + 0.01
        if c[i][2] < trig or atrs[i - 1] is None:
            continue
        s_ = trig - atr_mult * atrs[i - 1]
        t_ = trig + rr * (trig - s_)
        hist["trades"] += 1
        res = "open"
        for j in range(i + 1, min(len(c), i + 1 + horizon)):
            if c[j][1] <= s_ or c[j][3] <= s_:
                res = "stop"
                break
            if c[j][2] >= t_:
                res = "target"
                break
        hist[res] += 1
    decided = p_t + p_s
    share = p_t / decided if decided else None
    be = 1.0 / (1.0 + rr)
    if share is None:
        verdict, lean = "אין מספיק מקרים שהגיעו ליעד או לסטופ", 0
    else:
        gap = (share - be) * 100
        lean = 1 if gap >= 5 else -1 if gap <= -5 else 0
        verdict = ("נוטה ליעד" if gap >= 5 else "נוטה לסטופ" if gap <= -5 else "קרוב לנקודת האיזון")
        if abs(gap) >= 15:
            verdict = "נוטה בחוזקה " + ("ליעד" if gap > 0 else "לסטופ")
    hs = hist["target"] + hist["stop"]
    return {"replays": replays, "horizon": horizon, "rr": round(rr, 2),
            "p_target": round(p_t * 100, 1), "p_stop": round(p_s * 100, 1),
            "p_open": round((1 - p_t - p_s) * 100, 1),
            "avg_r": round(sum(end_r) / n, 2), "median_days": sorted(days_t)[len(days_t) // 2] if days_t else None,
            "breakeven": round(be * 100, 1), "share": round(share * 100, 1) if share is not None else None,
            "verdict": verdict, "lean": lean,
            "history": dict(hist, hit_rate=round(hist["target"] / hs * 100, 1) if hs else None)}


def _atr_series(c, n=14):
    tr = [c[0][2] - c[0][3]] + [max(c[i][2] - c[i][3], abs(c[i][2] - c[i - 1][4]), abs(c[i][3] - c[i - 1][4]))
                                for i in range(1, len(c))]
    out, a = [None] * len(c), None
    for i, x in enumerate(tr):
        if i == n - 1:
            a = sum(tr[:n]) / n
        elif i >= n:
            a = (a * (n - 1) + x) / n
        out[i] = a
    return out


def _atr(c):
    s = _atr_series(c)
    return s[-1] or 1.0


# ---------------------------------------------------------------- cache helpers

def _cached(name, max_age_h, fetch):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_h * 3600:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    data = fetch()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def _get_json(url, ua, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------- SEC (trader-analyst)

ITEMS_HE = {"1.01": "הסכם מהותי", "1.02": "סיום הסכם מהותי", "1.03": "פשיטת רגל", "1.05": "אירוע סייבר",
            "2.01": "רכישה או מכירה של נכסים", "2.02": "דוח רבעוני (תוצאות)", "2.03": "התחייבות פיננסית",
            "2.05": "עלויות ארגון מחדש", "2.06": "ירידת ערך", "3.01": "הודעת מחיקה מהמסחר", "3.02": "מכירת מניות לא רשומה",
            "4.01": "החלפת רואה חשבון", "4.02": "דוחות קודמים לא אמינים", "5.02": "שינוי בהנהלה", "5.03": "שינוי תקנון",
            "5.07": "הצבעת בעלי מניות", "7.01": "גילוי (Reg FD)", "8.01": "אירוע אחר", "9.01": "דוחות ונספחים"}
IMPORTANT = {"1.03", "1.05", "2.02", "2.06", "3.01", "4.02", "5.02"}


def sec_block(sym, ua):
    import facts  # trader-analyst
    tick = _cached("sec_tickers.json", 24 * 7, lambda: _get_json(facts.TICKERS_URL, ua))
    m = {v["ticker"].upper(): (str(v["cik_str"]).zfill(10), v.get("title")) for v in tick.values()}
    key = sym.upper().replace(".", "-")
    if key not in m:
        return None  # ETF, crypto or foreign filer without XBRL facts
    cik, title = m[key]
    cf = _cached("facts_%s.json" % cik, 12, lambda: _get_json(facts.FACTS_URL % cik, ua, 40))
    sub = _cached("subm_%s.json" % cik, 6, lambda: _get_json(facts.SUBMISSIONS_URL % cik, ua))
    out = {"cik": cik, "entity": title}
    try:
        res = facts.analyse(cf, quarters=6)
        out["quarters"] = res["rows"]
        out["ttm"] = res["ttm"]
        out["changes"] = facts.changes(res)
    except Exception as e:
        out["quarters"], out["error"] = [], str(e)[:120]
    rec = sub.get("filings", {}).get("recent", {})
    since = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    filings = []
    for i, form in enumerate(rec.get("form", [])):
        if rec["filingDate"][i] < since:
            break
        if form not in ("8-K", "10-Q", "10-K", "6-K", "20-F"):
            continue
        items = [x.strip() for x in (rec.get("items", [""] * (i + 1))[i] or "").split(",") if x.strip()]
        acc = rec["accessionNumber"][i]
        filings.append({"form": form, "date": rec["filingDate"][i], "items": items,
                        "items_he": [ITEMS_HE.get(x, x) for x in items if x != "9.01"],
                        "important": bool(set(items) & IMPORTANT),
                        "url": facts.ARCHIVE_URL % (int(cik), acc.replace("-", "")) + rec["primaryDocument"][i]})
    out["filings"] = filings[:8]
    return out


# ---------------------------------------------------------------- Congress (trader-pelosi)

def congress(watch, days=90):
    """{TICKER: [trade, ...]} for House PTRs filed in the last `days` days. Needs pdftotext."""
    import ptr  # trader-pelosi
    if not ptr.shutil.which("pdftotext"):
        return {"_note": "pdftotext not installed"}
    today = datetime.date.today()
    since = (today - datetime.timedelta(days=days)).isoformat()
    rows = []
    for year in sorted({today.year, (today - datetime.timedelta(days=days)).year}):
        try:
            rows += ptr.select(ptr.load_index(year), kind="P", since=since)
        except Exception:
            continue
    seen_path = os.path.join(CACHE, "ptr_parsed.json")
    try:
        with open(seen_path, encoding="utf-8") as fh:
            parsed = json.load(fh)
    except Exception:
        parsed = {}
    pdf_dir = os.path.join(CACHE, "ptr-pdfs")
    started = time.time()
    for r in rows:
        if r["doc_id"] in parsed or time.time() - started > 240:
            continue
        try:
            _, rep = ptr._fetch_and_read(r["doc_id"], r["year"], pdf_dir, filed=r["filed"])
            parsed[r["doc_id"]] = {"name": r["name"], "district": r["district"], "filed": r["filed"],
                                   "url": r["url"], "rows": (rep or {}).get("rows") or []}
            pdf = os.path.join(pdf_dir, "%s.pdf" % r["doc_id"])
            if os.path.exists(pdf):
                os.remove(pdf)  # keep the cache small: the parsed rows are enough
        except Exception:
            parsed[r["doc_id"]] = {"name": r["name"], "filed": r["filed"], "url": r["url"], "rows": []}
        time.sleep(0.2)
    os.makedirs(CACHE, exist_ok=True)
    with open(seen_path, "w", encoding="utf-8") as fh:
        json.dump(parsed, fh)
    wanted = {s.upper().replace("-", "."): s for s in watch}
    out = {}
    for doc, f in parsed.items():
        if (f.get("filed") or "") < since:
            continue
        for t in f["rows"]:
            tk = (t.get("ticker") or "").upper()
            if tk in wanted:
                out.setdefault(wanted[tk], []).append({
                    "member": f["name"], "district": f.get("district"), "owner": ptr.OWNERS.get(t.get("owner") or "", t.get("owner")),
                    "type": t.get("type_name") or t.get("type"), "date": t.get("date"), "filed": f["filed"],
                    "amount": t.get("amount"), "option": t.get("option"), "late": t.get("late"),
                    "days_to_disclose": t.get("days_to_disclose"), "url": f["url"]})
    for k in out:
        out[k].sort(key=lambda x: x.get("date") or "", reverse=True)
    out["_meta"] = {"filings": len([1 for f in parsed.values() if (f.get("filed") or "") >= since]), "since": since}
    return out
