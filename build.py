#!/usr/bin/env python3
"""Builds the static online dashboard into ./site (run by GitHub Actions on a schedule).

    python build.py          # real data from Yahoo Finance
    python build.py --demo   # synthetic data, for testing
"""
import datetime
import json
import os
import re
import sys
import time

import dashboard as D
import risk as RK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "site")
LWC = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"


def safe(sym):
    return re.sub(r"[^A-Za-z0-9._-]", "_", sym)


def main():
    D.DEMO = "--demo" in sys.argv
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    wl = D.read_watchlist()
    cg = {}
    if D.extras and not D.DEMO:
        try:
            cg = D.extras.congress(wl["symbols"])
            print("congress: %s" % (cg.get("_meta") or cg.get("_note")), flush=True)
        except Exception as e:
            print("congress skipped: %s" % str(e)[:160], flush=True)
    rows, ok, full = [], 0, {}
    for sym in wl["symbols"]:
        try:
            a = D.analyze(sym)
            a["congress"] = cg.get(sym, []) if "_note" not in cg else None
            a["ind_score"] = D.ind_score(a["tech"], a["plan"], a["fund"], a.get("tm"), a["congress"], sector=a.get("sector"))
            hist, plan2, dec = D.engine_block(a["tech"], a["fund"], a.get("sec"), a.get("ext"), a["ind_score"], a.get("event"))
            if hist is not None:
                dec["ext_alerts"] = (a.get("decision") or {}).get("ext_alerts")
                a["score"], a["plan2"], a["decision"] = hist, plan2, dec
            full[sym] = a
            q = D.quick(sym)
            q["congress"] = len(a["congress"] or [])
            q["ind"] = a["ind_score"]["value"] if a.get("ind_score") else None
            if hist is not None:
                q.update(score=hist["value"], score_label=hist["label"], decision_code=hist["code"], act=dec["verdict"], act_code=dec["code"],
                         event=(a.get("event") or {}).get("level"), on=(a.get("oneil") or {}).get("status"))
            rows.append(q)
            ok += 1
            print("ok   %-8s %s" % (sym, (a.get("decision") or {}).get("verdict", "")), flush=True)
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e)[:160]})
            print("FAIL %-8s %s" % (sym, str(e)[:160]), flush=True)
        if not D.DEMO:
            time.sleep(0.4)
    # watchlist overlap: most correlated symbols (daily returns, last 120 sessions)
    peers = RK.watchlist_peers({s: [(c[0], c[4]) for c in a["candles"][-200:]] for s, a in full.items()})
    for sym, a in full.items():
        a["peers"] = peers.get(sym)
        with open(os.path.join(OUT, "data", safe(sym) + ".json"), "w", encoding="utf-8") as fh:
            json.dump(a, fh, ensure_ascii=False, separators=(",", ":"))
    now = datetime.datetime.now(datetime.timezone.utc)
    index = {"meta": {"watchlist": wl, "demo": D.DEMO, "static": True,
                      "built_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "congress": cg.get("_meta"), "sec": bool(os.environ.get("SEC_USER_AGENT"))}, "rows": rows}
    with open(os.path.join(OUT, "data", "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace('<script src="/lwc.js"></script>',
                        '<script>window.STATIC_SITE=true</script><script src="%s"></script>' % LWC)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    print("Built %d of %d symbols at %s" % (ok, len(wl["symbols"]), index["meta"]["built_at"]))
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
