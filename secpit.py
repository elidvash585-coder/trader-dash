"""SEC fundamentals as they were KNOWN at each date (point in time). Standard library only.

Used by the research (backtest4/5) and by the live site, so both read the numbers the same way:
- each quarter's value is the FIRST one reported (a restatement filed later was not known before),
- usable from the day after the filing date,
- Q4 = annual value minus Q1..Q3, known when the 10-K was filed.
EPS uses diluted EPS (USD/shares); when a company does not report it, net income is used for growth.
"""
import datetime

REVENUE = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "RevenuesNetOfInterestExpense",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
           "SalesRevenueServicesNet"]
OPERATING = ["OperatingIncomeLoss"]
EPS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]
NET_INCOME = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]


def pit_quarters(cf, concepts, unit="USD"):
    """{period_end: (value, filed)} three-month values as first reported."""
    merged = {}
    for concept in concepts:
        node = cf.get("facts", {}).get("us-gaap", {}).get(concept)
        if not node or unit not in node.get("units", {}):
            continue
        first = {}
        for f in node["units"][unit]:
            if "start" not in f or (f.get("form") or "").split("/")[0] not in ("10-Q", "10-K"):
                continue
            k = (f["start"], f["end"])
            if k not in first or f["filed"] < first[k]["filed"]:
                first[k] = f
        q, years = {}, []
        for (st, en), f in first.items():
            days = (datetime.date.fromisoformat(en) - datetime.date.fromisoformat(st)).days
            if 80 <= days <= 100:
                q[en] = (float(f["val"]), f["filed"], st)
            elif 350 <= days <= 380:
                years.append((st, en, f))
        for st, en, f in years:
            if en in q:
                continue
            ins = sorted([v for e, v in q.items() if v[2] >= st and e < en], key=lambda v: v[2])
            if len(ins) == 3:
                q[en] = (float(f["val"]) - sum(v[0] for v in ins), max([f["filed"]] + [v[1] for v in ins]), st)
        for en, v in q.items():
            merged.setdefault(en, v)
    return {e: (v[0], v[1]) for e, v in merged.items()}


def _year_ago(ends, e):
    de = datetime.date.fromisoformat(e)
    c = [x for x in ends if abs((de - datetime.date.fromisoformat(x)).days - 365) <= 15]
    return c[0] if c else None


def growth_events(cf):
    """Chronological list of what became known at each filing:
    {"known": date the value became usable, "end": quarter end, "eps_yoy", "rev_yoy", "eps_ttm_yoy",
     "eps_pos", "rev_yoy_prev": [...the two quarters before], "margin", "margin_prev"}"""
    rev = pit_quarters(cf, REVENUE)
    eps = pit_quarters(cf, EPS, "USD/shares")
    basis = "eps"
    if len(eps) < 8:
        ni = pit_quarters(cf, NET_INCOME)
        if len(ni) > len(eps):     # only switch when net income actually has more history
            eps, basis = ni, "net_income"
    op = pit_quarters(cf, OPERATING)
    ends = sorted(set(rev) | set(eps))
    out = []
    for i, e in enumerate(ends):
        e0 = _year_ago(ends, e)
        if not e0:
            continue
        known = max(v[1] for v in (rev.get(e), eps.get(e), rev.get(e0), eps.get(e0)) if v)
        ev = {"end": e, "known": (datetime.date.fromisoformat(known) + datetime.timedelta(days=1)).isoformat(), "basis": basis}
        if e in eps and e0 in eps and eps[e0][0] > 0:
            ev["eps_yoy"] = eps[e][0] / eps[e0][0] - 1
        ev["eps_pos"] = eps[e][0] > 0 if e in eps else None
        if e in rev and e0 in rev and rev[e0][0] > 0:
            ev["rev_yoy"] = rev[e][0] / rev[e0][0] - 1
        if e in op and e in rev and rev[e][0] > 0:
            ev["margin"] = op[e][0] / rev[e][0]
        # trailing-12-month EPS growth (the A in CAN SLIM, from quarterly reports)
        last4 = [x for x in ends[:i + 1] if x in eps][-4:]
        prev4 = [x for x in ends[:i + 1] if x in eps][-8:-4]
        if len(last4) == 4 and len(prev4) == 4:
            t1, t0 = sum(eps[x][0] for x in last4), sum(eps[x][0] for x in prev4)
            ev["eps_ttm"], ev["eps_ttm_prev"] = t1, t0
            if t0 > 0:
                ev["eps_ttm_yoy"] = t1 / t0 - 1
        out.append(ev)
    # previous quarters' growth, for acceleration
    for i, ev in enumerate(out):
        ev["rev_yoy_prev"] = [o.get("rev_yoy") for o in out[max(0, i - 2):i]]
        ev["eps_yoy_prev"] = [o.get("eps_yoy") for o in out[max(0, i - 2):i]]
        ev["eps_ttm_hist"] = [o.get("eps_ttm") for o in out[max(0, i - 12):i + 1:4]]
    return sorted(out, key=lambda x: x["known"])


def known_at(events, day):
    """The latest event already known on `day` (ISO date), or None."""
    last = None
    for ev in events:
        if ev["known"] <= day:
            last = ev
        else:
            break
    return last
