"""Price features for Trader Dash: one definition shared by the live site and the research.

Standard library only. Every series is aligned with the input bars and uses only bars up to and
including its own index, so a value at index t is what was known at the close of day t.
(tests/test_features.py checks this by recomputing on truncated data.)

Swing points (pivots) need K bars on each side. A pivot at bar j is therefore CONFIRMED only at
bar j + K and is never used before that.
"""
import math

K_PIVOT = 3        # bars each side for a swing high / low
LOOK_RES = 120     # bars to look back for resistance
ROLL_PCT = 252     # bars for the ATR% percentile


# ---------------------------------------------------------------- basic series

def _rma(x, n):
    """Wilder smoothing, seeded the way Wilder defined it: the first value is the simple average of the
    first n values (None before that), then a += (v - a) / n. Seeding with the first single value instead
    (as before v3) made RSI up to 14 points and ATR up to 16 % wrong in a stock's first ~60 sessions."""
    out, a, seed = [None] * len(x), None, []
    for i, v in enumerate(x):
        if v is None:
            continue
        if a is None:
            seed.append(v)
            if len(seed) < n:
                continue
            a = sum(seed) / n
        else:
            a = a + (v - a) / n
        out[i] = a
    return out


def sma(x, n):
    out, s = [None] * len(x), 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(x, n):
    out, a, k = [None] * len(x), None, 2.0 / (n + 1)
    for i, v in enumerate(x):
        a = v if a is None else a + k * (v - a)
        out[i] = a
    return out


def true_range(h, l, c):
    return [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]


def atr(h, l, c, n=14):
    return _rma(true_range(h, l, c), n)


def rsi(c, n=14):
    up = [None] + [max(c[i] - c[i - 1], 0.0) for i in range(1, len(c))]
    dn = [None] + [max(c[i - 1] - c[i], 0.0) for i in range(1, len(c))]
    au, ad = _rma(up, n), _rma(dn, n)
    out = [None] * len(c)
    for i in range(n, len(c)):
        if au[i] is None or ad[i] is None:
            continue
        out[i] = 100.0 if ad[i] == 0 else 100 - 100 / (1 + au[i] / ad[i])
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


def adx(h, l, c, n=14):
    """ADX, +DI, -DI (Wilder). ADX measures trend STRENGTH, not direction."""
    N = len(c)
    pdm, ndm = [None] * N, [None] * N
    for i in range(1, N):
        upm, dnm = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = upm if upm > dnm and upm > 0 else 0.0
        ndm[i] = dnm if dnm > upm and dnm > 0 else 0.0
    tr = [None] + true_range(h, l, c)[1:]
    atr_, sp, sn = _rma(tr, n), _rma(pdm, n), _rma(ndm, n)
    pdi, ndi, dx = [None] * N, [None] * N, [None] * N
    for i in range(n, N):
        if atr_[i]:
            pdi[i] = 100 * sp[i] / atr_[i]
            ndi[i] = 100 * sn[i] / atr_[i]
            s = pdi[i] + ndi[i]
            dx[i] = 100 * abs(pdi[i] - ndi[i]) / s if s else 0.0
    adx_ = _rma(dx, n)
    out = [None] * N
    for i in range(N):
        if adx_[i] is not None and i >= 2 * n:
            out[i] = adx_[i]
    return out, pdi, ndi


def pct_rank(x, n):
    """Share (0-100) of the last n values that are <= the current one."""
    out = [None] * len(x)
    for i in range(len(x)):
        if x[i] is None or i < n // 2:
            continue
        w = [v for v in x[max(0, i - n + 1):i + 1] if v is not None]
        if len(w) >= n // 2:
            out[i] = 100.0 * sum(1 for v in w if v <= x[i]) / len(w)
    return out


def realized_vol(c, n=20):
    lr = [None] + [math.log(c[i] / c[i - 1]) if c[i] > 0 and c[i - 1] > 0 else 0.0 for i in range(1, len(c))]
    out = [None] * len(c)
    for i in range(n, len(c)):
        w = lr[i - n + 1:i + 1]
        m = sum(w) / n
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / (n - 1)) * math.sqrt(252) * 100
    return out


# ---------------------------------------------------------------- swing structure

def pivots(h, l, k=K_PIVOT):
    """(is_high, is_low) per bar. Only usable from bar j + k on (see confirmed_*)."""
    N = len(h)
    ph, pl = [False] * N, [False] * N
    for j in range(k, N - k):
        wh, wl = h[j - k:j + k + 1], l[j - k:j + k + 1]
        ph[j] = h[j] == max(wh) and h[j] > h[j - 1]
        pl[j] = l[j] == min(wl) and l[j] < l[j - 1]
    return ph, pl


def structure(h, l, c, a, k=K_PIVOT):
    """Per bar t, from pivots confirmed by t (pivot j <= t - k):
    last two swing highs / lows, the HH/HL state, break of structure, failed breakout,
    nearest swing high above the close within LOOK_RES bars (resistance) and the last swing low."""
    N = len(c)
    ph, pl = pivots(h, l, k)
    highs, lows = [], []          # (index, price) of confirmed pivots so far
    out = {k_: [None] * N for k_ in ("state", "bos_up", "bos_dn", "failed_bo", "res", "swing_low", "last_ph", "consol")}
    for t in range(N):
        j = t - k
        if j >= 0:
            if ph[j]:
                highs.append((j, h[j]))
            if pl[j]:
                lows.append((j, l[j]))
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1][1] > highs[-2][1]
            hl = lows[-1][1] > lows[-2][1]
            out["state"][t] = 2 if hh and hl else -2 if (not hh and not hl) else 0
        if highs:
            lv = highs[-1][1]
            out["last_ph"][t] = lv
            out["bos_up"][t] = c[t] > lv and (t == 0 or c[t - 1] <= lv)
            # failed breakout: a close above the last swing high in the last 5 bars, now back below it
            out["failed_bo"][t] = c[t] < lv and any(c[i] > lv for i in range(max(0, t - 5), t))
        if lows:
            lv = lows[-1][1]
            out["bos_dn"][t] = c[t] < lv and (t == 0 or c[t - 1] >= lv)
            below = [p for i_, p in reversed(lows) if p < c[t]]
            out["swing_low"][t] = below[0] if below else None
        above = [p for i_, p in highs if i_ >= t - LOOK_RES and p > c[t]]
        out["res"][t] = min(above) if above else None
        if t >= 20 and a[t]:
            out["consol"][t] = (max(h[t - 19:t + 1]) - min(l[t - 19:t + 1])) / a[t] < 4.0
    return out


def levels(h, l, c, a, look=250, k=K_PIVOT, max_n=3, end=None):
    """Main support / resistance levels known at the close of bar `end` (default: the last bar).
    Built from CONFIRMED swing points only (pivot j counts from bar j + k), clustered within 0.5 ATR.
    Each level: price (median of its swing points), touches, last (bar index), flip (it was a swing high
    AND a swing low, i.e. support that became resistance or the reverse).
    Up to max_n each side, nearest first, levels touched twice or more preferred; none are invented
    (a stock at a new high has no resistance). Also the most recent unfilled price gap on each side."""
    N = len(c) if end is None else end + 1
    if N < 2 * k + 5 or not a or a[N - 1] is None:
        return {"support": [], "resistance": [], "gap_up": None, "gap_down": None}
    atr_ = a[N - 1]
    tol = 0.5 * atr_
    lo = max(k, N - look)
    pts = []
    for j in range(lo, N - k):
        wh, wl = h[j - k:j + k + 1], l[j - k:j + k + 1]
        if h[j] == max(wh) and h[j] > h[j - 1]:
            pts.append((h[j], j, "h"))
        if l[j] == min(wl) and l[j] < l[j - 1]:
            pts.append((l[j], j, "l"))
    pts.sort()
    cl_ = []
    for p in pts:
        if cl_ and p[0] - cl_[-1][-1][0] <= tol and p[0] - cl_[-1][0][0] <= 2 * tol:
            cl_[-1].append(p)
        else:
            cl_.append([p])
    price = c[N - 1]
    lv = []
    for g in cl_:
        ps = sorted(x[0] for x in g)
        m = ps[len(ps) // 2] if len(ps) % 2 else (ps[len(ps) // 2 - 1] + ps[len(ps) // 2]) / 2
        kinds = {x[2] for x in g}
        lv.append({"price": round(m, 2), "touches": len(g), "last": max(x[1] for x in g), "flip": len(kinds) == 2,
                   "dist_atr": round((m - price) / atr_, 2)})

    def pick(side):
        s = [x for x in lv if (x["price"] < price if side == "s" else x["price"] > price)]
        s.sort(key=lambda x: abs(x["price"] - price))
        strong = [x for x in s if x["touches"] >= 2]
        out = strong[:max_n]
        if len(out) < max_n:  # fill with single swing points, nearest first, marked by touches = 1
            out += [x for x in s if x["touches"] < 2][:max_n - len(out)]
        return sorted(out, key=lambda x: abs(x["price"] - price))[:max_n]
    gap_up = gap_down = None
    for i in range(max(1, N - look), N):
        if l[i] > h[i - 1] and l[i] - h[i - 1] >= 0.2 * atr_ and min(l[i + 1:N] or [l[i]]) > h[i - 1]:
            gap_up = {"low": round(h[i - 1], 2), "high": round(l[i], 2), "bar": i}
        if h[i] < l[i - 1] and l[i - 1] - h[i] >= 0.2 * atr_ and max(h[i + 1:N] or [h[i]]) < l[i - 1]:
            gap_down = {"low": round(h[i], 2), "high": round(l[i - 1], 2), "bar": i}
    return {"support": pick("s"), "resistance": pick("r"), "gap_up": gap_up, "gap_down": gap_down}


# ---------------------------------------------------------------- volume

def obv(c, v):
    out, s = [0.0] * len(c), 0.0
    for i in range(1, len(c)):
        s += v[i] if c[i] > c[i - 1] else -v[i] if c[i] < c[i - 1] else 0.0
        out[i] = s
    return out


def avwap(h, l, c, v, anchor):
    """Anchored VWAP from bar `anchor` (daily bars: typical price x volume, an approximation of
    the intraday VWAP). None before the anchor."""
    out, pv, vv = [None] * len(c), 0.0, 0.0
    if anchor is None or anchor < 0:
        return out
    for i in range(anchor, len(c)):
        tp = (h[i] + l[i] + c[i]) / 3
        pv += tp * v[i]
        vv += v[i]
        out[i] = pv / vv if vv else tp
    return out


def volume_profile(h, l, v, look=120, bins=24):
    """APPROXIMATE profile from daily bars: each day's volume spread evenly over its high-low range.
    Real volume-at-price needs intraday data; the result is labelled as an estimate."""
    h, l, v = h[-look:], l[-look:], v[-look:]
    lo, hi = min(l), max(h)
    if hi <= lo:
        return None
    w = (hi - lo) / bins
    vol = [0.0] * bins
    for H, L, V in zip(h, l, v):
        if V <= 0:
            continue
        a, b = int((L - lo) / w), min(bins - 1, int((H - lo) / w))
        span = max(1, b - a + 1)
        for k in range(a, b + 1):
            vol[min(k, bins - 1)] += V / span
    poc = max(range(bins), key=lambda k: vol[k])
    tot, area, lo_k, hi_k = sum(vol), vol[poc], poc, poc
    while area < 0.7 * tot and (lo_k > 0 or hi_k < bins - 1):
        dn = vol[lo_k - 1] if lo_k > 0 else -1
        up = vol[hi_k + 1] if hi_k < bins - 1 else -1
        if up >= dn:
            hi_k += 1
            area += up
        else:
            lo_k -= 1
            area += dn
    return {"approx": True, "look": look, "lo": lo, "step": w, "vol": [round(x) for x in vol],
            "poc": lo + (poc + 0.5) * w, "va_low": lo + lo_k * w, "va_high": lo + (hi_k + 1) * w}


# ---------------------------------------------------------------- relative strength

def align(t, bt, bc):
    """Benchmark closes aligned to the stock's timestamps by calendar day (carry the last known close)."""
    day = lambda x: int(x // 86400)
    m = {day(a): b for a, b in zip(bt, bc) if b}
    out, last = [], None
    keys = sorted(m)
    kpos = 0
    for x in t:
        d = day(x)
        while kpos < len(keys) and keys[kpos] <= d:
            last = m[keys[kpos]]
            kpos += 1
        out.append(last)
    return out


def rel_strength(c, b, n):
    """Excess return over n bars vs the benchmark, in percent points."""
    out = [None] * len(c)
    for i in range(n, len(c)):
        if b[i] and b[i - n] and c[i - n]:
            out[i] = (c[i] / c[i - n] - b[i] / b[i - n]) * 100
    return out


# ---------------------------------------------------------------- everything at once

def compute(t, o, h, l, c, v, bench=None):
    """All feature series. bench: {"SPY": [closes aligned], "QQQ": [...], "SECTOR": [...]} (optional)."""
    N = len(c)
    a = atr(h, l, c)
    s20, s50, s150, s200 = sma(c, 20), sma(c, 50), sma(c, 150), sma(c, 200)
    ad, pdi, ndi = adx(h, l, c)
    st = structure(h, l, c, a)
    r = rsi(c)
    cc = cci(h, l, c)
    e12, e26 = ema(c, 12), ema(c, 26)
    mac = [x - y for x, y in zip(e12, e26)]
    sig = ema(mac, 9)
    atrp = [a[i] / c[i] * 100 if a[i] and c[i] else None for i in range(N)]
    ob = obv(c, v)
    f = {"atr": a, "sma20": s20, "sma50": s50, "sma150": s150, "sma200": s200, "adx": ad, "pdi": pdi, "ndi": ndi,
         "rsi": r, "cci": cc, "atr_pct": atrp, "atr_pctile": pct_rank(atrp, ROLL_PCT), "rv20": realized_vol(c),
         "macd_h": [(mac[i] - sig[i]) / a[i] if a[i] else None for i in range(N)]}
    f.update(st)
    for n_, s_ in (("d_sma20", s20), ("d_sma50", s50), ("d_sma150", s150), ("d_sma200", s200)):
        f[n_] = [(c[i] / s_[i] - 1) * 100 if s_[i] else None for i in range(N)]
    f["slope50"] = [(s50[i] / s50[i - 20] - 1) * 100 if i >= 20 and s50[i] and s50[i - 20] else None for i in range(N)]
    f["ext_atr"] = [(c[i] - s20[i]) / a[i] if s20[i] and a[i] else None for i in range(N)]
    f["room_atr"] = [((st["res"][i] - c[i]) / a[i]) if st["res"][i] and a[i] else None for i in range(N)]
    rv, ud, obs = [None] * N, [None] * N, [None] * N
    for i in range(21, N):
        avg = sum(v[i - 20:i]) / 20
        rv[i] = v[i] / avg if avg else None
        upv = sum(v[j] for j in range(i - 19, i + 1) if c[j] >= c[j - 1])
        dnv = sum(v[j] for j in range(i - 19, i + 1) if c[j] < c[j - 1])
        ud[i] = min(5.0, upv / dnv) if dnv else None
        obs[i] = 1 if ob[i] > ob[i - 20] else -1 if ob[i] < ob[i - 20] else 0
    f["rvol"], f["updown"], f["obv20"] = rv, ud, obs
    f["clv"] = [(c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5 for i in range(N)]
    f["body"] = [abs(c[i] - o[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.0 for i in range(N)]
    f["lwick"] = [(min(o[i], c[i]) - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.0 for i in range(N)]
    f["gap"] = [None] + [(o[i] / c[i - 1] - 1) * 100 if c[i - 1] else None for i in range(1, N)]
    for n_ in (20, 60, 120):
        f["ret%d" % n_] = [(c[i] / c[i - n_] - 1) * 100 if i >= n_ and c[i - n_] else None for i in range(N)]
    for name, b in (bench or {}).items():
        if not b:
            continue
        for n_ in (20, 60, 120):
            f["rs_%s_%d" % (name.lower(), n_)] = rel_strength(c, b, n_)
    return f


def last(f, key):
    s = f.get(key)
    if not s:
        return None
    x = s[-1]
    return round(x, 4) if isinstance(x, float) else x
