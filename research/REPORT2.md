# Strategy grid

89 stocks, 192 configs, 2022-11-08 to 2026-11-06. Train to 2024-08-27, validation to 2025-04-03, test after. Costs 0.10% round trip.

## Best config per setup (chosen on validation)

| setup | stop ATR | target R | days | train edge | val edge | test n | test R | test win | test edge |
|---|---|---|---|---|---|---|---|---|---|
| brk_high | 2.0 | 2.0 | 10 | -0.012 | -0.002 | 17307 | +0.074 | 50.2 | -0.010 |
| brk_20d | 3.0 | 1.0 | 20 | +0.010 | +0.009 | 4754 | +0.109 | 55.6 | -0.010 |
| ma20_up | 1.5 | 1.5 | 20 | -0.011 | +0.034 | 2048 | +0.102 | 45.7 | -0.002 |
| pullback | 1.0 | 3.0 | 10 | +0.027 | +0.176 | 1278 | +0.092 | 37.4 | +0.011 |
| oversold | 1.0 | 3.0 | 10 | +0.069 | +0.429 | 1127 | +0.222 | 43.8 | +0.140 |
| every_day | 2.0 | 3.0 | 10 | +0.000 | +0.000 | 34319 | +0.091 | 50.1 | +0.000 |

## Chosen

oversold, stop 1.0 ATR, target 3.0R, 10 days: test n 1127, mean R +0.222 [-0.041, 0.489], edge vs every day +0.140 [-0.096, 0.374]

Filter model kept: False. Validation R all +0.404 vs top 40% +0.383; test R all +0.222 vs top 40% +0.073

Weights: d_s200 +0.74, atr_pct +0.44, d_hi52 -0.43, d_s50 +0.25, room -0.16, channel -0.16, ch_pos -0.14, ret5 +0.14, updown +0.13, ret20 -0.12
