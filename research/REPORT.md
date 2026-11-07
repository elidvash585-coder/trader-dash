# Backtest of the entry checklist

42,735 trades, 89 stocks, 2022-11-08 to 2026-10-07 (train until 2025-03-17, test after). Plan: buy above the day's high, stop 1 ATR, target 2R, 20 days.

Baseline: win 36.3% (break-even 33.3%), mean R +0.064

## Each rule (all years)

| rule | passes | win pass / fail | R pass / fail | R diff [95%% CI] | train / test diff |
|---|---|---|---|---|---|
| trend | 41% | 35.3 / 37.0 | +0.029 / +0.088 | -0.060 [-0.144, +0.027] | -0.078 / -0.036 |
| channel | 53% | 35.6 / 37.2 | +0.041 / +0.090 | -0.049 [-0.149, +0.058] | -0.056 / -0.042 |
| momentum | 45% | 35.9 / 36.6 | +0.046 / +0.079 | -0.033 [-0.129, +0.073] | -0.015 / -0.061 |
| not_overbought | 90% | 36.3 / 36.6 | +0.065 / +0.054 | +0.011 [-0.098, +0.107] | +0.062 / -0.079 (flips) |
| volume | 57% | 36.1 / 36.6 | +0.051 / +0.082 | -0.031 [-0.093, +0.029] | -0.038 / -0.022 |
| room | 46% | 36.3 / 36.3 | +0.070 / +0.059 | +0.011 [-0.043, +0.059] | -0.011 / +0.043 (flips) |

## Out of sample (test period)

Equal weights AUC 0.496; learned rules AUC 0.500; learned full AUC 0.502 (0.5 = no skill)

| rules passed | n | win | R |
|---|---|---|---|
| 1 | 861 | 34.1 | +0.011 |
| 2 | 4561 | 35.6 | +0.087 |
| 3 | 5014 | 34.9 | +0.042 |
| 4 | 3802 | 36.4 | +0.065 |
| 5 | 2307 | 33.8 | -0.012 |
| 6 | 754 | 32.0 | -0.082 |

Model rules, score fifths (test):

| fifth | n | win | R |
|---|---|---|---|
| 1 | 3460 | 35.5 | +0.040 |
| 2 | 3460 | 35.6 | +0.042 |
| 3 | 3459 | 33.4 | +0.007 |
| 4 | 3460 | 37.4 | +0.135 |
| 5 | 3460 | 33.7 | -0.000 |

Weights: trend -0.04, channel -0.03, room -0.01, volume -0.01, not_overbought +0.00, momentum +0.00

Model full, score fifths (test):

| fifth | n | win | R |
|---|---|---|---|
| 1 | 3460 | 35.5 | +0.032 |
| 2 | 3460 | 35.8 | +0.068 |
| 3 | 3459 | 33.8 | +0.001 |
| 4 | 3460 | 33.5 | -0.001 |
| 5 | 3460 | 37.0 | +0.123 |

Weights: d_s50 -0.21, ch_move +0.14, ch_pos +0.11, cci -0.07, room_r -0.06, rsi +0.05, macd_h +0.04, ret20 +0.03, d_hi52 -0.02, ret5 +0.02, trend -0.02, room +0.02, updown +0.02, atr_pct +0.01, momentum -0.01, rvol +0.01, volume -0.01, obv20 -0.01, d_s200 -0.00, not_overbought +0.00, channel -0.00
