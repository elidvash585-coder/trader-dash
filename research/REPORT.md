# Backtest of the entry checklist

42,694 trades, 89 stocks, 2022-11-08 to 2026-10-06 (train until 2025-03-17, test after). Plan: buy above the day's high, stop 1 ATR, target 2R, 20 days.

Baseline: win 36.3% (break-even 33.3%), mean R +0.063

## Each rule (all years)

| rule | passes | win pass / fail | R pass / fail | R diff [95%% CI] | train / test diff |
|---|---|---|---|---|---|
| trend | 41% | 35.3 / 37.0 | +0.028 / +0.088 | -0.060 [-0.144, +0.027] | -0.078 / -0.037 |
| channel | 53% | 35.5 / 37.2 | +0.041 / +0.089 | -0.049 [-0.150, +0.058] | -0.056 / -0.042 |
| momentum | 45% | 35.9 / 36.6 | +0.045 / +0.078 | -0.033 [-0.129, +0.073] | -0.015 / -0.061 |
| not_overbought | 90% | 36.3 / 36.5 | +0.064 / +0.054 | +0.010 [-0.098, +0.107] | +0.062 / -0.079 (flips) |
| volume | 57% | 36.1 / 36.5 | +0.051 / +0.080 | -0.029 [-0.090, +0.029] | -0.038 / -0.019 |
| room | 46% | 36.3 / 36.3 | +0.069 / +0.059 | +0.010 [-0.044, +0.059] | -0.011 / +0.042 (flips) |

## Out of sample (test period)

Equal weights AUC 0.496; learned rules AUC 0.500; learned full AUC 0.502 (0.5 = no skill)

| rules passed | n | win | R |
|---|---|---|---|
| 1 | 857 | 34.1 | +0.009 |
| 2 | 4545 | 35.6 | +0.085 |
| 3 | 5001 | 34.8 | +0.041 |
| 4 | 3798 | 36.4 | +0.064 |
| 5 | 2305 | 33.7 | -0.013 |
| 6 | 752 | 31.9 | -0.082 |

Model rules, score fifths (test):

| fifth | n | win | R |
|---|---|---|---|
| 1 | 3452 | 35.4 | +0.037 |
| 2 | 3451 | 35.5 | +0.039 |
| 3 | 3452 | 33.4 | +0.009 |
| 4 | 3451 | 37.3 | +0.134 |
| 5 | 3452 | 33.6 | -0.003 |

Weights: trend -0.04, channel -0.03, room -0.01, volume -0.01, not_overbought +0.00, momentum +0.00

Model full, score fifths (test):

| fifth | n | win | R |
|---|---|---|---|
| 1 | 3452 | 35.4 | +0.031 |
| 2 | 3451 | 35.7 | +0.066 |
| 3 | 3452 | 33.7 | -0.002 |
| 4 | 3451 | 33.4 | -0.002 |
| 5 | 3452 | 36.9 | +0.121 |

Weights: d_s50 -0.21, ch_move +0.14, ch_pos +0.11, cci -0.07, room_r -0.06, rsi +0.05, macd_h +0.04, ret20 +0.03, d_hi52 -0.02, ret5 +0.02, trend -0.02, room +0.02, updown +0.02, atr_pct +0.01, momentum -0.01, rvol +0.01, volume -0.01, obv20 -0.01, d_s200 -0.00, not_overbought +0.00, channel -0.00
