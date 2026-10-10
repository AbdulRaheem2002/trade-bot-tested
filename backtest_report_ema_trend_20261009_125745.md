# Backtest Performance Report: EMA_TREND

**Universe:** BTC/USDT  
**Evaluation Period:** 2025-10-18 to 2026-10-07  
**Initial Capital:** $10,000.00  
**Final Equity:** $10,013.40  

## 1. Executive Summary & Benchmark Comparison

| Metric | Strategy Performance | Buy & Hold Benchmark | Relative Alpha |
|---|---|---|---|
| **Total Return** | **+0.13%** | -21.42% | +21.55% |
| **Max Drawdown** | **3.68%** | 49.48% | +45.79% |
| **Sharpe Ratio** | **0.05** | — | — |
| **Sortino Ratio** | **0.07** | — | — |
| **Total Closed Trades** | **14** | — | — |
| **Win Rate** | **14.3%** | — | — |
| **Profit Factor** | **0.36** | — | — |
| **Total Fees Paid** | **$26.78** | — | — |

## 2. Walk-Forward Analysis (Train / Test Split)

| Metric | In-Sample (IS - 70%) | Out-of-Sample (OOS - 30%) | Robustness Ratio |
|---|---|---|---|
| **Return %** | -1.65% | +1.81% | -1.10 |
| **Win Rate %** | 15.4% | 0.0% | — |
| **Profit Factor** | 0.53 | 0.00 | — |
| **Sharpe Ratio** | -0.70 | 0.87 | — |
| **Max Drawdown %** | 3.18% | 3.68% | — |
| **Trades Count** | 13 | 1 | — |

## 3. Trade Quality & Sizing Metrics

- **Average R-Multiple:** `-2.78 R`
- **Expectancy (R):** `-2.78 R` per trade
- **Expectancy (USD):** `$-19.60` per trade
- **Gross Profit:** `$156.42` | **Gross Loss:** `$430.86`
- **Exposure Time:** `0.0%` of total backtest horizon

## 4. Closed Trades Sample

| Time | Symbol | Side | Entry | Exit | Qty | Realized PnL | R | Reason |
|---|---|---|---|---|---|---|---|---|
| 2026-01-06 20:00 | BTC/USDT | long | $93,906.64 | $92,052.14 | 0.0210 | $-40.88 | -1.02 R | `invalidation` |
| 2026-01-08 08:00 | BTC/USDT | long | $92,150.55 | $90,084.05 | 0.0210 | $-45.29 | -1.94 R | `invalidation` |
| 2026-01-08 12:00 | BTC/USDT | long | $90,593.29 | $90,181.66 | 0.0210 | $-10.54 | -2.04 R | `strategy_close` |
| 2026-01-09 20:00 | BTC/USDT | long | $90,826.96 | $90,286.98 | 0.0210 | $-13.24 | -2.27 R | `strategy_close` |
| 2026-01-20 08:00 | BTC/USDT | long | $90,686.60 | $91,132.39 | 0.0210 | $+7.45 | -2.15 R | `strategy_close` |
| 2026-03-16 12:00 | BTC/USDT | long | $73,610.11 | $73,515.99 | 0.0260 | $-4.36 | -2.09 R | `strategy_close` |
| 2026-03-18 16:00 | BTC/USDT | long | $73,999.98 | $71,826.43 | 0.0260 | $-58.38 | -3.01 R | `invalidation` |
| 2026-03-18 20:00 | BTC/USDT | long | $71,503.55 | $71,036.84 | 0.0270 | $-14.52 | -2.88 R | `strategy_close` |
| 2026-03-19 08:00 | BTC/USDT | long | $71,282.16 | $70,156.07 | 0.0270 | $-32.30 | -3.44 R | `strategy_close` |
| 2026-03-26 08:00 | BTC/USDT | long | $71,372.20 | $70,052.40 | 0.0270 | $-37.53 | -4.09 R | `strategy_close` |
| 2026-05-18 00:00 | BTC/USDT | long | $71,823.86 | $77,418.94 | 0.0270 | $+148.98 | -1.98 R | `strategy_close` |
| 2026-05-21 12:00 | BTC/USDT | long | $78,117.76 | $77,150.51 | 0.0250 | $-26.11 | -3.73 R | `strategy_close` |
| 2026-05-22 08:00 | BTC/USDT | long | $77,818.87 | $77,369.36 | 0.0250 | $-13.17 | -4.14 R | `strategy_close` |
| 2026-07-16 04:00 | BTC/USDT | long | $69,464.87 | $64,723.90 | 0.0280 | $-134.56 | -4.10 R | `invalidation` |
