# Trader Dash

Stock dashboard for my TradingView watchlist: trend channel, support and resistance, CCI/RSI/MACD,
volume analysis, analysts, fundamentals and a rule-based entry checklist.

- Updates itself every 30 minutes while the US market is open (GitHub Actions), served by GitHub Pages.
- Edit `watchlist.json` to change the stocks, `app_config.json` for budget and risk.
- Data: Yahoo Finance (free, unofficial, can lag). Analysis tooling, not financial advice.
- Charts: TradingView Lightweight Charts. Scanner code adapted from day-trader-skill (MIT).
