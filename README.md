
![Alt text](allmight.jpeg)

These are the strategies being used
1. Funding Rate Strategy
Purpose: Trades based on crypto futures funding rates

Key Parameters:

Threshold: 0.0005 (enters when funding rate exceeds ± this value)

Exit Threshold: 0.0001 (exits when funding rate normalizes below this)

Symbols Monitored: BTCUSDT

Current Status:

Funding rate = 0.00010000 (below threshold → no trades)

2. Technical Strategy.....
Purpose: Uses RSI, MACD, and volume for entries/exits

Key Parameters:

RSI Buy: 35 (enters LONG if RSI < 35)

RSI Sell: 65 (enters SHORT if RSI > 65)

Volume Multiplier: 1.0 (requires volume > moving average)

Symbols Monitored:

BTCUSDT, ETHUSDT, 1000SHIBUSDT

Current Status:

No trades yet (conditions not met in logs)

3. Ratio Reversion Strategy
Purpose: Trades BTC/ETH ratio mean reversion

Key Parameters:

Entry Z-Score: 1.5 (enters when ratio deviates ±1.5σ from mean)

Exit Z-Score: 0.5 (exits when ratio returns to ±0.5σ)

History Window: 30 data points

Current Status:

Ratio = 42.1297 (latest)

Z-Score = -0.39 (too close to mean → no trade)

Waiting for:

Z-score to exceed ±1.5 (e.g., <-1.5 → long BTC/short ETH, >1.5 → short BTC/long ETH)

4. Liquidation Strategy
Purpose: Trades large liquidation clusters

Key Parameters:

Threshold: $1M liquidation volume (not triggered in testnet)

Cooldown: 1 hour after a signal

Symbols Monitored:

BTCUSDT, ETHUSDT, 1000SHIBUSDT

Current Status:

Testnet mode → returns 0 (no live liquidation data)
