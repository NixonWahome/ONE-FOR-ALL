import os
import time
import csv
import threading
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import ta
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from binance.client import Client
import requests

load_dotenv()
# Initialize decimal precision12332
#UpdAted decimal error fix now
getcontext().prec = 8


# ======== CONFIGURATION ========
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Binance
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    TESTNET = True  # Change to False for live trading

    # Trading Parameters
    LEVERAGE = 3
    MAX_ORDER_ATTEMPTS = 3
    ORDER_DELAY = 2
    MAX_CONCURRENT_TRADES = 3
    DAILY_LOSS_LIMIT = Decimal('-0.1')  # -10%
    MAX_POSITION_SIZE_PCT = Decimal('0.8')
    MIN_MARGIN_BUFFER = Decimal('0.1')

    # Strategy Symbols
    FUNDING_SYMBOLS = ["BTCUSDT", ]
    TECHNICAL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "1000SHIBUSDT"]

    # Risk Management
    RISK_PER_TRADE = Decimal('0.01')  # 1%


# ======== CORE TRADING CLASSES ========
class TradingCore:
    def __init__(self):
        # Initialize Binance client with proper testnet configuration
        if Config.TESTNET:
            self.client = Client(
                api_key=Config.BINANCE_API_KEY,
                api_secret=Config.BINANCE_API_SECRET,
                testnet=True
            )
        else:
            self.client = Client(
                api_key=Config.BINANCE_API_KEY,
                api_secret=Config.BINANCE_API_SECRET
            )

        # Initialize position mode (One-Way Mode)
        self.ensure_position_mode()

        self.active_trades = {}
        self.trade_history = []
        self.daily_pnl = Decimal('0')
        self.lock = threading.Lock()

    def get_available_margin(self):
        """Calculate available margin considering buffer"""
        balance = self.get_account_balance()
        available = balance * Config.MAX_POSITION_SIZE_PCT
        print(
            f"[MARGIN] Available: {float(available):.2f} (Balance: {float(balance):.2f}, Max %: {float(Config.MAX_POSITION_SIZE_PCT) * 100}%)")
        return available

    def close_all_positions(self):
        """Close all open positions before switching modes"""
        try:
            positions = self.client.futures_position_information()
            for pos in positions:
                position_amt = float(pos['positionAmt'])
                if position_amt != 0:
                    symbol = pos['symbol']
                    side = 'SELL' if position_amt > 0 else 'BUY'
                    quantity = abs(position_amt)

                    print(f"[SYSTEM] Closing {symbol} position: {side} {quantity}")

                    # For One-Way mode, we don't need to specify positionSide
                    order_params = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'MARKET',
                        'quantity': quantity,
                        'reduceOnly': True
                    }

                    self.client.futures_create_order(**order_params)
                    time.sleep(0.5)  # Add small delay between orders
            return True
        except Exception as e:
            print(f"[ERROR] Failed to close positions: {e}")
            return False

    def ensure_position_mode(self):
        """Ensure we're in One-Way Mode, closing positions if needed"""
        attempts = 0
        while attempts < Config.MAX_ORDER_ATTEMPTS:
            try:
                position_mode = self.client.futures_get_position_mode()
                print(f"[DEBUG] Current position mode: {position_mode}")

                if position_mode['dualSidePosition']:
                    # First close all positions
                    print("[SYSTEM] Closing all positions to switch modes...")
                    if not self.close_all_positions():
                        raise Exception("Failed to close positions")

                    # Then switch to One-Way Mode
                    print("[SYSTEM] Switching to One-Way Mode...")
                    self.client.futures_change_position_mode(dualSidePosition=False)
                    time.sleep(2)  # Wait for mode change to propagate

                    # Verify the change
                    new_mode = self.client.futures_get_position_mode()
                    if not new_mode['dualSidePosition']:
                        msg = "Successfully switched to One-Way Mode"
                        print(f"[SYSTEM] {msg}")
                        self.send_telegram(f"⚙️ {msg}")
                        return True
                    else:
                        raise Exception("Failed to verify mode switch")
                else:
                    print("[SYSTEM] Already in One-Way Mode")
                    return True

            except Exception as e:
                attempts += 1
                msg = f"Error setting position mode (attempt {attempts}/{Config.MAX_ORDER_ATTEMPTS}): {str(e)}"
                print(f"[ERROR] {msg}")
                self.send_telegram(f"⚠️ {msg}")
                time.sleep(Config.ORDER_DELAY * 2)

        msg = "Failed to set position mode after maximum attempts"
        print(f"[ERROR] {msg}")
        self.send_telegram(f"❌ {msg}")
        return False

    def send_telegram(self, message):
        """Send message to Telegram and print to terminal"""
        try:
            # Print to terminal first
            print(f"[TELEGRAM] {message}")

            # Then send to Telegram
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            params = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": message}
            requests.post(url, params=params, timeout=5)
        except Exception as e:
            print(f"[ERROR] Telegram error: {e}")

    def log_trade(self, trade_data):
        with self.lock:
            # Print trade to terminal
            trade_str = (
                f"[TRADE] Strategy: {trade_data['strategy']} | "
                f"Symbol: {trade_data['symbol']} | "
                f"Side: {trade_data['side']} | "
                f"Qty: {float(trade_data['quantity']):.4f} | "
                f"Entry: {float(trade_data['entry_price']):.2f}"
            )
            if 'exit_price' in trade_data:
                trade_str += (
                    f" | Exit: {float(trade_data['exit_price']):.2f} | "
                    f"PnL: {float(trade_data.get('pnl', 0)):.2f} | "
                    f"Reason: {trade_data.get('stop_reason', '')}"
                )
            print(trade_str)

            # Log to CSV file
            with open('combined_trades.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_data['timestamp'],
                    trade_data['strategy'],
                    trade_data['symbol'],
                    trade_data['side'],
                    float(trade_data['quantity']),
                    float(trade_data['entry_price']),
                    float(trade_data.get('exit_price', 0)),
                    float(trade_data.get('pnl', 0)),
                    trade_data.get('duration', ''),
                    trade_data.get('stop_reason', '')
                ])
            self.trade_history.append(trade_data)

    def get_current_price(self, symbol):
        for _ in range(Config.MAX_ORDER_ATTEMPTS):
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                price = Decimal(str(ticker['price']))
                print(f"[PRICE] {symbol}: {price}")
                return price
            except Exception as e:
                msg = f"Price error {symbol}: {e}"
                print(f"[ERROR] {msg}")
                self.send_telegram(f"⚠️ {msg}")
                time.sleep(Config.ORDER_DELAY)
        return Decimal("0")

    def check_risk_limits(self):
        if self.daily_pnl < Config.DAILY_LOSS_LIMIT:
            msg = f"DAILY LOSS LIMIT REACHED: {self.daily_pnl * 100:.2f}%"
            print(f"[RISK] 🛑 {msg}")
            self.send_telegram(f"🛑 {msg}")
            return False
        return True


class TradingStrategy(TradingCore):
    def __init__(self, name):
        super().__init__()
        self.strategy_name = name
        self.symbol_info_cache = {}

    def get_symbol_info(self, symbol):
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    data = {
                        'min_qty': Decimal(s['filters'][1]['minQty']),
                        'max_qty': Decimal(s['filters'][1]['maxQty']),
                        'step_size': Decimal(s['filters'][1]['stepSize']),
                        'price_precision': s['pricePrecision']
                    }
                    self.symbol_info_cache[symbol] = data
                    return data
        except Exception as e:
            msg = f"Error getting symbol info: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"⚠️ {msg}")
            return {
                'min_qty': Decimal('1'),
                'max_qty': Decimal('100000'),
                'step_size': Decimal('1'),
                'price_precision': 2
            }

    def calculate_position_size(self, symbol, price, atr=None):
        """Calculate position size with margin constraints"""
        symbol_data = self.get_symbol_info(symbol)
        available_margin = self.get_available_margin()
        price = Decimal(str(price))  # Ensure price is Decimal

        if atr is None:
            atr = price * Decimal('0.01')
        else:
            atr = Decimal(str(atr))  # Convert ATR to Decimal if it isn't already

        # Calculate based on risk management
        risk_amount = available_margin * Config.RISK_PER_TRADE
        risk_based_size = (risk_amount / (atr * Decimal('2'))) * Decimal(str(Config.LEVERAGE))

        # Calculate based on available margin
        margin_based_size = (available_margin * Decimal(str(Config.LEVERAGE))) / price

        # Use the smaller of the two
        size = min(risk_based_size, margin_based_size)

        # Apply exchange constraints
        size = max(symbol_data['min_qty'], min(size, symbol_data['max_qty']))
        size = round(size / symbol_data['step_size']) * symbol_data['step_size']

        print(
            f"[SIZE CALC] {symbol}: {size} (Risk-based: {float(risk_based_size):.4f}, Margin-based: {float(margin_based_size):.4f})")
        return size

    def get_account_balance(self):
        for _ in range(Config.MAX_ORDER_ATTEMPTS):
            try:
                balance = self.client.futures_account_balance()
                for item in balance:
                    if item['asset'] == 'USDT':
                        bal = Decimal(item['balance'])
                        print(f"[BALANCE] Current balance: {bal}")
                        return bal
                bal = Decimal('100')  # Fallback for testnet
                print(f"[BALANCE] Using testnet balance: {bal}")
                return bal
            except Exception as e:
                msg = f"Balance error: {e}"
                print(f"[ERROR] {msg}")
                self.send_telegram(f"⚠️ {msg}")
                time.sleep(Config.ORDER_DELAY)
        bal = Decimal('100')
        print(f"[BALANCE] Fallback balance: {bal}")
        return bal

    def can_trade(self, symbol):
        if not self.check_risk_limits():
            print(f"[TRADE] Cannot trade {symbol}: Risk limits exceeded")
            return False

        if symbol in self.active_trades:
            print(f"[TRADE] Cannot trade {symbol}: Already in active trade")
            return False

        if len(self.active_trades) >= Config.MAX_CONCURRENT_TRADES:
            print(f"[TRADE] Cannot trade {symbol}: Max concurrent trades reached")
            return False

        print(f"[TRADE] Can trade {symbol}")
        return True

    def exit_trade(self, symbol, reason):
        if symbol not in self.active_trades:
            msg = f"No active trade for {symbol} to exit"
            print(f"[TRADE] {msg}")
            self.send_telegram(f"ℹ️ {msg}")
            return

        trade = self.active_trades[symbol]
        exit_side = "SELL" if trade['side'] == "LONG" else "BUY"
        exit_price = self.get_current_price(symbol)

        try:
            print(f"[ORDER] Exiting {symbol} {trade['side']} position: {exit_side} {trade['quantity']} @ ~{exit_price}")

            # For One-Way mode, we don't need to specify positionSide
            order_params = {
                'symbol': symbol,
                'side': exit_side,
                'type': "MARKET",
                'quantity': float(trade['quantity']),
                'reduceOnly': True
            }

            self.client.futures_create_order(**order_params)

            if trade['side'] == "LONG":
                pnl = (exit_price - trade['entry_price']) * trade['quantity']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['quantity']

            self.daily_pnl += pnl / self.get_account_balance()

            trade.update({
                'exit_price': exit_price,
                'pnl': pnl,
                'duration': str(timedelta(seconds=time.time() - trade['last_update'])),
                'stop_reason': reason
            })

            self.log_trade(trade)
            msg = f"🔴 Exit {symbol}: {reason} | PnL: {pnl:.2f}"
            print(f"[TRADE] {msg}")
            self.send_telegram(msg)

            del self.active_trades[symbol]

        except Exception as e:
            msg = f"Exit failed {symbol}: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"❌ {msg}")


# ======== STRATEGY IMPLEMENTATIONS ========
class FundingRateStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("funding_rate")
        self.params = {
            "threshold": Decimal("0.0005"),
            "exit_threshold": Decimal("0.0001"),
            "stop_loss": Decimal("0.02"),
            "take_profit": Decimal("0.05")
        }

    def get_funding_rate(self, symbol):
        for _ in range(Config.MAX_ORDER_ATTEMPTS):
            try:
                funding = self.client.futures_funding_rate(symbol=symbol, limit=1)[0]
                rate = Decimal(str(funding['fundingRate']))
                print(f"[FUNDING] {symbol}: {rate}")
                return rate
            except Exception as e:
                msg = f"Funding error {symbol}: {e}"
                print(f"[ERROR] {msg}")
                self.send_telegram(f"⚠️ {msg}")
                time.sleep(Config.ORDER_DELAY)
        return Decimal("0")

    def check_conditions(self):
        print(f"[STRATEGY] Checking funding rate conditions")
        for symbol in Config.FUNDING_SYMBOLS:
            if not self.can_trade(symbol):
                continue

            rate = self.get_funding_rate(symbol)
            price = self.get_current_price(symbol)

            if rate > self.params["threshold"]:
                print(f"[SIGNAL] {symbol}: Funding rate {rate} > threshold {self.params['threshold']} - LONG")
                self.enter_trade(symbol, "LONG", price)
            elif rate < -self.params["threshold"]:
                print(f"[SIGNAL] {symbol}: Funding rate {rate} < threshold {-self.params['threshold']} - SHORT")
                self.enter_trade(symbol, "SHORT", price)

    def enter_trade(self, symbol, side, price, atr=None):
        """Execute trade with proper type conversion and margin verification"""
        try:
            # Ensure all numeric values are Decimal
            price = Decimal(str(price))
            atr = Decimal(str(atr)) if atr is not None else price * Decimal('0.01')

            # Set leverage first
            self.set_leverage(symbol)

            # Calculate position size (returns Decimal)
            quantity = self.calculate_position_size(symbol, price, atr)

            # Convert to float for API call only at the last moment
            quantity_float = float(quantity)

            # Final margin check
            required_margin = (quantity * price) / Decimal(str(Config.LEVERAGE))
            available_margin = self.get_available_margin()

            if required_margin > available_margin:
                raise ValueError(
                    f"Insufficient margin. Need {float(required_margin):.2f} USDT, "
                    f"have {float(available_margin):.2f} USDT"
                )

            # Prepare order parameters
            order_side = 'BUY' if side == 'LONG' else 'SELL'
            order_params = {
                'symbol': symbol,
                'side': order_side,
                'type': 'MARKET',
                'quantity': quantity_float  # Convert to float here
            }

            # Add positionSide for Hedge Mode
            if hasattr(self, 'position_mode') and self.position_mode.get('dualSidePosition', False):
                order_params['positionSide'] = 'LONG' if side == 'LONG' else 'SHORT'

            # Execute order
            print(f"[ORDER] Entering {symbol} {side}: {order_params}")
            self.client.futures_create_order(**order_params)

            # Record trade
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'strategy': self.strategy_name,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'atr': atr,
                'margin_used': required_margin,
                'last_update': time.time()
            }

            self.active_trades[symbol] = trade_data
            self.log_trade(trade_data)

            # Send notification
            msg = (f"✅ {self.strategy_name} {side} {symbol}\n"
                   f"Size: {quantity_float:.4f} @ {float(price):.2f}\n"
                   f"Margin: {float(required_margin):.2f} USDT")
            print(f"[TRADE] {msg.splitlines()[0]}")
            self.send_telegram(msg)
            return True

        except Exception as e:
            error_msg = f"Failed {side} {symbol}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.send_telegram(f"❌ {error_msg}")
            return False

    def manage_trades(self):
        print(f"[STRATEGY] Managing funding rate trades")
        for symbol, trade in list(self.active_trades.items()):
            if trade['strategy'] != self.strategy_name:
                continue

            current_price = self.get_current_price(symbol)
            if current_price == 0:
                continue

            if trade['side'] == "LONG":
                pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
            else:
                pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']

            if pnl_pct <= -self.params["stop_loss"]:
                print(f"[EXIT] {symbol}: Stop loss triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "SL")
            elif pnl_pct >= self.params["take_profit"]:
                print(f"[EXIT] {symbol}: Take profit triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "TP")
            else:
                rate = self.get_funding_rate(symbol)
                if abs(rate) < self.params["exit_threshold"]:
                    print(f"[EXIT] {symbol}: Funding rate normalized ({rate})")
                    self.exit_trade(symbol, "Funding normalized")


class TechnicalStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("technical")
        self.params = {
            "rsi_buy": 35,
            "rsi_sell": 65,
            "volume_multiplier": 1.0,
            "stop_loss": Decimal('0.03'),
            "take_profit": Decimal('0.015')
        }
        self.historical_data = {}

    def get_historical_data(self, symbol, interval='5m', lookback=100):
        if symbol in self.historical_data:
            return self.historical_data[symbol]

        try:
            print(f"[DATA] Fetching historical data for {symbol}")
            bars = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=lookback
            )
            df = pd.DataFrame(bars, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df = self.calculate_indicators(df)
            self.historical_data[symbol] = df
            return df
        except Exception as e:
            msg = f"Historical data error {symbol}: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"⚠️ {msg}")
            return None

    def calculate_indicators(self, df):
        try:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd_diff()
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14
            ).average_true_range()
            return df.fillna(0)
        except Exception as e:
            msg = f"Indicator error: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"⚠️ {msg}")
            return df

    def check_conditions(self):
        print(f"[STRATEGY] Checking technical conditions")
        for symbol in Config.TECHNICAL_SYMBOLS:
            if not self.can_trade(symbol):
                continue

            df = self.get_historical_data(symbol)
            if df is None or len(df) < 50:
                msg = f"Insufficient data for {symbol}"
                print(f"[DATA] {msg}")
                continue

            last_row = df.iloc[-1]
            price = self.get_current_price(symbol)

            if (last_row['rsi'] < self.params["rsi_buy"] and
                    last_row['macd'] < 0 and
                    last_row['close'] < last_row['vwap'] and
                    last_row['volume_ratio'] > self.params["volume_multiplier"]):

                atr = last_row['atr'] if not np.isnan(last_row['atr']) else price * Decimal('0.01')
                print(f"[SIGNAL] {symbol}: Bullish setup (RSI: {last_row['rsi']:.2f}, MACD: {last_row['macd']:.4f})")
                self.enter_trade(symbol, "LONG", price, atr)

            elif (last_row['rsi'] > self.params["rsi_sell"] and
                  last_row['macd'] > 0 and
                  last_row['close'] > last_row['vwap'] and
                  last_row['volume_ratio'] > self.params["volume_multiplier"]):

                atr = last_row['atr'] if not np.isnan(last_row['atr']) else price * Decimal('0.01')
                print(f"[SIGNAL] {symbol}: Bearish setup (RSI: {last_row['rsi']:.2f}, MACD: {last_row['macd']:.4f})")
                self.enter_trade(symbol, "SHORT", price, atr)

    def enter_trade(self, symbol, side, price, atr=None):
        """Completely type-safe trade execution with margin checks"""
        try:
            # 1. CONVERT ALL INPUTS TO DECIMAL FIRST
            def to_decimal(value):
                if isinstance(value, Decimal):
                    return value
                return Decimal(str(float(value)))  # Handle numpy floats etc.

            price = to_decimal(price)
            atr = to_decimal(atr) if atr is not None else price * Decimal('0.01')

            # 2. SET LEVERAGE (ensures it's configured)
            self.set_leverage(symbol)

            # 3. CALCULATE POSITION SIZE (returns Decimal)
            quantity = self.calculate_position_size(symbol, price, atr)
            if not isinstance(quantity, Decimal):
                quantity = to_decimal(quantity)

            # 4. MARGIN VERIFICATION (all Decimal math)
            leverage = to_decimal(Config.LEVERAGE)
            required_margin = (quantity * price) / leverage
            available_margin = self.get_available_margin()

            if required_margin > available_margin:
                raise ValueError(
                    f"Insufficient margin: Requires {float(required_margin):.2f} USDT, "
                    f"only {float(available_margin):.2f} available"
                )

            # 5. PREPARE ORDER (convert to float ONLY at last step)
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if side == 'LONG' else 'SELL',
                'type': 'MARKET',
                'quantity': float(quantity)  # Binance API requires float
            }

            # Hedge mode support
            if getattr(self, 'position_mode', {}).get('dualSidePosition', False):
                order_params['positionSide'] = 'LONG' if side == 'LONG' else 'SHORT'

            # 6. EXECUTE TRADE
            print(f"[ORDER] Submitting {order_params}")
            self.client.futures_create_order(**order_params)

            # 7. RECORD TRADE (keep as Decimal internally)
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'strategy': self.strategy_name,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'atr': atr,
                'margin_used': required_margin,
                'last_update': time.time()
            }
            self.active_trades[symbol] = trade_data
            self.log_trade(trade_data)

            # 8. SEND NOTIFICATION
            msg = (f"✅ {side} {symbol}\n"
                   f"Size: {float(quantity):.4f} @ {float(price):.2f}\n"
                   f"Margin: {float(required_margin):.2f} USDT")
            self.send_telegram(msg)
            return True

        except Exception as e:
            error_msg = f"Failed {side} {symbol}: {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.send_telegram(f"❌ {error_msg}")
            return False

    def manage_trades(self):
        print(f"[STRATEGY] Managing technical trades")
        for symbol, trade in list(self.active_trades.items()):
            if trade['strategy'] != self.strategy_name:
                continue

            current_price = self.get_current_price(symbol)
            if current_price == 0:
                continue

            if trade['side'] == "LONG":
                pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
            else:
                pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']

            print(f"[TRADE] {symbol} {trade['side']} PnL: {pnl_pct * 100:.2f}%")

            if pnl_pct <= -self.params["stop_loss"]:
                print(f"[EXIT] {symbol}: Stop loss triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "SL")
            elif pnl_pct >= self.params["take_profit"]:
                print(f"[EXIT] {symbol}: Take profit triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "TP")


class RatioReversionStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("ratio_reversion")
        self.params = {
            "entry_z": Decimal('1.5'),
            "exit_z": Decimal('0.5'),
            "stop_loss": Decimal('0.08'),
            "take_profit": Decimal('0.05')
        }
        self.ratio_history = []
        self.history_window = 30
        self.historical_data = {}

    def get_btc_eth_ratio(self):
        """Calculate current BTC/ETH price ratio"""
        for _ in range(Config.MAX_ORDER_ATTEMPTS):
            try:
                btc_price = self.get_current_price("BTCUSDT")
                eth_price = self.get_current_price("ETHUSDT")
                if eth_price == 0:
                    return Decimal('0')
                ratio = btc_price / eth_price
                print(f"[RATIO] BTC/ETH: {ratio:.4f}")
                return ratio
            except Exception as e:
                print(f"[ERROR] Ratio calculation failed: {e}")
                time.sleep(Config.ORDER_DELAY)
        return Decimal('0')

    def update_ratio_history(self):
        """Maintain rolling window of ratio history"""
        try:
            ratio = self.get_btc_eth_ratio()
            self.ratio_history.append(float(ratio))
            if len(self.ratio_history) > self.history_window:
                self.ratio_history.pop(0)
            print(f"[RATIO] History updated (length: {len(self.ratio_history)})")
        except Exception as e:
            print(f"[ERROR] Failed to update ratio history: {e}")

    def get_z_score(self):
        """Calculate z-score of current ratio vs history"""
        if len(self.ratio_history) < 5:
            print("[RATIO] Insufficient history for z-score")
            return Decimal('0')

        try:
            current_ratio = float(self.get_btc_eth_ratio())
            mean = np.mean(self.ratio_history)
            std = np.std(self.ratio_history)
            z_score = Decimal(str((current_ratio - mean) / std)) if std != 0 else Decimal('0')
            print(f"[RATIO] Z-Score: {z_score:.2f} (Mean: {mean:.4f}, Std: {std:.4f})")
            return z_score
        except Exception as e:
            print(f"[ERROR] Z-score calculation failed: {e}")
            return Decimal('0')

    def enter_trade(self, symbol, side, price, atr=None):
        """Enter a trade for the ratio reversion strategy"""
        try:
            price = Decimal(str(price))
            atr_decimal = Decimal(str(atr)) if atr is not None else price * Decimal('0.01')
            quantity = self.calculate_position_size(symbol, price, atr_decimal)

            if quantity <= 0:
                msg = f"Invalid quantity for {symbol}: {quantity}"
                print(f"[ERROR] {msg}")
                self.send_telegram(f"⚠️ {msg}")
                return False

            order_side = "BUY" if side == "LONG" else "SELL"

            print(f"[ORDER] Entering {symbol} {side}: {order_side} {quantity} @ ~{price}")
            self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type="MARKET",
                quantity=float(quantity)
            )

            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'strategy': self.strategy_name,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'atr': atr_decimal,
                'last_update': time.time()
            }

            self.active_trades[symbol] = trade_data
            self.log_trade(trade_data)

            msg = (f"🚀 {self.strategy_name} {side} {symbol}\n"
                   f"Size: {float(quantity):.4f} @ {float(price):.2f}\n"
                   f"ATR: {float(atr_decimal):.4f}")
            print(f"[TRADE] {msg.splitlines()[0]}")
            self.send_telegram(msg)
            return True

        except Exception as e:
            msg = f"Failed to enter {side} trade for {symbol}: {str(e)}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"❌ {msg}")
            return False

    def check_conditions(self):
        """Check for ratio reversion opportunities"""
        print(f"[STRATEGY] Checking ratio reversion conditions")

        if not all(s in Config.TECHNICAL_SYMBOLS for s in ["BTCUSDT", "ETHUSDT"]):
            print("[RATIO] Required symbols not in technical symbols list")
            return

        if "BTCUSDT" in self.active_trades or "ETHUSDT" in self.active_trades:
            print("[RATIO] Already in BTC/ETH trade")
            return

        self.update_ratio_history()
        z_score = self.get_z_score()
        btc_price = self.get_current_price("BTCUSDT")
        eth_price = self.get_current_price("ETHUSDT")

        btc_atr = self.get_historical_data("BTCUSDT")['atr'].iloc[
            -1] if "BTCUSDT" in self.historical_data else btc_price * Decimal('0.01')
        eth_atr = self.get_historical_data("ETHUSDT")['atr'].iloc[
            -1] if "ETHUSDT" in self.historical_data else eth_price * Decimal('0.01')

        if z_score < -self.params["entry_z"]:
            print(f"[SIGNAL] Z-Score {z_score:.2f} < entry threshold {-self.params['entry_z']} - Long BTC/Short ETH")
            if self.can_trade("BTCUSDT"):
                self.enter_trade("BTCUSDT", "LONG", btc_price, btc_atr)
            if self.can_trade("ETHUSDT"):
                self.enter_trade("ETHUSDT", "SHORT", eth_price, eth_atr)

        elif z_score > self.params["entry_z"]:
            print(f"[SIGNAL] Z-Score {z_score:.2f} > entry threshold {self.params['entry_z']} - Short BTC/Long ETH")
            if self.can_trade("BTCUSDT"):
                self.enter_trade("BTCUSDT", "SHORT", btc_price, btc_atr)
            if self.can_trade("ETHUSDT"):
                self.enter_trade("ETHUSDT", "LONG", eth_price, eth_atr)

    def manage_trades(self):
        """Manage open ratio reversion trades"""
        print(f"[STRATEGY] Managing ratio reversion trades")

        btc_trade = self.active_trades.get("BTCUSDT")
        eth_trade = self.active_trades.get("ETHUSDT")

        if not btc_trade or not eth_trade:
            print("[RATIO] No active BTC/ETH pair trade")
            return

        if btc_trade['strategy'] != self.strategy_name or eth_trade['strategy'] != self.strategy_name:
            print("[RATIO] Active trades not from ratio strategy")
            return

        current_z = self.get_z_score()
        if abs(current_z) < self.params["exit_z"]:
            print(f"[EXIT] Z-Score {current_z:.2f} normalized below {self.params['exit_z']}")
            self.exit_trade("BTCUSDT", "Ratio normalized")
            self.exit_trade("ETHUSDT", "Ratio normalized")
            return

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            trade = self.active_trades.get(symbol)
            if not trade:
                continue

            current_price = self.get_current_price(symbol)
            if trade['side'] == "LONG":
                pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
            else:
                pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']

            print(f"[TRADE] {symbol} {trade['side']} PnL: {pnl_pct * 100:.2f}%")

            if pnl_pct <= -self.params["stop_loss"]:
                print(f"[EXIT] {symbol}: Stop loss triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "SL")
            elif pnl_pct >= self.params["take_profit"]:
                print(f"[EXIT] {symbol}: Take profit triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "TP")

    def get_historical_data(self, symbol, interval='5m', lookback=100):
        if symbol in self.historical_data:
            return self.historical_data[symbol]

        try:
            print(f"[DATA] Fetching historical data for {symbol}")
            bars = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=lookback
            )
            df = pd.DataFrame(bars, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df = self.calculate_indicators(df)
            self.historical_data[symbol] = df
            return df
        except Exception as e:
            msg = f"Historical data error {symbol}: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"⚠️ {msg}")
            return None

    def calculate_indicators(self, df):
        try:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd_diff()
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14
            ).average_true_range()
            return df.fillna(0)
        except Exception as e:
            msg = f"Indicator error: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"⚠️ {msg}")
            return df


class LiquidationStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("liquidation")
        self.params = {
            "threshold": Decimal('1000000'),
            "stop_loss": Decimal('0.15'),
            "take_profit": Decimal('0.10'),
            "cooldown": 3600
        }
        self.last_signal_time = 0

    def get_liquidation_volume(self, symbol):
        try:
            if Config.TESTNET:
                print(f"[LIQUIDATION] Testnet - returning 0 for {symbol}")
                return Decimal('0')

            liquidations = self.client.futures_liquidation_orders(
                symbol=symbol,
                limit=10
            )
            vol = Decimal(str(sum(float(liq['qty']) for liq in liquidations)))
            print(f"[LIQUIDATION] {symbol}: {vol:,.0f}")
            return vol
        except Exception as e:
            msg = f"Liquidation error {symbol}: {e}"
            print(f"[ERROR] {msg}")
            self.send_telegram(f"⚠️ {msg}")
            return Decimal('0')

    def check_conditions(self):
        print(f"[STRATEGY] Checking liquidation conditions")
        if time.time() - self.last_signal_time < self.params["cooldown"]:
            remaining = self.params["cooldown"] - (time.time() - self.last_signal_time)
            print(f"[LIQUIDATION] In cooldown ({remaining:.0f}s remaining)")
            return

        for symbol in Config.TECHNICAL_SYMBOLS:
            if not self.can_trade(symbol):
                continue

            liquidations = self.get_liquidation_volume(symbol)
            if liquidations > self.params["threshold"]:
                price = self.get_current_price(symbol)
                atr = self.get_historical_data(symbol)['atr'].iloc[
                    -1] if symbol in self.historical_data else price * Decimal('0.02')

                print(
                    f"[SIGNAL] {symbol}: Large liquidations {liquidations:,.0f} > threshold {self.params['threshold']:,.0f}")
                self.enter_trade(symbol, "LONG", price, atr)
                self.last_signal_time = time.time()
                msg = f"🚨 Large liquidations detected: {symbol} ({liquidations:,.0f})"
                print(f"[ALERT] {msg}")
                self.send_telegram(msg)
                break

    def manage_trades(self):
        print(f"[STRATEGY] Managing liquidation trades")
        for symbol, trade in list(self.active_trades.items()):
            if trade['strategy'] != self.strategy_name:
                continue

            current_price = self.get_current_price(symbol)
            if current_price == 0:
                continue

            pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
            print(f"[TRADE] {symbol} {trade['side']} PnL: {pnl_pct * 100:.2f}%")

            if pnl_pct <= -self.params["stop_loss"]:
                print(f"[EXIT] {symbol}: Stop loss triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "SL")
            elif pnl_pct >= self.params["take_profit"]:
                print(f"[EXIT] {symbol}: Take profit triggered ({pnl_pct * 100:.2f}%)")
                self.exit_trade(symbol, "TP")


# ======== STRATEGY COORDINATOR ========
class StrategyManager:
    def __init__(self):
        self.strategies = [
            FundingRateStrategy(),
            TechnicalStrategy(),
            RatioReversionStrategy(),
            LiquidationStrategy()
        ]
        print("[SYSTEM] Strategy manager initialized")
        self.send_startup_message()

    def send_startup_message(self):
        msg = "🚀 Trading bot started successfully!"
        print(f"[SYSTEM] {msg}")
        for strategy in self.strategies:
            try:
                strategy.send_telegram(msg)
                break  # Only need to send once
            except:
                continue

    def run_cycle(self):
        cycle_start = time.time()
        print(f"\n[SYSTEM] Starting new cycle at {datetime.now().isoformat()}")

        # Update data
        for strategy in self.strategies:
            if hasattr(strategy, 'update_ratio_history'):
                strategy.update_ratio_history()
            if hasattr(strategy, 'get_historical_data'):
                for symbol in Config.TECHNICAL_SYMBOLS:
                    strategy.get_historical_data(symbol)

        # Check conditions
        for strategy in self.strategies:
            strategy.check_conditions()

        # Manage trades
        for strategy in self.strategies:
            strategy.manage_trades()

        # Clean up historical data
        for strategy in self.strategies:
            if hasattr(strategy, 'historical_data'):
                for symbol in list(strategy.historical_data.keys()):
                    if symbol not in Config.TECHNICAL_SYMBOLS:
                        del strategy.historical_data[symbol]

        cycle_duration = time.time() - cycle_start
        print(f"[SYSTEM] Cycle completed in {cycle_duration:.2f}s")


if __name__ == "__main__":
    print("[SYSTEM] Initializing trading bot...")
    manager = StrategyManager()
    while True:
        manager.run_cycle()
        sleep_time = 60 - datetime.now().second
        print(f"[SYSTEM] Sleeping for {sleep_time}s until next minute")
        time.sleep(sleep_time)
