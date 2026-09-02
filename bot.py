import asyncio
import json
import sys
import threading
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import pandas as pd
import requests
from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.error import BadRequest, Forbidden, TelegramError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

keepalive_app = Flask(__name__)


@keepalive_app.route("/")
def keepalive_home():
    return "ARIA Crypto Engine is running."


@keepalive_app.route("/health")
def keepalive_health():
    return "OK", 200


def run_flask():
    keepalive_app.run(host="0.0.0.0", port=10000)

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
LOG_FILE = "signals_log.json"
TRADES_FILE = "paper_trades.json"
WHALES_FILE = "whales_log.json"
ALERTS_FILE = "alerts_log.json"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
TIMEOUT = 10

WHALE_USD_THRESHOLD = 5_000_000
VOLUME_SPIKE_RATIO = 5.0
VOLUME_AVG_PERIOD = 20

RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed"),
]

SENTIMENT_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}

FEAR_GREED_EMOJI = {
    "Extreme Fear": "😱",
    "Fear": "😨",
    "Neutral": "😐",
    "Greed": "😊",
    "Extreme Greed": "🤩",
}

BULLISH_WORDS = [
    "bull", "bullish", "soar", "surge", "rally", "gain", "record",
    "breakout", "adoption", "buy", "upgrade", "inflows", "approval",
]
BEARISH_WORDS = [
    "bear", "bearish", "drop", "crash", "sell", "fall", "plunge",
    "ban", "lawsuit", "crackdown", "loss", "hack", "outflows", "downgrade",
]

TELEGRAM_TOKEN = "8877914394:AAHSPfZx3x2E2qQl929LedIj9NTfUnwgaMw"
CHAT_ID = "758980281"

SUPPORTED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
    "XRPUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
]
SYMBOL_DISPLAY = {s: s.replace("USDT", "/USDT") for s in SUPPORTED_SYMBOLS}
SHORT_MAP = {s.replace("USDT", ""): s for s in SUPPORTED_SYMBOLS}

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

COINLORE_SLUGS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binance-coin",
    "ADAUSDT": "cardano",
    "XRPUSDT": "ripple",
    "DOGEUSDT": "dogecoin",
    "DOTUSDT": "polkadot",
    "LINKUSDT": "chainlink",
    "LTCUSDT": "litecoin",
}

SYMBOL_KEYWORDS = {
    "BTCUSDT": ["btc", "bitcoin"],
    "ETHUSDT": ["eth", "ethereum"],
    "SOLUSDT": ["sol", "solana"],
    "DOGEUSDT": ["doge", "dogecoin"],
    "XRPUSDT": ["xrp", "ripple"],
    "ADAUSDT": ["ada", "cardano"],
    "BNBUSDT": ["bnb", "binance"],
    "DOTUSDT": ["dot", "polkadot"],
    "LINKUSDT": ["link", "chainlink"],
    "LTCUSDT": ["ltc", "litecoin"],
}

SIGNAL_EMOJI = {
    "STRONG BUY": "🟢",
    "BUY": "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
    "STRONG SELL": "🔴",
}

DEFAULT_NOTIONAL = 1000.0


def fetch_klines(symbol, interval="1h", limit=100):
    """Fetch klines from Binance and return a pandas DataFrame."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        print(f"⏱️  TIMEOUT: Request for {symbol} ({interval}) exceeded {TIMEOUT}s.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as exc:
        print(f"❌ ERROR: Failed to fetch {symbol} ({interval}) — {exc}")
        return pd.DataFrame()

    if not data or not isinstance(data, list):
        print(f"⚠️  WARNING: No data returned for {symbol} ({interval}).")
        return pd.DataFrame()

    rows = [
        {
            "Open": float(item[1]),
            "High": float(item[2]),
            "Low": float(item[3]),
            "Close": float(item[4]),
            "Volume": float(item[5]),
            "Date": pd.to_datetime(item[0], unit="ms"),
        }
        for item in data
    ]
    return pd.DataFrame(rows).set_index("Date")


def calculate_rsi(closes, period=14):
    """Calculate the RSI for the given sequence of closing prices."""
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(closes, period):
    """Calculate the EMA series for the given sequence of closing prices."""
    if len(closes) < period:
        return []

    multiplier = 2 / (period + 1)
    ema = [sum(closes[:period]) / period]

    for price in closes[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema


def calculate_macd(closes, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, histogram and crossover state."""
    if len(closes) < slow + signal:
        return {"macd": None, "signal": None, "histogram": None, "cross": None}

    series = pd.Series(closes)
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    cross = None
    if len(macd_line) >= 2:
        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            cross = "BULLISH"
        elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            cross = "BEARISH"

    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
        "cross": cross,
    }


def calculate_bollinger(closes, period=20, std=2):
    """Calculate Bollinger Bands (upper, middle, lower)."""
    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None}

    series = pd.Series(closes)
    middle = series.rolling(window=period).mean()
    sd = series.rolling(window=period).std()
    return {
        "upper": float((middle + std * sd).iloc[-1]),
        "middle": float(middle.iloc[-1]),
        "lower": float((middle - std * sd).iloc[-1]),
    }


def calculate_atr(df, period=14):
    """Calculate the Average True Range (latest value)."""
    if df.empty or len(df) < period + 1:
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(atr.iloc[-1])


def calculate_adx(df, period=14):
    """Calculate the Average Directional Index (latest value)."""
    if df.empty or len(df) < period * 2:
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return float(adx.iloc[-1])


def volume_confirmation(df, period=20, multiplier=1.5):
    """Return True if latest volume exceeds `multiplier` * SMA(volume)."""
    if df.empty or len(df) < period + 1:
        return False

    volumes = df["Volume"]
    sma_volume = volumes.rolling(window=period).mean()
    avg = float(sma_volume.iloc[-1])
    if avg <= 0:
        return False
    return float(volumes.iloc[-1]) > multiplier * avg


def detect_candlestick_pattern(df):
    """Detect basic patterns on the last closed candle."""
    if df.empty or len(df) < 2:
        return None

    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    open1, close1 = o[-1], c[-1]
    high1, low1 = h[-1], l[-1]
    open0, close0 = o[-2], c[-2]

    body1 = abs(close1 - open1)
    body0 = abs(close0 - open0)

    # Engulfing patterns
    if close0 < open0 and close1 > open1 and close1 >= open0 and open1 <= close0:
        return "bullish_engulfing"
    if close0 > open0 and close1 < open1 and close1 <= open0 and open1 >= close0:
        return "bearish_engulfing"

    # Hammer / Shooting Star
    if body1 == 0:
        return None
    lower_wick = min(open1, close1) - low1
    upper_wick = high1 - max(open1, close1)

    if lower_wick >= 2 * body1 and upper_wick <= body1:
        return "hammer"
    if upper_wick >= 2 * body1 and lower_wick <= body1:
        return "shooting_star"

    return None


def compute_consensus(price, rsi, macd_info, boll, ema50, ema200, vol_ok=False, pattern=None):
    """Dynamic weighted consensus engine.

    Each indicator contributes fairly; volume confirmation nudges the leading
    side by +1 instead of multiplying the whole score.
    """
    bullish = 0
    bearish = 0

    # RSI
    if rsi is not None:
        if rsi <= 35:
            bullish += 2
        elif rsi <= 45:
            bullish += 1
        elif rsi >= 65:
            bearish += 2
        elif rsi >= 55:
            bearish += 1

    # MACD cross + histogram
    if macd_info["cross"] == "BULLISH":
        bullish += 2
    elif macd_info["cross"] == "BEARISH":
        bearish += 2
    if macd_info["histogram"] is not None:
        if macd_info["histogram"] > 0:
            bullish += 1
        elif macd_info["histogram"] < 0:
            bearish += 1

    # Bollinger Bands
    if boll["lower"] is not None and price <= boll["lower"]:
        bullish += 2
    if boll["upper"] is not None and price >= boll["upper"]:
        bearish += 2

    # EMA trend
    if ema50 is not None:
        if price > ema50:
            bullish += 1
        else:
            bearish += 1
    if ema200 is not None:
        if price > ema200:
            bullish += 1
        else:
            bearish += 1

    # Candlestick pattern
    if pattern in ("bullish_engulfing", "hammer"):
        bullish += 2
    elif pattern in ("bearish_engulfing", "shooting_star"):
        bearish += 2

    net = bullish - bearish

    # Volume confirmation: add +1 to the leading side
    if vol_ok:
        if net > 0:
            net += 1
        elif net < 0:
            net -= 1

    if net >= 4:
        signal = "STRONG BUY"
    elif net >= 2:
        signal = "BUY"
    elif net <= -4:
        signal = "STRONG SELL"
    elif net <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(98, max(50, round(50 + abs(net) * 8)))
    return signal, confidence, bullish, bearish


def analyze_indicators(df):
    """Compute all indicators for a DataFrame and run the consensus engine."""
    if df.empty:
        return None

    closes = df["Close"].to_numpy()
    price = float(closes[-1])
    rsi = calculate_rsi(closes)
    macd_info = calculate_macd(closes)
    boll = calculate_bollinger(closes)
    ema50 = calculate_ema(closes, 50)
    ema200 = calculate_ema(closes, 200)
    atr = calculate_atr(df)
    adx = calculate_adx(df)
    vol_ok = volume_confirmation(df)
    pattern = detect_candlestick_pattern(df)

    ema50_last = ema50[-1] if ema50 else None
    ema200_last = ema200[-1] if ema200 else None

    signal, confidence, bullish, bearish = compute_consensus(
        price, rsi, macd_info, boll, ema50_last, ema200_last, vol_ok, pattern
    )

    return {
        "price": price,
        "rsi": rsi,
        "macd": macd_info,
        "boll": boll,
        "ema50": ema50_last,
        "ema200": ema200_last,
        "atr": atr,
        "adx": adx,
        "vol_ok": vol_ok,
        "pattern": pattern,
        "signal": signal,
        "confidence": confidence,
        "bullish": bullish,
        "bearish": bearish,
    }


def generate_atr_tp_sl(price, atr, signal):
    """Dynamic TP/SL based on ATR."""
    if atr is None or atr <= 0:
        return None

    if signal in ("BUY", "STRONG BUY"):
        return {
            "SL": price - 1.5 * atr,
            "TP1": price + 1.5 * atr,
            "TP2": price + 3 * atr,
        }
    if signal in ("SELL", "STRONG SELL"):
        return {
            "SL": price + 1.5 * atr,
            "TP1": price - 1.5 * atr,
            "TP2": price - 3 * atr,
        }
    return None


def timeframe_trend(df):
    """Return EMA50 slope direction ('bullish'/'bearish'/'neutral')."""
    if df.empty:
        return "neutral"

    ema50 = calculate_ema(df["Close"].to_numpy(), 50)
    if len(ema50) < 3:
        return "neutral"

    slope = ema50[-1] - ema50[-3]
    if slope > 0:
        return "bullish"
    if slope < 0:
        return "bearish"
    return "neutral"


def confluence_badge(direction, tf_trends):
    """Return 🟢/🔴/🟡 based on whether shorter/longer timeframes agree."""
    if direction == "neutral":
        return "🟡"

    bullish_count = sum(1 for t in tf_trends if t == "bullish")
    bearish_count = sum(1 for t in tf_trends if t == "bearish")

    if direction == "bullish":
        if bullish_count == len(tf_trends):
            return "🟢"
        if bearish_count == len(tf_trends):
            return "🔴"
    else:
        if bearish_count == len(tf_trends):
            return "🟢"
        if bullish_count == len(tf_trends):
            return "🔴"
    return "🟡"


def normalize_symbol(text):
    """Standardize user input into a Binance USDT pair."""
    if not text:
        return None

    symbol = text.strip().upper()
    if symbol in SUPPORTED_SYMBOLS:
        return symbol
    if symbol in SHORT_MAP:
        return SHORT_MAP[symbol]
    if symbol.endswith("USDT"):
        return symbol
    if symbol.isalpha() and 2 <= len(symbol) <= 12:
        return symbol + "USDT"
    return None


def get_price(symbol):
    df = fetch_klines(symbol, interval="1h", limit=1)
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def fetch_fear_greed():
    """Fetch the Crypto Fear & Greed Index from alternative.me."""
    try:
        response = requests.get(FEAR_GREED_URL, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        item = data["data"][0]
        return int(item["value"]), item.get("value_classification", "Neutral")
    except Exception as exc:
        print(f"❌ ERROR: Failed to fetch Fear & Greed index — {exc}")
        return None, None


def fear_greed_bar(value, segments=10):
    filled = round(value / 100 * segments)
    return "🟩" * filled + "⬜" * (segments - filled)


def load_whales():
    try:
        with open(WHALES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_whales(events):
    with open(WHALES_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


def record_whale_event(event):
    events = load_whales()
    key = event.get("key")
    if key and any(e.get("key") == key for e in events):
        return False
    events.append(event)
    save_whales(events)
    return True


def detect_volume_spike(symbol):
    """Detect a whale volume spike using Binance 1m data."""
    df = fetch_klines(symbol, interval="1m", limit=VOLUME_AVG_PERIOD + 2)
    if df.empty or len(df) < VOLUME_AVG_PERIOD + 1:
        return None

    volumes = df["Volume"].to_numpy()
    closes = df["Close"].to_numpy()

    avg_volume = float(volumes[:-1].mean())
    if avg_volume <= 0:
        return None

    latest_volume = float(volumes[-1])
    ratio = latest_volume / avg_volume
    if ratio < VOLUME_SPIKE_RATIO:
        return None

    price = float(closes[-1])
    prev_price = float(closes[-2]) if len(closes) >= 2 else price
    pct_change = (price - prev_price) / prev_price * 100 if prev_price else 0.0
    usd_value = latest_volume * price
    asset = symbol.replace("USDT", "")
    minute_key = df.index[-1].strftime("%Y%m%d%H%M")

    if pct_change >= 0.5:
        flow = "🟢 Binance ➔ Unknown Wallet"
        classification = "🟢 Exchange to Wallet (Accumulation/HODL Signal)"
        impact = "Strong accumulation — potential bullish momentum!"
    elif pct_change <= -0.5:
        flow = "🔴 Unknown Wallet ➔ Binance"
        classification = "🔴 Wallet to Exchange (Potential Dumping/Sell Pressure)"
        impact = "Expect high volatility & potential sell pressure!"
    else:
        flow = "🟡 Unknown Wallet ➔ Unknown Wallet"
        classification = "🟡 Large Internal Transfer (Volatility Warning)"
        impact = "High volatility expected — monitor closely!"

    return {
        "key": f"spike:{asset}:{minute_key}",
        "asset": asset,
        "symbol": symbol,
        "amount": latest_volume,
        "usd_value": usd_value,
        "ratio": round(ratio, 2),
        "price": price,
        "flow": flow,
        "classification": classification,
        "impact": impact,
        "source": "Volume Spike",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "unix": datetime.now(timezone.utc).timestamp(),
    }


def fetch_onchain_btc_txs():
    """Poll large unconfirmed BTC transactions from blockchain.info."""
    try:
        response = requests.get(
            "https://blockchain.info/unconfirmed-transactions?format=json",
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        txs = response.json().get("txs", [])
    except requests.exceptions.RequestException as exc:
        print(f"❌ ERROR: Failed to fetch on-chain BTC txs — {exc}")
        return []

    btc_price = get_price("BTCUSDT")
    if not btc_price:
        return []

    alerts = []
    for tx in txs[:100]:
        total_btc = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
        usd_value = total_btc * btc_price
        if usd_value < WHALE_USD_THRESHOLD:
            continue

        alerts.append({
            "key": f"onchain:{tx.get('hash', '')}",
            "asset": "BTC",
            "amount": total_btc,
            "usd_value": usd_value,
            "flow": "🟡 Unknown Wallet ➔ Unknown Wallet",
            "classification": "🟡 Large Internal Transfer (Volatility Warning)",
            "impact": "Large on-chain transfer — monitor for volatility.",
            "source": "On-chain",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "unix": datetime.now(timezone.utc).timestamp(),
        })
    return alerts


def fetch_large_transactions():
    """Combine volume-spike detection and on-chain large transactions."""
    alerts = []

    for symbol in ("BTCUSDT", "ETHUSDT"):
        try:
            spike = detect_volume_spike(symbol)
        except Exception as exc:
            print(f"❌ ERROR: Volume spike detection failed for {symbol} — {exc}")
            spike = None
        if spike:
            alerts.append(spike)

    try:
        alerts.extend(fetch_onchain_btc_txs())
    except Exception as exc:
        print(f"❌ ERROR: On-chain whale fetch failed — {exc}")

    return alerts


def format_whale_message(alert):
    asset = alert["asset"]
    amount = alert.get("amount")
    amount_txt = f"{amount:,.0f} {asset}" if amount else "N/A"
    usd_txt = f"${alert['usd_value']:,.0f} USD"

    return (
        "🚨 <b>WHALE MOVEMENT DETECTED!</b> 🚨\n\n"
        f"<b>Asset:</b> {asset}\n"
        f"<b>Amount:</b> {amount_txt} (~{usd_txt})\n"
        f"<b>Flow:</b> {alert['flow']}\n"
        f"<b>Classification:</b> {alert['classification']}\n"
        f"<b>Market Impact:</b> {alert['impact']}\n\n"
        f"<i>Tracked live by Aria Crypto Engine</i>"
    )


def is_chat_id_configured():
    return (
        bool(CHAT_ID)
        and "YOUR_CHAT_ID" not in CHAT_ID
        and "INSERT" not in CHAT_ID
    )


async def whale_tracker_loop(context: ContextTypes.DEFAULT_TYPE):
    if not is_chat_id_configured():
        print("📵 CHAT_ID is not configured — skipping whale alerts.")
        return

    try:
        alerts = fetch_large_transactions()
    except Exception as exc:
        print(f"❌ ERROR in whale_tracker_loop: {exc}")
        return

    for alert in alerts:
        if not record_whale_event(alert):
            continue

        message = format_whale_message(alert)
        try:
            await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")
            print(f"🐋 Whale alert sent: {alert['key']}")
        except Forbidden as exc:
            print(f"⛔ FORBIDDEN: Bot cannot message CHAT_ID {CHAT_ID}. "
                  f"Start a chat with the bot and set a valid CHAT_ID. — {exc}")
        except BadRequest as exc:
            print(f"⚠️ BAD REQUEST: Invalid CHAT_ID {CHAT_ID!r} — {exc}")
        except TelegramError as exc:
            print(f"❌ TELEGRAM ERROR sending whale alert — {exc}")
        except Exception as exc:
            print(f"❌ ERROR sending whale alert: {exc}")


def get_recent_whales(hours=24):
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    return [e for e in load_whales() if e.get("unix", 0) >= cutoff]


async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE):
    alerts = load_alerts()
    if not alerts:
        return

    remaining = []
    triggered_any = False

    for alert in alerts:
        price = get_price(alert["symbol"])
        if price is None:
            remaining.append(alert)
            continue

        triggered = (
            (alert["condition"] == "ABOVE" and price >= alert["target_price"])
            or (alert["condition"] == "BELOW" and price <= alert["target_price"])
        )

        if not triggered:
            remaining.append(alert)
            continue

        triggered_any = True
        display = alert["symbol"].replace("USDT", "/USDT")
        message = (
            f"🔔 <b>PRICE ALERT TRIGGERED</b>\n"
            f"{display} reached ${price:,.2f}\n"
            f"(Target: {alert['condition']} ${alert['target_price']:,.2f})"
        )
        try:
            await context.bot.send_message(chat_id=alert["chat_id"], text=message, parse_mode="HTML")
            print(f"🔔 Price alert triggered for {alert['id']} ({display})")
        except Exception as exc:
            print(f"❌ ERROR sending price alert {alert['id']}: {exc}")

    if triggered_any:
        save_alerts(remaining)


def format_position_alert(trade, symbol, price, signal, confidence):
    display = symbol.replace("USDT", "/USDT")
    return (
        "⚠️ <b>URGENT POSITION ALERT: TREND REVERSAL!</b> ⚠️\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 Trade: <b>#{trade['id']}</b>\n"
        f"💱 Symbol: <b>{display}</b>\n"
        f"📌 Direction: <b>{trade['direction']}</b> x{trade['leverage']}\n"
        f"💵 Entry: ${trade['entry_price']:,.2f} → Now: ${price:,.2f}\n"
        f"🚦 15m Consensus: <b>{signal}</b> (Confidence {confidence}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 <b>AI Advice:</b> Consider closing this position or tightening your stop-loss!"
    )


def format_liquidation_alert(trade, live_price):
    display = trade["symbol"].replace("USDT", "/USDT")
    return (
        "💀 <b>LIQUIDATION ALERT!</b> 💀\n\n"
        f"<b>Trade ID:</b> #{trade['id']} ({trade['direction']} {display})\n"
        f"<b>Liquidation Price:</b> ${live_price:,.2f}\n"
        f"<b>Final P&L:</b> {trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+,.2f})\n\n"
        f"<i>Your margin has been fully exhausted and the position was "
        f"forcefully closed by the engine.</i>"
    )


async def position_monitor_loop(context: ContextTypes.DEFAULT_TYPE):
    data = load_trades()
    active = data.get("active", [])
    if not active:
        return

    changed = False

    for trade in list(active):
        symbol = trade["symbol"]

        live_price = get_price(symbol)
        if live_price is not None:
            pnl_pct, _ = calc_pnl(trade, live_price)
            if pnl_pct <= -100.0:
                trade_id = trade["id"]
                finalize_trade(trade, live_price)
                data["active"].remove(trade)
                data["history"].append(trade)
                changed = True

                message = format_liquidation_alert(trade, live_price)
                try:
                    await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")
                    print(f"💀 Liquidation alert sent for trade {trade_id}")
                except Exception as exc:
                    print(f"❌ ERROR sending liquidation alert for {trade_id}: {exc}")
                continue

        df = fetch_klines(symbol, interval="15m", limit=300)
        result = analyze_indicators(df)
        if result is None:
            continue

        price = result["price"]
        signal = result["signal"]
        confidence = result["confidence"]

        direction = trade["direction"]
        reversal = (
            (direction == "LONG" and signal in ("SELL", "STRONG SELL"))
            or (direction == "SHORT" and signal in ("BUY", "STRONG BUY"))
        )

        warned = trade.get("warning_triggered", False)

        if reversal and not warned:
            message = format_position_alert(trade, symbol, price, signal, confidence)
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    f"🔒 Close Position Now (#{trade['id']})",
                    callback_data=f"close_quick:{trade['id']}",
                )
            ]])
            try:
                await context.bot.send_message(
                    chat_id=CHAT_ID, text=message, parse_mode="HTML", reply_markup=keyboard
                )
                print(f"⚠️ Position reversal warning sent for trade {trade['id']}")
            except Exception as exc:
                print(f"❌ ERROR sending position warning for {trade['id']}: {exc}")
            trade["warning_triggered"] = True
            changed = True
        elif not reversal and warned:
            trade["warning_triggered"] = False
            changed = True

    if changed:
        save_trades(data)


async def fetch_crypto_news(symbol, limit=3):
    """Fetch news from Iranian crypto RSS feeds, filtered by symbol.

    Returns a tuple (articles, matched) where `matched` is True when the
    headlines are specific to the requested symbol.
    """
    articles = await _fetch_rss_news(limit=50)
    if not articles:
        print("ℹ️ All news endpoints failed — using mock headlines.")
        return await asyncio.to_thread(_mock_news, symbol, limit), False

    filtered = _filter_articles_by_symbol(articles, symbol)
    if filtered:
        return filtered[:limit], True
    return articles[:limit], False


def _filter_articles_by_symbol(articles, symbol):
    keywords = [k.lower() for k in SYMBOL_KEYWORDS.get(symbol, [symbol.replace("USDT", "").lower()])]

    matches = []
    for article in articles:
        title = article["title"].lower()
        if any(keyword in title for keyword in keywords):
            matches.append(article)
    return matches


async def _fetch_rss_news(limit=3):
    headers = {"User-Agent": "Mozilla/5.0 (crypto-bot)"}

    def _fetch_source(source_name, url):
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        return source_name, response.text

    articles = []
    for source_name, url in RSS_FEEDS:
        try:
            name, xml_text = await asyncio.to_thread(_fetch_source, source_name, url)
            articles.extend(_parse_rss_feed(xml_text, name))
        except Exception as exc:
            print(f"❌ ERROR: RSS feed {source_name} failed — {exc}")

    return articles[:limit]


def _mock_news(symbol, limit=3):
    base = symbol.replace("USDT", "")
    price = get_price(symbol)
    price_txt = f"${price:,.2f}" if price else "key levels"

    bullish = False
    df = fetch_klines(symbol, interval="1h", limit=30)
    if not df.empty:
        rsi = calculate_rsi(df["Close"].to_numpy())
        if rsi is not None:
            bullish = rsi >= 50

    if bullish:
        headlines = [
            f"{base} surges above {price_txt} as demand rises",
            f"Institutional interest in {base} pushes price toward {price_txt}",
            f"{base} momentum builds as buyers defend {price_txt}",
        ]
    else:
        headlines = [
            f"{base} tests support at {price_txt} amid selling pressure",
            f"Market caution grows as {base} consolidates near {price_txt}",
            f"{base} faces resistance at {price_txt} as sentiment cools",
        ]

    return [
        {"title": headline, "source": "Market Simulation", "url": ""}
        for headline in headlines[:limit]
    ]


def _parse_rss_feed(xml_text, source_name):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    articles = []
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        if not title:
            continue
        articles.append({
            "title": title,
            "source": source_name,
            "url": link,
            "published": _relative_time(_parse_pub_date(item)),
        })
    return articles


def _parse_pub_date(item):
    pub_date = (item.findtext("pubDate") or "").strip()
    if pub_date:
        return pub_date

    for child in item:
        if child.tag.split("}")[-1] == "date":
            text = (child.text or "").strip()
            if text:
                return text
    return ""


def _relative_time(date_str):
    if not date_str:
        return "Recently"

    parsed = None
    try:
        parsed = parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return "Recently"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    seconds = int((datetime.now(timezone.utc) - parsed).total_seconds())
    if seconds < 0:
        return "Recently"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = seconds // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def analyze_sentiment_with_ai(news_text):
    """Analyze English news sentiment using keyword matching."""
    text = news_text.lower()
    bullish_count = sum(text.count(word) for word in BULLISH_WORDS)
    bearish_count = sum(text.count(word) for word in BEARISH_WORDS)

    if bullish_count > bearish_count:
        return {
            "sentiment": "BULLISH",
            "reason": f"Headlines contain predominantly bullish keywords (bullish {bullish_count}, bearish {bearish_count}).",
        }
    if bearish_count > bullish_count:
        return {
            "sentiment": "BEARISH",
            "reason": f"Headlines contain predominantly bearish keywords (bearish {bearish_count}, bullish {bullish_count}).",
        }
    return {
        "sentiment": "NEUTRAL",
        "reason": f"Mixed or neutral headlines with no strong directional signal (bullish {bullish_count}, bearish {bearish_count}).",
    }


def build_news_message(base, articles, sentiment, matched=True):
    emoji = SENTIMENT_EMOJI.get(sentiment["sentiment"], "🟡")
    lines = [f"📰 <b>MARKET NEWS — {base}</b>", "━━━━━━━━━━━━━━━━━━━━"]

    if not matched:
        lines.append(f"ℹ️ <i>No specific news found for {base}. Showing latest market news:</i>")

    for i, article in enumerate(articles, 1):
        if article["url"]:
            lines.append(f"{i}. <a href=\"{article['url']}\">{article['title']}</a>")
        else:
            lines.append(f"{i}. {article['title']}")
        lines.append(f"   📰 {article['source']} • 🕒 {article.get('published', 'Recently')}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📰 <b>AI Market Sentiment:</b> {sentiment['sentiment']} {emoji}")
    lines.append(f"🧠 <b>AI Analyst View:</b> {sentiment['reason']}")
    return "\n".join(lines)


async def run_news(symbol):
    base = symbol.replace("USDT", "")
    articles, matched = await fetch_crypto_news(symbol)

    if not articles:
        return None

    news_text = "\n".join(a["title"] for a in articles)
    sentiment = analyze_sentiment_with_ai(news_text)

    return build_news_message(base, articles, sentiment, matched)


def get_chart_url(symbol, timeframe="1h", df=None):
    """Return a direct chart image URL (QuickChart) for the given symbol/timeframe."""
    if df is None or df.empty:
        df = fetch_klines(symbol, interval=timeframe, limit=60)

    if df.empty:
        slug = COINLORE_SLUGS.get(symbol, symbol.replace("USDT", "").lower())
        return f"https://charts.coinlore.com/static/img/{slug}_1h.png"

    closes = [round(float(x), 2) for x in df["Close"].to_numpy()[-60:]]
    labels = [ts.strftime("%m/%d %H:%M") for ts in df.index[-60:]]

    config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": f"{symbol.replace('USDT', '/USDT')} · {timeframe}",
                "data": closes,
                "borderColor": "#00c076",
                "backgroundColor": "rgba(0,192,118,0.15)",
                "fill": True,
                "pointRadius": 0,
                "borderWidth": 2,
                "tension": 0.1,
            }],
        },
        "options": {
            "scales": {
                "x": {"ticks": {"display": False}, "grid": {"display": False}},
                "y": {"grid": {"color": "#2a2a3a"}, "ticks": {"color": "#9aa0a6"}},
            },
            "plugins": {"legend": {"labels": {"color": "#ffffff"}}},
        },
    }

    config_json = json.dumps(config, separators=(",", ":"))
    return f"https://quickchart.io/chart?width=600&height=400&bkg=12121c&c={quote(config_json)}"


def timeframe_keyboard(symbol):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(tf, callback_data=f"tf:{symbol}:{tf}") for tf in TIMEFRAMES],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")],
    ])


def load_log():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_log(entry):
    log = load_log()
    log.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_trades():
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data.setdefault("active", [])
    data.setdefault("history", [])
    return data


def save_trades(data):
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_alerts():
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_alerts(alerts):
    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)


def add_alert(chat_id, symbol, target_price, condition):
    alerts = load_alerts()
    alert = {
        "id": uuid.uuid4().hex[:8].upper(),
        "chat_id": int(chat_id),
        "symbol": symbol,
        "target_price": target_price,
        "condition": condition,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    alerts.append(alert)
    save_alerts(alerts)
    return alert


def alerts_for_chat(chat_id):
    return [a for a in load_alerts() if a["chat_id"] == int(chat_id)]


def clear_alerts_for_chat(chat_id):
    remaining = [a for a in load_alerts() if a["chat_id"] != int(chat_id)]
    save_alerts(remaining)


def open_trade(symbol, direction, leverage, entry_price, notional=DEFAULT_NOTIONAL):
    data = load_trades()
    trade = {
        "id": uuid.uuid4().hex[:8].upper(),
        "symbol": symbol,
        "direction": direction,
        "leverage": leverage,
        "entry_price": entry_price,
        "notional": notional,
        "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    data["active"].append(trade)
    save_trades(data)
    return trade


def calc_pnl(trade, current_price):
    entry = trade["entry_price"]
    leverage = trade["leverage"]
    if trade["direction"] == "LONG":
        pnl_frac = (current_price - entry) / entry * leverage
    else:
        pnl_frac = (entry - current_price) / entry * leverage
    return pnl_frac * 100, trade["notional"] * pnl_frac


def finalize_trade(trade, current_price):
    pnl_pct, pnl_usd = calc_pnl(trade, current_price)
    trade["close_price"] = current_price
    trade["close_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    trade["pnl_pct"] = round(pnl_pct, 2)
    trade["pnl_usd"] = round(pnl_usd, 2)
    return trade


def close_trade(trade_id, current_price):
    data = load_trades()
    trade = next((t for t in data["active"] if t["id"] == trade_id), None)
    if trade is None:
        return None

    finalize_trade(trade, current_price)
    data["active"] = [t for t in data["active"] if t["id"] != trade_id]
    data["history"].append(trade)
    save_trades(data)
    return trade


def run_analysis(symbol, timeframe="1h"):
    df = fetch_klines(symbol, interval=timeframe, limit=300)
    if df.empty:
        return None

    result = analyze_indicators(df)
    if result is None:
        return None

    df_15m = fetch_klines(symbol, interval="15m", limit=300)
    df_4h = fetch_klines(symbol, interval="4h", limit=300)
    tf_trends = [timeframe_trend(df_15m), timeframe_trend(df_4h)]

    direction = "neutral"
    if result["signal"] in ("BUY", "STRONG BUY"):
        direction = "bullish"
    elif result["signal"] in ("SELL", "STRONG SELL"):
        direction = "bearish"
    confluence = confluence_badge(direction, tf_trends)

    tp_sl = generate_atr_tp_sl(result["price"], result["atr"], result["signal"])

    caption = build_analysis_caption(symbol, result, tp_sl, confluence, timeframe)
    chart_url = get_chart_url(symbol, timeframe, df.tail(60))

    return caption, chart_url, {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbol": symbol,
        "timeframe": timeframe,
        "price": result["price"],
        "rsi": result["rsi"],
        "macd": result["macd"]["macd"],
        "macd_signal": result["macd"]["signal"],
        "macd_histogram": result["macd"]["histogram"],
        "bb_upper": result["boll"]["upper"],
        "bb_middle": result["boll"]["middle"],
        "bb_lower": result["boll"]["lower"],
        "ema50": result["ema50"],
        "ema200": result["ema200"],
        "adx": result["adx"],
        "atr": result["atr"],
        "bullish": result["bullish"],
        "bearish": result["bearish"],
        "signal": result["signal"],
        "confidence": result["confidence"],
    }


def build_analysis_caption(symbol, result, tp_sl, confluence, timeframe="1h"):
    display = symbol.replace("USDT", "/USDT")
    price = result["price"]
    rsi = result["rsi"]
    macd_info = result["macd"]
    boll = result["boll"]
    ema50 = result["ema50"]
    ema200 = result["ema200"]
    adx = result["adx"]
    atr = result["atr"]
    vol_ok = result["vol_ok"]
    pattern = result["pattern"]
    signal = result["signal"]
    confidence = result["confidence"]
    bullish = result["bullish"]
    bearish = result["bearish"]

    rsi_txt = f"{rsi:.2f}" if rsi is not None else "N/A"
    adx_txt = f"{adx:.2f}" if adx is not None else "N/A"
    atr_txt = f"{atr:,.2f}" if atr is not None else "N/A"
    macd_txt = f"{macd_info['macd']:,.2f}" if macd_info["macd"] is not None else "N/A"
    sig_txt = f"{macd_info['signal']:,.2f}" if macd_info["signal"] is not None else "N/A"
    hist_txt = f"{macd_info['histogram']:,.2f}" if macd_info["histogram"] is not None else "N/A"
    cross_txt = macd_info["cross"] or "NEUTRAL"
    bb_upper_txt = f"{boll['upper']:,.2f}" if boll["upper"] is not None else "N/A"
    bb_mid_txt = f"{boll['middle']:,.2f}" if boll["middle"] is not None else "N/A"
    bb_lower_txt = f"{boll['lower']:,.2f}" if boll["lower"] is not None else "N/A"
    ema50_txt = f"{ema50:,.2f}" if ema50 is not None else "N/A"
    ema200_txt = f"{ema200:,.2f}" if ema200 is not None else "N/A"
    vol_txt = "✅" if vol_ok else "❌"
    pattern_txt = pattern.replace("_", " ").title() if pattern else "None"

    lines = [
        f"📊 <b>QUANT v4 ANALYSIS — {display} ({timeframe})</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💵 <b>Price</b>: ${price:,.2f}",
        f"🔀 <b>Confluence</b>: {confluence}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📈 <b>RSI(14)</b>: {rsi_txt}  ·  <b>ADX(14)</b>: {adx_txt}",
        f"🎚 <b>ATR(14)</b>: {atr_txt}",
        f"🔀 <b>MACD</b> (12,26,9)",
        f"   Line: {macd_txt} · Signal: {sig_txt}",
        f"   Histogram: {hist_txt} · Cross: <b>{cross_txt}</b>",
        f"🎗 <b>Bollinger Bands</b> (20,2)",
        f"   Upper: {bb_upper_txt} · Middle: {bb_mid_txt} · Lower: {bb_lower_txt}",
        f"📉 <b>EMA(50)</b>: {ema50_txt} · <b>EMA(200)</b>: {ema200_txt}",
        f"📊 <b>Volume Confirm</b>: {vol_txt} · <b>Pattern</b>: {pattern_txt}",
    ]

    if tp_sl:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("🎯 <b>Dynamic TP/SL (ATR)</b>")
        lines.append(f"   🟢 TP1: ${tp_sl['TP1']:,.2f} · 🟢 TP2: ${tp_sl['TP2']:,.2f}")
        lines.append(f"   🔴 SL: ${tp_sl['SL']:,.2f}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🐂 Bullish: <b>{bullish}</b> · 🐻 Bearish: <b>{bearish}</b>")
    lines.append(f"🚦 <b>Signal</b>: {SIGNAL_EMOJI[signal]} <b>{signal}</b>")
    lines.append(f"🧠 <b>AI Confidence</b>: <b>{confidence}%</b>")

    return "\n".join(lines)


def welcome_text():
    return (
        "🌌 <b>ARIA CRYPTO ENGINE v3.0</b> 🌌\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Your AI-powered trading concierge.\n\n"
        "✨ <b>Commands</b> ✨\n"
        "🪙 <code>/price [symbol]</code> — live price\n"
        "📊 <code>/analyze [symbol]</code> — analysis + chart\n"
        "📈 <code>/trade SYM LONG/SHORT [lev]</code> — paper trade\n"
        "💼 <code>/portfolio</code> — positions &amp; P&amp;L\n"
        "🔔 <code>/alert SYM price</code> — price alert\n"
        "📋 <code>/myalerts</code> — your alerts\n"
        "🧮 <code>/risk capital risk% entry sl</code> — position size\n"
        "🌡️ <code>/market</code> — fear &amp; greed overview\n"
        "☀️ <code>/daily</code> — executive summary\n"
        "🐋 <code>/whales</code> — whale movements (24h)\n"
        "📰 <code>/news [symbol]</code> — news &amp; sentiment\n"
        "🔥 <code>/signals</code> — high-confidence scanner\n\n"
        "⚡ <i>Use the menu below.</i>"
    )


def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("💰 Live Prices", callback_data="prices"),
            InlineKeyboardButton("📊 Run Analysis", callback_data="analyze"),
        ],
        [
            InlineKeyboardButton("📈 Paper Trade", callback_data="trade"),
            InlineKeyboardButton("💼 My Portfolio", callback_data="portfolio"),
        ],
        [
            InlineKeyboardButton("📰 Market News", callback_data="news"),
            InlineKeyboardButton("🌡️ Market Overview", callback_data="market"),
        ],
        [
            InlineKeyboardButton("🔔 Price Alerts", callback_data="alerts"),
            InlineKeyboardButton("🧮 Risk Calculator", callback_data="risk"),
        ],
        [
            InlineKeyboardButton("🔥 Gold Signals", callback_data="signals"),
            InlineKeyboardButton("🐋 Whale Tracker", callback_data="whales"),
        ],
        [
            InlineKeyboardButton("☀️ Daily Brief", callback_data="daily"),
            InlineKeyboardButton("❓ Help & Guide", callback_data="help"),
        ],
    ])


def submenu_keyboard(extra_buttons=None):
    buttons = [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")]
    rows = []
    if extra_buttons:
        rows.extend(extra_buttons)
    rows.append(buttons)
    return InlineKeyboardMarkup(rows)


def symbol_keyboard(prefix, symbols):
    rows = []
    for i in range(0, len(symbols), 2):
        rows.append([
            InlineKeyboardButton(SYMBOL_DISPLAY[s], callback_data=f"{prefix}:{s}")
            for s in symbols[i:i + 2]
        ])
    rows.append([InlineKeyboardButton("🔙 Main Menu", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


def direction_keyboard(symbol):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 LONG", callback_data=f"setup:LONG:{symbol}"),
            InlineKeyboardButton("🔴 SHORT", callback_data=f"setup:SHORT:{symbol}"),
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="trade")],
    ])


def leverage_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("5x", callback_data="lev:5"),
            InlineKeyboardButton("10x", callback_data="lev:10"),
            InlineKeyboardButton("20x", callback_data="lev:20"),
            InlineKeyboardButton("50x", callback_data="lev:50"),
        ],
        [InlineKeyboardButton("✏️ Custom Leverage", callback_data="lev:custom")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_trade")],
    ])


def size_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("$100", callback_data="size:100"),
            InlineKeyboardButton("$500", callback_data="size:500"),
            InlineKeyboardButton("$1000", callback_data="size:1000"),
            InlineKeyboardButton("$5000", callback_data="size:5000"),
        ],
        [InlineKeyboardButton("✏️ Custom Size", callback_data="size:custom")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_trade")],
    ])


def confirm_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Confirm Trade", callback_data="confirm_trade"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel_trade"),
        ],
    ])


def scan_symbol(symbol):
    df = fetch_klines(symbol, interval="1h", limit=300)
    result = analyze_indicators(df)
    if result is None:
        return None
    return {
        "symbol": symbol,
        "price": result["price"],
        "signal": result["signal"],
        "confidence": result["confidence"],
    }


def build_signals_message(high):
    lines = ["🔥 <b>GOLD SIGNALS — HIGH CONFIDENCE (≥75%)</b>", "━━━━━━━━━━━━━━━━━━━━"]
    buttons = []

    for r in high:
        display = r["symbol"].replace("USDT", "/USDT")
        emoji = SIGNAL_EMOJI.get(r["signal"], "🟡")
        lines.append(
            f"{emoji} <b>{display}</b> — {r['signal']} · {r['confidence']}% · ${r['price']:,.2f}"
        )
        buttons.append([
            InlineKeyboardButton(f"🟢 LONG {display}", callback_data=f"setup:LONG:{r['symbol']}"),
            InlineKeyboardButton(f"🔴 SHORT {display}", callback_data=f"setup:SHORT:{r['symbol']}"),
        ])

    buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="menu")])
    return "\n".join(lines), InlineKeyboardMarkup(buttons)


def build_portfolio_report():
    data = load_trades()
    active = data["active"]

    if not active:
        return (
            "💼 <b>MY PAPER PORTFOLIO</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "No open paper trades. Use 📈 <i>Open Paper Trade</i> to begin."
        )

    lines = ["💼 <b>MY PAPER PORTFOLIO</b>", "━━━━━━━━━━━━━━━━━━━━"]
    total_usd = 0.0
    total_notional = 0.0

    for t in active:
        price = get_price(t["symbol"])
        if price is None:
            lines.append(f"#{t['id']} <b>{t['symbol']}</b> {t['direction']} x{t['leverage']} — price unavailable")
            continue
        pnl_pct, pnl_usd = calc_pnl(t, price)
        total_usd += pnl_usd
        total_notional += t["notional"]
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        lines.append(f"{emoji} <b>{t['symbol']}</b> {t['direction']} x{t['leverage']}  #{t['id']}")
        lines.append(f"   Entry ${t['entry_price']:,.2f} → Now ${price:,.2f}")
        lines.append(f"   P&amp;L {pnl_pct:+.2f}% (${pnl_usd:+,.2f})")

    total_pct = (total_usd / total_notional * 100) if total_notional else 0.0
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"💰 <b>Total P&amp;L</b>: ${total_usd:+,.2f} ({total_pct:+.2f}%)")
    return "\n".join(lines)


def portfolio_keyboard():
    active = load_trades().get("active", [])
    if not active:
        return None

    rows = []
    for trade in active:
        display = trade["symbol"].replace("USDT", "/USDT")
        rows.append([
            InlineKeyboardButton(
                f"🔒 Close {display} {trade['direction']}",
                callback_data=f"close_quick:{trade['id']}",
            )
        ])
    rows.append([InlineKeyboardButton("🔙 Main Menu", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


def build_market_overview():
    lines = ["🌡️ <b>MARKET OVERVIEW</b>", "━━━━━━━━━━━━━━━━━━━━"]

    value, classification = fetch_fear_greed()
    if value is not None:
        emoji = FEAR_GREED_EMOJI.get(classification, "😐")
        lines.append(f"🧭 <b>Fear &amp; Greed Index:</b> {value}/100 {emoji}")
        lines.append(f"   <i>{classification}</i>")
        lines.append(f"   {fear_greed_bar(value)}")
    else:
        lines.append("🧭 <b>Fear &amp; Greed:</b> unavailable")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("💵 <b>Top Coins</b>")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]:
        price = get_price(symbol)
        display = symbol.replace("USDT", "/USDT")
        lines.append(f"   {display}: ${price:,.2f}" if price is not None else f"   {display}: N/A")

    return "\n".join(lines)


def build_alerts_message(chat_id):
    alerts = alerts_for_chat(chat_id)
    if not alerts:
        return (
            "🔔 <b>PRICE ALERTS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "No active alerts. Tap <b>➕ Create New Alert</b> to set one.",
            False,
        )

    lines = ["🔔 <b>YOUR PRICE ALERTS</b>", "━━━━━━━━━━━━━━━━━━━━"]
    for alert in alerts:
        display = alert["symbol"].replace("USDT", "/USDT")
        lines.append(
            f"#{alert['id']} <b>{display}</b> {alert['condition']} "
            f"${alert['target_price']:,.2f}"
        )
    return "\n".join(lines), True


async def build_daily_summary():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"☀️ <b>DAILY EXECUTIVE SUMMARY</b> — {now}", "━━━━━━━━━━━━━━━━━━━━"]

    value, classification = fetch_fear_greed()
    if value is not None:
        emoji = FEAR_GREED_EMOJI.get(classification, "😐")
        lines.append(f"🧭 <b>Fear &amp; Greed:</b> {value}/100 — {classification} {emoji}")
    else:
        lines.append("🧭 <b>Fear &amp; Greed:</b> unavailable")

    lines.append("💵 <b>Top Coins</b>")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        price = get_price(symbol)
        display = symbol.replace("USDT", "/USDT")
        lines.append(f"   {display}: ${price:,.2f}" if price is not None else f"   {display}: N/A")

    whales = get_recent_whales(24)
    lines.append(f"🐋 <b>Whale Activity (24h):</b> {len(whales)} movement(s)")

    try:
        articles, _ = await fetch_crypto_news("BTCUSDT")
        if articles:
            sentiment = analyze_sentiment_with_ai("\n".join(a["title"] for a in articles))
            lines.append(
                f"📰 <b>News Sentiment:</b> {sentiment['sentiment']} "
                f"{SENTIMENT_EMOJI[sentiment['sentiment']]}"
            )
    except Exception as exc:
        print(f"❌ ERROR in daily summary (news): {exc}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚡ <i>Generated by Aria Crypto Engine v3.0</i>")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        welcome_text(), parse_mode="HTML", reply_markup=main_menu_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        welcome_text(), parse_mode="HTML", reply_markup=main_menu_keyboard()
    )


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = normalize_symbol(context.args[0]) if context.args else "BTCUSDT"

    if symbol is None:
        await update.message.reply_text(
            f"❌ <b>Invalid symbol</b>: <code>{context.args[0]}</code>",
            parse_mode="HTML",
        )
        return

    price = get_price(symbol)
    if price is None:
        await update.message.reply_text(
            f"⚠️ <b>Unable to fetch price</b> for <b>{symbol.replace('USDT', '/USDT')}</b>.",
            parse_mode="HTML",
        )
        return

    await update.message.reply_text(
        f"💰 <b>{symbol.replace('USDT', '/USDT')}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💵 Price: <b>${price:,.2f}</b>",
        parse_mode="HTML",
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = normalize_symbol(context.args[0]) if context.args else "BTCUSDT"
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    if timeframe not in TIMEFRAMES:
        timeframe = "1h"

    if symbol is None:
        await update.message.reply_text(
            f"❌ <b>Invalid symbol</b>: <code>{context.args[0]}</code>",
            parse_mode="HTML",
        )
        return

    try:
        result = run_analysis(symbol, timeframe)
    except Exception as exc:
        print(f"❌ ERROR in /analyze: {exc}")
        result = None

    if result is None:
        await update.message.reply_text(
            f"⚠️ <b>Analysis failed</b> for <b>{symbol.replace('USDT', '/USDT')}</b>. "
            f"Check the symbol or try again later.",
            parse_mode="HTML",
        )
        return

    caption, chart_url, log_entry = result
    save_log(log_entry)
    await update.message.reply_photo(
        photo=chart_url, caption=caption, parse_mode="HTML",
        reply_markup=timeframe_keyboard(symbol),
    )


async def trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "📈 <b>Usage</b>: <code>/trade SYM LONG/SHORT [leverage]</code>\n"
            "Example: <code>/trade BTC LONG 10</code>",
            parse_mode="HTML",
        )
        return

    symbol = normalize_symbol(args[0])
    direction = args[1].upper()

    if symbol is None:
        await update.message.reply_text(
            f"❌ <b>Invalid symbol</b>: <code>{args[0]}</code>", parse_mode="HTML"
        )
        return

    if direction not in ("LONG", "SHORT"):
        await update.message.reply_text(
            "❌ <b>Direction</b> must be <b>LONG</b> or <b>SHORT</b>.",
            parse_mode="HTML",
        )
        return

    leverage = None
    if len(args) > 2:
        try:
            leverage = float(args[2])
        except ValueError:
            leverage = None
        if leverage is not None and (leverage <= 0 or leverage > 100):
            await update.message.reply_text(
                "❌ <b>Leverage</b> must be between 1 and 100.", parse_mode="HTML"
            )
            return

    display = symbol.replace("USDT", "/USDT")
    context.user_data["pending_trade"] = {
        "symbol": symbol,
        "direction": direction,
        "leverage": leverage,
        "notional": None,
    }
    context.user_data["state"] = None

    if leverage is not None:
        await update.message.reply_text(
            f"📈 <b>{display}</b> {direction} — Step 2/2\n"
            f"Leverage: <b>x{leverage:g}</b>\n"
            f"Choose your <b>Margin Size</b>:",
            parse_mode="HTML",
            reply_markup=size_keyboard(),
        )
    else:
        await update.message.reply_text(
            f"📈 <b>{display}</b> {direction} — Step 1/2\n"
            f"Choose your <b>Leverage</b>:",
            parse_mode="HTML",
            reply_markup=leverage_keyboard(),
        )


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        results = await asyncio.gather(
            *[asyncio.to_thread(scan_symbol, s) for s in SUPPORTED_SYMBOLS]
        )
    except Exception as exc:
        print(f"❌ ERROR in /signals: {exc}")
        await update.message.reply_text("⚠️ <b>Signal scan failed.</b> Try again later.", parse_mode="HTML")
        return

    high = [r for r in results if r is not None and r["confidence"] >= 75]

    if not high:
        await update.message.reply_text(
            "🔥 <b>GOLD SIGNALS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "No high-confidence signals (≥75%) right now.",
            parse_mode="HTML",
            reply_markup=submenu_keyboard(),
        )
        return

    high.sort(key=lambda r: r["confidence"], reverse=True)
    text, keyboard = build_signals_message(high)
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        build_portfolio_report(),
        parse_mode="HTML",
        reply_markup=portfolio_keyboard() or main_menu_keyboard(),
    )


async def close_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "🔒 <b>Usage</b>: <code>/close &lt;trade_id&gt;</code>\n"
            "Example: <code>/close AB12CD34</code>",
            parse_mode="HTML",
        )
        return

    trade_id = context.args[0].upper()
    data = load_trades()
    trade = next((t for t in data["active"] if t["id"] == trade_id), None)

    if trade is None:
        await update.message.reply_text(
            f"❌ No active trade with ID <b>{trade_id}</b>.", parse_mode="HTML"
        )
        return

    price = get_price(trade["symbol"])
    if price is None:
        await update.message.reply_text(
            f"⚠️ Unable to fetch price for <b>{trade['symbol']}</b>. Trade not closed.",
            parse_mode="HTML",
        )
        return

    closed = close_trade(trade_id, price)
    emoji = "🟢" if closed["pnl_usd"] >= 0 else "🔴"
    await update.message.reply_text(
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔖 ID: <b>{closed['id']}</b>\n"
        f"💱 Symbol: <b>{closed['symbol']}</b>\n"
        f"📌 Direction: <b>{closed['direction']}</b> x{closed['leverage']}\n"
        f"💵 Entry: <b>${closed['entry_price']:,.2f}</b>\n"
        f"💵 Exit: <b>${closed['close_price']:,.2f}</b>\n"
        f"📊 P&amp;L: <b>{closed['pnl_pct']:+.2f}%</b> (${closed['pnl_usd']:+,.2f})",
        parse_mode="HTML",
    )


def build_whales_message():
    events = get_recent_whales(24)

    if not events:
        return (
            "🐋 <b>WHALE ACTIVITY — LAST 24H</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "No whale movements detected yet. Check back soon."
        )

    lines = ["🐋 <b>WHALE ACTIVITY — LAST 24H</b>", "━━━━━━━━━━━━━━━━━━━━"]
    for e in events[-15:]:
        usd = f"${e['usd_value']:,.0f}"
        lines.append(f"{e['flow']}\n   <b>{e['asset']}</b> · {usd} · {e['timestamp']}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📡 <i>{len(events)} event(s) tracked live by Aria Crypto Engine</i>")
    return "\n".join(lines)


async def whales_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        build_whales_message(), parse_mode="HTML", reply_markup=submenu_keyboard()
    )


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = normalize_symbol(context.args[0]) if context.args else "BTCUSDT"

    if symbol is None:
        await update.message.reply_text(
            f"❌ <b>Invalid symbol</b>: <code>{context.args[0]}</code>",
            parse_mode="HTML",
        )
        return

    try:
        message = await run_news(symbol)
    except Exception as exc:
        print(f"❌ ERROR in /news: {exc}")
        message = None

    if message is None:
        await update.message.reply_text(
            f"⚠️ <b>No news found</b> for <b>{symbol.replace('USDT', '/USDT')}</b>. "
            f"Try again later.",
            parse_mode="HTML",
        )
        return

    await update.message.reply_text(message, parse_mode="HTML")


async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text(
            "🔔 <b>Usage</b>: <code>/alert SYM price</code>\n"
            "Example: <code>/alert BTC 80000</code>",
            parse_mode="HTML",
        )
        return

    symbol = normalize_symbol(context.args[0])
    if symbol is None:
        await update.message.reply_text(
            f"❌ <b>Invalid symbol</b>: <code>{context.args[0]}</code>", parse_mode="HTML"
        )
        return

    try:
        target = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ <b>Price</b> must be a number.", parse_mode="HTML")
        return

    current = get_price(symbol)
    if current is None:
        await update.message.reply_text(
            f"⚠️ Unable to fetch price for <b>{symbol.replace('USDT', '/USDT')}</b>.",
            parse_mode="HTML",
        )
        return

    condition = "ABOVE" if target > current else "BELOW"
    alert = add_alert(update.effective_chat.id, symbol, target, condition)

    await update.message.reply_text(
        f"🔔 <b>ALERT SET</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💱 {symbol.replace('USDT', '/USDT')}\n"
        f"🎯 Target: <b>{condition} ${target:,.2f}</b>\n"
        f"💵 Current: ${current:,.2f}\n"
        f"🔖 ID: <b>{alert['id']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 View with /myalerts",
        parse_mode="HTML",
    )


async def myalerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, has_alerts = build_alerts_message(update.effective_chat.id)
    keyboard = None
    if has_alerts:
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🗑️ Clear All Alerts", callback_data="clear_alerts")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")],
        ])
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 4:
        await update.message.reply_text(
            "🧮 <b>Usage</b>: <code>/risk capital risk% entry stop_loss</code>\n"
            "Example: <code>/risk 1000 2 79000 77500</code>",
            parse_mode="HTML",
        )
        return

    try:
        capital = float(context.args[0])
        risk_pct = float(context.args[1])
        entry = float(context.args[2])
        stop_loss = float(context.args[3])
    except ValueError:
        await update.message.reply_text("❌ All inputs must be numbers.", parse_mode="HTML")
        return

    if min(capital, risk_pct, entry, stop_loss) <= 0:
        await update.message.reply_text("❌ Values must be positive.", parse_mode="HTML")
        return

    risk_amount = capital * risk_pct / 100
    price_diff = abs(entry - stop_loss) / entry * 100
    if price_diff == 0:
        await update.message.reply_text(
            "❌ Entry and stop loss are identical.", parse_mode="HTML"
        )
        return

    position_size = risk_amount / (price_diff / 100)
    max_leverage = min(100, max(1, round(100 / price_diff)))

    await update.message.reply_text(
        f"🧮 <b>POSITION RISK CALCULATOR</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Capital: <b>${capital:,.2f}</b>\n"
        f"🎯 Risk %: <b>{risk_pct:.2f}%</b>\n"
        f"💸 Risk Amount: <b>${risk_amount:,.2f}</b>\n"
        f"📊 Price Diff: <b>{price_diff:.2f}%</b>\n"
        f"🏦 Max Position Size: <b>${position_size:,.2f}</b>\n"
        f"⚡ Recommended Max Leverage: <b>x{max_leverage}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ <i>Never risk more than 1-2% per trade. "
        f"Set your stop loss before entering.</i>",
        parse_mode="HTML",
    )


async def market_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        build_market_overview(), parse_mode="HTML", reply_markup=submenu_keyboard()
    )


async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = await build_daily_summary()
    except Exception as exc:
        print(f"❌ ERROR in /daily: {exc}")
        text = "⚠️ <b>Daily summary failed.</b> Please try again later."
    await update.message.reply_text(
        text, parse_mode="HTML", reply_markup=submenu_keyboard()
    )


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.user_data.get("state")
    text = update.message.text.strip()

    if state == "CUSTOM_LEVERAGE":
        pending = context.user_data.get("pending_trade")
        if not pending:
            context.user_data["state"] = None
            await update.message.reply_text(
                "⚠️ Session expired. Use the menu below.", parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
            return
        try:
            value = float(text)
        except ValueError:
            await update.message.reply_text("❌ Please send a valid number.", parse_mode="HTML")
            return
        if value <= 0 or value > 100:
            await update.message.reply_text(
                "❌ Leverage must be between 1 and 100. Try again.", parse_mode="HTML"
            )
            return
        pending["leverage"] = value
        context.user_data["state"] = None
        await update.message.reply_text(
            f"⚡ Leverage set to <b>x{value:g}</b>.\n"
            f"Choose your <b>Margin Size</b>:",
            parse_mode="HTML", reply_markup=size_keyboard(),
        )

    elif state == "CUSTOM_MARGIN":
        pending = context.user_data.get("pending_trade")
        if not pending:
            context.user_data["state"] = None
            await update.message.reply_text(
                "⚠️ Session expired. Use the menu below.", parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
            return
        try:
            value = float(text)
        except ValueError:
            await update.message.reply_text("❌ Please send a valid number.", parse_mode="HTML")
            return
        if value <= 0:
            await update.message.reply_text("❌ Margin must be positive.", parse_mode="HTML")
            return
        pending["notional"] = value
        context.user_data["state"] = None
        display = pending["symbol"].replace("USDT", "/USDT")
        await update.message.reply_text(
            f"💰 Margin set to <b>${value:,.2f}</b>.\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Confirm opening {pending['direction']} on {display} with "
            f"x{pending['leverage']:g} and ${value:,.2f}?",
            parse_mode="HTML", reply_markup=confirm_keyboard(),
        )

    elif state == "WAITING_PRICE":
        flow = context.user_data.get("alert_flow")
        if not isinstance(flow, dict) or "symbol" not in flow:
            context.user_data.pop("alert_flow", None)
            context.user_data["state"] = None
            await update.message.reply_text(
                "⚠️ Alert session expired. Use the menu below.", parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
            return
        try:
            target = float(text)
        except ValueError:
            await update.message.reply_text(
                "❌ Please send a valid number for the target price.", parse_mode="HTML"
            )
            return
        if target <= 0:
            await update.message.reply_text("❌ Price must be positive.", parse_mode="HTML")
            return

        symbol = flow["symbol"]
        current = flow["current"]
        condition = "ABOVE" if target > current else "BELOW"
        alert = add_alert(update.effective_chat.id, symbol, target, condition)

        context.user_data.pop("alert_flow", None)
        context.user_data["state"] = None
        display = symbol.replace("USDT", "/USDT")
        await update.message.reply_text(
            f"🔔 <b>ALERT SET</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💱 {display}\n"
            f"🎯 Target: <b>{condition} ${target:,.2f}</b>\n"
            f"💵 Current: ${current:,.2f}\n"
            f"🔖 ID: <b>{alert['id']}</b>",
            parse_mode="HTML", reply_markup=submenu_keyboard(),
        )

    else:
        await update.message.reply_text(
            "🤖 I respond to buttons. Tap <b>Main Menu</b> below to navigate:",
            parse_mode="HTML", reply_markup=main_menu_keyboard(),
        )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except (TimedOut, BadRequest, TelegramError):
        pass

    data = query.data
    try:
        if data == "menu":
            await query.edit_message_text(
                welcome_text(), parse_mode="HTML", reply_markup=main_menu_keyboard()
            )

        elif data == "prices":
            await query.edit_message_text(
                "💰 <b>Live Prices</b> — select a coin:",
                parse_mode="HTML",
                reply_markup=symbol_keyboard("price", SUPPORTED_SYMBOLS[:5]),
            )

        elif data == "analyze":
            await query.edit_message_text(
                "📊 <b>Run Analysis</b> — select a coin:",
                parse_mode="HTML",
                reply_markup=symbol_keyboard("analyze", SUPPORTED_SYMBOLS),
            )

        elif data == "trade":
            await query.edit_message_text(
                "📈 <b>Open Paper Trade</b> — select a coin:",
                parse_mode="HTML",
                reply_markup=symbol_keyboard("trade", SUPPORTED_SYMBOLS),
            )

        elif data == "portfolio":
            await query.edit_message_text(
                build_portfolio_report(),
                parse_mode="HTML",
                reply_markup=portfolio_keyboard() or main_menu_keyboard(),
            )

        elif data == "news":
            await query.edit_message_text(
                "📰 <b>Market News</b> — select a coin:",
                parse_mode="HTML",
                reply_markup=symbol_keyboard("news", SUPPORTED_SYMBOLS),
            )

        elif data.startswith("news:"):
            symbol = data.split(":", 1)[1]
            message = await run_news(symbol)
            if message is None:
                await query.edit_message_text(
                    f"⚠️ <b>No news found</b> for <b>{symbol.replace('USDT', '/USDT')}</b>.",
                    parse_mode="HTML",
                )
                return
            await query.edit_message_text(
                message, parse_mode="HTML", reply_markup=main_menu_keyboard()
            )

        elif data == "market":
            await query.edit_message_text(
                build_market_overview(), parse_mode="HTML", reply_markup=submenu_keyboard()
            )

        elif data == "alerts":
            await query.edit_message_text(
                "🔔 <b>Price Alerts</b> — choose an action:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("➕ Create New Alert", callback_data="alert_new")],
                    [InlineKeyboardButton("🗑️ View/Clear Active Alerts", callback_data="alert_list")],
                    [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")],
                ]),
            )

        elif data == "alert_new":
            context.user_data["alert_flow"] = "SELECT_COIN"
            await query.edit_message_text(
                "🔔 <b>Create New Alert</b> — select a coin:",
                parse_mode="HTML",
                reply_markup=symbol_keyboard("alertpick", SUPPORTED_SYMBOLS),
            )

        elif data.startswith("alertpick:"):
            symbol = data.split(":", 1)[1]
            price = get_price(symbol)
            if price is None:
                await query.edit_message_text(
                    f"⚠️ Unable to fetch price for <b>{symbol.replace('USDT', '/USDT')}</b>.",
                    parse_mode="HTML", reply_markup=submenu_keyboard(),
                )
                return
            display = symbol.replace("USDT", "/USDT")
            context.user_data["alert_flow"] = {"symbol": symbol, "current": price}
            context.user_data["state"] = "WAITING_PRICE"
            await query.edit_message_text(
                f"💵 <b>{display}</b> current price: ${price:,.2f}\n\n"
                f"Please type the target price for <b>{display}</b>:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("❌ Cancel", callback_data="menu")],
                ]),
            )

        elif data == "alert_list":
            text, has_alerts = build_alerts_message(query.message.chat_id)
            keyboard = None
            if has_alerts:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🗑️ Clear All Alerts", callback_data="clear_alerts")],
                    [InlineKeyboardButton("🔙 Main Menu", callback_data="menu")],
                ])
            await query.edit_message_text(
                text, parse_mode="HTML", reply_markup=keyboard or submenu_keyboard()
            )

        elif data == "clear_alerts":
            clear_alerts_for_chat(query.message.chat_id)
            await query.edit_message_text(
                "🔔 <b>All alerts cleared.</b>",
                parse_mode="HTML",
                reply_markup=submenu_keyboard(),
            )

        elif data == "whales":
            await query.edit_message_text(
                build_whales_message(), parse_mode="HTML", reply_markup=submenu_keyboard()
            )

        elif data == "risk":
            await query.edit_message_text(
                "🧮 <b>POSITION RISK CALCULATOR</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Usage: <code>/risk capital risk% entry stop_loss</code>\n\n"
                "Example: <code>/risk 1000 2 79000 77500</code>",
                parse_mode="HTML",
                reply_markup=submenu_keyboard(),
            )

        elif data == "daily":
            text = await build_daily_summary()
            await query.edit_message_text(
                text, parse_mode="HTML", reply_markup=submenu_keyboard()
            )

        elif data == "help":
            await query.edit_message_text(
                welcome_text(), parse_mode="HTML", reply_markup=main_menu_keyboard()
            )

        elif data.startswith("price:"):
            symbol = data.split(":", 1)[1]
            price = get_price(symbol)
            if price is None:
                text = f"⚠️ Unable to fetch price for <b>{symbol.replace('USDT', '/USDT')}</b>."
            else:
                text = (
                    f"💰 <b>{symbol.replace('USDT', '/USDT')}</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💵 Price: <b>${price:,.2f}</b>"
                )
            await query.edit_message_text(
                text, parse_mode="HTML",
                reply_markup=symbol_keyboard("price", SUPPORTED_SYMBOLS[:5]),
            )

        elif data.startswith("analyze:"):
            symbol = data.split(":", 1)[1]
            result = run_analysis(symbol, "1h")
            if result is None:
                await query.edit_message_text(
                    f"⚠️ <b>Analysis failed</b> for <b>{symbol.replace('USDT', '/USDT')}</b>.",
                    parse_mode="HTML",
                )
                return
            caption, chart_url, log_entry = result
            save_log(log_entry)
            await query.message.reply_photo(
                photo=chart_url, caption=caption, parse_mode="HTML",
                reply_markup=timeframe_keyboard(symbol),
            )

        elif data.startswith("tf:"):
            _, symbol, timeframe = data.split(":")
            if timeframe not in TIMEFRAMES:
                timeframe = "1h"
            result = run_analysis(symbol, timeframe)
            if result is None:
                await query.answer("Analysis failed. Try again.")
                return
            caption, chart_url, log_entry = result
            save_log(log_entry)
            await query.edit_message_media(
                media=InputMediaPhoto(media=chart_url, caption=caption, parse_mode="HTML"),
                reply_markup=timeframe_keyboard(symbol),
            )

        elif data.startswith("trade:"):
            symbol = data.split(":", 1)[1]
            await query.edit_message_text(
                f"📈 <b>{symbol.replace('USDT', '/USDT')}</b> — choose direction:",
                parse_mode="HTML",
                reply_markup=direction_keyboard(symbol),
            )

        elif data == "signals":
            results = await asyncio.gather(
                *[asyncio.to_thread(scan_symbol, s) for s in SUPPORTED_SYMBOLS]
            )
            high = [r for r in results if r is not None and r["confidence"] >= 75]
            if not high:
                await query.edit_message_text(
                    "🔥 <b>GOLD SIGNALS</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                    "No high-confidence signals (≥75%) right now.",
                    parse_mode="HTML", reply_markup=submenu_keyboard(),
                )
                return
            high.sort(key=lambda r: r["confidence"], reverse=True)
            text, keyboard = build_signals_message(high)
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=keyboard)

        elif data.startswith("setup:"):
            _, direction, symbol = data.split(":")
            context.user_data["pending_trade"] = {
                "symbol": symbol,
                "direction": direction,
                "leverage": None,
                "notional": None,
            }
            context.user_data["state"] = None
            await query.edit_message_text(
                f"📈 <b>{symbol.replace('USDT', '/USDT')}</b> {direction} — Step 1/2\n"
                f"Choose your <b>Leverage</b>:",
                parse_mode="HTML", reply_markup=leverage_keyboard(),
            )

        elif data.startswith("lev:"):
            value = data.split(":", 1)[1]
            pending = context.user_data.get("pending_trade")
            if not pending:
                await query.edit_message_text(
                    "⚠️ Session expired. Start a new trade.",
                    parse_mode="HTML", reply_markup=main_menu_keyboard(),
                )
                return
            if value == "custom":
                context.user_data["state"] = "CUSTOM_LEVERAGE"
                await query.edit_message_text(
                    "✏️ Send your custom leverage (e.g. <code>15</code>):", parse_mode="HTML"
                )
                return
            pending["leverage"] = float(value)
            display = pending["symbol"].replace("USDT", "/USDT")
            await query.edit_message_text(
                f"📈 <b>{display}</b> {pending['direction']} — Step 2/2\n"
                f"Leverage: <b>x{value}</b>\n"
                f"Choose your <b>Margin Size</b>:",
                parse_mode="HTML", reply_markup=size_keyboard(),
            )

        elif data.startswith("size:"):
            value = data.split(":", 1)[1]
            pending = context.user_data.get("pending_trade")
            if not pending:
                await query.edit_message_text(
                    "⚠️ Session expired. Start a new trade.",
                    parse_mode="HTML", reply_markup=main_menu_keyboard(),
                )
                return
            if value == "custom":
                context.user_data["state"] = "CUSTOM_MARGIN"
                await query.edit_message_text(
                    "✏️ Send your custom margin size in USD (e.g. <code>2500</code>):",
                    parse_mode="HTML",
                )
                return
            pending["notional"] = float(value)
            display = pending["symbol"].replace("USDT", "/USDT")
            await query.edit_message_text(
                f"📈 <b>Confirm Trade</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💱 {display}\n"
                f"📌 Direction: <b>{pending['direction']}</b>\n"
                f"⚡ Leverage: <b>x{pending['leverage']:g}</b>\n"
                f"💰 Margin: <b>${pending['notional']:,.2f}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Confirm opening {pending['direction']} on {display} with "
                f"x{pending['leverage']:g} and ${pending['notional']:,.2f}?",
                parse_mode="HTML", reply_markup=confirm_keyboard(),
            )

        elif data == "confirm_trade":
            pending = context.user_data.pop("pending_trade", None)
            context.user_data["state"] = None
            if not pending:
                await query.edit_message_text(
                    "⚠️ Session expired. Start a new trade.",
                    parse_mode="HTML", reply_markup=main_menu_keyboard(),
                )
                return
            entry = get_price(pending["symbol"])
            if entry is None:
                await query.edit_message_text(
                    f"⚠️ Unable to fetch price for <b>{pending['symbol'].replace('USDT', '/USDT')}</b>.",
                    parse_mode="HTML", reply_markup=main_menu_keyboard(),
                )
                return
            trade = open_trade(
                pending["symbol"], pending["direction"], pending["leverage"],
                entry, pending["notional"],
            )
            emoji = "🟢" if pending["direction"] == "LONG" else "🔴"
            await query.edit_message_text(
                f"{emoji} <b>PAPER TRADE OPENED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🔖 ID: <b>{trade['id']}</b>\n"
                f"💱 Symbol: <b>{trade['symbol']}</b>\n"
                f"📌 Direction: <b>{pending['direction']}</b>\n"
                f"⚡ Leverage: <b>x{pending['leverage']:g}</b>\n"
                f"💰 Margin: <b>${trade['notional']:,.2f}</b>\n"
                f"💵 Entry: <b>${entry:,.2f}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 /portfolio · 🔒 /close {trade['id']}",
                parse_mode="HTML", reply_markup=main_menu_keyboard(),
            )

        elif data == "cancel_trade":
            context.user_data.pop("pending_trade", None)
            context.user_data["state"] = None
            await query.edit_message_text(
                "❌ Trade cancelled.", parse_mode="HTML", reply_markup=main_menu_keyboard()
            )

        elif data.startswith("close_quick:"):
            trade_id = data.split(":", 1)[1]
            trade = next((t for t in load_trades()["active"] if t["id"] == trade_id), None)
            if trade is None:
                await query.edit_message_text(
                    "❌ Trade not found or already closed.",
                    parse_mode="HTML", reply_markup=main_menu_keyboard(),
                )
                return
            price = get_price(trade["symbol"])
            if price is None:
                await query.edit_message_text(
                    "⚠️ Unable to fetch price to close trade.",
                    parse_mode="HTML", reply_markup=main_menu_keyboard(),
                )
                return
            closed = close_trade(trade_id, price)
            confirmation = (
                f"✅ <b>POSITION CLOSED</b>\n"
                f"<b>ID:</b> #{closed['id']}\n"
                f"<b>Final P&L:</b> {closed['pnl_pct']:+.2f}% (${closed['pnl_usd']:+,.2f})"
            )
            keyboard = portfolio_keyboard()
            if keyboard:
                await query.edit_message_text(
                    confirmation + "\n\n" + build_portfolio_report(),
                    parse_mode="HTML", reply_markup=keyboard,
                )
            else:
                await query.edit_message_text(
                    confirmation, parse_mode="HTML", reply_markup=main_menu_keyboard()
                )

        else:
            await query.edit_message_text(
                "❓ Unknown action. Use the menu.", reply_markup=main_menu_keyboard()
            )

    except Exception as exc:
        print(f"❌ ERROR in button_handler: {exc}")


def main():
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .pool_timeout(30)
        .get_updates_read_timeout(30)
        .get_updates_write_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_pool_timeout(30)
        .build()
    )

    application.job_queue.scheduler.configure(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 30,
        },
        **application.job_queue.scheduler_configuration,
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("trade", trade_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("close", close_command))
    application.add_handler(CommandHandler("whales", whales_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("alert", alert_command))
    application.add_handler(CommandHandler("myalerts", myalerts_command))
    application.add_handler(CommandHandler("risk", risk_command))
    application.add_handler(CommandHandler("market", market_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))

    application.job_queue.run_repeating(whale_tracker_loop, interval=120, first=10)
    application.job_queue.run_repeating(check_price_alerts, interval=30, first=5)
    application.job_queue.run_repeating(position_monitor_loop, interval=60, first=15)

    threading.Thread(target=run_flask, daemon=True).start()
    print("🌐 Flask server started on port 10000")
    print("🤖 Telegram Bot is online and listening...")
    application.run_polling()


if __name__ == "__main__":
    main()
