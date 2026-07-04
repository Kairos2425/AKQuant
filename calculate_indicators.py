#!/usr/bin/env python3
"""
Calculate all technical indicators for 中芯国际 (688981).
Outputs a comprehensive JSON file for the web dashboard.
"""

import pandas as pd
import numpy as np
import json
import os


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        return super().default(obj)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    """Load the daily stock data."""
    df = pd.read_csv(os.path.join(DATA_DIR, "688981_daily.csv"),
                     index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def calc_ma(df, periods=[5, 10, 20, 60]):
    """Calculate Moving Averages."""
    for p in periods:
        df[f'MA{p}'] = df['close'].rolling(window=p).mean()
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD_DIF'] = ema_fast - ema_slow
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD_HIST'] = 2 * (df['MACD_DIF'] - df['MACD_DEA'])
    return df


def calc_rsi(df, period=14):
    """Calculate RSI indicator."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    return df


def calc_kdj(df, n=9, m1=3, m2=3):
    """Calculate KDJ indicator."""
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()

    rsv = ((df['close'] - low_min) / (high_max - low_min)) * 100
    df['K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/m2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df


def calc_bollinger(df, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    df['BOLL_MID'] = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['BOLL_UP'] = df['BOLL_MID'] + std_dev * std
    df['BOLL_DN'] = df['BOLL_MID'] - std_dev * std
    df['BOLL_WIDTH'] = (df['BOLL_UP'] - df['BOLL_DN']) / df['BOLL_MID'] * 100
    return df


def calc_support_resistance(df, window=20, threshold=0.03):
    """Calculate Support and Resistance levels using local extrema."""
    # Find local maxima (resistance) and minima (support)
    highs = df['high'].values
    lows = df['low'].values

    resistance_levels = []
    support_levels = []

    for i in range(window, len(df) - window):
        # Local maximum
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance_levels.append(highs[i])
        # Local minimum
        if lows[i] == min(lows[i-window:i+window+1]):
            support_levels.append(lows[i])

    # Cluster nearby levels
    def cluster_levels(levels, threshold_pct=threshold):
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = []
        current_group = [levels[0]]
        for lvl in levels[1:]:
            if lvl - current_group[-1] <= current_group[-1] * threshold_pct:
                current_group.append(lvl)
            else:
                clustered.append(np.mean(current_group))
                current_group = [lvl]
        clustered.append(np.mean(current_group))
        return sorted(clustered, reverse=True)

    resistance = cluster_levels(resistance_levels)
    support = cluster_levels(support_levels)

    # Filter to relevant levels near current price
    current_price = df['close'].iloc[-1]
    resistance = [r for r in resistance if r > current_price][:3]
    support = [s for s in support if s < current_price][:3]

    return resistance, support


def calc_volume_indicators(df):
    """Calculate volume-based indicators."""
    df['VOL_MA5'] = df['volume'].rolling(window=5).mean()
    df['VOL_MA20'] = df['volume'].rolling(window=20).mean()
    df['VOL_RATIO'] = df['volume'] / df['VOL_MA20']

    # Volume-price trend
    df['VPT'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
    return df


def calc_atr(df, period=14):
    """Calculate Average True Range for volatility."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/period, adjust=False).mean()
    return df


def generate_summary(df):
    """Generate quantitative summary statistics."""
    recent = df.tail(20)
    current_price = df['close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]

    # Price change stats
    chg_1d = df['pct_chg'].iloc[-1]
    chg_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
    chg_20d = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0

    # Volatility
    volatility_20d = recent['pct_chg'].std()

    # Volume trend
    vol_ratio = df['volume'].iloc[-1] / df['VOL_MA20'].iloc[-1] if not pd.isna(df['VOL_MA20'].iloc[-1]) else 0

    # Trend strength
    above_ma = sum([
        current_price > df['MA5'].iloc[-1] if not pd.isna(df['MA5'].iloc[-1]) else False,
        current_price > ma20 if not pd.isna(ma20) else False,
        current_price > ma60 if not pd.isna(ma60) else False,
    ])

    return {
        "current_price": round(current_price, 2),
        "chg_1d": round(chg_1d, 2),
        "chg_5d": round(chg_5d, 2),
        "chg_20d": round(chg_20d, 2),
        "volatility_20d": round(volatility_20d, 2),
        "vol_ratio": round(vol_ratio, 2),
        "above_ma_count": above_ma,
        "trend": "强势" if above_ma >= 3 else ("偏强" if above_ma >= 2 else ("偏弱" if above_ma >= 1 else "弱势")),
        "ma20_position": round((current_price / ma20 - 1) * 100, 2) if not pd.isna(ma20) else None,
        "ma60_position": round((current_price / ma60 - 1) * 100, 2) if not pd.isna(ma60) else None,
        "rsi": round(df['RSI'].iloc[-1], 2) if not pd.isna(df['RSI'].iloc[-1]) else None,
    }


def main():
    print("Calculating technical indicators for 中芯国际 (688981)...")

    df = load_data()
    print(f"  Loaded {len(df)} trading days")

    # Calculate all indicators
    df = calc_ma(df)
    df = calc_macd(df)
    df = calc_rsi(df)
    df = calc_kdj(df)
    df = calc_bollinger(df)
    df = calc_volume_indicators(df)
    df = calc_atr(df)

    # Calculate support & resistance
    resistance, support = calc_support_resistance(df)
    summary = generate_summary(df)

    print(f"  Resistance levels: {[round(r, 2) for r in resistance]}")
    print(f"  Support levels: {[round(s, 2) for s in support]}")
    print(f"  Current trend: {summary['trend']}")
    print(f"  RSI(14): {summary['rsi']}")

    # Prepare data for JSON export
    # Replace NaN with null for JSON compatibility
    df_export = df.copy()
    df_export.index = df_export.index.strftime("%Y-%m-%d")

    # Build export object
    export = {
        "metadata": {
            "symbol": "688981",
            "name": "中芯国际",
            "market": "科创板",
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(df),
            "date_range": [df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")],
        },
        "summary": summary,
        "support_resistance": {
            "resistance": [round(r, 2) for r in resistance],
            "support": [round(s, 2) for s in support],
        },
        "ohlc": [],
        "indicators": [],
    }

    # Build row-by-row data (more efficient for Plotly)
    for idx, row in df_export.iterrows():
        entry = {
            "date": idx,
            "open": round(row["open"], 2) if not pd.isna(row["open"]) else None,
            "high": round(row["high"], 2) if not pd.isna(row["high"]) else None,
            "low": round(row["low"], 2) if not pd.isna(row["low"]) else None,
            "close": round(row["close"], 2) if not pd.isna(row["close"]) else None,
            "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
            "amount": round(row["amount"], 2) if not pd.isna(row["amount"]) else None,
            "pct_chg": round(row["pct_chg"], 2) if not pd.isna(row["pct_chg"]) else None,
            "turnover": round(row["turnover"], 2) if not pd.isna(row["turnover"]) else None,
            "amplitude": round(row["amplitude"], 2) if not pd.isna(row["amplitude"]) else None,
            # MAs
            "MA5": round(row["MA5"], 2) if not pd.isna(row["MA5"]) else None,
            "MA10": round(row["MA10"], 2) if not pd.isna(row["MA10"]) else None,
            "MA20": round(row["MA20"], 2) if not pd.isna(row["MA20"]) else None,
            "MA60": round(row["MA60"], 2) if not pd.isna(row["MA60"]) else None,
            # MACD
            "MACD_DIF": round(row["MACD_DIF"], 4) if not pd.isna(row["MACD_DIF"]) else None,
            "MACD_DEA": round(row["MACD_DEA"], 4) if not pd.isna(row["MACD_DEA"]) else None,
            "MACD_HIST": round(row["MACD_HIST"], 4) if not pd.isna(row["MACD_HIST"]) else None,
            # RSI
            "RSI": round(row["RSI"], 2) if not pd.isna(row["RSI"]) else None,
            # KDJ
            "K": round(row["K"], 2) if not pd.isna(row["K"]) else None,
            "D": round(row["D"], 2) if not pd.isna(row["D"]) else None,
            "J": round(row["J"], 2) if not pd.isna(row["J"]) else None,
            # Bollinger
            "BOLL_UP": round(row["BOLL_UP"], 2) if not pd.isna(row["BOLL_UP"]) else None,
            "BOLL_MID": round(row["BOLL_MID"], 2) if not pd.isna(row["BOLL_MID"]) else None,
            "BOLL_DN": round(row["BOLL_DN"], 2) if not pd.isna(row["BOLL_DN"]) else None,
            "BOLL_WIDTH": round(row["BOLL_WIDTH"], 2) if not pd.isna(row["BOLL_WIDTH"]) else None,
            # Volume
            "VOL_MA5": int(row["VOL_MA5"]) if not pd.isna(row["VOL_MA5"]) else 0,
            "VOL_MA20": int(row["VOL_MA20"]) if not pd.isna(row["VOL_MA20"]) else 0,
            "VOL_RATIO": round(row["VOL_RATIO"], 2) if not pd.isna(row["VOL_RATIO"]) else None,
            # Others
            "ATR": round(row["ATR"], 4) if not pd.isna(row["ATR"]) else None,
        }
        export["ohlc"].append(entry)

    # Save to JSON
    output_path = os.path.join(DATA_DIR, "dashboard_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, cls=NpEncoder)
    print(f"\n  Dashboard data saved to: {output_path}")
    print(f"  Total data points: {len(export['ohlc'])}")
    print("Done!")


if __name__ == "__main__":
    main()
