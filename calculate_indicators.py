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


def generate_analysis(df, summary, resistance, support):
    """Generate comprehensive quantitative analysis text for each module."""
    current_price = summary["current_price"]
    analysis = {}

    # ========================================
    # 1. K-line & MA Trend Analysis
    # ========================================
    ma5 = df['MA5'].iloc[-1]
    ma10 = df['MA10'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    last_close = df['close'].iloc[-1]

    # MA alignment
    mas = {'MA5': ma5, 'MA10': ma10, 'MA20': ma20, 'MA60': ma60}
    valid_mas = {k: v for k, v in mas.items() if not pd.isna(v)}
    sorted_mas = sorted(valid_mas.items(), key=lambda x: x[1], reverse=True)
    ideal_order = ['MA5', 'MA10', 'MA20', 'MA60']
    current_order = [m[0] for m in sorted_mas]

    if current_order == ideal_order and last_close > ma5:
        ma_align = "多头完美排列：MA5>MA10>MA20>MA60，价格站上所有均线，处于强势上升通道"
        ma_signal = "bullish"
    elif last_close > ma5 and last_close > ma20:
        ma_align = "短中期均线多头排列，价格位于MA5和MA20上方，中期趋势向好但短期可能震荡"
        ma_signal = "bullish"
    elif last_close < ma60:
        ma_align = f"价格跌破MA60（¥{ma60:.2f}）长期均线，中长期趋势偏弱，需关注MA60能否收复"
        ma_signal = "bearish"
    elif last_close > ma60 and last_close < ma20:
        ma_align = f"价格在MA20（¥{ma20:.2f}）与MA60（¥{ma60:.2f}）之间震荡整理，方向待选择"
        ma_signal = "neutral"
    else:
        ma_align = "均线交织缠绕，趋势不明确，处于震荡格局"
        ma_signal = "neutral"

    # MA cross signals (recent)
    cross_signals = []
    for i in range(-5, 0):
        if pd.isna(df['MA5'].iloc[i]) or pd.isna(df['MA10'].iloc[i]):
            continue
        if i > -len(df):
            prev_ma5 = df['MA5'].iloc[i-1]
            prev_ma10 = df['MA10'].iloc[i-1]
            curr_ma5 = df['MA5'].iloc[i]
            curr_ma10 = df['MA10'].iloc[i]
            if prev_ma5 <= prev_ma10 and curr_ma5 > curr_ma10:
                cross_signals.append(f"{df.index[i].strftime('%m/%d')} MA5上穿MA10（金叉）")
            elif prev_ma5 >= prev_ma10 and curr_ma5 < curr_ma10:
                cross_signals.append(f"{df.index[i].strftime('%m/%d')} MA5下穿MA10（死叉）")

    # Candlestick analysis (last 3 days)
    last3 = df.tail(3)
    body_sizes = abs(last3['close'] - last3['open'])
    avg_body = body_sizes.mean()
    latest_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    latest_range = df['high'].iloc[-1] - df['low'].iloc[-1]

    if latest_body < latest_range * 0.2:
        candle_pattern = "最近交易日出现十字星形态，多空力量均衡，变盘信号值得关注"
    elif df['close'].iloc[-1] > df['open'].iloc[-1] and latest_body > avg_body * 1.5:
        candle_pattern = "最近交易日收出实体阳线，多头力量增强"
    elif df['close'].iloc[-1] < df['open'].iloc[-1] and latest_body > avg_body * 1.5:
        candle_pattern = "最近交易日收出实体阴线，空头力量占优"
    else:
        candle_pattern = "K线实体适中，多空博弈较为均衡"

    # ATR volatility
    atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else 0
    atr_pct = (atr / current_price) * 100

    analysis["kline_ma"] = {
        "title": "K线趋势与均线分析",
        "ma_alignment": ma_align,
        "ma_signal": ma_signal,
        "cross_signals": cross_signals[-3:] if cross_signals else ["近期无明显均线交叉信号"],
        "candle_pattern": candle_pattern,
        "atr_pct": round(atr_pct, 2),
        "price_vs_ma5": f"价格{'高于' if last_close > ma5 else '低于'}MA5（¥{ma5:.2f}），偏离{abs((last_close/ma5-1)*100):.1f}%",
        "price_vs_ma20": f"价格{'高于' if last_close > ma20 else '低于'}MA20（¥{ma20:.2f}），偏离{abs((last_close/ma20-1)*100):.1f}%",
        "price_vs_ma60": f"价格{'高于' if last_close > ma60 else '低于'}MA60（¥{ma60:.2f}），偏离{abs((last_close/ma60-1)*100):.1f}%",
        "verdict": "短期看多，中期谨慎看多" if ma_signal == "bullish" else ("短期偏空，中长期需观察" if ma_signal == "bearish" else "震荡整理，方向待明确"),
    }

    # ========================================
    # 2. MACD Analysis
    # ========================================
    dif = df['MACD_DIF'].iloc[-1]
    dea = df['MACD_DEA'].iloc[-1]
    hist = df['MACD_HIST'].iloc[-1]
    prev_hist = df['MACD_HIST'].iloc[-2] if len(df) > 2 else 0

    if dif > 0 and dea > 0:
        macd_zone = "DIF与DEA均在零轴上方，中期趋势偏多"
    elif dif < 0 and dea < 0:
        macd_zone = "DIF与DEA均在零轴下方，中期趋势偏空"
    else:
        macd_zone = "DIF与DEA在零轴附近交错，趋势方向不明确"

    if dif > dea:
        macd_position = f"DIF（{dif:.3f}）在DEA（{dea:.3f}）上方，MACD柱状图为{'红' if hist > 0 else '绿'}柱，多头占优"
    else:
        macd_position = f"DIF（{dif:.3f}）在DEA（{dea:.3f}）下方，MACD柱状图为{'红' if hist > 0 else '绿'}柱，空头占优"

    # Histogram analysis
    if hist > 0 and hist > prev_hist:
        hist_trend = "红柱持续放大，上涨动能增强"
    elif hist > 0 and hist < prev_hist:
        hist_trend = "红柱缩短，上涨动能减弱，可能出现顶背离"
    elif hist < 0 and hist < prev_hist:
        hist_trend = "绿柱持续放大，下跌动能增强"
    elif hist < 0 and hist > prev_hist:
        hist_trend = "绿柱缩短，下跌动能减弱，可能出现底背离"
    else:
        hist_trend = "柱状图变化不明显"

    # MACD cross
    macd_cross = ""
    for i in range(-3, 0):
        if pd.isna(df['MACD_DIF'].iloc[i]) or pd.isna(df['MACD_DEA'].iloc[i]):
            continue
        if i > -len(df):
            p_dif = df['MACD_DIF'].iloc[i-1]
            p_dea = df['MACD_DEA'].iloc[i-1]
            c_dif = df['MACD_DIF'].iloc[i]
            c_dea = df['MACD_DEA'].iloc[i]
            if p_dif <= p_dea and c_dif > c_dea:
                macd_cross = f"{df.index[i].strftime('%m/%d')} MACD金叉（DIF上穿DEA），买入信号"
                break
            elif p_dif >= p_dea and c_dif < c_dea:
                macd_cross = f"{df.index[i].strftime('%m/%d')} MACD死叉（DIF下穿DEA），卖出信号"
                break
    if not macd_cross:
        macd_cross = "近期无MACD交叉信号"

    analysis["macd"] = {
        "title": "MACD指标分析",
        "zone": macd_zone,
        "position": macd_position,
        "hist_trend": hist_trend,
        "cross": macd_cross,
        "values": f"DIF={dif:.4f} DEA={dea:.4f} HIST={hist:.4f}",
        "verdict": "MACD偏多，持仓或逢低关注" if dif > dea and dif > 0 else ("MACD偏空，观望或减仓" if dif < dea and dif < 0 else "MACD中性，等待方向选择"),
    }

    # ========================================
    # 3. RSI Analysis
    # ========================================
    rsi = df['RSI'].iloc[-1]
    rsi_5d_ago = df['RSI'].iloc[-6] if len(df) > 6 else rsi

    if rsi > 80:
        rsi_zone = f"RSI={rsi:.1f}，处于严重超买区间（>80），回调风险较大"
        rsi_signal = "bearish"
    elif rsi > 70:
        rsi_zone = f"RSI={rsi:.1f}，处于超买区间（70-80），短期有技术回调需求"
        rsi_signal = "bearish"
    elif rsi > 50:
        rsi_zone = f"RSI={rsi:.1f}，处于偏强区间（50-70），多头占据主动"
        rsi_signal = "bullish"
    elif rsi > 30:
        rsi_zone = f"RSI={rsi:.1f}，处于偏弱区间（30-50），空头力量偏强"
        rsi_signal = "bearish"
    elif rsi > 20:
        rsi_zone = f"RSI={rsi:.1f}，处于超卖区间（20-30），短期有技术反弹需求"
        rsi_signal = "bullish"
    else:
        rsi_zone = f"RSI={rsi:.1f}，处于严重超卖区间（<20），反弹概率较大"
        rsi_signal = "bullish"

    rsi_trend = f"RSI五日变化：{rsi_5d_ago:.1f}→{rsi:.1f}，{'上升' if rsi > rsi_5d_ago else '下降'}趋势"

    # RSI divergence check
    price_5d = df['close'].iloc[-6] if len(df) > 6 else df['close'].iloc[-1]
    if df['close'].iloc[-1] > price_5d and rsi < rsi_5d_ago:
        rsi_divergence = "⚠ 注意：近5日价格走高但RSI走低，存在顶背离风险"
    elif df['close'].iloc[-1] < price_5d and rsi > rsi_5d_ago:
        rsi_divergence = "💡 提示：近5日价格走低但RSI走高，存在底背离机会"
    else:
        rsi_divergence = "RSI与价格走势同步，无明显背离"

    analysis["rsi"] = {
        "title": "RSI相对强弱分析",
        "zone": rsi_zone,
        "trend": rsi_trend,
        "divergence": rsi_divergence,
        "signal": rsi_signal,
        "verdict": "RSI显示短期超买，注意回调" if rsi > 70 else ("RSI显示短期超卖，关注反弹机会" if rsi < 30 else "RSI处于中性区间，顺势操作"),
    }

    # ========================================
    # 4. KDJ Analysis
    # ========================================
    k = df['K'].iloc[-1]
    d = df['D'].iloc[-1]
    j = df['J'].iloc[-1]
    prev_k = df['K'].iloc[-2] if len(df) > 2 else k
    prev_d = df['D'].iloc[-2] if len(df) > 2 else d

    if j > 100:
        kdj_zone = f"J值={j:.1f}，超过100进入超买区，短期回调概率大"
    elif j < 0:
        kdj_zone = f"J值={j:.1f}，跌破0进入超卖区，短期反弹概率大"
    else:
        kdj_zone = f"K={k:.1f} D={d:.1f} J={j:.1f}，KDJ处于正常波动区间"

    if prev_k <= prev_d and k > d:
        kdj_cross = f"K线（{k:.1f}）上穿D线（{d:.1f}），KDJ金叉，短线看多信号"
    elif prev_k >= prev_d and k < d:
        kdj_cross = f"K线（{k:.1f}）下穿D线（{d:.1f}），KDJ死叉，短线看空信号"
    else:
        kdj_cross = f"K线（{k:.1f}）与D线（{d:.1f}）维持{'多头' if k > d else '空头'}排列"

    analysis["kdj"] = {
        "title": "KDJ随机指标分析",
        "zone": kdj_zone,
        "cross": kdj_cross,
        "values": f"K={k:.2f} D={d:.2f} J={j:.2f}",
        "verdict": "KDJ短线看多，可关注低吸机会" if k > d and j < 80 else ("KDJ短线偏空，等待超卖信号" if k < d and j > 20 else "KDJ中性，等待明确信号"),
    }

    # ========================================
    # 5. Bollinger Bands Analysis
    # ========================================
    bb_up = df['BOLL_UP'].iloc[-1]
    bb_mid = df['BOLL_MID'].iloc[-1]
    bb_dn = df['BOLL_DN'].iloc[-1]
    bb_width = df['BOLL_WIDTH'].iloc[-1]
    bb_width_10d = df['BOLL_WIDTH'].iloc[-11] if len(df) > 11 else bb_width

    price_in_band = (last_close - bb_dn) / (bb_up - bb_dn) * 100  # 0-100

    if price_in_band > 90:
        bb_position = f"价格（¥{last_close:.2f}）接近布林带上轨（¥{bb_up:.2f}），处于强势区域但面临上轨压力"
    elif price_in_band > 60:
        bb_position = f"价格位于布林带中上轨之间，运行偏强"
    elif price_in_band > 30:
        bb_position = f"价格位于布林带中下轨之间，运行偏弱"
    elif price_in_band > 10:
        bb_position = f"价格接近布林带下轨（¥{bb_dn:.2f}），处于弱势区域但有下轨支撑"
    else:
        bb_position = f"价格触及布林带下轨（¥{bb_dn:.2f}），超跌反弹概率加大"

    if bb_width < bb_width_10d * 0.8:
        bb_squeeze = f"布林带宽度收窄（{bb_width:.1f}% vs 10日前{bb_width_10d:.1f}%），波动率压缩，可能酝酿突破行情"
    elif bb_width > bb_width_10d * 1.3:
        bb_squeeze = f"布林带宽度扩张（{bb_width:.1f}% vs 10日前{bb_width_10d:.1f}%），波动率放大，趋势行情可能延续"
    else:
        bb_squeeze = f"布林带宽度稳定（{bb_width:.1f}%），波动率正常"

    analysis["bollinger"] = {
        "title": "布林带指标分析",
        "position": bb_position,
        "squeeze": bb_squeeze,
        "band_values": f"上轨¥{bb_up:.2f} 中轨¥{bb_mid:.2f} 下轨¥{bb_dn:.2f}",
        "width_pct": round(bb_width, 1),
        "verdict": "布林带上轨附近，注意压力位" if price_in_band > 85 else ("布林带下轨附近，关注支撑反弹" if price_in_band < 15 else "布林带中轨附近，震荡格局"),
    }

    # ========================================
    # 6. Volume Analysis
    # ========================================
    vol = df['volume'].iloc[-1]
    vol_ma5 = df['VOL_MA5'].iloc[-1]
    vol_ma20 = df['VOL_MA20'].iloc[-1]
    vol_ratio = summary['vol_ratio']
    pct_chg = df['pct_chg'].iloc[-1]

    # Volume trend
    vol_5d_avg = df['volume'].tail(5).mean()
    vol_20d_avg = df['volume'].tail(20).mean()
    vol_60d_avg = df['volume'].tail(60).mean() if len(df) >= 60 else vol_20d_avg

    if vol_5d_avg > vol_20d_avg * 1.3:
        vol_trend = "近5日均量显著高于20日均量，资金活跃度提升，关注放量方向"
    elif vol_5d_avg < vol_20d_avg * 0.7:
        vol_trend = "近5日均量显著低于20日均量，市场交投清淡，观望情绪浓厚"
    else:
        vol_trend = "近期成交量围绕20日均量波动，交投平稳"

    # Price-volume relationship
    if pct_chg > 2 and vol_ratio > 1.5:
        pv_relation = "放量上涨，量价配合良好，上涨动力充足"
    elif pct_chg > 0 and vol_ratio < 0.7:
        pv_relation = "缩量上涨，上涨缺乏量能支持，警惕回调"
    elif pct_chg < -2 and vol_ratio > 1.5:
        pv_relation = "放量下跌，抛压较重，短期回避"
    elif pct_chg < 0 and vol_ratio < 0.7:
        pv_relation = "缩量下跌，抛压减轻，可能出现企稳信号"
    else:
        pv_relation = "量价关系正常，未出现明显异常信号"

    # Key volume events
    vol_max_idx = df['volume'].tail(60).idxmax()
    vol_max_date = vol_max_idx.strftime('%m/%d')
    vol_max_ratio = df.loc[vol_max_idx, 'volume'] / df['VOL_MA20'].loc[vol_max_idx]

    analysis["volume"] = {
        "title": "成交量与资金分析",
        "trend": vol_trend,
        "pv_relation": pv_relation,
        "vol_ratio": round(vol_ratio, 2),
        "recent_avg_5d": f"5日均量：{vol_5d_avg/10000:.0f}万手",
        "recent_avg_20d": f"20日均量：{vol_20d_avg/10000:.0f}万手",
        "max_vol_event": f"近60日最大量出现在{vol_max_date}，量比{vol_max_ratio:.1f}倍",
        "verdict": "放量配合上涨，资金参与度高" if pct_chg > 0 and vol_ratio > 1.2 else ("缩量或下跌放量，需谨慎观察" if pct_chg < 0 or vol_ratio < 0.8 else "量能平稳，维持现有趋势判断"),
    }

    # ========================================
    # 7. Support & Resistance Analysis
    # ========================================
    sr_lines = []
    for r in resistance[:3]:
        dist = (r / current_price - 1) * 100
        sr_lines.append({
            "level": r, "type": "resistance",
            "dist_pct": round(dist, 1),
            "desc": f"阻力位¥{r:.2f}，距当前价+{dist:.1f}%，{'接近阻力，突破需放量配合' if dist < 5 else '上方空间充足'}"
        })
    for s in support[:3]:
        dist = (current_price / s - 1) * 100
        sr_lines.append({
            "level": s, "type": "support",
            "dist_pct": round(dist, 1),
            "desc": f"支撑位¥{s:.2f}，距当前价-{dist:.1f}%，{'接近支撑，关注是否有效守住' if dist < 5 else '下方缓冲充足'}"
        })

    # Risk/Reward
    nearest_r = resistance[0] if resistance else current_price * 1.1
    nearest_s = support[0] if support else current_price * 0.9
    upside = (nearest_r / current_price - 1) * 100
    downside = (1 - nearest_s / current_price) * 100
    rr_ratio = upside / downside if downside > 0 else 999

    analysis["sr"] = {
        "title": "支撑阻力与风控分析",
        "levels": sr_lines,
        "upside_pct": round(upside, 1),
        "downside_pct": round(downside, 1),
        "rr_ratio": round(rr_ratio, 1),
        "verdict": f"风险收益比{rr_ratio:.1f}:1，{'盈亏比良好，适合参与' if rr_ratio > 2 else ('盈亏比一般，仓位需谨慎' if rr_ratio > 1 else '风险大于收益，建议观望')}",
    }

    # ========================================
    # 8. Comprehensive Multi-Factor Assessment
    # ========================================
    signals = []
    # K-line / MA
    if last_close > ma5 and last_close > ma20:
        signals.append({"name": "均线趋势", "signal": "偏多", "weight": 2, "cls": "bullish"})
    elif last_close < ma60:
        signals.append({"name": "均线趋势", "signal": "偏空", "weight": 2, "cls": "bearish"})
    else:
        signals.append({"name": "均线趋势", "signal": "中性", "weight": 0, "cls": "neutral"})

    # MACD
    if dif > dea and dif > 0:
        signals.append({"name": "MACD", "signal": "偏多", "weight": 2, "cls": "bullish"})
    elif dif < dea and dif < 0:
        signals.append({"name": "MACD", "signal": "偏空", "weight": 2, "cls": "bearish"})
    else:
        signals.append({"name": "MACD", "signal": "中性", "weight": 0, "cls": "neutral"})

    # RSI
    if 40 < rsi < 60:
        signals.append({"name": "RSI", "signal": "中性", "weight": 0, "cls": "neutral"})
    elif rsi > 70:
        signals.append({"name": "RSI", "signal": "超买", "weight": -1, "cls": "bearish"})
    elif rsi < 30:
        signals.append({"name": "RSI", "signal": "超卖", "weight": 1, "cls": "bullish"})
    elif rsi > 60:
        signals.append({"name": "RSI", "signal": "偏强", "weight": 1, "cls": "bullish"})
    else:
        signals.append({"name": "RSI", "signal": "偏弱", "weight": -1, "cls": "bearish"})

    # Volume
    if vol_ratio > 1.5 and pct_chg > 0:
        signals.append({"name": "成交量", "signal": "放量上涨", "weight": 2, "cls": "bullish"})
    elif vol_ratio > 1.5 and pct_chg < 0:
        signals.append({"name": "成交量", "signal": "放量下跌", "weight": -2, "cls": "bearish"})
    elif vol_ratio < 0.5:
        signals.append({"name": "成交量", "signal": "缩量", "weight": -1, "cls": "bearish"})
    else:
        signals.append({"name": "成交量", "signal": "正常", "weight": 0, "cls": "neutral"})

    # KDJ
    if k > d and j < 80:
        signals.append({"name": "KDJ", "signal": "金叉偏多", "weight": 1, "cls": "bullish"})
    elif k < d and j > 20:
        signals.append({"name": "KDJ", "signal": "死叉偏空", "weight": -1, "cls": "bearish"})
    else:
        signals.append({"name": "KDJ", "signal": "中性", "weight": 0, "cls": "neutral"})

    # Bollinger
    if price_in_band > 80:
        signals.append({"name": "布林带", "signal": "上轨压力", "weight": -1, "cls": "bearish"})
    elif price_in_band < 20:
        signals.append({"name": "布林带", "signal": "下轨支撑", "weight": 1, "cls": "bullish"})
    else:
        signals.append({"name": "布林带", "signal": "中轨运行", "weight": 0, "cls": "neutral"})

    # SR
    if upside > downside * 2:
        signals.append({"name": "支撑阻力", "signal": "盈亏比优", "weight": 1, "cls": "bullish"})
    elif downside > upside * 2:
        signals.append({"name": "支撑阻力", "signal": "盈亏比差", "weight": -1, "cls": "bearish"})
    else:
        signals.append({"name": "支撑阻力", "signal": "平衡", "weight": 0, "cls": "neutral"})

    total_weight = sum(s["weight"] for s in signals)
    max_weight = sum(abs(s["weight"]) for s in signals)

    if total_weight >= 7:
        overall = "综合量化评分显示多头信号强烈，多个指标共振看多，可积极关注做多机会，注意控制仓位"
        overall_cls = "bullish"
    elif total_weight >= 3:
        overall = "综合评分偏多，多数指标指向多头方向，可适当参与，关注关键阻力位突破情况"
        overall_cls = "bullish"
    elif total_weight >= -2:
        overall = "综合评分中性，多空信号交织，市场处于震荡格局，建议控制仓位等待方向明确"
        overall_cls = "neutral"
    elif total_weight >= -6:
        overall = "综合评分偏空，多数指标指向空头方向，建议减仓或观望，等待企稳信号"
        overall_cls = "bearish"
    else:
        overall = "综合评分显示空头信号强烈，多个指标共振看空，建议规避风险，耐心等待底部信号"
        overall_cls = "bearish"

    analysis["comprehensive"] = {
        "title": "多因子综合研判",
        "signals": signals,
        "total_weight": total_weight,
        "max_weight": max_weight,
        "score_pct": round((total_weight + max_weight) / (2 * max_weight) * 100, 1) if max_weight > 0 else 50,
        "overall": overall,
        "overall_cls": overall_cls,
        "risk_level": "低" if abs(total_weight) >= 7 else ("中" if abs(total_weight) >= 3 else "高"),
        "suggestion": "建议仓位：30-50%，关注量能配合" if overall_cls == "bullish" else (
            "建议仓位：10-20%，观望为主" if overall_cls == "neutral" else "建议仓位：0-10%，规避风险为主"),
        "key_points": [
            f"当前价格¥{current_price:.2f}处于{'上升' if last_close > ma20 else '下降'}趋势中",
            f"MACD{'多头' if dif > dea else '空头'}排列，{'动能' + ('增强' if abs(hist) > abs(prev_hist) else '减弱')}",
            f"RSI={rsi:.1f}，{'偏强' if rsi > 50 else '偏弱'}运行",
            f"量比{vol_ratio:.2f}，{'活跃' if vol_ratio > 1 else '清淡'}",
            f"盈亏比{rr_ratio:.1f}:1，{'适合' if rr_ratio > 2 else '需谨慎'}参与",
        ],
    }

    return analysis


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

    # Generate comprehensive analysis
    print("\n  Generating analysis texts...")
    analysis = generate_analysis(df, summary, resistance, support)
    print(f"  Analysis generated: {len(analysis)} modules")
    print(f"  Overall assessment: {analysis['comprehensive']['overall_cls']}")

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
        "analysis": analysis,
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
