#!/usr/bin/env python3
"""
Generate closing price chart for 中芯国际 (688981).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "688981_daily.csv"),
                     index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot closing price
    ax.plot(df.index, df['close'], color='#3987e5', linewidth=1.8, label='每日收盘价', zorder=3)

    # Fill area under curve
    ax.fill_between(df.index, df['close'], df['close'].min() * 0.95,
                     alpha=0.08, color='#3987e5')

    # Add MA lines
    if 'close' in df.columns:
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        ax.plot(df.index, df['MA20'], color='#c98500', linewidth=1.2,
                linestyle='--', label='MA20 均线', alpha=0.8)
        ax.plot(df.index, df['MA60'], color='#d55181', linewidth=1.2,
                linestyle='--', label='MA60 均线', alpha=0.8)

    # Mark high and low
    high_idx = df['close'].idxmax()
    low_idx = df['close'].idxmin()
    ax.scatter(high_idx, df['close'].max(), color='#e66767', s=80, zorder=5,
               label=f'最高: ¥{df["close"].max():.2f}')
    ax.scatter(low_idx, df['close'].min(), color='#199e70', s=80, zorder=5,
               label=f'最低: ¥{df["close"].min():.2f}')

    ax.annotate(f'¥{df["close"].max():.2f}',
                xy=(high_idx, df['close'].max()),
                xytext=(0, 12), textcoords='offset points',
                fontsize=10, color='#e66767', ha='center', fontweight='bold')
    ax.annotate(f'¥{df["close"].min():.2f}',
                xy=(low_idx, df['close'].min()),
                xytext=(0, -16), textcoords='offset points',
                fontsize=10, color='#199e70', ha='center', fontweight='bold')

    # Latest price annotation
    latest = df['close'].iloc[-1]
    ax.annotate(f'最新: ¥{latest:.2f}',
                xy=(df.index[-1], latest),
                xytext=(10, 0), textcoords='offset points',
                fontsize=11, color='#ffffff', ha='left', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#3987e5', alpha=0.9))

    ax.set_title('中芯国际 (688981) 过去一年每日收盘价', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('价格 (¥)', fontsize=11)
    leg = ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    fig.patch.set_facecolor('#1a1a19')
    ax.set_facecolor('#1a1a19')
    ax.tick_params(colors='#c3c2b7')
    ax.xaxis.label.set_color('#c3c2b7')
    ax.yaxis.label.set_color('#c3c2b7')
    ax.title.set_color('#ffffff')
    leg.get_frame().set_facecolor('#222221')
    leg.get_frame().set_edgecolor('#3a3a36')
    for text in leg.get_texts():
        text.set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#3a3a36')
    ax.spines['left'].set_color('#3a3a36')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    output_path = os.path.join(DATA_DIR, "closing_price_chart.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a19')
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    main()
