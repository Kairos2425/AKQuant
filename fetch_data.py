#!/usr/bin/env python3
"""
Fetch 中芯国际 (688981) daily stock data using multiple sources.
Primary: East Money API (direct HTTP)
Backup: tushare / akshare
"""

import json
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_via_curl(symbol="688981", market=1, start_date="20250701", end_date="20260704"):
    """Fetch stock data directly via curl to bypass Python proxy issues."""
    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116"
        f"&ut=7eea3edcaed734bea9cbfc24409ed989"
        f"&klt=101&fqt=1"
        f"&secid={market}.{symbol}"
        f"&beg={start_date}&end={end_date}"
    )

    result = subprocess.run(
        ["curl", "-s", url],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")

    data = json.loads(result.stdout)

    if data.get("data") and data["data"].get("klines"):
        klines = data["data"]["klines"]
        columns = ["date", "open", "close", "high", "low",
                    "volume", "amount", "amplitude", "pct_chg",
                    "change", "turnover"]

        rows = []
        for line in klines:
            parts = line.split(",")
            rows.append({
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": int(parts[5]),
                "amount": float(parts[6]),
                "amplitude": float(parts[7]),
                "pct_chg": float(parts[8]),
                "change": float(parts[9]),
                "turnover": float(parts[10]),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    return None


def try_akshare():
    """Try fetching via akshare as backup."""
    try:
        import os
        # Try to bypass proxy
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ.pop(key, None)

        import akshare as ak
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(
            symbol="688981", period="daily",
            start_date=start_date, end_date=end_date,
            adjust="qfq"
        )

        # Rename columns to standard
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude",
            "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover"
        }
        df.rename(columns=col_map, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        print(f"akshare fallback failed: {e}")
        return None


def main():
    print("=" * 60)
    print("  中芯国际 (688981) 股票数据获取")
    print("=" * 60)

    df = None

    # Method 1: Direct curl (most reliable)
    print("\n[1/3] Trying direct curl to East Money API...")
    try:
        df = fetch_via_curl()
        print(f"  ✓ Success: {len(df)} trading days fetched")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Method 2: akshare
    if df is None or len(df) == 0:
        print("\n[2/3] Trying akshare...")
        try:
            df = try_akshare()
            if df is not None and len(df) > 0:
                print(f"  ✓ Success: {len(df)} trading days fetched")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    if df is None or len(df) == 0:
        print("\n[ERROR] All data sources failed.")
        sys.exit(1)

    # Filter to last ~1 year
    one_year_ago = datetime.now() - timedelta(days=365)
    df = df[df.index >= one_year_ago].copy()

    print(f"\n  Final dataset: {len(df)} trading days")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: ¥{df['close'].min():.2f} - ¥{df['close'].max():.2f}")

    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "data", "688981_daily.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"\n  ✓ CSV saved to: {csv_path}")

    # Save to JSON for web
    json_path = os.path.join(os.path.dirname(__file__), "data", "688981_daily.json")
    df_json = df.copy()
    df_json.index = df_json.index.strftime("%Y-%m-%d")
    df_json.to_json(json_path, orient="index", force_ascii=False, indent=2)
    print(f"  ✓ JSON saved to: {json_path}")

    # Quick stats
    print(f"\n{'='*60}")
    print("  QUICK STATS")
    print(f"{'='*60}")
    print(f"  最新收盘价: ¥{df['close'].iloc[-1]:.2f}")
    print(f"  一年最高:   ¥{df['high'].max():.2f}  ({df['high'].idxmax().strftime('%Y-%m-%d')})")
    print(f"  一年最低:   ¥{df['low'].min():.2f}  ({df['low'].idxmin().strftime('%Y-%m-%d')})")
    print(f"  平均成交量: {df['volume'].mean():,.0f} 股")
    print(f"  平均换手率: {df['turnover'].mean():.2f}%")

    return df


if __name__ == "__main__":
    df = main()
