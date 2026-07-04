# AKQuant - 量化分析仪表盘

**中芯国际 (688981.SH)** 实时量化分析平台，基于 Python + Plotly.js 构建。

## ✨ 功能

- 📈 **K线蜡烛图** — 含 MA5/MA10/MA20/MA60 均线系统
- 📊 **MACD 指标** — DIF/DEA/柱状图
- 📉 **RSI 相对强弱** — 超买/超卖区间标注
- 🎯 **KDJ 随机指标** — K/D/J 三线
- 📐 **布林带** — 上轨/中轨/下轨 + 价格走势
- ⚡ **支撑与阻力位** — 局部极值聚类分析
- 📦 **成交量分析** — 含 MA5/MA20 均量线
- 🧠 **量化综合分析** — 多维度信号研判
- 📋 **原始数据表格** — 最近20交易日明细

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy matplotlib

# 2. 获取数据
python fetch_data.py

# 3. 计算技术指标
python calculate_indicators.py

# 4. 生成收盘价曲线图
python plot_closing_price.py

# 5. 打开仪表盘
# 用浏览器打开 dashboard.html
```

## 📁 项目结构

```
AKQuant/
├── fetch_data.py            # 数据获取 (curl → East Money API)
├── calculate_indicators.py  # 技术指标计算
├── plot_closing_price.py    # 收盘价曲线图
├── dashboard.html           # 主仪表盘 (自包含HTML)
├── data/
│   ├── raw_data.json        # 原始API数据
│   ├── 688981_daily.csv     # 每日交易数据 CSV
│   ├── 688981_daily.json    # 每日交易数据 JSON
│   ├── dashboard_data.json  # 完整仪表盘数据 (含所有指标)
│   └── closing_price_chart.png  # 收盘价曲线图
└── README.md
```

## 📊 技术指标参数

| 指标 | 参数 |
|------|------|
| MA 均线 | 5, 10, 20, 60 日 |
| MACD | (12, 26, 9) |
| RSI | 14 日 |
| KDJ | (9, 3, 3) |
| Bollinger Bands | (20, 2) |
| S/R Levels | 窗口20, 阈值3% |

## ⚠️ 免责声明

本工具仅用于技术研究和学习目的，不构成任何投资建议。股市有风险，投资需谨慎。

## 📄 数据来源

- East Money API (push2his.eastmoney.com)
- 备选: akshare

---

**AKQuant** © 2026 — Advanced Quantitative Trading Dashboard
