"""
拉取 5 年全量训练数据，覆盖 GPU 训练所需的所有数据类型。
用于在本地拉满数据后打包传输到 H800（H800 无法访问 Binance）。

数据类型:
  1. K线 (OHLCV) — 4 个交易对 × 4 个周期
  2. Mark Price K线 — 永续合约标记价格 (basis 计算)
  3. Funding Rate — 资金费率历史 (ML 特征)
  4. Open Interest — 持仓量 (ML 特征)

运行: python3 fetch_5year_data.py [--skip-klines] [--only-funding] [--only-oi]
预计耗时: 15-30 分钟（受 Binance API 限速影响）
"""

import os
import sys
import time
import pandas as pd
from binance_fetcher import (
    fetch_binance_klines,
    fetch_mark_price_klines,
    fetch_funding_rate_history,
    fetch_open_interest_history,
)

# ── 配置 ──────────────────────────────────────────────────────────────
DAYS_KLINE = 1826       # K线: 5 年
DAYS_FUNDING = 1826     # Funding: 5 年 (Binance 2019 年上线永续合约)
DAYS_OI = 365           # OI: Binance OI 历史 API 最多约 1 年

SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"]

INTERVALS = ["15m", "1h", "4h", "24h"]
MARK_INTERVALS = ["15m", "1h"]

# OI API 支持的粒度 — 拉取 1h 和 4h 即可覆盖训练需求
OI_INTERVALS = ["1h", "4h"]

# 解析命令行参数
args = set(sys.argv[1:])
SKIP_KLINES = "--skip-klines" in args
ONLY_FUNDING = "--only-funding" in args
ONLY_OI = "--only-oi" in args

# ── 统计 ──────────────────────────────────────────────────────────────
results = {"kline": [], "mark": [], "funding": [], "oi": []}
errors = []
t0 = time.time()


def report(symbol, interval, df, kind="kline"):
    if df is None or len(df) == 0:
        errors.append(f"[FAIL] {symbol} {interval} {kind}")
        return
    start = df.index[0].strftime('%Y-%m-%d')
    end = df.index[-1].strftime('%Y-%m-%d')

    dir_map = {"kline": "klines", "mark": "mark_klines",
               "funding": "funding_rates", "oi": "open_interest"}
    if kind == "funding":
        path = os.path.join('data', dir_map[kind], f'{symbol}_funding.parquet')
    elif kind == "oi":
        path = os.path.join('data', dir_map[kind], symbol, f'{interval}.parquet')
    else:
        path = os.path.join('data', dir_map[kind], symbol, f'{interval}.parquet')

    size = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
    results[kind].append(
        f"  ✓ {symbol:10} {interval:5} {len(df):7,}条  {start} ~ {end}  ({size}KB)")


print("=" * 70)
print(f"拉取训练全量数据 → H800 离线训练")
print(f"  K线:     {DAYS_KLINE} 天 × {len(SYMBOLS)} 交易对 × {len(INTERVALS)} 周期")
print(f"  Mark:    {DAYS_KLINE} 天 × ETHUSDT × {len(MARK_INTERVALS)} 周期")
print(f"  Funding: {DAYS_FUNDING} 天 × {len(SYMBOLS)} 交易对")
print(f"  OI:      {DAYS_OI} 天 × {len(SYMBOLS)} 交易对 × {len(OI_INTERVALS)} 周期")
print("=" * 70)

# ── 1. 主 K 线 ────────────────────────────────────────────────────────
if not SKIP_KLINES and not ONLY_FUNDING and not ONLY_OI:
    for symbol in SYMBOLS:
        print(f"\n[K线 - {symbol}]")
        for interval in INTERVALS:
            print(f"  → {interval} ...", end=" ", flush=True)
            try:
                df = fetch_binance_klines(symbol, interval, days=DAYS_KLINE, force_api=True)
                report(symbol, interval, df, kind="kline")
                n = len(df) if df is not None else 0
                print(f"{n:,}条")
            except Exception as e:
                errors.append(f"[ERROR] {symbol} kline {interval}: {e}")
                print(f"失败: {e}")

    # ── 2. Mark Price K 线 ────────────────────────────────────────────
    print(f"\n[Mark Price K线 - ETHUSDT]")
    for interval in MARK_INTERVALS:
        print(f"  → {interval} ...", end=" ", flush=True)
        try:
            df = fetch_mark_price_klines("ETHUSDT", interval, days=DAYS_KLINE, force_api=True)
            if df is not None and len(df) > 0:
                report("ETHUSDT", interval, df, kind="mark")
                print(f"{len(df):,}条")
            else:
                print("无数据")
        except Exception as e:
            errors.append(f"[ERROR] ETHUSDT mark {interval}: {e}")
            print(f"失败: {e}")

# ── 3. Funding Rate 历史 ──────────────────────────────────────────────
if not ONLY_OI:
    for symbol in SYMBOLS:
        print(f"\n[Funding Rate - {symbol}]")
        print(f"  → 拉取 {DAYS_FUNDING} 天 ...", end=" ", flush=True)
        try:
            df = fetch_funding_rate_history(symbol, days=DAYS_FUNDING, force_api=True)
            if df is not None and len(df) > 0:
                report(symbol, "8h", df, kind="funding")
                print(f"{len(df):,}条")
            else:
                print("无数据")
        except Exception as e:
            errors.append(f"[ERROR] {symbol} funding: {e}")
            print(f"失败: {e}")

# ── 4. Open Interest 历史 ─────────────────────────────────────────────
if not ONLY_FUNDING:
    for symbol in SYMBOLS:
        print(f"\n[Open Interest - {symbol}]")
        for interval in OI_INTERVALS:
            print(f"  → {interval} ({DAYS_OI}天) ...", end=" ", flush=True)
            try:
                df = fetch_open_interest_history(symbol, interval=interval,
                                                 days=DAYS_OI, force_api=True)
                if df is not None and len(df) > 0:
                    report(symbol, interval, df, kind="oi")
                    print(f"{len(df):,}条")
                else:
                    print("无数据")
            except Exception as e:
                errors.append(f"[ERROR] {symbol} OI {interval}: {e}")
                print(f"失败: {e}")

# ── 汇总 ───────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"完成！总耗时 {elapsed:.0f}s ({elapsed/60:.1f}分钟)")

for kind, label in [("kline", "K线"), ("mark", "Mark Price"),
                     ("funding", "Funding Rate"), ("oi", "Open Interest")]:
    if results[kind]:
        print(f"\n{label}:")
        for r in results[kind]:
            print(r)

if errors:
    print(f"\n失败 ({len(errors)}):")
    for e in errors:
        print(f"  {e}")

# ── 磁盘占用统计 ──────────────────────────────────────────────────────
total_kb = 0
for root, dirs, files in os.walk('data'):
    for f in files:
        if f.endswith('.parquet'):
            total_kb += os.path.getsize(os.path.join(root, f)) // 1024
print(f"\nParquet 总占用: {total_kb / 1024:.1f} MB")

# ── 打包提示 ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("下一步: 打包传输到 H800")
print("  ./pack_for_h800.sh")
print("  # 或手动:")
print("  tar -czf macd_train_data.tar.gz data/ *.py requirements*.txt")
print("  scp -J jumphost macd_train_data.tar.gz h800:/path/to/work/")
