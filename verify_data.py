"""
验证 H800 上的训练数据完整性。
在数据传输到 H800 后运行，确认所有训练所需文件齐全、格式正确、时间范围覆盖。

运行: python3 verify_data.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── 期望的数据清单 ──
REQUIRED = {
    "klines": {
        "symbols": ["ETHUSDT"],
        "intervals": ["15m", "1h", "4h", "24h"],
        "min_bars": 5000,
        "columns": ["open", "high", "low", "close", "volume"],
    },
    "mark_klines": {
        "symbols": ["ETHUSDT"],
        "intervals": ["15m", "1h"],
        "min_bars": 1000,
        "columns": ["mark_open", "mark_high", "mark_low", "mark_close"],
    },
}

OPTIONAL = {
    "klines_extra": {
        "symbols": ["BTCUSDT", "SOLUSDT", "BNBUSDT"],
        "intervals": ["15m", "1h", "4h", "24h"],
    },
    "funding_rates": {
        "symbols": ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"],
        "min_records": 500,
    },
    "open_interest": {
        "symbols": ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"],
        "intervals": ["1h", "4h"],
        "min_records": 500,
    },
}


def check_parquet(path, min_rows=100, required_cols=None):
    """读取 parquet 并检查基本健康度"""
    if not os.path.exists(path):
        return None, "文件不存在"
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return None, f"读取失败: {e}"

    if len(df) < min_rows:
        return df, f"数据不足: {len(df)} 条 (需 ≥{min_rows})"

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return df, f"缺少列: {missing}"

    nan_pct = df.isna().mean().max()
    if nan_pct > 0.1:
        worst_col = df.isna().mean().idxmax()
        return df, f"NaN 过多: {worst_col} = {nan_pct:.1%}"

    return df, None


def main():
    print("=" * 70)
    print(" 训练数据完整性验证")
    print(f" 数据目录: {DATA_DIR}")
    print("=" * 70)

    ok_count = 0
    warn_count = 0
    fail_count = 0
    total_size_kb = 0

    # ── 必需数据 ──
    print("\n[必需数据]")

    for symbol in REQUIRED["klines"]["symbols"]:
        for interval in REQUIRED["klines"]["intervals"]:
            path = os.path.join(DATA_DIR, 'klines', symbol, f'{interval}.parquet')
            df, err = check_parquet(
                path,
                min_rows=REQUIRED["klines"]["min_bars"],
                required_cols=REQUIRED["klines"]["columns"]
            )
            if err:
                print(f"  ✗ K线 {symbol}/{interval}: {err}")
                fail_count += 1
            else:
                size_kb = os.path.getsize(path) // 1024
                total_size_kb += size_kb
                span_days = (df.index[-1] - df.index[0]).days
                print(f"  ✓ K线 {symbol}/{interval}: {len(df):,} 条, "
                      f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} "
                      f"({span_days} 天, {size_kb}KB)")
                ok_count += 1

    for symbol in REQUIRED["mark_klines"]["symbols"]:
        for interval in REQUIRED["mark_klines"]["intervals"]:
            path = os.path.join(DATA_DIR, 'mark_klines', symbol, f'{interval}.parquet')
            df, err = check_parquet(
                path,
                min_rows=REQUIRED["mark_klines"]["min_bars"],
                required_cols=REQUIRED["mark_klines"]["columns"]
            )
            if err:
                print(f"  ✗ Mark {symbol}/{interval}: {err}")
                fail_count += 1
            else:
                size_kb = os.path.getsize(path) // 1024
                total_size_kb += size_kb
                print(f"  ✓ Mark {symbol}/{interval}: {len(df):,} 条, "
                      f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} "
                      f"({size_kb}KB)")
                ok_count += 1

    # ── 可选数据 ──
    print("\n[可选数据]")

    for symbol in OPTIONAL["klines_extra"]["symbols"]:
        for interval in OPTIONAL["klines_extra"]["intervals"]:
            path = os.path.join(DATA_DIR, 'klines', symbol, f'{interval}.parquet')
            df, err = check_parquet(path, min_rows=1000)
            if err:
                print(f"  ○ K线 {symbol}/{interval}: {err}")
                warn_count += 1
            else:
                size_kb = os.path.getsize(path) // 1024
                total_size_kb += size_kb
                print(f"  ✓ K线 {symbol}/{interval}: {len(df):,} 条")
                ok_count += 1

    for symbol in OPTIONAL["funding_rates"]["symbols"]:
        path = os.path.join(DATA_DIR, 'funding_rates', f'{symbol}_funding.parquet')
        df, err = check_parquet(path, min_rows=OPTIONAL["funding_rates"]["min_records"])
        if err:
            print(f"  ○ Funding {symbol}: {err}")
            warn_count += 1
        else:
            size_kb = os.path.getsize(path) // 1024
            total_size_kb += size_kb
            print(f"  ✓ Funding {symbol}: {len(df):,} 条, "
                  f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
            ok_count += 1

    for symbol in OPTIONAL["open_interest"]["symbols"]:
        for interval in OPTIONAL["open_interest"]["intervals"]:
            path = os.path.join(DATA_DIR, 'open_interest', symbol, f'{interval}.parquet')
            df, err = check_parquet(path, min_rows=OPTIONAL["open_interest"]["min_records"])
            if err:
                print(f"  ○ OI {symbol}/{interval}: {err}")
                warn_count += 1
            else:
                size_kb = os.path.getsize(path) // 1024
                total_size_kb += size_kb
                print(f"  ✓ OI {symbol}/{interval}: {len(df):,} 条")
                ok_count += 1

    # ── GPU 环境 (可选检测) ──
    print("\n[GPU 环境]")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
            print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB, "
                  f"BF16: {torch.cuda.is_bf16_supported()}")
        else:
            print(f"  ○ PyTorch {torch.__version__}, 无 GPU (CPU 模式)")
    except ImportError:
        print("  ○ PyTorch 未安装")

    try:
        import lightgbm as lgb
        print(f"  ✓ LightGBM {lgb.__version__}")
    except ImportError:
        print("  ○ LightGBM 未安装")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print(f" 汇总: ✓ {ok_count} 通过  ○ {warn_count} 可选缺失  ✗ {fail_count} 必需缺失")
    print(f" 数据总大小: {total_size_kb / 1024:.1f} MB")

    if fail_count > 0:
        print("\n ✗ 必需数据缺失！请在本机运行 fetch_5year_data.py 后重新打包传输。")
        return 1
    elif warn_count > 0:
        print("\n ○ 可选数据部分缺失，基础训练可正常进行。")
        print("   如需完整训练，在本机补充后重新打包。")
        return 0
    else:
        print("\n ✓ 所有数据就绪，可以开始训练！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
