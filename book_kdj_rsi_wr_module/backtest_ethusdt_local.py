from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODULE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kline_store import load_klines
from strategy_futures import FuturesEngine

try:
    from book_kdj_rsi_wr_module.indicators import TripleSwordConfig, add_triple_sword_features
except ImportError:
    from indicators import TripleSwordConfig, add_triple_sword_features


@dataclass(slots=True)
class StrategyParams:
    symbol: str = "ETHUSDT"
    interval: str = "1h"
    start: str = "2025-01-01"
    end: str = "2026-01-31"
    min_confluence: int = 2
    risk_frac_score2: float = 0.22
    risk_frac_score3: float = 0.35
    max_hold_bars: int = 96
    stop_loss_margin_pct: float = 0.25
    take_profit_margin_pct: float = 0.55
    trail_trigger_margin_pct: float = 0.35
    trail_pullback_margin_pct: float = 0.15
    max_leverage: int = 3
    initial_usdt: float = 100_000.0


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=89, adjust=False).mean()
    out["ret_1"] = out["close"].pct_change()

    close_prev = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - close_prev).abs(),
            (out["low"] - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=2).mean()
    out["atr_pct"] = (out["atr14"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _load_local_klines(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    df = load_klines(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        with_indicators=False,
        allow_api_fallback=False,
    )
    if df is None or len(df) < 300:
        raise RuntimeError(
            f"本地K线不足: symbol={symbol}, interval={interval}, "
            "请先检查 data/klines 下是否有对应 parquet 文件。"
        )
    return df


def _margin_and_leverage(
    eng: FuturesEngine,
    score: int,
    params: StrategyParams,
) -> tuple[float, int]:
    avail = eng.available_margin()
    if score >= 3:
        return avail * params.risk_frac_score3, min(params.max_leverage, 3)
    return avail * params.risk_frac_score2, min(params.max_leverage, 2)


def _run_strategy(df: pd.DataFrame, params: StrategyParams) -> dict:
    trade_start = pd.Timestamp(params.start)
    trade_end = pd.Timestamp(params.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # 纯合约资金账户，避免现货头寸干扰
    eng = FuturesEngine(
        name=f"TripleSword ETHUSDT Local ({params.interval})",
        initial_usdt=params.initial_usdt,
        initial_eth_value=0,
        max_leverage=params.max_leverage,
        use_spot=False,
    )

    long_entry_idx = None
    short_entry_idx = None
    long_peak_margin_pnl = -1e9
    short_peak_margin_pnl = -1e9
    cooldown = 0

    # warmup + 交易窗口
    start_idx = 120
    for idx in range(start_idx, len(df)):
        dt = df.index[idx]
        price = float(df["close"].iloc[idx])

        if dt < trade_start or dt > trade_end:
            continue

        row = df.iloc[idx]
        buy_score = int(row["buy_score"])
        sell_score = int(row["sell_score"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        atr_pct = float(row["atr_pct"])

        bull_trend = price > ema_slow and ema_fast > ema_slow
        bear_trend = price < ema_slow and ema_fast < ema_slow
        can_long = buy_score >= params.min_confluence and bull_trend
        can_short = sell_score >= params.min_confluence and bear_trend

        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        # 持仓管理 - 多头
        if eng.futures_long:
            long_pos = eng.futures_long
            margin_pnl = long_pos.calc_pnl(price) / max(long_pos.margin, 1e-9)
            long_peak_margin_pnl = max(long_peak_margin_pnl, margin_pnl)
            hold_bars = idx - int(long_entry_idx or idx)

            dynamic_stop = max(params.stop_loss_margin_pct, atr_pct * 4.0)
            should_stop = margin_pnl <= -dynamic_stop
            should_take = margin_pnl >= params.take_profit_margin_pct
            should_trail = (
                long_peak_margin_pnl >= params.trail_trigger_margin_pct
                and (long_peak_margin_pnl - margin_pnl) >= params.trail_pullback_margin_pct
            )
            should_timeout = hold_bars >= params.max_hold_bars
            should_flip = can_short and sell_score >= params.min_confluence

            if should_stop:
                eng.close_long(price, dt, f"LONG止损 marginPnL={margin_pnl:+.2f}")
                cooldown = 3
            elif should_take:
                eng.close_long(price, dt, f"LONG止盈 marginPnL={margin_pnl:+.2f}")
                cooldown = 2
            elif should_trail:
                eng.close_long(
                    price,
                    dt,
                    f"LONG回撤止盈 peak={long_peak_margin_pnl:+.2f}, now={margin_pnl:+.2f}",
                )
                cooldown = 2
            elif should_timeout:
                eng.close_long(price, dt, f"LONG超时 {hold_bars} bars")
                cooldown = 1
            elif should_flip:
                eng.close_long(price, dt, f"LONG反向平仓 sell_score={sell_score}")
                cooldown = 1

            if not eng.futures_long:
                long_entry_idx = None
                long_peak_margin_pnl = -1e9

        # 持仓管理 - 空头
        if eng.futures_short:
            short_pos = eng.futures_short
            margin_pnl = short_pos.calc_pnl(price) / max(short_pos.margin, 1e-9)
            short_peak_margin_pnl = max(short_peak_margin_pnl, margin_pnl)
            hold_bars = idx - int(short_entry_idx or idx)

            dynamic_stop = max(params.stop_loss_margin_pct, atr_pct * 4.0)
            should_stop = margin_pnl <= -dynamic_stop
            should_take = margin_pnl >= params.take_profit_margin_pct
            should_trail = (
                short_peak_margin_pnl >= params.trail_trigger_margin_pct
                and (short_peak_margin_pnl - margin_pnl) >= params.trail_pullback_margin_pct
            )
            should_timeout = hold_bars >= params.max_hold_bars
            should_flip = can_long and buy_score >= params.min_confluence

            if should_stop:
                eng.close_short(price, dt, f"SHORT止损 marginPnL={margin_pnl:+.2f}")
                cooldown = 3
            elif should_take:
                eng.close_short(price, dt, f"SHORT止盈 marginPnL={margin_pnl:+.2f}")
                cooldown = 2
            elif should_trail:
                eng.close_short(
                    price,
                    dt,
                    f"SHORT回撤止盈 peak={short_peak_margin_pnl:+.2f}, now={margin_pnl:+.2f}",
                )
                cooldown = 2
            elif should_timeout:
                eng.close_short(price, dt, f"SHORT超时 {hold_bars} bars")
                cooldown = 1
            elif should_flip:
                eng.close_short(price, dt, f"SHORT反向平仓 buy_score={buy_score}")
                cooldown = 1

            if not eng.futures_short:
                short_entry_idx = None
                short_peak_margin_pnl = -1e9

        # 开仓逻辑
        if cooldown == 0:
            if can_long and not eng.futures_long and not eng.futures_short:
                margin, lev = _margin_and_leverage(eng, buy_score, params)
                if margin >= 200:
                    eng.open_long(
                        price,
                        dt,
                        margin=margin,
                        leverage=lev,
                        reason=f"LONG入场 buy_score={buy_score}, trend=bull",
                    )
                    long_entry_idx = idx
                    long_peak_margin_pnl = 0.0
                    cooldown = 2
            elif can_short and not eng.futures_short and not eng.futures_long:
                margin, lev = _margin_and_leverage(eng, sell_score, params)
                if margin >= 200:
                    eng.open_short(
                        price,
                        dt,
                        margin=margin,
                        leverage=lev,
                        reason=f"SHORT入场 sell_score={sell_score}, trend=bear",
                    )
                    short_entry_idx = idx
                    short_peak_margin_pnl = 0.0
                    cooldown = 2

        eng.record_history(dt, price)

    # 收盘前平仓
    if len(df) > 0:
        final_dt = min(df.index[-1], trade_end)
        final_price = float(df.loc[df.index <= final_dt, "close"].iloc[-1])
    else:
        raise RuntimeError("empty dataframe")

    if eng.futures_long:
        eng.close_long(final_price, final_dt, "回测结束平多")
    if eng.futures_short:
        eng.close_short(final_price, final_dt, "回测结束平空")

    trade_df = df[(df.index >= trade_start) & (df.index <= trade_end)].copy()
    result = eng.get_result(trade_df)

    # 覆盖基准: 使用全资金买入ETH并持有
    if len(trade_df) > 1:
        first_price = float(trade_df["close"].iloc[0])
        last_price = float(trade_df["close"].iloc[-1])
        bh_return = (last_price / first_price - 1.0) * 100.0
        result["buy_hold_return"] = round(bh_return, 2)
        result["alpha"] = round(result["strategy_return"] - result["buy_hold_return"], 2)

    result["metadata"] = {
        "symbol": params.symbol,
        "interval": params.interval,
        "start": params.start,
        "end": params.end,
        "bars_total": int(len(df)),
        "bars_traded": int(len(trade_df)),
        "generated_at": datetime.now().isoformat(),
        "params": asdict(params),
        "local_data_only": True,
    }
    return result


def _export(result: dict, df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = out_dir / f"ethusdt_triple_sword_local_{ts}"

    # 结果主文件
    result_path = Path(str(prefix) + "_result.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 交易记录
    trades = result.get("trades", [])
    trades_df = pd.DataFrame(trades)
    trades_path = Path(str(prefix) + "_trades.csv")
    if len(trades_df) > 0:
        trades_df.to_csv(trades_path, index=False)
    else:
        trades_path.write_text("", encoding="utf-8")

    # 信号快照
    signal_cols = [
        "open",
        "high",
        "low",
        "close",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "rsi",
        "wr",
        "buy_score",
        "sell_score",
        "triple_sword_decision",
        "ema_fast",
        "ema_slow",
        "atr_pct",
    ]
    available_cols = [c for c in signal_cols if c in df.columns]
    signal_df = df[available_cols].copy().reset_index(names="date")
    signals_path = Path(str(prefix) + "_signals.csv")
    signal_df.to_csv(signals_path, index=False)

    return {
        "result": str(result_path),
        "trades": str(trades_path),
        "signals": str(signals_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH/USDT 本地K线三剑客独立回测")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "data", "backtests"))
    parser.add_argument("--initial-usdt", type=float, default=100_000.0)
    parser.add_argument("--max-leverage", type=int, default=3)
    parser.add_argument("--min-confluence", type=int, default=2)
    parser.add_argument("--risk-frac-score2", type=float, default=0.22)
    parser.add_argument("--risk-frac-score3", type=float, default=0.35)
    parser.add_argument("--max-hold-bars", type=int, default=96)
    parser.add_argument("--stop-loss-margin-pct", type=float, default=0.25)
    parser.add_argument("--take-profit-margin-pct", type=float, default=0.55)
    parser.add_argument("--trail-trigger-margin-pct", type=float, default=0.35)
    parser.add_argument("--trail-pullback-margin-pct", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = StrategyParams(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        min_confluence=args.min_confluence,
        risk_frac_score2=args.risk_frac_score2,
        risk_frac_score3=args.risk_frac_score3,
        max_hold_bars=args.max_hold_bars,
        stop_loss_margin_pct=args.stop_loss_margin_pct,
        take_profit_margin_pct=args.take_profit_margin_pct,
        trail_trigger_margin_pct=args.trail_trigger_margin_pct,
        trail_pullback_margin_pct=args.trail_pullback_margin_pct,
        max_leverage=args.max_leverage,
        initial_usdt=args.initial_usdt,
    )

    print(
        f"[backtest] start symbol={params.symbol} interval={params.interval} "
        f"window={params.start}~{params.end} (local-only)"
    )
    df = _load_local_klines(
        symbol=params.symbol,
        interval=params.interval,
        start=params.start,
        end=params.end,
    )
    print(f"[backtest] loaded bars={len(df)}")

    feature_cfg = TripleSwordConfig(min_confluence=params.min_confluence)
    df = add_triple_sword_features(df, feature_cfg)
    df = _add_context_features(df)

    result = _run_strategy(df, params)
    paths = _export(result, df, Path(args.out_dir))

    print("[backtest] done")
    print(
        f"[backtest] return={result['strategy_return']:+.2f}% "
        f"bh={result['buy_hold_return']:+.2f}% alpha={result['alpha']:+.2f}% "
        f"dd={result['max_drawdown']:.2f}% trades={result['total_trades']}"
    )
    print(f"[backtest] result: {paths['result']}")
    print(f"[backtest] trades: {paths['trades']}")
    print(f"[backtest] signals: {paths['signals']}")


if __name__ == "__main__":
    main()
