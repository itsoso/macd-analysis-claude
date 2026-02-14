from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .indicators import TripleSwordConfig, add_triple_sword_features
except ImportError:
    from indicators import TripleSwordConfig, add_triple_sword_features


@dataclass(slots=True)
class Trade:
    side: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    ret_pct: float


def generate_sample_ohlc(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = 100.0 + rng.normal(0.03, 1.0, size=rows).cumsum()
    high = base + rng.uniform(0.1, 1.5, size=rows)
    low = base - rng.uniform(0.1, 1.5, size=rows)
    open_ = base + rng.normal(0, 0.4, size=rows)
    close = base + rng.normal(0, 0.4, size=rows)
    ts = pd.date_range("2022-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": 1_000.0}
    )


def run_long_only_backtest(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0005,
) -> tuple[pd.DataFrame, dict]:
    cash = float(initial_cash)
    position = 0.0
    entry_price = 0.0
    entry_idx = -1
    trades: list[Trade] = []
    equity_curve: list[float] = []

    for i, row in df.iterrows():
        price = float(row["close"])
        signal = row["triple_sword_decision"]

        if position <= 0 and signal == "BUY":
            qty = (cash * (1.0 - fee_rate)) / price
            position = qty
            entry_price = price
            entry_idx = int(i)
            cash = 0.0
        elif position > 0 and signal == "SELL":
            cash = position * price * (1.0 - fee_rate)
            ret_pct = (price - entry_price) / entry_price * 100.0
            trades.append(
                Trade(
                    side="LONG",
                    entry_idx=entry_idx,
                    exit_idx=int(i),
                    entry_price=entry_price,
                    exit_price=price,
                    ret_pct=ret_pct,
                )
            )
            position = 0.0
            entry_price = 0.0
            entry_idx = -1

        equity = cash if position <= 0 else position * price
        equity_curve.append(equity)

    if position > 0:
        last_price = float(df["close"].iloc[-1])
        cash = position * last_price * (1.0 - fee_rate)
        ret_pct = (last_price - entry_price) / entry_price * 100.0
        trades.append(
            Trade(
                side="LONG",
                entry_idx=entry_idx,
                exit_idx=int(df.index[-1]),
                entry_price=entry_price,
                exit_price=last_price,
                ret_pct=ret_pct,
            )
        )

    wins = sum(1 for t in trades if t.ret_pct > 0)
    result = {
        "initial_cash": initial_cash,
        "final_equity": cash,
        "total_return_pct": (cash / initial_cash - 1.0) * 100.0,
        "trade_count": len(trades),
        "win_rate_pct": (wins / len(trades) * 100.0) if trades else 0.0,
    }

    out = df.copy()
    out["equity"] = equity_curve
    out.attrs["trades"] = [asdict(t) for t in trades]
    return out, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KDJ/RSI/WR 三剑客独立演示回测")
    parser.add_argument("--csv", help="行情 CSV（需有 high/low/close）")
    parser.add_argument("--out-dir", default="book_kdj_rsi_wr_module/outputs")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--generate-sample", action="store_true", help="使用随机样本数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_sample:
        df = generate_sample_ohlc()
    else:
        if not args.csv:
            raise ValueError("must provide --csv or use --generate-sample")
        df = pd.read_csv(args.csv)

    df = add_triple_sword_features(df, TripleSwordConfig())
    marked, report = run_long_only_backtest(
        df=df,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
    )

    signal_path = out_dir / "triple_sword_signals.csv"
    report_path = out_dir / "triple_sword_backtest_report.json"
    trades_path = out_dir / "triple_sword_trades.json"

    marked.to_csv(signal_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    trades_path.write_text(json.dumps(marked.attrs.get("trades", []), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[strategy] done")
    print(f"[strategy] report: {report_path}")
    print(f"[strategy] signals: {signal_path}")
    print(f"[strategy] trades: {trades_path}")
    print(f"[strategy] final_equity={report['final_equity']:.2f}, return={report['total_return_pct']:.2f}%")


if __name__ == "__main__":
    main()
