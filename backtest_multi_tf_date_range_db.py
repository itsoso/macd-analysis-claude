"""
多周期联合决策 - 固定日期区间回测（独立页面数据源，SQLite 持久化）。

目标:
1) 支持指定起止日期（如 2025-01-01 ~ 2026-01-31）
2) 输出每日持仓/盈亏轨迹 + 每日交易汇总 + 逐笔完整交易明细
3) 结果写入 SQLite，避免与现有 JSON 页面的并发冲突
"""

import argparse
import json
import os
import socket
from datetime import datetime

import pandas as pd

from binance_fetcher import fetch_binance_klines
from date_range_report import (
    TRADE_RECORD_FIELDS,
    build_daily_equity_positions,
    build_daily_trade_summary,
    normalize_trade_records,
    save_report_to_db,
)
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from optimize_six_book import (
    _build_tf_score_index,
    compute_signals_six,
    run_strategy_multi_tf,
)
from backtest_multi_tf_30d_7d import _apply_conservative_risk


def _parse_start_date(date_text):
    ts = pd.Timestamp(date_text)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _parse_end_date(date_text):
    ts = pd.Timestamp(date_text)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    # 纯日期默认扩展到当天 23:59:59
    if "T" not in str(date_text) and " " not in str(date_text):
        ts = ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def _load_choice(choice_file):
    with open(choice_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    top = (data.get("results_60d") or [None])[0]
    if not top:
        raise RuntimeError(f"未在 {choice_file} 中找到 results_60d TOP1 组合")
    return {
        "combo_name": top.get("combo_name", "多周期组合"),
        "primary_tf": top["primary_tf"],
        "decision_tfs": top["decision_tfs"],
    }


def _load_base_config(opt_file):
    with open(opt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = data.get("global_best") or {}
    cfg = dict(g.get("config") or {})
    if not cfg:
        raise RuntimeError(f"未在 {opt_file} 中找到 global_best.config")
    return cfg


def _fetch_tf_data(symbol, tf, fetch_days, warmup_start, end_dt):
    df = fetch_binance_klines(symbol, interval=tf, days=fetch_days)
    if df is None or len(df) < 120:
        return None
    df = add_all_indicators(df)
    add_moving_averages(df, timeframe=tf)
    df = df[(df.index >= warmup_start) & (df.index <= end_dt)].copy()
    if len(df) < 120:
        return None
    return df


def _to_daily_records(df):
    out = []
    for dt, row in df.iterrows():
        r = {"date": dt.date().isoformat()}
        for k, v in row.to_dict().items():
            if isinstance(v, (bool, str)) or v is None:
                r[k] = v
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = v
        out.append(r)
    return out


def run_range_backtest(args):
    start_dt = _parse_start_date(args.start_date)
    end_dt = _parse_end_date(args.end_date)
    if end_dt < start_dt:
        raise ValueError("结束日期不能早于开始日期")

    choice = _load_choice(args.choice_file)
    if args.primary_tf:
        choice["primary_tf"] = args.primary_tf
    if args.decision_tfs:
        choice["decision_tfs"] = [x.strip() for x in args.decision_tfs.split(",") if x.strip()]

    base_config = _load_base_config(args.opt_file)
    if not args.disable_conservative:
        base_config = _apply_conservative_risk(base_config)

    # 默认启用与当前实盘口径一致的门控/保护
    base_config.update(
        {
            "name": f"date_range_{start_dt.date()}_{end_dt.date()}",
            "use_live_gate": True,
            "coverage_min": 0.5,
            "consensus_min_strength": 40,
            "use_regime_aware": True,
            "use_protections": True,
            "prot_loss_streak_limit": 3,
            "prot_loss_streak_cooldown_bars": 24,
            "prot_daily_loss_limit_pct": 0.03,
            "prot_global_dd_limit_pct": 0.12,
            "prot_close_on_global_halt": True,
        }
    )

    warmup_start = start_dt - pd.Timedelta(days=args.warmup_days)
    now = pd.Timestamp.now().tz_localize(None)
    fetch_days = max(90, int((now - warmup_start).days + 5))

    needed_tfs = sorted(set([choice["primary_tf"]] + list(choice["decision_tfs"])))
    print("=" * 100)
    print("固定日期区间多周期回测 (DB)")
    print(f"区间: {start_dt} ~ {end_dt}")
    print(f"组合: {choice['combo_name']} | 主周期 {choice['primary_tf']} | 决策 {choice['decision_tfs']}")
    print(f"抓取天数: {fetch_days} 天 (含预热 {args.warmup_days} 天)")
    print("=" * 100)

    all_data = {}
    for tf in needed_tfs:
        print(f"获取 {tf} 数据...")
        df = _fetch_tf_data(args.symbol, tf, fetch_days, warmup_start, end_dt)
        if df is None:
            raise RuntimeError(f"{tf} 数据不足，无法完成回测")
        all_data[tf] = df
        print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")

    all_signals = {
        tf: compute_signals_six(all_data[tf], tf, all_data, max_bars=0)
        for tf in needed_tfs
    }
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, base_config)

    primary_df = all_data[choice["primary_tf"]]
    result = run_strategy_multi_tf(
        primary_df=primary_df,
        tf_score_map=tf_score_map,
        decision_tfs=choice["decision_tfs"],
        config=base_config,
        primary_tf=choice["primary_tf"],
        trade_days=0,  # 使用显式日期窗口
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
    )

    trades = normalize_trade_records(result.get("trades", []))
    history = result.get("history", [])
    daily_pos = build_daily_equity_positions(history, trades, start_dt, end_dt)
    daily_trade = build_daily_trade_summary(trades, start_dt, end_dt)
    daily = daily_pos.join(daily_trade, how="left").fillna(0.0)

    daily_records = _to_daily_records(daily)
    run_meta = {
        "runner": args.runner,
        "host": socket.gethostname(),
        "symbol": args.symbol,
        "combo_name": choice["combo_name"],
        "primary_tf": choice["primary_tf"],
        "decision_tfs": choice["decision_tfs"],
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "warmup_days": int(args.warmup_days),
        "generated_at": datetime.now().isoformat(),
    }
    summary = {
        "strategy_return": float(result.get("strategy_return", 0.0)),
        "buy_hold_return": float(result.get("buy_hold_return", 0.0)),
        "alpha": float(result.get("alpha", 0.0)),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
        "total_trades": int(result.get("total_trades", 0)),
        "liquidations": int(result.get("liquidations", 0)),
        "total_cost": float((result.get("fees") or {}).get("total_costs", 0.0)),
        "start_equity": float(daily_records[0]["total"]) if daily_records else 0.0,
        "end_equity": float(daily_records[-1]["total"]) if daily_records else 0.0,
        "days": len(daily_records),
    }

    payload = {
        "run_meta": run_meta,
        "summary": summary,
        "daily_records": daily_records,
        "trades": trades,
    }

    run_id = save_report_to_db(args.db_path, payload)

    export_dir = args.export_dir or os.path.join("data", "backtests", "date_range_exports")
    os.makedirs(export_dir, exist_ok=True)
    daily_csv = os.path.join(export_dir, f"date_range_daily_run_{run_id}.csv")
    trades_csv = os.path.join(export_dir, f"date_range_trades_run_{run_id}.csv")
    pd.DataFrame(daily_records).to_csv(daily_csv, index=False)
    pd.DataFrame(trades).to_csv(trades_csv, index=False)

    print(f"\n回测完成: run_id={run_id}")
    print(
        "收益: 策略={:+.2f}% 基准={:+.2f}% Alpha={:+.2f}% 回撤={:.2f}% 交易={}笔".format(
            summary["strategy_return"],
            summary["buy_hold_return"],
            summary["alpha"],
            summary["max_drawdown"],
            summary["total_trades"],
        )
    )
    print(f"DB: {os.path.abspath(args.db_path)}")
    print(f"导出: {os.path.abspath(daily_csv)}")
    print(f"导出: {os.path.abspath(trades_csv)}")

    return run_id


def build_parser():
    parser = argparse.ArgumentParser(description="固定日期区间多周期回测并写入SQLite")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="起始日期(含)")
    parser.add_argument("--end-date", type=str, default="2026-01-31", help="结束日期(含)")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="交易对")
    parser.add_argument("--runner", type=str, default="local", help="执行来源标识")
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.path.join("data", "backtests", "multi_tf_date_range_reports.db"),
        help="SQLite 数据库路径",
    )
    parser.add_argument(
        "--choice-file",
        type=str,
        default=os.path.join("data", "backtests", "backtest_multi_tf_30d_7d_result.json"),
        help="用于读取TOP组合的结果文件",
    )
    parser.add_argument(
        "--opt-file",
        type=str,
        default="optimize_six_book_result.json",
        help="用于读取基础参数配置的优化结果文件",
    )
    parser.add_argument("--primary-tf", type=str, default="", help="覆盖主周期 (可选)")
    parser.add_argument("--decision-tfs", type=str, default="", help="覆盖决策周期,逗号分隔 (可选)")
    parser.add_argument("--warmup-days", type=int, default=45, help="回测前预热天数")
    parser.add_argument("--disable-conservative", action="store_true", help="关闭保守仓位约束")
    parser.add_argument("--export-dir", type=str, default="", help="CSV导出目录")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_range_backtest(args)
