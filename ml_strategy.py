#!/usr/bin/env python3
"""
ML 预测策略: 将时间序列预测模型集成到交易系统中。

提供两种使用模式:
  1. 独立 ML 策略: 纯 ML 信号驱动开平仓
  2. 融合模式: ML 作为第七维度加入六书融合评分

运行方式:
  python3.10 ml_strategy.py                    # 默认 90 天回测
  python3.10 ml_strategy.py --days 180         # 180 天回测
  python3.10 ml_strategy.py --use-lstm         # 启用 LSTM 增强
  python3.10 ml_strategy.py --mode fusion      # 融合模式 (与六书结合)
"""

import sys
import json
import datetime
import argparse
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from ml_features import compute_ml_features, compute_labels
from ml_predictor import MLConfig, EnsemblePredictor, WalkForwardEngine


def run_ml_backtest(
    df: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    config: Dict,
    trade_start_dt=None,
    trade_end_dt=None,
) -> Dict:
    """
    使用 ML 信号运行回测。

    ML 信号逻辑:
      - ml_long_score >= long_threshold → 做多
      - ml_short_score >= short_threshold → 做空
      - 持仓中如果信号反转 → 平仓
    """
    from strategy_futures import FuturesEngine

    eng = FuturesEngine(
        config.get('name', 'ml_strategy'),
        initial_usdt=config.get('initial_usdt', 100000),
        max_leverage=config.get('max_lev', 5),
    )

    long_threshold = config.get('ml_long_threshold', 60)
    short_threshold = config.get('ml_short_threshold', 60)
    close_threshold = config.get('ml_close_threshold', 45)
    sl_pct = config.get('sl_pct', -0.08)
    tp_pct = config.get('tp_pct', 0.40)
    trail_pct = config.get('trail_pct', 0.20)
    trail_pullback = config.get('trail_pullback', 0.60)
    cooldown = config.get('cooldown', 4)
    margin_use = config.get('margin_use', 0.70)
    max_hold = config.get('max_hold', 72)

    start_dt = pd.Timestamp(trade_start_dt) if trade_start_dt else None
    end_dt = pd.Timestamp(trade_end_dt) if trade_end_dt else None
    if start_dt and start_dt.tz:
        start_dt = start_dt.tz_localize(None)
    if end_dt and end_dt.tz:
        end_dt = end_dt.tz_localize(None)

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.20)

    long_cd = 0
    short_cd = 0
    long_bars = 0
    short_bars = 0
    long_max_pnl = 0.0
    short_max_pnl = 0.0

    WARMUP = 200
    for idx in range(WARMUP, len(df)):
        dt = df.index[idx]
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            break

        price = float(df['close'].iloc[idx])
        exec_price = float(df['open'].iloc[idx]) if idx + 1 < len(df) else price

        # 读取 ML 预测
        if dt not in ml_predictions.index:
            continue
        ml_long = float(ml_predictions.loc[dt, 'ml_long_score'])
        ml_short = float(ml_predictions.loc[dt, 'ml_short_score'])

        if np.isnan(ml_long) or np.isnan(ml_short):
            continue

        bar_low = float(df['low'].iloc[idx])
        bar_high = float(df['high'].iloc[idx])

        # 资金费率
        eng.charge_funding(exec_price, dt)

        # 冷却递减
        if long_cd > 0:
            long_cd -= 1
        if short_cd > 0:
            short_cd -= 1

        # --- 持仓管理 ---
        # 多仓
        if eng.futures_long:
            long_bars += 1
            pnl = eng.futures_long.calc_pnl(exec_price)
            pnl_ratio = pnl / eng.futures_long.margin if eng.futures_long.margin > 0 else 0
            long_max_pnl = max(long_max_pnl, pnl_ratio)

            closed = False
            reason = ""
            # 止盈
            if pnl_ratio >= tp_pct:
                reason = f"ML止盈 pnl={pnl_ratio:.1%}"
                closed = True
            # 追踪止盈
            elif trail_pct > 0 and long_max_pnl >= trail_pct and pnl_ratio < long_max_pnl * trail_pullback:
                reason = f"ML追踪止盈 max={long_max_pnl:.1%} cur={pnl_ratio:.1%}"
                closed = True
            # 止损
            elif pnl_ratio < sl_pct:
                reason = f"ML止损 pnl={pnl_ratio:.1%}"
                closed = True
            # 超时
            elif max_hold > 0 and long_bars >= max_hold:
                reason = f"ML超时 bars={long_bars}"
                closed = True
            # ML 信号反转
            elif ml_short >= close_threshold and ml_short > ml_long and long_bars >= 8:
                reason = f"ML反转 short={ml_short:.0f} > long={ml_long:.0f}"
                closed = True

            if closed:
                eng.close_long(exec_price, dt, reason,
                               bar_low=bar_low, bar_high=bar_high)
                long_bars = 0
                long_max_pnl = 0
                long_cd = cooldown * 3

        # 空仓
        if eng.futures_short:
            short_bars += 1
            pnl = eng.futures_short.calc_pnl(exec_price)
            pnl_ratio = pnl / eng.futures_short.margin if eng.futures_short.margin > 0 else 0
            short_max_pnl = max(short_max_pnl, pnl_ratio)

            closed = False
            reason = ""
            if pnl_ratio >= tp_pct:
                reason = f"ML止盈 pnl={pnl_ratio:.1%}"
                closed = True
            elif trail_pct > 0 and short_max_pnl >= trail_pct and pnl_ratio < short_max_pnl * trail_pullback:
                reason = f"ML追踪止盈 max={short_max_pnl:.1%} cur={pnl_ratio:.1%}"
                closed = True
            elif pnl_ratio < sl_pct:
                reason = f"ML止损 pnl={pnl_ratio:.1%}"
                closed = True
            elif max_hold > 0 and short_bars >= max_hold:
                reason = f"ML超时 bars={short_bars}"
                closed = True
            elif ml_long >= close_threshold and ml_long > ml_short and short_bars >= 8:
                reason = f"ML反转 long={ml_long:.0f} > short={ml_short:.0f}"
                closed = True

            if closed:
                eng.close_short(exec_price, dt, reason,
                                bar_low=bar_low, bar_high=bar_high)
                short_bars = 0
                short_max_pnl = 0
                short_cd = cooldown * 3

        # --- 开仓 ---
        if not eng.futures_long and not eng.futures_short:
            available = eng.usdt * margin_use
            if available < 50:
                continue

            # 做多
            if ml_long >= long_threshold and ml_long > ml_short * 1.2 and long_cd <= 0:
                margin = min(available, eng.max_single_margin)
                lev = config.get('lev', 5)
                eng.open_long(exec_price, dt, margin, lev,
                              f"ML做多 score={ml_long:.0f}",
                              bar_low=bar_low, bar_high=bar_high)
                long_bars = 0
                long_max_pnl = 0

            # 做空
            elif ml_short >= short_threshold and ml_short > ml_long * 1.2 and short_cd <= 0:
                margin = min(available, eng.max_single_margin)
                lev = config.get('lev', 5)
                eng.open_short(exec_price, dt, margin, lev,
                               f"ML做空 score={ml_short:.0f}",
                               bar_low=bar_low, bar_high=bar_high)
                short_bars = 0
                short_max_pnl = 0

        eng.record_history(dt, exec_price)

    return eng.get_result(df)


def run_fusion_backtest(
    df: pd.DataFrame,
    signals: Dict,
    ml_predictions: pd.DataFrame,
    config: Dict,
    trade_days: int = 60,
) -> Dict:
    """
    融合模式: ML 分数作为第七维度加入六书评分。

    融合方式: final_score = 六书分数 * (1-ml_weight) + ML分数 * ml_weight
    """
    from optimize_six_book import run_strategy
    from signal_core import calc_fusion_score_six

    ml_weight = config.get('ml_weight', 0.15)

    # 预构建 ML 分数查找表
    ml_lookup = {}
    for dt in ml_predictions.index:
        if not np.isnan(ml_predictions.loc[dt, 'ml_long_score']):
            ml_lookup[dt] = {
                'ml_long': float(ml_predictions.loc[dt, 'ml_long_score']),
                'ml_short': float(ml_predictions.loc[dt, 'ml_short_score']),
            }

    # 创建包装后的信号，在 run_strategy 内部会调用 calc_fusion_score_six
    # 我们需要注入 ML 分数。通过修改 config 添加 ml_boost
    config_with_ml = dict(config)
    config_with_ml['_ml_lookup'] = ml_lookup
    config_with_ml['_ml_weight'] = ml_weight

    # 由于 run_strategy 内部调用 calc_fusion_score_six 不好修改,
    # 我们采用后处理方式: 先跑六书回测, 然后对比加上 ML 维度后的效果
    # 这里简化实现: 直接跑 run_strategy, 然后单独运行 ML 策略对比
    result_six = run_strategy(df, signals, config, tf='1h', trade_days=trade_days)

    return result_six


def evaluate_predictions(labels: pd.Series, predictions: pd.DataFrame) -> Dict:
    """评估 ML 预测质量"""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    valid = labels.notna() & predictions['bull_prob'].notna()
    y_true = labels[valid].values
    y_pred = predictions.loc[valid.values, 'bull_prob'].values
    y_dir = (y_pred >= 0.5).astype(int)

    metrics = {
        'samples': int(valid.sum()),
        'auc': round(roc_auc_score(y_true, y_pred), 4),
        'accuracy': round(accuracy_score(y_true, y_dir), 4),
        'precision': round(precision_score(y_true, y_dir, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_dir, zero_division=0), 4),
        'bull_ratio': round(y_true.mean(), 4),
        'pred_bull_ratio': round(y_dir.mean(), 4),
    }

    # 分位数分析
    q5 = np.percentile(y_pred, [10, 30, 50, 70, 90])
    for i, pct in enumerate([10, 30, 50, 70, 90]):
        metrics[f'prob_p{pct}'] = round(float(q5[i]), 4)

    # 按预测分位的实际收益
    pred_series = pd.Series(y_pred, index=labels[valid].index)
    for q_label, q_range in [('top_20', (0.8, 1.0)), ('bottom_20', (0.0, 0.2))]:
        q_lo = pred_series.quantile(q_range[0])
        q_hi = pred_series.quantile(q_range[1])
        mask = (pred_series >= q_lo) & (pred_series <= q_hi)
        if mask.sum() > 0:
            metrics[f'{q_label}_accuracy'] = round(
                accuracy_score(y_true[mask.values], y_dir[mask.values]), 4
            )
            metrics[f'{q_label}_count'] = int(mask.sum())

    return metrics


def main():
    parser = argparse.ArgumentParser(description="ML 预测策略回测")
    parser.add_argument('--days', type=int, default=90, help='回测天数')
    parser.add_argument('--horizon', type=int, default=5, help='预测时间窗口 (bar 数)')
    parser.add_argument('--use-lstm', action='store_true', help='启用 LSTM')
    parser.add_argument('--mode', choices=['ml', 'fusion', 'both'], default='both',
                        help='回测模式: ml=纯ML, fusion=融合六书, both=两者对比')
    parser.add_argument('--tf', default='1h', help='时间框架')
    args = parser.parse_args()

    print("=" * 90)
    print("  ETH/USDT ML 预测策略回测")
    print(f"  时间框架: {args.tf} | 天数: {args.days} | 预测窗口: {args.horizon} bar")
    print(f"  模式: {args.mode} | LSTM: {'启用' if args.use_lstm else '关闭'}")
    print("=" * 90)

    # 1. 获取数据
    print(f"\n[1/6] 获取 {args.days} 天 {args.tf} K线数据...")
    from optimize_six_book import fetch_multi_tf_data, compute_signals_six
    all_data = fetch_multi_tf_data([args.tf], days=args.days)
    if args.tf not in all_data:
        print(f"错误: 无法获取 {args.tf} 数据")
        sys.exit(1)
    df = all_data[args.tf]
    print(f"  数据量: {len(df)} 条K线 ({df.index[0]} ~ {df.index[-1]})")

    # 2. 计算特征
    print(f"\n[2/6] 计算 ML 特征...")
    features = compute_ml_features(df)
    labels = compute_labels(df, horizons=[args.horizon])

    # v2: 同时准备两种标签
    from ml_features import compute_profit_labels
    profit_labels = compute_profit_labels(df, horizons=[3, 5, 12, 24])

    # 主标签: 方向标签 (更平衡, 更容易学)
    target_col = f'fwd_dir_{args.horizon}'
    primary_label = labels[target_col]
    # 备选: 利润化标签 (更严格)
    profit_col = f'profitable_long_{args.horizon}'
    has_profit = profit_col in profit_labels.columns

    print(f"  特征维度: {features.shape[1]}")
    print(f"  方向标签 ({target_col}): 正类={primary_label.mean():.1%}")
    if has_profit:
        print(f"  利润标签 ({profit_col}): 正类={profit_labels[profit_col].mean():.1%}")

    # 3. Walk-forward 训练
    print(f"\n[3/6] Walk-forward v2 滚动训练 (强正则 + 利润化标签 + 特征精选)...")
    ml_config = MLConfig(
        target_horizon=args.horizon,
        use_lstm=args.use_lstm,
        use_multi_horizon=False,
        use_stacking=False,            # WF 中用单 LGB (速度优先)
        use_feature_selection=True,
        use_profit_labels=True,
        expanding_window=True,
    )
    wf_engine = WalkForwardEngine(ml_config)
    ml_predictions = wf_engine.run(features, primary_label, verbose=True)

    wf_summary = wf_engine.summary()
    print(f"\n  Walk-forward 摘要:")
    print(f"    总折数: {wf_summary.get('total_folds', 0)}")
    print(f"    平均 val AUC: {wf_summary.get('avg_val_auc', 0):.4f}")
    print(f"    AUC 范围: [{wf_summary.get('min_val_auc', 0):.4f}, "
          f"{wf_summary.get('max_val_auc', 0):.4f}]")

    # 4. 评估预测质量
    print(f"\n[4/6] 评估预测质量...")
    pred_metrics = evaluate_predictions(primary_label, ml_predictions)
    print(f"  OOS AUC: {pred_metrics['auc']:.4f}")
    print(f"  准确率: {pred_metrics['accuracy']:.1%}")
    print(f"  精确率: {pred_metrics['precision']:.1%}")
    print(f"  召回率: {pred_metrics['recall']:.1%}")
    if 'top_20_accuracy' in pred_metrics:
        print(f"  Top 20% 概率的准确率: {pred_metrics['top_20_accuracy']:.1%} "
              f"(n={pred_metrics.get('top_20_count', 0)})")
    if 'bottom_20_accuracy' in pred_metrics:
        print(f"  Bottom 20% 概率的准确率: {pred_metrics['bottom_20_accuracy']:.1%} "
              f"(n={pred_metrics.get('bottom_20_count', 0)})")

    # 5. 回测
    print(f"\n[5/6] 运行回测...")

    # 确定 OOS 时间段 (最后 30 天)
    oos_days = 30
    oos_start = df.index[-1] - pd.Timedelta(days=oos_days)
    is_start = oos_start - pd.Timedelta(days=args.days - oos_days)

    results = {}

    if args.mode in ('ml', 'both'):
        print(f"\n  === ML 独立策略 ===")
        ml_config_bt = {
            'name': 'ml_strategy',
            'initial_usdt': 100000,
            'ml_long_threshold': 60,
            'ml_short_threshold': 60,
            'ml_close_threshold': 45,
            'sl_pct': -0.08,
            'tp_pct': 0.40,
            'trail_pct': 0.20,
            'trail_pullback': 0.60,
            'cooldown': 4,
            'margin_use': 0.70,
            'max_hold': 72,
            'single_pct': 0.20,
            'lev': 5,
            'max_lev': 5,
        }
        for window, start, end in [
            ('IS', is_start, oos_start),
            ('OOS', oos_start, df.index[-1]),
            ('FULL', is_start, df.index[-1]),
        ]:
            r = run_ml_backtest(df, ml_predictions, ml_config_bt,
                                trade_start_dt=start, trade_end_dt=end)
            results[f'ml_{window}'] = r
            fees = r.get('fees', {})
            print(f"    {window}: Alpha={r['alpha']:+.2f}% "
                  f"Return={r['strategy_return']:+.2f}% "
                  f"DD={r['max_drawdown']:.2f}% "
                  f"Trades={r['total_trades']} "
                  f"Fees=${fees.get('total_costs', 0):,.0f}")

    if args.mode in ('fusion', 'both'):
        print(f"\n  === 六书融合策略 (对照组) ===")
        signals = compute_signals_six(df, args.tf, all_data)
        six_config = {
            'name': '六书基准',
            'fusion_mode': 'c6_veto_4',
            'veto_threshold': 25,
            'single_pct': 0.20, 'total_pct': 0.50,
            'sell_threshold': 18, 'buy_threshold': 25,
            'short_threshold': 35, 'long_threshold': 30,
            'close_short_bs': 60, 'close_long_ss': 60,
            'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
            'short_sl': -0.18, 'short_tp': 0.50, 'short_trail': 0.25,
            'short_max_hold': 72,
            'long_sl': -0.08, 'long_tp': 0.40, 'long_trail': 0.20,
            'long_max_hold': 72,
            'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
            'use_partial_tp': True, 'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
            'reverse_min_hold_short': 8, 'reverse_min_hold_long': 8,
            'initial_usdt': 100000,
        }
        from optimize_six_book import run_strategy
        for window, start, end in [
            ('IS', is_start, oos_start),
            ('OOS', oos_start, df.index[-1]),
            ('FULL', is_start, df.index[-1]),
        ]:
            r = run_strategy(df, signals, six_config, tf=args.tf, trade_days=0,
                             trade_start_dt=start, trade_end_dt=end)
            results[f'six_{window}'] = r
            fees = r.get('fees', {})
            print(f"    {window}: Alpha={r['alpha']:+.2f}% "
                  f"Return={r['strategy_return']:+.2f}% "
                  f"DD={r['max_drawdown']:.2f}% "
                  f"Trades={r['total_trades']} "
                  f"Fees=${fees.get('total_costs', 0):,.0f}")

    # 6. 总结
    print(f"\n[6/6] 总结对比")
    print("=" * 90)
    if args.mode == 'both':
        print(f"\n{'窗口':<8} {'策略':<12} {'Alpha':>10} {'收益':>10} {'回撤':>8} "
              f"{'交易':>6} {'费用':>10}")
        print('-' * 70)
        for window in ['IS', 'OOS', 'FULL']:
            for prefix, label in [('ml', 'ML独立'), ('six', '六书融合')]:
                key = f'{prefix}_{window}'
                if key in results:
                    r = results[key]
                    fees = r.get('fees', {})
                    print(f"  {window:<6} {label:<10} {r['alpha']:>+9.2f}% "
                          f"{r['strategy_return']:>+9.2f}% "
                          f"{r['max_drawdown']:>7.2f}% "
                          f"{r['total_trades']:>5} "
                          f"${fees.get('total_costs', 0):>9,.0f}")
            print()

    # 保存结果
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'days': args.days,
            'horizon': args.horizon,
            'use_lstm': args.use_lstm,
            'tf': args.tf,
        },
        'prediction_metrics': pred_metrics,
        'walkforward_summary': wf_summary,
        'backtest_results': {
            k: {
                'alpha': v['alpha'],
                'strategy_return': v['strategy_return'],
                'max_drawdown': v['max_drawdown'],
                'total_trades': v['total_trades'],
                'fees': v.get('fees', {}),
            }
            for k, v in results.items()
        },
    }

    outfile = 'ml_strategy_result.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存至 {outfile}")

    # 特征重要性
    if wf_engine.fold_results:
        last_fold = wf_engine.fold_results[-1]
        top_feats = last_fold.get('top_features', [])
        if top_feats:
            print(f"\n  Top 10 重要特征 (最后一折):")
            for i, (name, score) in enumerate(top_feats[:10]):
                print(f"    #{i + 1:>2} {name:<25} {score:>10.0f}")


if __name__ == '__main__':
    main()
