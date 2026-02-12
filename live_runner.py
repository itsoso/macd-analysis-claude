#!/usr/bin/env python3
"""
实盘交易系统启动入口
六书融合策略 ETH/USDT

用法:
  python live_runner.py --phase paper                    # Phase 1: 纸上交易
  python live_runner.py --phase testnet                  # Phase 2: 测试网实盘
  python live_runner.py --phase small_live               # Phase 3: 小资金实盘
  python live_runner.py --phase scale_up                 # Phase 4: 逐步加仓
  python live_runner.py --config live_trading_config.json # 从配置文件启动
  python live_runner.py --status                         # 查看状态
  python live_runner.py --kill-switch                    # 紧急平仓
  python live_runner.py --generate-config                # 生成配置模板
  python live_runner.py --test-connection                # 测试 API 连接
  python live_runner.py --test-signal                    # 测试信号计算
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_config import (
    LiveTradingConfig, TradingPhase, StrategyConfig,
    create_default_config
)
from live_trading_engine import LiveTradingEngine


BANNER = """
╔══════════════════════════════════════════════════════════╗
║           六书融合策略 · 实盘交易系统                      ║
║   Divergence + MA + Candlestick + Bollinger + VP + KDJ   ║
║                   ETH/USDT Futures                       ║
╚══════════════════════════════════════════════════════════╝
"""

PHASE_INFO = {
    "paper": {
        "name": "Phase 1: 纸上交易",
        "desc": "实时运行策略但不执行交易，仅记录信号和模拟结果",
        "risk": "零风险 - 无真实资金参与",
        "duration": "建议运行 1-2 周",
    },
    "testnet": {
        "name": "Phase 2: 测试网实盘",
        "desc": "在 Binance Testnet 上执行真实 API 调用",
        "risk": "零风险 - 使用测试网模拟资金",
        "duration": "建议运行 1-2 周",
    },
    "small_live": {
        "name": "Phase 3: 小资金实盘",
        "desc": "$500-1000 真实资金，2x 杠杆",
        "risk": "中等风险 - 最大损失限于投入资金",
        "duration": "建议运行 2-4 周",
    },
    "scale_up": {
        "name": "Phase 4: 逐步加仓",
        "desc": "盈利后逐步放大资金规模",
        "risk": "较高风险 - 需持续监控",
        "duration": "持续运行，每 2 周评估",
    },
}


def print_phase_info(phase: str):
    """打印阶段信息"""
    info = PHASE_INFO.get(phase, {})
    print(f"\n  📋 {info.get('name', phase)}")
    print(f"  📝 {info.get('desc', '')}")
    print(f"  ⚠️  {info.get('risk', '')}")
    print(f"  ⏱️  {info.get('duration', '')}")
    print()


def cmd_run(args):
    """运行交易引擎"""
    print(BANNER)

    # 加载配置
    if args.config and os.path.exists(args.config):
        print(f"  从配置文件加载: {args.config}")
        config = LiveTradingConfig.load(args.config)
    else:
        phase = TradingPhase(args.phase)
        config = create_default_config(phase)

        # 尝试加载优化结果
        opt_file = "optimize_six_book_result.json"
        if os.path.exists(opt_file):
            try:
                config.strategy = StrategyConfig.from_optimize_result(opt_file)
                print(f"  ✓ 已加载优化策略参数: {opt_file}")
            except Exception as e:
                print(f"  ⚠ 优化参数加载失败，使用默认值: {e}")

        # 命令行覆盖
        if args.symbol:
            config.strategy.symbol = args.symbol
        if args.timeframe:
            config.strategy.timeframe = args.timeframe
        if args.capital:
            config.initial_capital = args.capital
        if args.leverage:
            config.strategy.leverage = args.leverage

    print_phase_info(config.phase.value)

    # 安全确认
    if config.phase in (TradingPhase.SMALL_LIVE, TradingPhase.SCALE_UP):
        print("  ⚠️  警告: 此阶段涉及真实资金!")
        print(f"  💰 初始资金: ${config.initial_capital:.2f}")
        print(f"  📊 杠杆: {config.strategy.leverage}x")
        print(f"  🛡️  最大日亏损: {config.risk.max_daily_loss_pct:.0%}")
        print(f"  🛡️  最大回撤: {config.risk.max_drawdown_pct:.0%}")
        print()

        if not args.yes:
            confirm = input("  确认启动? (输入 YES 继续): ")
            if confirm != "YES":
                print("  已取消")
                return

    # 创建并运行引擎
    engine = LiveTradingEngine(config)

    print(f"\n  🚀 引擎启动中...\n")
    print(f"  按 Ctrl+C 优雅停止\n")

    engine.run()


def cmd_generate_config(args):
    """生成配置模板"""
    config = create_default_config(TradingPhase.PAPER)
    path = args.output or "live_trading_config.json"
    config.save_template(path)
    print(f"\n  ✓ 配置模板已生成: {path}")
    print(f"  请编辑此文件填入 API key 和自定义参数\n")


def cmd_test_connection(args):
    """测试 API 连接"""
    print(BANNER)
    print("  测试 API 连接...\n")

    if args.config and os.path.exists(args.config):
        config = LiveTradingConfig.load(args.config)
    else:
        config = create_default_config(
            TradingPhase(args.phase or "paper")
        )

    from order_manager import create_order_manager
    from trading_logger import TradingLogger

    logger = TradingLogger(log_dir="logs/test", name="test")
    om = create_order_manager(config, logger)

    if om.test_connection():
        print("\n  ✅ API 连接成功!")

        # 获取余额 — 优先从引擎状态文件读取真实余额
        try:
            engine_state_path = os.path.join(
                config.data_dir or "data", "engine_state.json"
            )
            if os.path.exists(engine_state_path):
                import json as _json
                with open(engine_state_path, 'r') as _f:
                    _state = _json.load(_f)
                usdt = _state.get("usdt", 0)
                frozen = _state.get("frozen_margin", 0)
                equity = _state.get("equity", usdt)
                print(f"  💰 USDT 余额: ${usdt:.2f}")
                print(f"  💰 可用余额: ${usdt - frozen:.2f}")
                if frozen > 0:
                    print(f"  🔒 冻结保证金: ${frozen:.2f}")
                print(f"  📈 当前权益: ${equity:.2f}")
            else:
                balance = om.get_balance()
                print(f"  💰 USDT 余额: ${balance['balance']:.2f}")
                print(f"  💰 可用余额: ${balance['available']:.2f}")
        except Exception:
            try:
                balance = om.get_balance()
                print(f"  💰 USDT 余额: ${balance['balance']:.2f}")
                print(f"  💰 可用余额: ${balance['available']:.2f}")
            except Exception:
                pass

        # 获取价格
        try:
            price = om.get_current_price(config.strategy.symbol)
            print(f"  📊 {config.strategy.symbol}: ${price:.2f}")
        except Exception:
            pass
    else:
        print("\n  ❌ API 连接失败")


def cmd_test_signal(args):
    """测试信号计算"""
    print(BANNER)
    print("  测试信号计算...\n")

    config = create_default_config(TradingPhase.PAPER)

    # 加载优化参数
    opt_file = "optimize_six_book_result.json"
    if os.path.exists(opt_file):
        try:
            config.strategy = StrategyConfig.from_optimize_result(opt_file)
        except Exception:
            pass

    if args.timeframe:
        config.strategy.timeframe = args.timeframe

    from live_signal_generator import LiveSignalGenerator
    from trading_logger import TradingLogger

    logger = TradingLogger(log_dir="logs/test", name="test")
    gen = LiveSignalGenerator(config.strategy, logger)

    print(f"  获取 {config.strategy.symbol} {config.strategy.timeframe} 数据...")
    gen.refresh_data(force=True)

    print(f"  计算信号...")
    sig = gen.compute_latest_signal()

    if sig:
        print(f"\n  ═══ 最新信号 ═══")
        print(f"  时间: {sig.timestamp}")
        print(f"  价格: ${sig.price:.2f}")
        print(f"  卖出分数: {sig.sell_score:.1f}")
        print(f"  买入分数: {sig.buy_score:.1f}")
        print(f"  冲突: {'是' if sig.conflict else '否'}")
        print(f"\n  六维分量:")
        for k, v in sig.components.items():
            print(f"    {k}: {v:.1f}")

        # 评估动作
        sig = gen.evaluate_action(sig)
        print(f"\n  推荐动作: {sig.action}")
        print(f"  原因: {sig.reason}")

        # 最近 5 根K线信号
        print(f"\n  ═══ 最近 5 根K线信号 ═══")
        df = gen._df
        for i in range(max(0, len(df) - 5), len(df)):
            dt = df.index[i]
            from optimize_six_book import calc_fusion_score_six
            ss, bs = calc_fusion_score_six(
                gen._signals, df, i, dt,
                {'fusion_mode': config.strategy.fusion_mode,
                 'veto_threshold': config.strategy.veto_threshold,
                 'kdj_bonus': config.strategy.kdj_bonus}
            )
            print(f"    {dt}  close=${df['close'].iloc[i]:.2f}  "
                  f"SS={ss:.1f}  BS={bs:.1f}")
    else:
        print("  ❌ 信号计算失败")


def _compute_single_tf(tf, base_config):
    """在线程中计算单个时间框架的信号（供 cmd_test_signal_multi 并行调用）"""
    import copy
    from live_signal_generator import LiveSignalGenerator
    from trading_logger import TradingLogger

    t0 = time.time()
    result = {"tf": tf, "ok": False, "elapsed": 0}
    try:
        cfg = copy.deepcopy(base_config)
        cfg.strategy.timeframe = tf
        logger = TradingLogger(log_dir="logs/test", name=f"test_{tf}")
        gen = LiveSignalGenerator(cfg.strategy, logger)
        gen.refresh_data(force=True)
        sig = gen.compute_latest_signal()
        if sig:
            sig = gen.evaluate_action(sig)
            result.update({
                "ok": True,
                "timestamp": sig.timestamp,
                "price": float(sig.price),
                "sell_score": round(float(sig.sell_score), 1),
                "buy_score": round(float(sig.buy_score), 1),
                "action": sig.action,
                "reason": sig.reason,
                "conflict": bool(sig.conflict),
                "components": {k: round(float(v), 1) for k, v in sig.components.items()},
                "bars": len(gen._df) if gen._df is not None else 0,
            })
        else:
            result["error"] = "信号计算失败"
    except Exception as e:
        result["error"] = str(e)
    result["elapsed"] = round(time.time() - t0, 1)
    return result


def cmd_test_signal_multi(args):
    """多时间框架并行信号计算"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(BANNER)

    # 解析目标时间框架
    if args.timeframe:
        timeframes = [t.strip() for t in args.timeframe.split(',') if t.strip()]
    else:
        timeframes = ['15m', '30m', '1h', '4h', '8h']

    print(f"  多时间框架并行信号检测  ({len(timeframes)} 个周期)")
    print(f"  周期: {', '.join(timeframes)}\n")

    config = create_default_config(TradingPhase.PAPER)
    opt_file = "optimize_six_book_result.json"
    if os.path.exists(opt_file):
        try:
            config.strategy = StrategyConfig.from_optimize_result(opt_file)
        except Exception:
            pass

    t_start = time.time()
    results = []

    # 并行计算（最多 4 个线程，避免 API 限流）
    max_workers = min(4, len(timeframes))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_compute_single_tf, tf, config): tf
            for tf in timeframes
        }
        for fut in as_completed(futures):
            tf_name = futures[fut]
            try:
                r = fut.result(timeout=90)
                results.append(r)
                status = "✅" if r["ok"] else "❌"
                print(f"  {status} {tf_name:>5s}  ({r['elapsed']:.1f}s)")
            except Exception as e:
                results.append({"tf": tf_name, "ok": False, "error": str(e), "elapsed": 0})
                print(f"  ❌ {tf_name:>5s}  异常: {e}")

    # 按原始顺序排列
    tf_order = {tf: i for i, tf in enumerate(timeframes)}
    results.sort(key=lambda r: tf_order.get(r["tf"], 999))

    total_time = round(time.time() - t_start, 1)
    print(f"\n  总耗时: {total_time}s (并行 {max_workers} 线程)")

    # 打印汇总表
    print(f"\n  {'─' * 78}")
    print(f"  {'周期':>5s}  {'价格':>10s}  {'卖出分':>6s}  {'买入分':>6s}  {'动作':<12s}  {'原因'}")
    print(f"  {'─' * 78}")

    for r in results:
        if r["ok"]:
            action_display = r["action"]
            if "LONG" in action_display:
                action_display = f"🟢 {action_display}"
            elif "SHORT" in action_display:
                action_display = f"🔴 {action_display}"
            else:
                action_display = f"⚪ {action_display}"
            print(f"  {r['tf']:>5s}  ${r['price']:>9.2f}  "
                  f"{r['sell_score']:>6.1f}  {r['buy_score']:>6.1f}  "
                  f"{action_display:<12s}  {r.get('reason', '')}")
        else:
            print(f"  {r['tf']:>5s}  {'--':>10s}  {'--':>6s}  {'--':>6s}  "
                  f"{'❌ 失败':<12s}  {r.get('error', '')}")

    print(f"  {'─' * 78}")

    # ================================================================
    #   智能多周期加权共识算法
    # ================================================================
    consensus = compute_weighted_consensus(results, timeframes)

    # 打印共识报告
    print(f"\n  ═══ 多周期智能共识 ═══")

    # 分类列表
    if consensus["long_tfs"]:
        print(f"  🟢 做多: {', '.join(consensus['long_tfs'])}")
    if consensus["short_tfs"]:
        print(f"  🔴 做空: {', '.join(consensus['short_tfs'])}")
    if consensus["hold_tfs"]:
        print(f"  ⚪ 观望: {', '.join(consensus['hold_tfs'])}")

    # 加权得分
    ws = consensus["weighted_scores"]
    print(f"\n  加权得分: 多={ws['long']:.1f}  空={ws['short']:.1f}  "
          f"净值={ws['net']:+.1f}  (满分100)")

    # 共振链
    for chain in consensus.get("resonance_chains", []):
        arrow = " → ".join(chain["chain"])
        icon = "🟢" if chain["direction"] == "long" else "🔴"
        print(f"  {icon} 共振链: {arrow}  "
              f"(连续{chain['length']}级, 含≥4h={chain['has_4h_plus']})")

    # 大周期信号
    lg = consensus.get("large_tf_signal", {})
    if lg.get("direction") != "neutral":
        icon = "🟢" if lg["direction"] == "long" else "🔴"
        print(f"  {icon} 大周期(≥4h): {lg['direction']} ({', '.join(lg['tfs'])})")

    # 最终决策
    d = consensus["decision"]
    strength_bar = "█" * int(d["strength"] / 10) + "░" * (10 - int(d["strength"] / 10))
    dir_icon = {"long": "🟢", "short": "🔴", "hold": "⚪"}.get(d["direction"], "⚪")
    print(f"\n  {dir_icon} 决策: {d['label']}")
    print(f"  💪 强度: [{strength_bar}] {d['strength']:.0f}/100")
    print(f"  📝 理由: {d['reason']}")

    # JSON 输出（供 API 使用）
    if args.output:
        import json as _json
        output = {
            "timeframes": timeframes,
            "results": results,
            "consensus": consensus,
            "total_elapsed": total_time,
        }
        with open(args.output, 'w') as f:
            _json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {args.output}")

    return results


# ================================================================
#   智能多周期加权共识算法
# ================================================================

# 从共享模块导入共识算法和常量
from multi_tf_consensus import (
    compute_weighted_consensus,
    TF_ORDER as _TF_ORDER,
    TF_WEIGHT as _TF_WEIGHT,
    TF_MINUTES as _TF_MINUTES,
)


def cmd_status(args):
    """查看引擎状态"""
    data_dir = args.data_dir or "data/live"
    state_file = os.path.join(data_dir, "engine_state.json")
    risk_file = os.path.join(data_dir, "risk_state.json")
    perf_file = os.path.join(data_dir, "performance.json")

    print(BANNER)
    print("  ═══ 引擎状态 ═══\n")

    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        print(f"  USDT: ${state.get('usdt', 0):.2f}")
        print(f"  冻结保证金: ${state.get('frozen_margin', 0):.2f}")
        print(f"  总K线: {state.get('total_bars', 0)}")
        print(f"  持仓: {list(state.get('positions', {}).keys()) or '无'}")
        print(f"  保存时间: {state.get('saved_at', 'N/A')}")

        for side, pos in state.get("positions", {}).items():
            print(f"\n  {side} 持仓:")
            print(f"    入场价: ${pos.get('entry_price', 0):.2f}")
            print(f"    数量: {pos.get('quantity', 0):.4f}")
            print(f"    保证金: ${pos.get('margin', 0):.2f}")
            print(f"    持仓K线: {pos.get('bars_held', 0)}")
    else:
        print("  无引擎状态文件")

    if os.path.exists(risk_file):
        with open(risk_file) as f:
            risk = json.load(f)
        print(f"\n  ═══ 风控状态 ═══\n")
        print(f"  暂停: {'是 - ' + risk.get('pause_reason', '') if risk.get('is_paused') else '否'}")
        print(f"  Kill Switch: {'激活' if risk.get('kill_switch_active') else '未激活'}")
        print(f"  日盈亏: ${risk.get('daily_pnl', 0):.2f}")
        print(f"  连续亏损: {risk.get('consecutive_losses', 0)}")
        print(f"  最大回撤: {risk.get('max_drawdown', 0):.2%}")
        print(f"  总交易: {risk.get('total_trades', 0)}")
        print(f"  总盈亏: ${risk.get('total_pnl', 0):.2f}")

    if os.path.exists(perf_file):
        from performance_tracker import PerformanceTracker
        tracker = PerformanceTracker(initial_capital=0, data_dir=data_dir)
        summary = tracker.get_summary()
        print(f"\n  ═══ 绩效汇总 ═══\n")
        print(f"  总收益率: {summary['total_return']:.2%}")
        print(f"  胜率: {summary['win_rate']:.0%}")
        print(f"  总手续费: ${summary['total_fees']:.2f}")
        print(f"  平均滑点: {summary['avg_slippage']:.3%}")

    print()


def cmd_kill_switch(args):
    """紧急平仓"""
    print("\n  🚨 Kill Switch 紧急平仓\n")

    if not args.yes:
        confirm = input("  确认执行紧急平仓? (输入 YES 继续): ")
        if confirm != "YES":
            print("  已取消")
            return

    if args.config and os.path.exists(args.config):
        config = LiveTradingConfig.load(args.config)
    else:
        print("  ❌ 需要提供配置文件 (--config)")
        return

    from order_manager import create_order_manager
    from trading_logger import TradingLogger

    logger = TradingLogger(log_dir="logs/emergency", name="emergency")
    om = create_order_manager(config, logger)

    print("  正在平仓所有持仓...")
    results = om.close_all_positions(config.strategy.symbol)

    for r in results:
        if r.get("error"):
            print(f"  ❌ {r['symbol']}: {r['error']}")
        else:
            print(f"  ✅ {r['symbol']}: 已平仓")

    print("\n  Kill Switch 执行完毕")


def main():
    parser = argparse.ArgumentParser(
        description="六书融合策略 · 实盘交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
阶段说明:
  paper       Phase 1: 纸上交易（推荐起点）
  testnet     Phase 2: Binance 测试网
  small_live  Phase 3: 小资金实盘 ($500-1000)
  scale_up    Phase 4: 逐步加仓

示例:
  python live_runner.py --phase paper
  python live_runner.py --phase paper --timeframe 4h
  python live_runner.py --config live_trading_config.json
  python live_runner.py --test-signal --timeframe 1h
  python live_runner.py --test-signal-multi --timeframe 10m,15m,30m,1h,4h,8h
  python live_runner.py --generate-config
        """
    )

    # 运行模式
    parser.add_argument("--phase", type=str, default="paper",
                       choices=["paper", "testnet", "small_live", "scale_up"],
                       help="交易阶段")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径")

    # 交易参数覆盖
    parser.add_argument("--symbol", type=str, default=None,
                       help="交易对 (默认: ETHUSDT)")
    parser.add_argument("--timeframe", type=str, default=None,
                       help="时间框架 (如: 1h, 4h)")
    parser.add_argument("--capital", type=float, default=None,
                       help="初始资金")
    parser.add_argument("--leverage", type=int, default=None,
                       help="杠杆倍数")

    # 工具命令
    parser.add_argument("--generate-config", action="store_true",
                       help="生成配置模板")
    parser.add_argument("--test-connection", action="store_true",
                       help="测试 API 连接")
    parser.add_argument("--test-signal", action="store_true",
                       help="测试信号计算")
    parser.add_argument("--test-signal-multi", action="store_true",
                       help="多时间框架并行信号检测 (--timeframe 逗号分隔)")
    parser.add_argument("--status", action="store_true",
                       help="查看引擎状态")
    parser.add_argument("--kill-switch", action="store_true",
                       help="紧急平仓")

    # 其他
    parser.add_argument("--yes", "-y", action="store_true",
                       help="跳过确认提示")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="数据目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="输出文件路径")

    args = parser.parse_args()

    # 分发命令
    if args.generate_config:
        cmd_generate_config(args)
    elif args.test_connection:
        cmd_test_connection(args)
    elif args.test_signal_multi:
        cmd_test_signal_multi(args)
    elif args.test_signal:
        cmd_test_signal(args)
    elif args.status:
        cmd_status(args)
    elif args.kill_switch:
        cmd_kill_switch(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
