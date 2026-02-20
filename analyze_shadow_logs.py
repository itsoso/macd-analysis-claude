"""
ML Shadow 日志分析脚本

解析 logs/live/trade_YYYYMMDD.jsonl 中的 SIGNAL 记录，统计：
1. ML 预测质量: bull_prob 分布 vs 实际价格走势
2. Regime 分类准确率 vs 事后波动率
3. Shadow 模式覆盖率 (有 ML 数据的信号占比)
4. ML 错误率 (ml_error 字段)
5. 各模型(LGB/LSTM/TFT)方向一致率

用法:
    python analyze_shadow_logs.py                  # 分析全部 logs/live/*.jsonl
    python analyze_shadow_logs.py --days 7         # 最近7天
    python analyze_shadow_logs.py --log trade_20260216.jsonl
    python analyze_shadow_logs.py --min-bars 50    # 最少N条信号才输出
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'live')

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
WARN = '\033[93m!\033[0m'
INFO = '\033[94m·\033[0m'


def load_signals(log_files):
    """从 JSONL 文件加载 SIGNAL 记录"""
    signals = []
    for path in log_files:
        try:
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get('level') == 'SIGNAL':
                            signals.append(rec)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"  {WARN} 读取失败 {path}: {e}")
    return signals


def summarize_coverage(signals):
    """统计 ML shadow 覆盖率"""
    total = len(signals)
    with_ml = sum(1 for s in signals
                  if 'ml_bull_prob' in s.get('data', {}).get('components', {}))
    with_error = sum(1 for s in signals
                     if 'ml_error' in s.get('data', {}).get('components', {}))
    no_ml = total - with_ml - with_error

    print(f"\n── 信号覆盖率 (共 {total} 条信号) ──────────────")
    if total == 0:
        print(f"  {WARN} 无 SIGNAL 记录")
        return

    pct_ml = with_ml / total * 100
    pct_err = with_error / total * 100
    icon_ml = PASS if pct_ml > 80 else (WARN if pct_ml > 0 else FAIL)
    print(f"  {icon_ml} 含 ML 预测: {with_ml}/{total} ({pct_ml:.1f}%)")
    if with_error:
        print(f"  {FAIL} ML 报错: {with_error}/{total} ({pct_err:.1f}%)")
    if no_ml and not with_error:
        print(f"  {WARN} 无 ML 数据且无报错 {no_ml} 条")
        print(f"       → 可能原因: 模型未加载 (部署时间 < 训练时间) 或 use_ml_enhancement=False")


def analyze_bull_prob(signals):
    """分析 bull_prob 分布"""
    ml_sigs = [s for s in signals
               if 'ml_bull_prob' in s.get('data', {}).get('components', {})]
    if not ml_sigs:
        print(f"\n── Bull Prob 分析 ───────────────────────────")
        print(f"  {WARN} 无 ML 预测数据，跳过")
        return

    probs = [float(s['data']['components']['ml_bull_prob']) for s in ml_sigs]
    directions = [s['data']['components'].get('ml_direction', 'neutral') for s in ml_sigs]

    def _dir_bucket(raw):
        d = str(raw).lower()
        if d.startswith('bullish') or d == 'long':
            return 'bull'
        if d.startswith('bearish') or d == 'short':
            return 'bear'
        return 'neutral'

    bull_count = sum(1 for d in directions if _dir_bucket(d) == 'bull')
    bear_count = sum(1 for d in directions if _dir_bucket(d) == 'bear')
    neutral_count = sum(1 for d in directions if _dir_bucket(d) == 'neutral')

    import statistics
    print(f"\n── Bull Prob 分析 (N={len(probs)}) ─────────────")
    print(f"  {INFO} 均值: {statistics.mean(probs):.3f}  "
          f"中位数: {statistics.median(probs):.3f}  "
          f"标准差: {statistics.stdev(probs) if len(probs)>1 else 0:.3f}")
    print(f"  {INFO} 方向分布: 看多={bull_count}  看空={bear_count}  中性={neutral_count}")

    buckets = {'< 0.42 (bear)': 0, '0.42-0.58 (neutral)': 0, '> 0.58 (bull)': 0}
    for p in probs:
        if p < 0.42:
            buckets['< 0.42 (bear)'] += 1
        elif p > 0.58:
            buckets['> 0.58 (bull)'] += 1
        else:
            buckets['0.42-0.58 (neutral)'] += 1
    for k, v in buckets.items():
        bar = '█' * (v * 30 // max(buckets.values(), default=1))
        print(f"  {INFO} {k:20s} {v:4d} |{bar}")

    # 模型间一致率
    lgb_probs = [s['data']['components'].get('ml_lgb_prob', None) for s in ml_sigs]
    lstm_probs = [s['data']['components'].get('ml_lstm_prob', None) for s in ml_sigs]
    tft_probs = [s['data']['components'].get('ml_tft_prob', None) for s in ml_sigs]

    def _agree_rate(p_list_a, p_list_b, threshold=0.5):
        pairs = [(a, b) for a, b in zip(p_list_a, p_list_b)
                 if a not in (None, '-') and b not in (None, '-')]
        if not pairs:
            return None, 0
        agree = sum(1 for a, b in pairs
                    if (float(a) > threshold) == (float(b) > threshold))
        return agree / len(pairs), len(pairs)

    lgb_lstm_rate, n1 = _agree_rate(lgb_probs, lstm_probs)
    lgb_tft_rate, n2 = _agree_rate(lgb_probs, tft_probs)
    if lgb_lstm_rate is not None:
        print(f"  {INFO} LGB-LSTM 方向一致率: {lgb_lstm_rate:.1%} (N={n1})")
    if lgb_tft_rate is not None:
        print(f"  {INFO} LGB-TFT  方向一致率: {lgb_tft_rate:.1%} (N={n2})")


def analyze_regime(signals):
    """统计 regime 分类分布"""
    ml_sigs = [s for s in signals
               if 'ml_regime' in s.get('data', {}).get('components', {})]
    if not ml_sigs:
        return

    regime_counts = defaultdict(int)
    for s in ml_sigs:
        r = s['data']['components'].get('ml_regime', '-')
        regime_counts[r] += 1

    print(f"\n── Regime 分布 (N={len(ml_sigs)}) ──────────────")
    total = sum(regime_counts.values())
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = '█' * int(pct / 3)
        print(f"  {INFO} {regime:15s} {count:4d} ({pct:5.1f}%) |{bar}")


def analyze_ml_errors(signals):
    """分析 ML 错误信息"""
    error_sigs = [s for s in signals
                  if 'ml_error' in s.get('data', {}).get('components', {})]
    if not error_sigs:
        return

    print(f"\n── ML 错误分析 ({len(error_sigs)} 条) ───────────")
    error_counts = defaultdict(int)
    for s in error_sigs:
        err = s['data']['components'].get('ml_error', '?')
        # 取错误前60字符作为 key
        error_counts[err[:60]] += 1
    for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {FAIL} [{cnt:3d}x] {err}")


def analyze_kelly(signals):
    """统计 Kelly 仓位分布"""
    kelly_vals = []
    for s in signals:
        c = s.get('data', {}).get('components', {})
        v = c.get('ml_kelly_fraction', None)
        if v not in (None, '-'):
            try:
                kelly_vals.append(float(v))
            except (ValueError, TypeError):
                pass
    if not kelly_vals:
        return

    import statistics
    print(f"\n── Kelly 仓位分布 (N={len(kelly_vals)}) ─────────")
    print(f"  {INFO} 均值: {statistics.mean(kelly_vals):.3f}  "
          f"最小: {min(kelly_vals):.3f}  最大: {max(kelly_vals):.3f}")
    buckets = [(0, 0.3, '0.0-0.3 (保守)'), (0.3, 0.6, '0.3-0.6 (中等)'),
               (0.6, 0.8, '0.6-0.8 (积极)'), (0.8, 1.1, '0.8-1.0 (激进)')]
    for lo, hi, label in buckets:
        cnt = sum(1 for v in kelly_vals if lo <= v < hi)
        bar = '█' * (cnt * 30 // max(len(kelly_vals), 1))
        print(f"  {INFO} {label:20s} {cnt:4d} |{bar}")


def print_deployment_advice(signals):
    """根据数据给出部署建议"""
    total = len(signals)
    if total == 0:
        return

    ml_sigs = [s for s in signals
               if 'ml_bull_prob' in s.get('data', {}).get('components', {})]
    error_sigs = [s for s in signals
                  if 'ml_error' in s.get('data', {}).get('components', {})]

    print(f"\n── 部署建议 ─────────────────────────────────")
    pct = len(ml_sigs) / total * 100

    if pct == 0 and not error_sigs:
        print(f"  {FAIL} ML 完全未运行")
        print(f"       1. 确认服务器已部署 data/ml_models/ (含 lgb_direction_model_1h.txt / tft_1h.pt / stacking_meta_1h.*)")
        print(f"       2. 确认 DB 策略配置 use_ml_enhancement=True")
        print(f"       3. 重启服务: systemctl restart macd-engine (必要时同时重启 macd-analysis)")
    elif error_sigs:
        err_sample = error_sigs[0]['data']['components'].get('ml_error', '')
        err_l = err_sample.lower()
        if 'lightgbm' in err_l or 'module' in err_l:
            print(f"  {FAIL} 依赖缺失: {err_sample}")
            print(f"       pip install lightgbm  (服务器上执行)")
        elif 'sklearn' in err_l:
            print(f"  {FAIL} 依赖缺失: {err_sample}")
            print(f"       pip install scikit-learn  (服务器上执行)")
        else:
            print(f"  {WARN} ML 报错，检查具体错误信息")
    elif pct < 50:
        print(f"  {WARN} ML 覆盖率较低 ({pct:.0f}%)，部分信号无 ML 增强")
    else:
        shadow_sigs = [s for s in ml_sigs
                       if s['data']['components'].get('ml_shadow', True)]
        if all(s['data']['components'].get('ml_shadow', True) for s in ml_sigs):
            print(f"  {PASS} ML 运行正常 (shadow 模式)，预测数据可用")
            print(f"       验证效果后，可将 ml_enhancement_shadow_mode=False 激活实盘增强")
        else:
            print(f"  {PASS} ML 已激活 (非 shadow 模式)，信号正在被增强")


def main():
    parser = argparse.ArgumentParser(description='ML Shadow 日志分析')
    parser.add_argument('--days', type=int, default=0, help='最近N天日志 (0=全部)')
    parser.add_argument('--log', type=str, default='', help='指定单个日志文件名')
    parser.add_argument('--min-bars', type=int, default=0, help='最少信号条数')
    args = parser.parse_args()

    print("=" * 52)
    print("  ML Shadow 日志分析")
    print(f"  LOG_DIR: {LOG_DIR}")
    print("=" * 52)

    if not os.path.exists(LOG_DIR):
        print(f"\n  {FAIL} 日志目录不存在: {LOG_DIR}")
        print("       实盘引擎尚未运行，或日志路径已更改。")
        sys.exit(1)

    # 收集日志文件
    if args.log:
        log_path = args.log if os.path.isabs(args.log) else os.path.join(LOG_DIR, args.log)
        log_files = [log_path]
    else:
        all_files = sorted([
            os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR)
            if f.endswith('.jsonl')
        ])
        if args.days > 0:
            cutoff = datetime.now() - timedelta(days=args.days)
            log_files = [f for f in all_files
                         if datetime.fromtimestamp(os.path.getmtime(f)) >= cutoff]
        else:
            log_files = all_files

    if not log_files:
        print(f"\n  {WARN} 无匹配的日志文件")
        sys.exit(0)

    print(f"\n  分析文件: {len(log_files)} 个")
    for f in log_files[-5:]:
        print(f"    · {os.path.basename(f)}")
    if len(log_files) > 5:
        print(f"    ... (前 {len(log_files)-5} 个省略)")

    signals = load_signals(log_files)

    if args.min_bars > 0 and len(signals) < args.min_bars:
        print(f"\n  {WARN} 信号数 {len(signals)} < 最小要求 {args.min_bars}，退出")
        sys.exit(0)

    summarize_coverage(signals)
    analyze_bull_prob(signals)
    analyze_regime(signals)
    analyze_kelly(signals)
    analyze_ml_errors(signals)
    print_deployment_advice(signals)
    print()


if __name__ == '__main__':
    main()
