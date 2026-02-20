"""
ML 模型健康检查脚本

在部署新模型到服务器后、或重启服务前运行此脚本，验证：
1. 所有模型文件是否存在并能成功加载
2. 运行配置中的 ML 开关是否启用
3. 特征计算管线是否正常
4. enhance_signal() 端到端推理是否无报错
5. 最新 live 日志 SIGNAL 是否携带 ml_* 字段

用法:
    python check_ml_health.py                    # 用本地 data/klines 数据测试
    python check_ml_health.py --verbose          # 显示详细特征/预测值
    python check_ml_health.py --timeframe 4h     # 指定 stacking 目标周期
    python check_ml_health.py --fix-stacking-alias  # 自动修复 stacking 默认别名到目标周期
    python check_ml_health.py --skip-live-check  # 跳过 live 日志检查
"""

import os
import sys
import json
import time
import argparse
import traceback
import subprocess
from datetime import datetime
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ml_models')
LIVE_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'live')

# 期望存在的模型文件 (必须 / 可选)
REQUIRED_FILE_GROUPS = [
    ('lgb_direction_model_1h.txt', 'lgb_direction_model.txt'),
]
OPTIONAL_FILES = [
    'lstm_1h.pt',
    'lstm_1h.onnx',
    'tft_1h.pt',
    'tft_1h.onnx',
    'tft_1h.meta.json',
    'lgb_cross_asset_1h.txt',
    'lgb_cross_asset_1h.txt.meta.json',
    'vol_regime_model.txt',
    'trend_regime_model.txt',
    'regime_config.json',
    'quantile_config.json',
    'stacking_meta.json',
    'stacking_meta.pkl',
    'stacking_meta_1h.json',
    'stacking_meta_1h.pkl',
    'stacking_meta_4h.json',
    'stacking_meta_4h.pkl',
    'stacking_meta_24h.json',
    'stacking_meta_24h.pkl',
    'ensemble_config.json',
]

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
WARN = '\033[93m!\033[0m'
INFO = '\033[94m·\033[0m'


def check_files():
    """检查模型文件是否存在"""
    print("\n── 模型文件检查 ─────────────────────────────")
    ok = True
    for group in REQUIRED_FILE_GROUPS:
        found = None
        for f in group:
            path = os.path.join(MODEL_DIR, f)
            if os.path.exists(path):
                found = f
                size = os.path.getsize(path)
                mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
                print(f"  {PASS} {f}  ({size//1024}KB, {mtime})")
                break
        if not found:
            print(f"  {FAIL} {' / '.join(group)}  [缺失 — 必须文件组]")
            ok = False

    missing_optional = []
    for f in OPTIONAL_FILES:
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
            print(f"  {PASS} {f}  ({size//1024}KB, {mtime})")
        else:
            missing_optional.append(f)
    if missing_optional:
        print(f"  {WARN} 以下可选文件缺失 (功能降级): {', '.join(missing_optional)}")

    return ok


def check_runtime_config():
    """检查 DB/配置中的运行时 ML 开关"""
    print("\n── 运行配置检查 ─────────────────────────────")
    try:
        from live_config import LiveTradingConfig
        cfg = LiveTradingConfig.load_from_db()
        s = cfg.strategy
        shadow_raw = getattr(s, 'ml_enhancement_shadow_mode', None)
        shadow_mode = None if shadow_raw is None else bool(shadow_raw)
        print(f"  {INFO} phase={cfg.phase.value} symbol={s.symbol} timeframe={s.timeframe}")
        print(f"  {INFO} use_ml_enhancement={getattr(s, 'use_ml_enhancement', False)}")
        print(f"  {INFO} ml_enhancement_shadow_mode={shadow_mode if shadow_mode is not None else 'unknown'}")
        print(f"  {INFO} ml_gpu_inference_url={bool(getattr(s, 'ml_gpu_inference_url', ''))}")
        return (
            True,
            s.timeframe,
            bool(getattr(s, 'use_ml_enhancement', False)),
            shadow_mode,
        )
    except Exception as e:
        print(f"  {WARN} 读取运行配置失败: {e}")
        return False, "1h", False, None


def check_stacking_alias_consistency(target_tf="1h", auto_fix=False):
    """调用 scripts/sync_stacking_alias.py 做别名一致性检查/修复。"""
    print("\n── Stacking 别名一致性检查 ─────────────────")
    script = os.path.join(os.path.dirname(__file__), "scripts", "sync_stacking_alias.py")
    if not os.path.exists(script):
        print(f"  {FAIL} 脚本缺失: {script}")
        return False

    cmd = [sys.executable, script, "--tf", target_tf]
    if not auto_fix:
        cmd.append("--check-only")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = (proc.stdout or "") + (proc.stderr or "")
        for line in output.splitlines():
            if line.strip():
                print(f"  {line}")

        if proc.returncode == 0:
            return True
        # check-only 模式下 returncode=1 表示发现不一致
        return False
    except Exception as e:
        print(f"  {FAIL} 执行失败: {e}")
        return False


def check_stacking_artifacts(target_tf="1h"):
    """检查 stacking 候选与指标门槛"""
    print("\n── Stacking 工件检查 ─────────────────────────")
    metas = sorted(glob(os.path.join(MODEL_DIR, "stacking_meta*.json")))
    if not metas:
        print(f"  {WARN} 未发现 stacking_meta*.json")
        return False

    selected = None
    for p in metas:
        name = os.path.basename(p)
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  {FAIL} {name}: 读取失败 {e}")
            continue

        tf = d.get("timeframe", "?")
        val = d.get("val_auc")
        test = d.get("test_auc")
        oof = d.get("oof_meta_auc")
        pkl = (d.get("model_files") or {}).get("meta", name.replace(".json", ".pkl"))
        pkl_ok = os.path.exists(os.path.join(MODEL_DIR, pkl))
        print(f"  {INFO} {name}: tf={tf} val={val} test={test} oof={oof} pkl={'Y' if pkl_ok else 'N'}")
        if tf == target_tf and pkl_ok and selected is None:
            selected = name

    if selected:
        print(f"  {PASS} 建议候选: {selected} (target_tf={target_tf})")
        return True
    print(f"  {WARN} 未找到与 target_tf={target_tf} 匹配且可用的 stacking 候选")
    return False


def check_latest_live_signal():
    """检查最新 live SIGNAL 的 ML 字段与时间新鲜度"""
    print("\n── Live 日志检查 ─────────────────────────────")
    files = sorted(glob(os.path.join(LIVE_LOG_DIR, "trade_*.jsonl")))
    if not files:
        print(f"  {WARN} 无 trade_*.jsonl 日志")
        return False

    latest_file = files[-1]
    latest_signal = None
    try:
        with open(latest_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("level") == "SIGNAL":
                    latest_signal = rec
    except Exception as e:
        print(f"  {FAIL} 读取日志失败: {e}")
        return False

    print(f"  {INFO} 文件: {os.path.basename(latest_file)}")
    if latest_signal is None:
        print(f"  {WARN} 未找到 SIGNAL 记录")
        return False

    ts = latest_signal.get("timestamp")
    data = latest_signal.get("data", {})
    sig_bar = data.get("timestamp")
    comps = data.get("components", {}) or {}
    ml_keys = [k for k in comps.keys() if k.startswith("ml_")]
    print(f"  {INFO} 最新记录时间: {ts}")
    print(f"  {INFO} 信号bar时间: {sig_bar}")
    print(f"  {INFO} ml_* 字段数: {len(ml_keys)}")

    if not ml_keys:
        print(f"  {FAIL} 最新 SIGNAL 没有 ml_* 字段，ML 分支未生效")
        return False

    ml_enabled = comps.get("ml_enabled")
    ml_available = comps.get("ml_available")
    ml_reason = comps.get("ml_reason", "")
    print(f"  {INFO} ml_enabled={ml_enabled} ml_available={ml_available} reason={ml_reason}")

    if "ml_bull_prob" not in comps and bool(ml_enabled):
        print(f"  {WARN} ML 已启用但未产出 bull_prob，请检查模型加载/特征覆盖")

    ml_err = comps.get("ml_error", "")
    if ml_err:
        print(f"  {FAIL} ml_error: {ml_err}")
        return False

    try:
        now_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        bar_dt = datetime.strptime(sig_bar, "%Y-%m-%d %H:%M:%S")
        lag_h = (now_dt - bar_dt).total_seconds() / 3600.0
        print(f"  {INFO} signal_lag={lag_h:.1f}h")
        if lag_h > 24:
            print(f"  {WARN} 信号bar滞后超过24h，请检查数据刷新/网络")
    except Exception:
        pass
    print(f"  {PASS} 最新 SIGNAL 已携带 ML 字段")
    return True


def check_model_loading(target_tf="1h", verbose=False):
    """测试 MLSignalEnhancer 加载"""
    print("\n── 模型加载检查 ─────────────────────────────")
    try:
        from ml_live_integration import MLSignalEnhancer
        t0 = time.time()
        enhancer = MLSignalEnhancer(stacking_timeframe=target_tf)
        result = enhancer.load_model()
        elapsed = time.time() - t0

        checks = {
            'LGB 方向': enhancer._direction_model is not None,
            '跨资产 LGB': enhancer._cross_asset_model is not None,
            'LSTM (路径注册)': enhancer._lstm_meta is not None,
            'TFT (路径注册)': enhancer._tft_meta is not None,
            'Regime': enhancer._regime_model is not None,
            '分位数': enhancer._quantile_model is not None,
            'Stacking': enhancer._stacking_meta_model is not None,
        }
        core_loaded = checks['LGB 方向'] or checks['Stacking']
        any_loaded = any(checks.values())
        for name, loaded in checks.items():
            icon = PASS if loaded else WARN
            print(f"  {icon} {name}: {'已加载' if loaded else '未加载 (降级)'}")

        if enhancer._stacking_meta_model is not None:
            tf = (enhancer._stacking_config or {}).get('timeframe', '?')
            va = (enhancer._stacking_config or {}).get('val_auc', '?')
            ta = (enhancer._stacking_config or {}).get('test_auc', '?')
            print(f"  {INFO} Stacking 详情: tf={tf}, val_auc={va}, test_auc={ta}")
        elif getattr(enhancer, "_stacking_disabled_reason", None):
            print(f"  {WARN} Stacking 禁用原因: {enhancer._stacking_disabled_reason}")
        if verbose:
            print(f"  {INFO} Stacking 候选: {list(enhancer._iter_stacking_candidates())}")

        print(f"  {INFO} 加载耗时: {elapsed:.2f}s")
        if enhancer._stacking_meta_model is None:
            print(f"  {WARN} Stacking 未加载: 需要先运行 train_gpu.py --mode stacking --tf {target_tf}")
        if not core_loaded:
            print(f"  {FAIL} 核心方向模型未加载 (需 LGB 或 Stacking 至少一个可用)")
        return enhancer, core_loaded
    except ImportError as e:
        print(f"  {FAIL} 导入失败: {e}")
        print("       请确认 lightgbm/torch 已安装: pip install lightgbm torch")
        return None, False
    except Exception as e:
        print(f"  {FAIL} 加载异常: {e}")
        traceback.print_exc()
        return None, False


def check_feature_pipeline(verbose=False):
    """测试特征计算管线"""
    print("\n── 特征计算检查 ─────────────────────────────")
    try:
        import pandas as pd
        import numpy as np

        # 构造最小测试 DataFrame (120 行, 模拟 1h K 线)
        n = 120
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        base = 2500.0
        rng = np.random.default_rng(42)
        close = base + rng.normal(0, 20, n).cumsum()
        df = pd.DataFrame({
            'open':   close * (1 + rng.normal(0, 0.001, n)),
            'high':   close * (1 + abs(rng.normal(0, 0.003, n))),
            'low':    close * (1 - abs(rng.normal(0, 0.003, n))),
            'close':  close,
            'volume': rng.exponential(1000, n),
            'quote_volume': rng.exponential(2500000, n),
            'taker_buy_volume': rng.exponential(500, n),
            'taker_buy_quote_volume': rng.exponential(1250000, n),
            'funding_rate': rng.normal(0.0001, 0.0002, n),
            'open_interest': rng.exponential(500000, n),
        }, index=idx)

        from ml_features import compute_ml_features
        t0 = time.time()
        feats = compute_ml_features(df)
        elapsed = time.time() - t0

        if feats is None or len(feats) == 0:
            print(f"  {FAIL} 特征计算返回空")
            return False

        nan_cols = feats.columns[feats.iloc[-1].isna()].tolist()
        print(f"  {PASS} 特征维度: {feats.shape[1]} 列, {len(feats)} 行 ({elapsed:.2f}s)")
        if nan_cols:
            print(f"  {WARN} 最后一行含 NaN: {nan_cols[:10]}{'...' if len(nan_cols)>10 else ''}")
        else:
            print(f"  {PASS} 最后一行无 NaN")
        if verbose:
            print(f"  {INFO} 特征列: {list(feats.columns[:20])}...")
        return True
    except Exception as e:
        print(f"  {FAIL} 特征计算失败: {e}")
        traceback.print_exc()
        return False


def check_end_to_end(enhancer, target_tf="1h", verbose=False, runtime_shadow_mode=None):
    """端到端推理测试"""
    print("\n── 端到端推理检查 ───────────────────────────")
    if enhancer is None:
        print(f"  {WARN} 跳过 (模型未加载)")
        return False
    try:
        import pandas as pd
        import numpy as np
        from indicators import add_all_indicators
        from ma_indicators import add_moving_averages

        n = 200
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        rng = np.random.default_rng(99)
        close = 2500.0 + rng.normal(0, 20, n).cumsum()
        df = pd.DataFrame({
            'open':   close * (1 + rng.normal(0, 0.001, n)),
            'high':   close * (1 + abs(rng.normal(0, 0.003, n))),
            'low':    close * (1 - abs(rng.normal(0, 0.003, n))),
            'close':  close,
            'volume': rng.exponential(1000, n),
            'quote_volume': rng.exponential(2500000, n),
            'taker_buy_volume': rng.exponential(500, n),
            'taker_buy_quote_volume': rng.exponential(1250000, n),
            'funding_rate': rng.normal(0.0001, 0.0002, n),
            'open_interest': rng.exponential(500000, n),
        }, index=idx)
        df = add_all_indicators(df)
        add_moving_averages(df, timeframe=target_tf)

        t0 = time.time()
        ml_ss, ml_bs, ml_info = enhancer.enhance_signal(30.0, 25.0, df)
        elapsed = time.time() - t0

        if 'ml_error' in ml_info:
            print(f"  {FAIL} enhance_signal 报错: {ml_info['ml_error']}")
            return False

        bull_prob = ml_info.get('bull_prob', '?')
        regime = ml_info.get('regime', '?')
        direction = ml_info.get('direction_action', '?')
        # shadow 状态优先以运行配置为准。若两侧都缺失则标记 unknown，避免误报。
        shadow_from_info = ml_info.get('shadow_mode')
        if runtime_shadow_mode is not None:
            shadow = bool(runtime_shadow_mode)
            shadow_src = "runtime_config"
        elif shadow_from_info is not None:
            shadow = bool(shadow_from_info)
            shadow_src = "ml_info"
        else:
            shadow = None
            shadow_src = "unknown"
        ml_ver = ml_info.get('ml_version', '?')

        print(f"  {PASS} enhance_signal 完成 ({elapsed:.2f}s)")
        print(f"  {INFO} bull_prob={bull_prob}  regime={regime}  direction={direction}")
        print(f"  {INFO} SS: 30.0 → {ml_ss:.2f},  BS: 25.0 → {ml_bs:.2f}")
        print(f"  {INFO} shadow_mode={shadow if shadow is not None else 'unknown'} ({shadow_src})  ml_version={ml_ver}")

        if verbose:
            for k, v in ml_info.items():
                if k not in ('shadow_mode', 'ml_version'):
                    print(f"       {k}: {v}")

        if shadow is True:
            print(f"  {WARN} 当前为 shadow 模式，ML 不影响实际信号")
            print(f"       启用方法: live_config.py → ml_enhancement_shadow_mode=False")
        elif shadow is None:
            print(f"  {WARN} 无法判定 shadow 状态（配置与 ml_info 均未提供）")
        return True
    except Exception as e:
        print(f"  {FAIL} 端到端测试失败: {e}")
        traceback.print_exc()
        return False


def print_summary(results):
    print("\n── 汇总 ─────────────────────────────────────")
    all_ok = all(results.values())
    for name, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon} {name}")
    if all_ok:
        print(f"\n  {PASS} 全部通过，ML 系统可正常运行")
    else:
        print(f"\n  {WARN} 存在问题，请根据上方输出排查")
    print()


def main():
    parser = argparse.ArgumentParser(description='ML 模型健康检查')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细输出')
    parser.add_argument('--timeframe', type=str, default='', help='Stacking 目标周期 (默认从DB读取)')
    parser.add_argument('--skip-live-check', action='store_true', help='跳过 live 日志检查')
    parser.add_argument('--fix-stacking-alias', action='store_true', help='若别名不一致则自动修复到目标周期')
    args = parser.parse_args()

    print("=" * 52)
    print("  ML 模型健康检查")
    print(f"  MODEL_DIR: {MODEL_DIR}")
    print("=" * 52)

    results = {}
    cfg_ok, cfg_tf, cfg_ml, cfg_shadow = check_runtime_config()
    target_tf = (args.timeframe or cfg_tf or '1h').strip()
    results['运行配置'] = cfg_ok
    results['Stacking别名'] = check_stacking_alias_consistency(
        target_tf, auto_fix=bool(args.fix_stacking_alias)
    )
    results['文件完整性'] = check_files()
    results['Stacking工件'] = check_stacking_artifacts(target_tf)
    enhancer, loaded = check_model_loading(target_tf, args.verbose)
    results['模型加载'] = loaded
    results['特征管线'] = check_feature_pipeline(args.verbose)
    results['端到端推理'] = check_end_to_end(
        enhancer, target_tf, args.verbose, runtime_shadow_mode=cfg_shadow
    )
    if not args.skip_live_check:
        results['Live日志ML字段'] = check_latest_live_signal()
    elif cfg_ml:
        print(f"\n  {INFO} 已跳过 live 日志检查 (--skip-live-check)")
    print_summary(results)


if __name__ == '__main__':
    main()
