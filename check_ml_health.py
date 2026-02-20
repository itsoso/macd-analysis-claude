"""
ML 模型健康检查脚本

在部署新模型到服务器后、或重启服务前运行此脚本，验证：
1. 所有模型文件是否存在并能成功加载
2. 特征计算管线是否正常
3. enhance_signal() 端到端推理是否无报错
4. Stacking / LGB / LSTM / TFT 各层是否独立可用

用法:
    python check_ml_health.py                    # 用本地 data/klines 数据测试
    python check_ml_health.py --verbose          # 显示详细特征/预测值
"""

import os
import sys
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ml_models')

# 期望存在的模型文件 (必须 / 可选)
REQUIRED_FILES = [
    'lgb_direction_model.txt',
    'lgb_direction_model.txt.meta.json',
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
    for f in REQUIRED_FILES:
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
            print(f"  {PASS} {f}  ({size//1024}KB, {mtime})")
        else:
            print(f"  {FAIL} {f}  [缺失 — 必须文件]")
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


def check_model_loading(verbose=False):
    """测试 MLSignalEnhancer 加载"""
    print("\n── 模型加载检查 ─────────────────────────────")
    try:
        from ml_live_integration import MLSignalEnhancer
        t0 = time.time()
        enhancer = MLSignalEnhancer()
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

        print(f"  {INFO} 加载耗时: {elapsed:.2f}s")
        if enhancer._stacking_meta_model is None:
            print(f"  {WARN} Stacking 未加载: 需要先运行 train_gpu.py --mode stacking --tf 1h")
        return enhancer, any_loaded
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


def check_end_to_end(enhancer, verbose=False):
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
        add_moving_averages(df, timeframe='1h')

        t0 = time.time()
        ml_ss, ml_bs, ml_info = enhancer.enhance_signal(30.0, 25.0, df)
        elapsed = time.time() - t0

        if 'ml_error' in ml_info:
            print(f"  {FAIL} enhance_signal 报错: {ml_info['ml_error']}")
            return False

        bull_prob = ml_info.get('bull_prob', '?')
        regime = ml_info.get('regime', '?')
        direction = ml_info.get('direction_action', '?')
        shadow = ml_info.get('shadow_mode', True)
        ml_ver = ml_info.get('ml_version', '?')

        print(f"  {PASS} enhance_signal 完成 ({elapsed:.2f}s)")
        print(f"  {INFO} bull_prob={bull_prob}  regime={regime}  direction={direction}")
        print(f"  {INFO} SS: 30.0 → {ml_ss:.2f},  BS: 25.0 → {ml_bs:.2f}")
        print(f"  {INFO} shadow_mode={shadow}  ml_version={ml_ver}")

        if verbose:
            for k, v in ml_info.items():
                if k not in ('shadow_mode', 'ml_version'):
                    print(f"       {k}: {v}")

        if shadow:
            print(f"  {WARN} 当前为 shadow 模式，ML 不影响实际信号")
            print(f"       启用方法: live_config.py → ml_enhancement_shadow_mode=False")
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
    args = parser.parse_args()

    print("=" * 52)
    print("  ML 模型健康检查")
    print(f"  MODEL_DIR: {MODEL_DIR}")
    print("=" * 52)

    results = {}
    results['文件完整性'] = check_files()
    enhancer, loaded = check_model_loading(args.verbose)
    results['模型加载'] = loaded
    results['特征管线'] = check_feature_pipeline(args.verbose)
    results['端到端推理'] = check_end_to_end(enhancer, args.verbose)
    print_summary(results)


if __name__ == '__main__':
    main()
