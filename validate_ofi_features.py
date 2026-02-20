"""
OFI 微结构特征扩展验证

用法:
    python3 validate_ofi_features.py           # 默认 2年数据
    python3 validate_ofi_features.py --years 3  # 3年数据
    python3 validate_ofi_features.py --full      # 全量数据 (5年)
"""
import os, sys, json, argparse
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from ml_features import compute_ml_features, compute_profit_labels

MICRO_FEATS = [
    'ofi', 'ofi_ma5', 'ofi_std5', 'cum_ofi', 'cum_ofi_slope',
    'large_trade_ratio', 'buy_sell_pressure',
    'vwap_dist_change', 'vwap_dist_ma5', 'above_vwap_streak',
]
LGB_PARAMS = dict(
    objective='binary', metric='auc',
    num_leaves=31, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, min_child_samples=50,
    reg_alpha=0.1, reg_lambda=0.1,
    verbose=-1, n_jobs=-1, seed=42,
)


def walk_forward_auc(X, y, n_folds=5, purge_bars=24):
    n = len(X)
    fold_size = n // (n_folds + 1)
    aucs, imps = [], []
    for fold in range(n_folds):
        te_start = fold_size + fold * fold_size
        te_end = te_start + fold_size
        if te_end > n:
            break
        tr_end = te_start - purge_bars
        if tr_end < 200:
            continue
        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        X_te, y_te = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        aucs.append(roc_auc_score(y_te, model.predict(X_te)))
        imp = dict(zip(X.columns, model.feature_importance('gain')))
        imps.append(imp)
    return np.mean(aucs), np.std(aucs), aucs, imps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=float, default=2.0, help='使用最近N年数据')
    parser.add_argument('--full', action='store_true', help='使用全量数据')
    parser.add_argument('--folds', type=int, default=5, help='Walk-Forward 折数')
    args = parser.parse_args()

    print("=" * 60)
    print("  OFI 微结构特征扩展验证 (Walk-Forward)")
    print("=" * 60)

    df = pd.read_parquet('data/klines/ETHUSDT/1h.parquet')
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')].sort_index()

    if not args.full:
        cutoff = df.index[-1] - pd.Timedelta(days=365 * args.years)
        df = df[df.index >= cutoff].copy()

    print(f"\n数据范围: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df):,} 行)")

    df = add_all_indicators(df)
    df = add_moving_averages(df)
    feats = compute_ml_features(df)

    # 加载 funding / OI
    fp = f'data/funding_rates/ETHUSDT_funding.parquet'
    if os.path.exists(fp):
        fund = pd.read_parquet(fp)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        feats = feats.join(fund[['funding_rate']].rename(columns={'funding_rate': '_fr_raw'}), how='left')
        if '_fr_raw' in feats.columns:
            feats['funding_rate'] = feats.get('funding_rate', feats['_fr_raw'])
            feats.drop(columns=['_fr_raw'], inplace=True, errors='ignore')

    labels = compute_profit_labels(df, horizons=[5], cost_pct=0.0015)
    valid = feats.notna().all(axis=1) & labels['profitable_long_5'].notna()
    X = feats[valid].copy()
    y = labels.loc[valid, 'profitable_long_5'].copy()

    micro_in = [f for f in MICRO_FEATS if f in X.columns]
    base_cols = [c for c in X.columns if c not in MICRO_FEATS]
    impr_cols = list(X.columns)

    print(f"有效样本: {len(X):,}  正类比例: {y.mean():.3f}")
    print(f"基线特征: {len(base_cols)} 维  改进特征: {len(impr_cols)} 维 (+{len(micro_in)} 微结构)")
    print(f"\n训练中... ({args.folds} 折 Walk-Forward)")

    base_mean, base_std, base_aucs, _ = walk_forward_auc(X[base_cols], y, args.folds)
    print(f"  基线完成: {base_mean:.4f} ± {base_std:.4f}")

    impr_mean, impr_std, impr_aucs, impr_imps = walk_forward_auc(X[impr_cols], y, args.folds)
    print(f"  改进完成: {impr_mean:.4f} ± {impr_std:.4f}")

    delta = impr_mean - base_mean
    sign = '+' if delta >= 0 else ''

    print(f"\n── 结果 ──────────────────────────────────────")
    print(f"  基线  ({len(base_cols):2d}维): AUC={base_mean:.4f} ± {base_std:.4f}")
    print(f"         逐折: {[f'{a:.4f}' for a in base_aucs]}")
    print(f"  改进  ({len(impr_cols):2d}维): AUC={impr_mean:.4f} ± {impr_std:.4f}")
    print(f"         逐折: {[f'{a:.4f}' for a in impr_aucs]}")
    print(f"  提升:  {sign}{delta:.4f} ({sign}{delta/base_mean*100:.2f}%)")

    # 微结构特征平均重要性排名
    if impr_imps:
        avg_imp = {}
        for imp in impr_imps:
            for k, v in imp.items():
                avg_imp[k] = avg_imp.get(k, 0) + v / len(impr_imps)
        sorted_feats = sorted(avg_imp.items(), key=lambda x: -x[1])
        micro_ranks = {f: i+1 for i, (f, _) in enumerate(sorted_feats) if f in micro_in}
        print(f"\n  微结构特征重要性排名 (Top 10 内):")
        for f in micro_in:
            rank = micro_ranks.get(f, '?')
            imp_val = avg_imp.get(f, 0)
            in_top10 = '★' if isinstance(rank, int) and rank <= 10 else ' '
            print(f"    {in_top10} #{rank:>3}  {f:<25} gain={imp_val:.1f}")

    conclusion = '✅ OFI 微结构特征有效 — 建议采用' if delta > 0.001 else \
                 '⚠ 微弱提升 (<0.001) — 在更长数据上验证' if delta > 0 else \
                 '❌ 无提升 — 重新检查特征实现'
    print(f"\n  {conclusion}")

    result = {
        'data_years': args.years if not args.full else 'full',
        'n_samples': len(X),
        'n_folds': args.folds,
        'baseline_auc': round(base_mean, 4),
        'baseline_std': round(base_std, 4),
        'baseline_aucs': [round(a, 4) for a in base_aucs],
        'improved_auc': round(impr_mean, 4),
        'improved_std': round(impr_std, 4),
        'improved_aucs': [round(a, 4) for a in impr_aucs],
        'delta_auc': round(delta, 4),
        'delta_pct': round(delta / base_mean * 100, 2),
        'n_base_features': len(base_cols),
        'n_improved_features': len(impr_cols),
        'micro_features_found': micro_in,
    }
    out_path = 'validate_ofi_result.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")


if __name__ == '__main__':
    main()
