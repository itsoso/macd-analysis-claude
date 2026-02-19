"""
评估与回测对比

- 分类指标: Accuracy, F1, AUC-ROC
- 回测模拟: 用 ML 信号模拟交易，计算 Sharpe
- 对比: ML 信号 vs 六书规则信号 vs 买入持有
- 特征重要性分析
- ONNX 导出
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.config import (
    PRIMARY_TF, LOOKBACK, D_MODEL, N_HEADS, D_FF,
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT, STATIC_DIM,
    FORECAST_HORIZON, BATCH_SIZE,
    ENSEMBLE_TFT_WEIGHT, ENSEMBLE_LGB_WEIGHT,
    DATA_DIR, CHECKPOINT_DIR,
)
from ml.models import TemporalFusionTransformer, TimeSeriesDataset


def load_tft_model(tf: str, device: torch.device):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"tft_best_{tf}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_features = ckpt["n_features"]

    model = TemporalFusionTransformer(
        n_time_features=n_features,
        n_static=STATIC_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        forecast_horizon=FORECAST_HORIZON,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_tft(model, dataloader, device):
    """TFT 批量预测, 返回概率和回归值"""
    all_probs = []
    all_regs = []
    all_labels = []
    all_reg_labels = []

    with torch.no_grad():
        for x_time, x_static, cls_label, reg_label in dataloader:
            x_time = x_time.to(device)
            x_static = x_static.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                cls_logits, reg_pred, _, _ = model(x_time, x_static)
            probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_regs.append(reg_pred[:, 0].cpu().numpy())
            all_labels.append(cls_label.numpy())
            all_reg_labels.append(reg_label.numpy())

    return (np.concatenate(all_probs),
            np.concatenate(all_regs),
            np.concatenate(all_labels),
            np.concatenate(all_reg_labels))


def predict_lgb(tf: str, X: np.ndarray):
    """LightGBM 预测概率"""
    import lightgbm as lgb
    model_path = os.path.join(CHECKPOINT_DIR, f"lgb_{tf}.txt")
    model = lgb.Booster(model_file=model_path)
    probs = model.predict(np.nan_to_num(X, nan=0.0))
    return probs  # (N, 3)


def evaluate_classification(probs, labels, name=""):
    """计算分类指标"""
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {name} 分类评估")
    print(f"{'='*50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(classification_report(labels, preds,
                                target_names=["short", "hold", "long"],
                                zero_division=0))

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
            print(f"  AUC-ROC (weighted): {auc:.4f}")
    except Exception:
        pass

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def simulate_trading(probs, reg_preds, actual_returns, labels):
    """
    简单交易模拟:
      - 预测 long 且概率 > 0.5 → 做多
      - 预测 short 且概率 > 0.5 → 做空
      - 否则不操作
    计算 Sharpe ratio
    """
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)

    positions = np.zeros(len(preds))
    positions[(preds == 2) & (max_probs > 0.5)] = 1.0   # long
    positions[(preds == 0) & (max_probs > 0.5)] = -1.0   # short

    strategy_returns = positions * actual_returns
    strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

    total_return = (1 + strategy_returns).prod() - 1
    n_trades = np.sum(positions != 0)

    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365 * 24)
    else:
        sharpe = 0.0

    buy_hold_return = (1 + actual_returns[~np.isnan(actual_returns)]).prod() - 1

    print(f"\n  交易模拟:")
    print(f"    总收益率:     {total_return*100:.2f}%")
    print(f"    买入持有:     {buy_hold_return*100:.2f}%")
    print(f"    Sharpe:       {sharpe:.3f}")
    print(f"    交易次数:     {int(n_trades)} / {len(preds)}")
    print(f"    做多: {np.sum(positions==1)}, 做空: {np.sum(positions==-1)}, 观望: {np.sum(positions==0)}")

    return {
        "total_return": float(total_return),
        "buy_hold_return": float(buy_hold_return),
        "sharpe": float(sharpe),
        "n_trades": int(n_trades),
    }


def export_onnx(model, n_features, tf, device):
    """导出 ONNX 模型"""
    model.eval()
    dummy_time = torch.randn(1, LOOKBACK, n_features, device=device)
    dummy_static = torch.zeros(1, STATIC_DIM, device=device)

    onnx_path = os.path.join(CHECKPOINT_DIR, f"tft_{tf}.onnx")
    torch.onnx.export(
        model,
        (dummy_time, dummy_static),
        onnx_path,
        input_names=["x_time", "x_static"],
        output_names=["cls_logits", "reg_pred", "attn_weights", "var_weights"],
        dynamic_axes={
            "x_time": {0: "batch"},
            "x_static": {0: "batch"},
            "cls_logits": {0: "batch"},
            "reg_pred": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[export] ONNX 模型保存到 {onnx_path}")
    return onnx_path


def evaluate_all(tf: str = PRIMARY_TF):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    meta = json.load(open(os.path.join(DATA_DIR, f"meta_{tf}.json")))
    feat_cols = meta["feat_cols"]
    time_feat_cols = [c for c in feat_cols if c != "regime"]

    test_df = pd.read_parquet(os.path.join(DATA_DIR, f"test_{tf}.parquet"))
    print(f"[eval] 测试集: {len(test_df)} bars")

    # ── TFT ──
    tft_results = {}
    try:
        model, ckpt = load_tft_model(tf, device)
        test_ds = TimeSeriesDataset(test_df, time_feat_cols, meta["label_cols"], LOOKBACK)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )
        tft_probs, tft_regs, labels, reg_labels = predict_tft(model, test_loader, device)
        tft_results["cls"] = evaluate_classification(tft_probs, labels, "TFT")
        tft_results["trade"] = simulate_trading(tft_probs, tft_regs, reg_labels, labels)

        # ONNX 导出
        export_onnx(model, len(time_feat_cols), tf, device)
    except FileNotFoundError:
        print("[eval] TFT 模型不存在, 跳过")

    # ── LightGBM ──
    lgb_results = {}
    try:
        X_test = test_df[feat_cols].values
        lgb_probs = predict_lgb(tf, X_test)
        y_test = test_df["cls_label"].values.astype(int)
        reg_test = test_df["reg_label"].values

        lgb_results["cls"] = evaluate_classification(lgb_probs, y_test, "LightGBM")
        lgb_results["trade"] = simulate_trading(lgb_probs, None, reg_test, y_test)
    except FileNotFoundError:
        print("[eval] LightGBM 模型不存在, 跳过")

    # ── 集成 ──
    if tft_results and lgb_results:
        # 对齐长度 (TFT 因 lookback 少了前 LOOKBACK 个样本)
        offset = len(test_df) - len(tft_probs)
        lgb_probs_aligned = lgb_probs[offset:]
        y_aligned = y_test[offset:]
        reg_aligned = reg_test[offset:]

        ensemble_probs = (ENSEMBLE_TFT_WEIGHT * tft_probs +
                          ENSEMBLE_LGB_WEIGHT * lgb_probs_aligned)
        ens_cls = evaluate_classification(ensemble_probs, y_aligned, "Ensemble (TFT+LGB)")
        ens_trade = simulate_trading(ensemble_probs, tft_regs, reg_aligned, y_aligned)

        results = {
            "tft": tft_results,
            "lgb": lgb_results,
            "ensemble": {"cls": ens_cls, "trade": ens_trade},
        }
    else:
        results = {"tft": tft_results, "lgb": lgb_results}

    # 保存结果
    results_path = os.path.join(CHECKPOINT_DIR, f"eval_results_{tf}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[eval] 评估结果保存到 {results_path}")

    return results


if __name__ == "__main__":
    evaluate_all()
