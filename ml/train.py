"""
训练主脚本 — 在 H800 上运行

用法:
    python -m ml.train                    # 默认训练 TFT + LightGBM
    python -m ml.train --model tft        # 仅 TFT
    python -m ml.train --model lgb        # 仅 LightGBM
    python -m ml.train --prepare          # 先准备数据再训练
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.config import (
    PRIMARY_TF, LOOKBACK, D_MODEL, N_HEADS, D_FF,
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT, STATIC_DIM,
    FORECAST_HORIZON, BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE,
    LR, WEIGHT_DECAY, FOCAL_ALPHA, FOCAL_GAMMA,
    HUBER_DELTA, CLS_WEIGHT, REG_WEIGHT,
    LGB_PARAMS, LGB_ROUNDS, LGB_EARLY_STOP,
    DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
)
from ml.models import TemporalFusionTransformer, TimeSeriesDataset, FocalLoss


# ══════════════════════════════════════════════════════════
#  TFT 训练
# ══════════════════════════════════════════════════════════

def train_tft(tf: str = PRIMARY_TF):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载数据
    meta = json.load(open(os.path.join(DATA_DIR, f"meta_{tf}.json")))
    feat_cols = meta["feat_cols"]
    label_cols = meta["label_cols"]

    train_df = pd.read_parquet(os.path.join(DATA_DIR, f"train_{tf}.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, f"val_{tf}.parquet"))

    print(f"[train] 训练集: {len(train_df)}, 验证集: {len(val_df)}")
    print(f"  特征: {len(feat_cols)}")

    # 去掉标签列以外的特征列中的 regime (regime 用作 static 而非 time-varying)
    time_feat_cols = [c for c in feat_cols if c != "regime"]

    train_ds = TimeSeriesDataset(train_df, time_feat_cols, label_cols, LOOKBACK)
    val_ds = TimeSeriesDataset(val_df, time_feat_cols, label_cols, LOOKBACK)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 模型
    n_features = len(time_feat_cols)
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] TFT 参数量: {n_params:,}")

    # 损失
    cls_criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    reg_criterion = nn.HuberLoss(delta=HUBER_DELTA)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 混合精度
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # TensorBoard
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_cls_correct = 0
        train_total = 0

        for batch_idx, (x_time, x_static, cls_label, reg_label) in enumerate(train_loader):
            x_time = x_time.to(device)
            x_static = x_static.to(device)
            cls_label = cls_label.to(device)
            reg_label = reg_label.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                cls_logits, reg_pred, _, _ = model(x_time, x_static)
                cls_loss = cls_criterion(cls_logits, cls_label)
                # 回归: 只取第一步预测 vs 标签
                reg_loss = reg_criterion(reg_pred[:, 0], reg_label)
                loss = CLS_WEIGHT * cls_loss + REG_WEIGHT * reg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * x_time.size(0)
            pred = cls_logits.argmax(dim=-1)
            train_cls_correct += (pred == cls_label).sum().item()
            train_total += x_time.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_acc = train_cls_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_cls_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_time, x_static, cls_label, reg_label in val_loader:
                x_time = x_time.to(device)
                x_static = x_static.to(device)
                cls_label = cls_label.to(device)
                reg_label = reg_label.to(device)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    cls_logits, reg_pred, _, _ = model(x_time, x_static)
                    cls_loss = cls_criterion(cls_logits, cls_label)
                    reg_loss = reg_criterion(reg_pred[:, 0], reg_label)
                    loss = CLS_WEIGHT * cls_loss + REG_WEIGHT * reg_loss

                val_loss_sum += loss.item() * x_time.size(0)
                pred = cls_logits.argmax(dim=-1)
                val_cls_correct += (pred == cls_label).sum().item()
                val_total += x_time.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_cls_correct / max(val_total, 1)
        elapsed = time.time() - t0

        # 日志
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"tft_best_{tf}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "n_features": n_features,
                "feat_cols": time_feat_cols,
            }, ckpt_path)
            print(f"  ✓ 保存 best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    writer.close()
    print(f"\n[train] TFT 训练完成. Best val_loss={best_val_loss:.4f}")
    return os.path.join(CHECKPOINT_DIR, f"tft_best_{tf}.pt")


# ══════════════════════════════════════════════════════════
#  LightGBM 训练
# ══════════════════════════════════════════════════════════

def train_lightgbm(tf: str = PRIMARY_TF):
    import lightgbm as lgb

    meta = json.load(open(os.path.join(DATA_DIR, f"meta_{tf}.json")))
    feat_cols = meta["feat_cols"]

    train_df = pd.read_parquet(os.path.join(DATA_DIR, f"train_{tf}.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, f"val_{tf}.parquet"))

    # 扁平特征 (无滑动窗口, 直接用当前 bar 特征)
    X_train = train_df[feat_cols].values
    y_train = train_df["cls_label"].values.astype(int)
    X_val = val_df[feat_cols].values
    y_val = val_df["cls_label"].values.astype(int)

    # 处理 NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    print(f"[train] LightGBM — 训练: {X_train.shape}, 验证: {X_val.shape}")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.log_evaluation(100),
        lgb.early_stopping(LGB_EARLY_STOP),
    ]

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # 保存
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, f"lgb_{tf}.txt")
    model.save_model(model_path)
    print(f"[train] LightGBM 保存到 {model_path}")

    # 特征重要性
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    print("\n[train] Top-20 特征重要性:")
    for name, imp in feat_imp[:20]:
        print(f"  {name:40s} {imp:.1f}")

    # 保存重要性
    imp_path = os.path.join(CHECKPOINT_DIR, f"lgb_importance_{tf}.json")
    with open(imp_path, "w") as f:
        json.dump(feat_imp, f, indent=2)

    return model_path


# ══════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ML 训练")
    parser.add_argument("--model", choices=["tft", "lgb", "all"], default="all")
    parser.add_argument("--tf", default=PRIMARY_TF)
    parser.add_argument("--prepare", action="store_true", help="先准备数据")
    args = parser.parse_args()

    if args.prepare:
        from ml.prepare_data import prepare_dataset
        prepare_dataset(tf=args.tf)

    if args.model in ("tft", "all"):
        print("\n" + "=" * 60)
        print("  训练 Temporal Fusion Transformer")
        print("=" * 60)
        train_tft(tf=args.tf)

    if args.model in ("lgb", "all"):
        print("\n" + "=" * 60)
        print("  训练 LightGBM")
        print("=" * 60)
        train_lightgbm(tf=args.tf)


if __name__ == "__main__":
    main()
