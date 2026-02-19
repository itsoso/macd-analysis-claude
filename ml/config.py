"""ML 训练超参数配置"""

# ── 数据 ──
SYMBOL = "ETHUSDT"
TIMEFRAMES = ["1h", "4h", "8h", "24h"]
PRIMARY_TF = "1h"                # 主预测周期
FETCH_DAYS = 730                 # 拉取 2 年数据
LABEL_HORIZON = {                # 未来 N 根 K 线作为标签
    "1h": 6,
    "4h": 3,
    "8h": 2,
    "24h": 1,
}
LABEL_THRESHOLD = 0.015          # 涨跌幅 > 1.5% 视为 long/short, 其余 hold

# ── 时间序列切分 ──
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── TFT 模型 ──
LOOKBACK = 96                    # 输入窗口: 过去 96 根 K 线
FORECAST_HORIZON = 6             # 预测未来 6 步
D_MODEL = 128                    # 隐藏维度
N_HEADS = 4                      # 注意力头数
D_FF = 256                       # FFN 维度
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 1
DROPOUT = 0.1
STATIC_DIM = 3                   # regime one-hot (趋势/震荡/高波动)

# ── 训练 ──
BATCH_SIZE = 512
EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_ALPHA = 0.25               # Focal Loss alpha
FOCAL_GAMMA = 2.0                # Focal Loss gamma
HUBER_DELTA = 0.02               # Huber Loss delta (收益率尺度)
CLS_WEIGHT = 0.6                 # 分类损失权重
REG_WEIGHT = 0.4                 # 回归损失权重

# ── LightGBM ──
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,               # long / hold / short
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "device": "cpu",              # LightGBM CPU 即可
    "num_threads": 16,
}
LGB_ROUNDS = 2000
LGB_EARLY_STOP = 50

# ── 集成 ──
ENSEMBLE_TFT_WEIGHT = 0.6
ENSEMBLE_LGB_WEIGHT = 0.4

# ── 路径 ──
DATA_DIR = "ml/data"
CHECKPOINT_DIR = "ml/checkpoints"
LOG_DIR = "ml/logs"
