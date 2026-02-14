# 《炒股指标三剑客》独立模块

这个目录是独立模块，不会修改或并入你现有策略主流程。  
目标是把你提供的书做成:

1. 逐页 OCR 文本抽取  
2. 逐页解读（页级摘要、规则、代码映射）  
3. KDJ / RSI / WR 三指标代码实现与简单回测演示

## 目录说明

- `ocr_extract.py`: 扫描版 PDF 按页 OCR。
- `page_interpret.py`: 按页生成结构化解读。
- `indicators.py`: KDJ/RSI/WR + 三剑客融合信号。
- `strategy_demo.py`: 独立演示回测（不依赖主工程策略）。
- `backtest_ethusdt_local.py`: ETH/USDT 独立策略回测（只读本地 K 线）。
- `run_pipeline.py`: 一键执行 OCR + 逐页解读。

## 快速开始

```bash
.venv/bin/pip install -r book_kdj_rsi_wr_module/requirements.txt
```

### 1) 逐页抽取 + 逐页解读（整本）

```bash
.venv/bin/python -m book_kdj_rsi_wr_module.run_pipeline \
  --pdf "/Users/liqiuhua/Downloads/炒股指标三剑客 KDJ、RSI、WR入门与技巧 三大经典指标灵活运用轻松判断个股顶底 (永良，韦铭锋著, 永良, author) 9787542952998(已优化).pdf" \
  --out-dir "/Users/liqiuhua/work/personal/macd-analysis-claude/book_kdj_rsi_wr_module/outputs"
```

输出文件:

- `outputs/ocr_pages.jsonl`: 每页 OCR 文本
- `outputs/page_interpretation.jsonl`: 每页解读结构化结果
- `outputs/page_interpretation.md`: 可直接阅读的逐页解读文档

### 2) 三指标代码演示回测

用随机样本数据演示:

```bash
.venv/bin/python -m book_kdj_rsi_wr_module.strategy_demo \
  --generate-sample \
  --out-dir "/Users/liqiuhua/work/personal/macd-analysis-claude/book_kdj_rsi_wr_module/outputs"
```

或者用你自己的 CSV（至少包含 `high`, `low`, `close`）:

```bash
.venv/bin/python -m book_kdj_rsi_wr_module.strategy_demo \
  --csv "/path/to/your_ohlc.csv" \
  --out-dir "/Users/liqiuhua/work/personal/macd-analysis-claude/book_kdj_rsi_wr_module/outputs"
```

### 3) ETH/USDT 本地数据独立回测

只读取本地 Parquet，不走 API 回退：

```bash
.venv/bin/python /Users/liqiuhua/work/personal/macd-analysis-claude/book_kdj_rsi_wr_module/backtest_ethusdt_local.py \
  --symbol ETHUSDT \
  --interval 1h \
  --start 2025-01-01 \
  --end 2026-01-31
```

默认输出到:

- `/Users/liqiuhua/work/personal/macd-analysis-claude/data/backtests/*_result.json`
- `/Users/liqiuhua/work/personal/macd-analysis-claude/data/backtests/*_trades.csv`
- `/Users/liqiuhua/work/personal/macd-analysis-claude/data/backtests/*_signals.csv`

## 说明

- 这是 OCR 自动解读，扫描噪声较高的页会有识别误差。
- 逐页解读的 `置信度` 是启发式指标，便于你优先复核低置信页面。
- 融合信号采用“最小共振数”规则（默认至少 2 个指标同向）。
