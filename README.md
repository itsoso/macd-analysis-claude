# 背离技术分析 - 代码实现

基于《背离技术分析》(江南小隐 著) 全书内容的 Python 代码实现。

## 项目概述

本项目将书中的背离分析理论系统地转化为可执行的 Python 代码，覆盖全书五个章节的核心内容：

| 章节 | 模块 | 实现内容 |
|------|------|----------|
| 第二章 几何形态背离 | `pattern_divergence.py` | 趋势K线背离、幅度背离、时间度背离、均线交叉背离、均线相交面积背离 |
| 第三章 MACD指标背离 | `macd_divergence.py` | 黄白线背离、彩柱线长度背离(当堆/邻堆/隔堆)、彩柱线面积背离、黄白线相交面积背离、金叉死叉 |
| 第四章 背离与背驰 | `exhaustion.py` | 背驰识别(隔堆背离计数+零轴返回检测)、卖点用背离/买点用背驰操作原则 |
| 第五章 其他背离 | `kdj/cci/rsi/volume_price` | KDJ顶底背离、CCI顺势指标背离、RSI强弱指标背离、量价背离(价升量减/价跌量增/地量) |

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 模拟数据演示
python main.py --demo

# 分析A股个股 (需要akshare)
python main.py --symbol 600519 --start 2024-01-01

# 从CSV文件分析
python main.py --csv your_data.csv --save output.png
```

## 在代码中使用

```python
from data_fetcher import fetch_stock_data
from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer
from visualization import plot_comprehensive_analysis

# 获取数据
df = fetch_stock_data('600519', start_date='2024-01-01')

# 计算指标
df = add_all_indicators(df)

# 综合分析
analyzer = ComprehensiveAnalyzer(df)
results = analyzer.analyze_all()

# 打印报告
analyzer.print_report(results)

# 可视化
plot_comprehensive_analysis(df, results, title='贵州茅台', save_path='maotai.png')
```

## 核心操作原则

### 卖点用背离（宁早勿晚）
满足以下任一条件即考虑卖出：
- MACD黄白线远离零轴后返回，重新上升不创新高 + 已有一次隔堆背离
- 虽未返回零轴，但已有两次以上隔堆背离

### 买点用背驰（宁迟勿早）
需同时满足两个条件才考虑买入：
- 产生了两次以上隔堆背离
- MACD黄白线两次返回零轴后再度底背离

## 项目结构

```
macd-analysis-claude/
├── main.py                 # 主程序入口
├── config.py               # 全局参数配置
├── indicators.py           # 技术指标计算 (MACD/KDJ/CCI/RSI/均线)
├── data_fetcher.py         # 数据获取 (akshare/CSV/模拟)
├── visualization.py        # 可视化图表
├── divergence/
│   ├── __init__.py
│   ├── pattern_divergence.py    # 第二章: 几何形态背离
│   ├── macd_divergence.py       # 第三章: MACD指标背离
│   ├── exhaustion.py            # 第四章: 背离与背驰
│   ├── kdj_divergence.py        # 第五章: KDJ背离
│   ├── cci_divergence.py        # 第五章: CCI背离
│   ├── rsi_divergence.py        # 第五章: RSI背离
│   ├── volume_price_divergence.py # 第五章: 量价背离
│   └── comprehensive.py        # 综合分析
├── requirements.txt
└── README.md
```

## 免责声明

本项目仅供技术学习和研究使用，不构成任何投资建议。股票投资有风险，入市需谨慎。
