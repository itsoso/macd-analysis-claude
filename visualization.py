"""
可视化模块
绘制K线图 + 技术指标 + 背离标记
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import config as cfg


def plot_comprehensive_analysis(df: pd.DataFrame, results: dict,
                                 title: str = '背离技术分析',
                                 save_path: str = None):
    """
    绘制综合背离分析图表

    布局:
    - 子图1: K线图 + 均线 + 背离标记
    - 子图2: 成交量
    - 子图3: MACD (DIF/DEA/柱状图)
    - 子图4: KDJ
    - 子图5: RSI
    """
    fig, axes = plt.subplots(5, 1, figsize=(18, 24),
                              gridspec_kw={'height_ratios': [4, 1, 2, 1.5, 1.5]},
                              sharex=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    x = np.arange(len(df))
    dates = df.index

    # ============================================================
    # 子图1: K线图 + 均线
    # ============================================================
    ax1 = axes[0]
    ax1.set_title(title, fontsize=16, fontweight='bold')

    # 绘制K线
    for i in range(len(df)):
        color = 'red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green'

        # 实体
        body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
        body_top = max(df['open'].iloc[i], df['close'].iloc[i])
        ax1.bar(i, body_top - body_bottom, bottom=body_bottom, width=0.6,
                color=color, edgecolor=color, alpha=0.8)
        # 上下影线
        ax1.vlines(i, df['low'].iloc[i], body_bottom, color=color, linewidth=0.8)
        ax1.vlines(i, body_top, df['high'].iloc[i], color=color, linewidth=0.8)

    # 绘制均线
    ma_colors = {5: '#FF6600', 10: '#0066FF', 30: '#9933FF', 60: '#FF0099'}
    for period, color in ma_colors.items():
        col = f'MA{period}'
        if col in df.columns:
            ax1.plot(x, df[col], color=color, linewidth=1, alpha=0.7, label=f'MA{period}')

    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)

    # 标记背离信号
    _mark_signals_on_chart(ax1, df, results)

    # ============================================================
    # 子图2: 成交量
    # ============================================================
    ax2 = axes[1]
    colors = ['red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green'
              for i in range(len(df))]
    ax2.bar(x, df['volume'], color=colors, alpha=0.6, width=0.6)
    ax2.set_ylabel('成交量')
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # 子图3: MACD
    # ============================================================
    ax3 = axes[2]
    if 'DIF' in df.columns:
        ax3.plot(x, df['DIF'], color='white', linewidth=1.2, label='DIF')
        ax3.plot(x, df['DEA'], color='yellow', linewidth=1.2, label='DEA')

        # 彩柱线
        bar_colors = ['red' if v >= 0 else 'green' for v in df['MACD_BAR']]
        ax3.bar(x, df['MACD_BAR'], color=bar_colors, alpha=0.6, width=0.6)

        # 零轴
        ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax3.legend(loc='upper left', fontsize=8)

    ax3.set_ylabel('MACD')
    ax3.set_facecolor('#1a1a2e')
    ax3.grid(True, alpha=0.2)

    # 标记MACD背离
    _mark_macd_signals(ax3, df, results)

    # ============================================================
    # 子图4: KDJ
    # ============================================================
    ax4 = axes[3]
    if 'K' in df.columns:
        ax4.plot(x, df['K'], color='white', linewidth=1, label='K')
        ax4.plot(x, df['D'], color='yellow', linewidth=1, label='D')
        ax4.plot(x, df['J'], color='purple', linewidth=0.8, alpha=0.7, label='J')
        ax4.axhline(y=80, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax4.axhline(y=20, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
        ax4.axhline(y=50, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.set_ylim(-20, 120)

    ax4.set_ylabel('KDJ')
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # 子图5: RSI
    # ============================================================
    ax5 = axes[4]
    if 'RSI6' in df.columns:
        ax5.plot(x, df['RSI6'], color='white', linewidth=1, label='RSI6')
    if 'RSI12' in df.columns:
        ax5.plot(x, df['RSI12'], color='yellow', linewidth=1, label='RSI12')
    ax5.axhline(y=80, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax5.axhline(y=20, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax5.axhline(y=50, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
    ax5.legend(loc='upper left', fontsize=8)
    ax5.set_ylabel('RSI')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)

    # X轴日期标签
    step = max(len(df) // 20, 1)
    tick_positions = list(range(0, len(df), step))
    ax5.set_xticks(tick_positions)
    if hasattr(dates, 'strftime'):
        ax5.set_xticklabels([dates[i].strftime('%m-%d') for i in tick_positions],
                            rotation=45, fontsize=8)
    else:
        ax5.set_xticklabels([str(dates[i])[-5:] for i in tick_positions],
                            rotation=45, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图表已保存: {save_path}")
    else:
        plt.savefig('divergence_analysis.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("图表已保存: divergence_analysis.png")

    plt.close()


def _mark_signals_on_chart(ax, df, results):
    """在K线图上标记背离信号"""
    # 收集所有信号
    all_top_signals = []
    all_bottom_signals = []

    # 形态背离
    pattern = results.get('pattern', {})
    for key in ['kline_divergence', 'amplitude_divergence', 'time_divergence',
                 'ma_cross_divergence', 'ma_area_divergence']:
        for sig in pattern.get(key, []):
            if sig['direction'] == 'top':
                all_top_signals.append(sig)
            else:
                all_bottom_signals.append(sig)

    # MACD背离
    macd = results.get('macd', {})
    for key in ['dif_dea_divergence', 'bar_length_divergence', 'bar_area_divergence']:
        for sig in macd.get(key, []):
            if sig['direction'] == 'top':
                all_top_signals.append(sig)
            else:
                all_bottom_signals.append(sig)

    # 背驰信号
    exhaustion = results.get('exhaustion', {})
    for sig in exhaustion.get('sell_signals', []):
        all_top_signals.append({**sig, 'is_exhaustion': True})
    for sig in exhaustion.get('buy_signals', []):
        all_bottom_signals.append({**sig, 'is_exhaustion': True})

    # 去重(同一位置附近只标记一次)
    marked_top = set()
    marked_bottom = set()

    for sig in all_top_signals:
        idx = sig['idx']
        bucket = idx // 3
        if bucket in marked_top:
            continue
        marked_top.add(bucket)

        marker = '★' if sig.get('is_exhaustion') else '▼'
        color = 'darkred' if sig.get('is_exhaustion') else 'red'
        size = 12 if sig.get('is_exhaustion') else 8

        ax.annotate(marker, xy=(idx, df['high'].iloc[idx]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=size, color=color, ha='center', va='bottom',
                    fontweight='bold')

    for sig in all_bottom_signals:
        idx = sig['idx']
        bucket = idx // 3
        if bucket in marked_bottom:
            continue
        marked_bottom.add(bucket)

        marker = '★' if sig.get('is_exhaustion') else '▲'
        color = 'darkgreen' if sig.get('is_exhaustion') else 'green'
        size = 12 if sig.get('is_exhaustion') else 8

        ax.annotate(marker, xy=(idx, df['low'].iloc[idx]),
                    xytext=(0, -10), textcoords='offset points',
                    fontsize=size, color=color, ha='center', va='top',
                    fontweight='bold')


def _mark_macd_signals(ax, df, results):
    """在MACD图上标记背离信号"""
    macd = results.get('macd', {})

    for sig in macd.get('dif_dea_divergence', []):
        idx = sig['idx']
        prev_idx = sig.get('prev_idx', idx)
        dif_val = sig.get('dif_curr', df['DIF'].iloc[idx])

        color = 'red' if sig['direction'] == 'top' else 'lime'
        ax.annotate('D', xy=(idx, dif_val),
                    fontsize=8, color=color, ha='center', va='bottom',
                    fontweight='bold')

        # 画连线表示背离
        if prev_idx != idx and prev_idx < len(df):
            prev_dif = sig.get('dif_prev', df['DIF'].iloc[prev_idx])
            ax.plot([prev_idx, idx], [prev_dif, dif_val],
                    color=color, linewidth=1.5, linestyle='--', alpha=0.7)
