#!/usr/bin/env python3
"""
裸K线交易法回测 — 许佳聪《裸K线交易法》策略实现
=====================================================

纯价格行为(Price Action)策略，不使用任何技术指标。
仅依赖K线形态 + 关键位(支撑/阻力) + 趋势结构做出交易决策。

数据: ETH/USDT 日线, 2025-01-01 ~ 2026-01-31
交易: 合约双向(做多/做空), 逐仓, 可配置杠杆

回测遵循严格无偏原则:
- 信号在K线收盘时生成, 下一根K线开盘执行 (无同Bar偏差)
- 含手续费(0.05%)+滑点(0.1%)+资金费率(0.01%/8h)
- 每笔交易记录完整明细, 供人工review
"""

import os
import sys
import json
import socket
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from naked_kline_strategy import (
    Signal, KeyLevel,
    find_swing_points, identify_trend, find_key_levels,
    is_at_key_level, scan_bar, generate_daily_signals,
    calc_position_size,
    detect_pin_bar, detect_inside_bar, detect_engulfing,
    detect_fakey, detect_two_bar_reversal,
)

# ======================================================
#   轻量级合约引擎 (独立于主项目)
# ======================================================

class NakedKlinePosition:
    """合约仓位"""
    def __init__(self, direction, entry_price, quantity, leverage, margin, stop_loss, tp1, tp2, reason):
        self.direction = direction      # 'long' / 'short'
        self.entry_price = entry_price
        self.quantity = quantity
        self.leverage = leverage
        self.margin = margin            # 投入保证金
        self.stop_loss = stop_loss
        self.take_profit_1 = tp1
        self.take_profit_2 = tp2
        self.reason = reason
        self.open_time = None
        self.bars_held = 0

    def calc_pnl(self, price):
        if self.direction == 'long':
            return (price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - price) * self.quantity

    def pnl_ratio(self, price):
        """相对于保证金的盈亏比"""
        pnl = self.calc_pnl(price)
        return pnl / self.margin if self.margin > 0 else 0


class NakedKlineEngine:
    """裸K线策略回测引擎"""

    TAKER_FEE = 0.0005     # 0.05%
    SLIPPAGE = 0.001        # 0.1%
    FUNDING_RATE = 0.0001   # 0.01% / 8h
    MAINTENANCE_RATE = 0.05 # 5% 维持保证金率

    def __init__(self, initial_usdt=100000, max_leverage=3, risk_per_trade=0.02,
                 max_bars_hold=30):
        self.initial_usdt = initial_usdt
        self.usdt = initial_usdt
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade
        self.max_bars_hold = max_bars_hold

        self.position: Optional[NakedKlinePosition] = None
        self.frozen_margin = 0.0

        # 统计
        self.trades: List[Dict] = []
        self.daily_records: List[Dict] = []
        self.history: List[Dict] = []
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_funding = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

    def available_usdt(self):
        """可用资金 (usdt已扣除冻结保证金)"""
        return self.usdt

    def total_value(self, price):
        """总资产 = 可用资金 + 冻结保证金 + 未实现盈亏"""
        val = self.usdt + self.frozen_margin
        if self.position:
            val += self.position.calc_pnl(price)
        return val

    def open_position(self, direction, price, dt, signal: Signal, leverage=None):
        """开仓"""
        if self.position is not None:
            return  # 已有持仓

        lev = leverage or min(self.max_leverage, 3)
        avail = self.available_usdt()
        risk_amount = avail * self.risk_per_trade
        price_risk = abs(signal.entry_price - signal.stop_loss)

        if price_risk <= 0 or price_risk / price > 0.10:
            return  # 止损距离不合理

        # 用书中的资金管理: 风险金额 / 止损距离 = 仓位
        qty = risk_amount / price_risk
        notional = qty * price
        margin = notional / lev

        # 保证金不超过可用余额的30%
        max_margin = avail * 0.30
        if margin > max_margin:
            margin = max_margin
            notional = margin * lev
            qty = notional / price

        # 绝对上限: 单笔名义价值不超过 $500,000 (ETH流动性约束)
        if notional > 500000:
            notional = 500000
            qty = notional / price
            margin = notional / lev

        if margin < 100 or qty * price < 200:
            return
        if margin > avail:
            return

        # 含滑点的实际成交价
        if direction == 'long':
            exec_price = price * (1 + self.SLIPPAGE)
        else:
            exec_price = price * (1 - self.SLIPPAGE)

        fee = notional * self.TAKER_FEE
        slippage_cost = notional * self.SLIPPAGE

        # 从可用资金中扣除保证金和手续费
        self.usdt -= (margin + fee)
        self.frozen_margin += margin
        self.total_fees += fee
        self.total_slippage += slippage_cost

        self.position = NakedKlinePosition(
            direction=direction,
            entry_price=exec_price,
            quantity=qty,
            leverage=lev,
            margin=margin,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            reason=f"{signal.pattern}|{signal.notes}",
        )
        self.position.open_time = dt
        self.position.bars_held = 0

        self._record_trade(dt, price, exec_price, 'OPEN', direction, qty,
                           notional, fee, slippage_cost, lev, margin, 0, signal)

    def close_position(self, price, dt, reason, use_open_price=True):
        """平仓"""
        if self.position is None:
            return

        pos = self.position
        if pos.direction == 'long':
            exec_price = price * (1 - self.SLIPPAGE)
        else:
            exec_price = price * (1 + self.SLIPPAGE)

        notional = pos.quantity * exec_price
        fee = notional * self.TAKER_FEE
        slippage_cost = abs(pos.quantity * price * self.SLIPPAGE)

        if pos.direction == 'long':
            pnl = (exec_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exec_price) * pos.quantity

        self.usdt += pos.margin + pnl - fee
        self.frozen_margin -= pos.margin
        self.total_fees += fee
        self.total_slippage += slippage_cost
        self.total_pnl += pnl

        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self._record_trade(dt, price, exec_price, 'CLOSE', pos.direction,
                           pos.quantity, notional, fee, slippage_cost,
                           pos.leverage, pos.margin, pnl, None,
                           entry_price=pos.entry_price,
                           close_reason=reason,
                           bars_held=pos.bars_held)

        self.position = None
        self.frozen_margin = max(0, self.frozen_margin)  # 安全保护

    def check_stop_loss(self, high, low, dt):
        """检查止损/止盈/强平"""
        if self.position is None:
            return

        pos = self.position

        # 强平检查
        if pos.direction == 'long':
            worst_price = low
            notional = pos.quantity * worst_price
            remaining = pos.margin + (worst_price - pos.entry_price) * pos.quantity
            if remaining < notional * self.MAINTENANCE_RATE:
                self.close_position(worst_price, dt, '强制平仓(爆仓)')
                return
        else:
            worst_price = high
            notional = pos.quantity * worst_price
            remaining = pos.margin + (pos.entry_price - worst_price) * pos.quantity
            if remaining < notional * self.MAINTENANCE_RATE:
                self.close_position(worst_price, dt, '强制平仓(爆仓)')
                return

        # 止损
        if pos.direction == 'long' and low <= pos.stop_loss:
            self.close_position(pos.stop_loss, dt, '触发止损')
            return
        if pos.direction == 'short' and high >= pos.stop_loss:
            self.close_position(pos.stop_loss, dt, '触发止损')
            return

        # 止盈1 (盈亏比 1:1.5)
        if pos.direction == 'long' and high >= pos.take_profit_1:
            self.close_position(pos.take_profit_1, dt, '触发止盈(1:1.5)')
            return
        if pos.direction == 'short' and low <= pos.take_profit_1:
            self.close_position(pos.take_profit_1, dt, '触发止盈(1:1.5)')
            return

    def apply_funding(self, price, dt):
        """每日结算资金费率 (日线每bar约结算3次 = 24h/8h)"""
        if self.position is None:
            return
        notional = self.position.quantity * price
        funding = notional * self.FUNDING_RATE * 3  # 3 次/天
        self.usdt -= funding
        self.total_funding += funding

    def record_daily(self, dt, o, h, l, c, trend, signal_desc=''):
        """记录每日状态 (含完整 OHLC)"""
        total = self.total_value(c)
        pos_info = None
        if self.position:
            pos = self.position
            pos_info = {
                'direction': pos.direction,
                'entry_price': round(pos.entry_price, 2),
                'quantity': round(pos.quantity, 4),
                'unrealized_pnl': round(pos.calc_pnl(c), 2),
                'pnl_ratio': round(pos.pnl_ratio(c) * 100, 2),
                'stop_loss': round(pos.stop_loss, 2),
                'tp1': round(pos.take_profit_1, 2),
                'bars_held': pos.bars_held,
            }

        self.daily_records.append({
            'date': dt.strftime('%Y-%m-%d'),
            'open': round(float(o), 2),
            'high': round(float(h), 2),
            'low': round(float(l), 2),
            'close': round(float(c), 2),
            'trend': trend,
            'signal': signal_desc,
            'position': pos_info,
            'total_value': round(total, 2),
            'usdt': round(self.usdt, 2),
            'return_pct': round((total / self.initial_usdt - 1) * 100, 2),
        })

        self.history.append({
            'time': dt.isoformat(),
            'total': total,
            'price': float(c),
        })

    def _record_trade(self, dt, market_price, exec_price, action, direction,
                      quantity, notional, fee, slippage_cost, leverage, margin,
                      pnl, signal=None, entry_price=None, close_reason=None,
                      bars_held=None):
        """记录完整交易明细"""
        trade = {
            'time': dt.strftime('%Y-%m-%d %H:%M'),
            'action': action,
            'direction': direction,
            'market_price': round(market_price, 2),
            'exec_price': round(exec_price, 2),
            'quantity': round(quantity, 6),
            'notional_value': round(notional, 2),
            'fee': round(fee, 4),
            'slippage_cost': round(slippage_cost, 4),
            'total_cost': round(fee + slippage_cost, 4),
            'leverage': leverage,
            'margin': round(margin, 2),
            'pnl': round(pnl, 2),
            'after_total': round(self.total_value(market_price), 2),
            'after_usdt': round(self.usdt, 2),
            'after_available': round(self.available_usdt(), 2),
        }

        if signal:
            trade['pattern'] = signal.pattern
            trade['signal_strength'] = signal.strength
            trade['at_key_level'] = signal.at_key_level
            trade['with_trend'] = signal.with_trend
            trade['trend'] = signal.trend
            trade['stop_loss'] = round(signal.stop_loss, 2)
            trade['tp1'] = round(signal.take_profit_1, 2)
            trade['tp2'] = round(signal.take_profit_2, 2)
            trade['risk_reward'] = signal.risk_reward

        if entry_price:
            trade['entry_price'] = round(entry_price, 2)
        if close_reason:
            trade['close_reason'] = close_reason
        if bars_held is not None:
            trade['bars_held'] = bars_held

        # 仓位快照
        if self.position:
            trade['has_position'] = True
            trade['pos_direction'] = self.position.direction
            trade['pos_entry'] = round(self.position.entry_price, 2)
            trade['pos_qty'] = round(self.position.quantity, 6)
        else:
            trade['has_position'] = False

        self.trades.append(trade)

    def get_summary(self):
        """获取回测汇总"""
        if not self.history:
            return {}

        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades * 100 if total_trades > 0 else 0

        # 最大回撤
        peak = self.initial_usdt
        max_dd = 0
        for h in self.history:
            val = h['total']
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        final_total = self.history[-1]['total']
        total_return = (final_total / self.initial_usdt - 1) * 100

        # 年化 (简化)
        days = len(self.daily_records)
        annual_return = total_return * 365 / days if days > 0 else 0

        # 盈亏比统计
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0 and t['action'] == 'CLOSE']
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0 and t['action'] == 'CLOSE']
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        return {
            'strategy_name': '裸K线交易法 (许佳聪)',
            'initial_capital': self.initial_usdt,
            'final_capital': round(final_total, 2),
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate_pct': round(win_rate, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_fees': round(self.total_fees, 2),
            'total_slippage': round(self.total_slippage, 2),
            'total_funding': round(self.total_funding, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'trade_days': days,
        }


# ======================================================
#   回测主流程
# ======================================================

def run_backtest(df: pd.DataFrame, leverage: int = 3,
                 risk_pct: float = 0.02,
                 max_hold: int = 30,
                 initial_capital: float = 100000,
                 cooldown_bars: int = 2) -> NakedKlineEngine:
    """
    执行裸K线策略回测。

    核心执行逻辑 (无偏):
    1. bar[i] 收盘时: 生成信号 (基于 bar[i] 及之前数据)
    2. bar[i+1] 开盘时: 执行交易 (用 bar[i+1] 的 open 作为成交价)
    3. bar[i+1] 期间: 检查止损/止盈/强平 (用 bar[i+1] 的 high/low)
    """
    engine = NakedKlineEngine(
        initial_usdt=initial_capital,
        max_leverage=leverage,
        risk_per_trade=risk_pct,
        max_bars_hold=max_hold,
    )

    warmup = 60  # 预热期: 建立关键位和趋势
    pending_signal: Optional[Signal] = None
    cooldown = 0
    last_close_trade_idx = -1  # 跟踪最后平仓的交易索引, 防止重复触发冷却

    for idx in range(warmup, len(df)):
        dt = df.index[idx]
        o = float(df['open'].iloc[idx])
        h = float(df['high'].iloc[idx])
        l = float(df['low'].iloc[idx])
        c = float(df['close'].iloc[idx])

        # ═══════════════════════════════════════
        #  Phase 1: 执行待定信号 (使用当前bar的open)
        # ═══════════════════════════════════════
        if pending_signal is not None and cooldown <= 0 and engine.position is None:
            sig = pending_signal
            pending_signal = None  # 消耗掉

            # 确认信号仍有效 (开盘价没有跳过止损)
            valid = False
            if sig.direction == 'long' and o > sig.stop_loss:
                valid = True
            elif sig.direction == 'short' and o < sig.stop_loss:
                valid = True

            if valid:
                engine.open_position(sig.direction, o, dt, sig, leverage)
        else:
            pending_signal = None  # 冷却期/已持仓时消耗信号

        if cooldown > 0:
            cooldown -= 1

        # ═══════════════════════════════════════
        #  Phase 2: 持仓管理 (当前bar内)
        # ═══════════════════════════════════════
        had_position = engine.position is not None

        if engine.position:
            engine.position.bars_held += 1

            # 止损/止盈/强平检查
            engine.check_stop_loss(h, l, dt)

            # 超时平仓
            if engine.position and engine.position.bars_held >= max_hold:
                engine.close_position(c, dt, f'持仓超时({max_hold}日)')

            # 资金费率
            if engine.position:
                engine.apply_funding(c, dt)

        # 检测本轮是否发生了平仓 (仅在这个bar内触发一次冷却)
        if had_position and engine.position is None:
            trade_count = len(engine.trades)
            if trade_count > last_close_trade_idx:
                cooldown = cooldown_bars
                last_close_trade_idx = trade_count

        # ═══════════════════════════════════════
        #  Phase 3: 生成当前bar的信号 (收盘后)
        #           → 留到下一根bar执行
        # ═══════════════════════════════════════
        trend = identify_trend(df, idx)
        key_levels = find_key_levels(df, idx)
        signals = scan_bar(df, idx, key_levels, trend)

        signal_desc = ''
        if signals:
            best = signals[0]  # 取最强信号
            signal_desc = f"{best.notes}({best.direction}) S={best.strength}"

            # 只有没持仓且没冷却时才准备信号
            if engine.position is None and cooldown <= 0:
                pending_signal = best

            # 持仓时: 检查反向信号平仓 (需要较强信号)
            if engine.position and best.direction != engine.position.direction:
                if best.strength >= 60:
                    engine.close_position(c, dt, f'反向信号平仓: {best.notes}')
                    if engine.position is None:
                        cooldown = cooldown_bars
                        last_close_trade_idx = len(engine.trades)

        # 记录每日状态 (完整 OHLC)
        engine.record_daily(dt, o, h, l, c, trend, signal_desc)

    # 收尾: 如有持仓则平仓
    if engine.position and len(df) > 0:
        engine.close_position(float(df['close'].iloc[-1]), df.index[-1], '回测结束平仓')

    return engine


# ======================================================
#   数据获取
# ======================================================

def fetch_data(start_date='2025-01-01', end_date='2026-01-31') -> pd.DataFrame:
    """获取ETH/USDT日线数据"""
    # 计算需要的天数 (含预热60天)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days + 90  # 多取90天做预热

    print(f"📡 获取 ETH/USDT 日线数据 (最近 {total_days} 天) ...")
    df = fetch_binance_klines(
        symbol='ETHUSDT',
        interval='1d',
        days=total_days,
        limit_per_request=1000,
    )

    if df is None or len(df) == 0:
        raise RuntimeError("获取K线数据失败")

    # 裁剪到目标范围 (含预热期)
    pre_start = start_dt - timedelta(days=70)
    mask = (df.index >= pd.Timestamp(pre_start)) & (df.index <= pd.Timestamp(end_dt))
    df = df[mask].copy()

    print(f"  ✅ 获取 {len(df)} 根日线 [{df.index[0].date()} ~ {df.index[-1].date()}]")
    return df


# ======================================================
#   章节解读生成
# ======================================================

def generate_book_analysis():
    """生成《裸K线交易法》的章节解读"""
    return {
        'book_title': '裸K线交易法',
        'author': '许佳聪',
        'core_philosophy': '不使用任何技术指标，纯粹依靠K线形态、市场结构（趋势）和关键位（支撑/阻力）来做交易决策。强调"少即是多"的极简主义交易哲学。',
        'chapters': [
            {
                'chapter': '第一章: 认识裸K线交易',
                'summary': '裸K线交易法的核心理念是剥离所有技术指标干扰，只关注价格本身。价格是市场参与者行为的最终表现，任何指标都是对价格的二次加工，存在滞后性。',
                'key_points': [
                    '价格是所有信息的最终体现',
                    '技术指标是价格的衍生物，存在滞后',
                    '交易需要简单可重复的系统',
                ],
                'code_mapping': '本模块的设计理念：naked_kline_strategy.py 不导入任何技术指标库',
            },
            {
                'chapter': '第二章: 市场结构与趋势',
                'summary': '用摆动高点(Swing High)和摆动低点(Swing Low)定义市场结构。上升趋势=更高的高点+更高的低点(HH+HL)；下降趋势=更低的低点+更低的高点(LL+LH)。',
                'key_points': [
                    'HH+HL = 上升趋势',
                    'LL+LH = 下降趋势',
                    '趋势不明确时定义为震荡/盘整',
                    '顺势交易是第一原则',
                ],
                'code_mapping': 'find_swing_points() + identify_trend() 函数实现',
            },
            {
                'chapter': '第三章: 关键位(支撑与阻力)',
                'summary': '关键位是价格反复测试但未能突破的水平区域。支撑位在下方托住价格，阻力位在上方压制价格。关键位的强度取决于被测试次数和时间跨度。',
                'key_points': [
                    '关键位是价格多次触及的区域',
                    '多次测试的关键位更强',
                    '旧支撑变新阻力（角色互换）',
                    '交易信号在关键位附近更有效',
                ],
                'code_mapping': 'find_key_levels() + is_at_key_level() 函数实现',
            },
            {
                'chapter': '第四章: Pin Bar (影线反转信号)',
                'summary': 'Pin Bar是最经典的裸K线信号。长影线表示价格被强力推回，暗示反转。看涨Pin Bar(锤子线)有长下影线；看跌Pin Bar(射击之星)有长上影线。',
                'key_points': [
                    '影线长度 >= 实体的2倍',
                    '实体位于K线一端',
                    '出现在关键位附近最有效',
                    '顺势方向的Pin Bar胜率最高',
                ],
                'code_mapping': 'detect_pin_bar() 函数实现',
            },
            {
                'chapter': '第五章: Inside Bar (内包线)',
                'summary': 'Inside Bar代表市场犹豫和蓄能。当前K线完全被前一根K线包含，暗示即将突破。交易方向由突破方向和趋势决定。',
                'key_points': [
                    '子线的高点低于母线高点',
                    '子线的低点高于母线低点',
                    '压缩越极端，突破力度越大',
                    '母线大实体 + 小子线 = 最强信号',
                ],
                'code_mapping': 'detect_inside_bar() 函数实现',
            },
            {
                'chapter': '第六章: 吞没线 (Engulfing)',
                'summary': '吞没线/外包线是强势反转信号。当前K线完全覆盖前一根K线的范围，实体吞没实体更有力度。',
                'key_points': [
                    '当前K线的高低点完全包含前一根',
                    '看涨吞没: 阳线吞没阴线',
                    '看跌吞没: 阴线吞没阳线',
                    '实体对实体的吞没更强',
                ],
                'code_mapping': 'detect_engulfing() 函数实现',
            },
            {
                'chapter': '第七章: Fakey (假突破)',
                'summary': 'Fakey是书中最精妙的信号。先出现Inside Bar，然后假突破母线范围后收回。这意味着突破失败，应反向交易。Fakey信号的胜率最高。',
                'key_points': [
                    'Inside Bar + 假突破 + 收回 = Fakey',
                    '突破幅度越大但收回越完整，信号越强',
                    '是对散户追突破心理的逆向利用',
                    '在关键位出现的Fakey是最强信号',
                ],
                'code_mapping': 'detect_fakey() 函数实现',
            },
            {
                'chapter': '第八章: 双K反转',
                'summary': '两根连续方向相反的大实体K线形成反转。第二根收复第一根大部分失地，整体效果类似一个大Pin Bar。',
                'key_points': [
                    '两根大实体K线方向相反',
                    '第二根收复第一根60%以上幅度',
                    '配合关键位使用效果更好',
                ],
                'code_mapping': 'detect_two_bar_reversal() 函数实现',
            },
            {
                'chapter': '第九章: 交易管理',
                'summary': '入场、止损、止盈的完整管理体系。止损放在形态的极端点（如Pin Bar的影线尽头），止盈至少1:1.5的盈亏比。',
                'key_points': [
                    '入场: 形态确认后在下一根K线开盘',
                    '止损: 放在形态极端点下方/上方',
                    '止盈: 至少1:1.5盈亏比',
                    '移动止损: 在盈利后逐步收紧',
                ],
                'code_mapping': '回测引擎中的 check_stop_loss() + 开/平仓逻辑',
            },
            {
                'chapter': '第十章: 资金管理',
                'summary': '每笔交易最多承受账户2%的风险。仓位大小 = 风险金额 / 止损距离。这确保即使连续亏损也不会伤筋动骨。',
                'key_points': [
                    '单笔风险 <= 账户的2%',
                    '仓位 = 风险金额 / 价格距离',
                    '连续10次亏损也只损失20%',
                    '资金管理比胜率更重要',
                ],
                'code_mapping': 'calc_position_size() + 引擎中的仓位控制逻辑',
            },
        ],
    }


# ======================================================
#   信号统计分析
# ======================================================

def analyze_signals(engine: NakedKlineEngine, all_signals: List[Signal]):
    """统计各形态的信号质量"""
    pattern_stats = {}

    for trade in engine.trades:
        if trade['action'] != 'CLOSE':
            continue

        # 找到对应的开仓交易
        pattern = None
        for t in engine.trades:
            if (t['action'] == 'OPEN' and
                t.get('time', '') < trade.get('time', '') and
                t['direction'] == trade['direction']):
                pattern = t.get('pattern', 'unknown')

        if pattern is None:
            continue

        if pattern not in pattern_stats:
            pattern_stats[pattern] = {
                'name': pattern,
                'total': 0, 'wins': 0, 'losses': 0,
                'total_pnl': 0, 'avg_pnl': 0,
                'at_key_level_count': 0, 'with_trend_count': 0,
            }

        stats = pattern_stats[pattern]
        stats['total'] += 1
        pnl = trade.get('pnl', 0)
        stats['total_pnl'] += pnl
        if pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

    # 计算均值
    for k, v in pattern_stats.items():
        if v['total'] > 0:
            v['avg_pnl'] = round(v['total_pnl'] / v['total'], 2)
            v['win_rate'] = round(v['wins'] / v['total'] * 100, 2)
        v['total_pnl'] = round(v['total_pnl'], 2)

    return list(pattern_stats.values())


# ======================================================
#   主函数
# ======================================================

def main():
    parser = argparse.ArgumentParser(description='裸K线交易法回测')
    parser.add_argument('--start', default='2025-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2026-01-31', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金(USDT)')
    parser.add_argument('--leverage', type=int, default=3, help='杠杆倍数')
    parser.add_argument('--risk', type=float, default=0.02, help='每笔风险比例')
    parser.add_argument('--max-hold', type=int, default=30, help='最大持仓天数')
    parser.add_argument('--results-dir', default=None, help='结果输出目录')
    parser.add_argument('--runner', default='local', help='执行者标识')
    args = parser.parse_args()

    print("=" * 60)
    print("  裸K线交易法 · 许佳聪 — ETH/USDT 回测")
    print("  纯价格行为策略，零指标")
    print("=" * 60)
    print(f"  回测区间: {args.start} ~ {args.end}")
    print(f"  初始资金: ${args.capital:,.0f}")
    print(f"  杠杆: {args.leverage}x | 单笔风险: {args.risk*100:.0f}%")
    print(f"  最大持仓: {args.max_hold} 天")
    print()

    # 获取数据
    df = fetch_data(args.start, args.end)

    # 裁剪回测范围
    bt_start = pd.Timestamp(args.start)
    bt_end = pd.Timestamp(args.end)

    # 生成所有信号 (独立统计用)
    all_signals = generate_daily_signals(df, warmup=60)
    print(f"\n📊 共检测到 {len(all_signals)} 个裸K线信号")
    for pat in set(s.pattern for s in all_signals):
        count = sum(1 for s in all_signals if s.pattern == pat)
        print(f"  · {pat}: {count} 个")

    # 运行回测
    print(f"\n🔥 开始回测 ...")
    engine = run_backtest(
        df=df,
        leverage=args.leverage,
        risk_pct=args.risk,
        max_hold=args.max_hold,
        initial_capital=args.capital,
        cooldown_bars=2,
    )

    # 汇总结果
    summary = engine.get_summary()
    pattern_stats = analyze_signals(engine, all_signals)
    book_analysis = generate_book_analysis()

    print(f"\n{'='*60}")
    print(f"  回测结果汇总")
    print(f"{'='*60}")
    print(f"  总收益率:     {summary['total_return_pct']:+.2f}%")
    print(f"  年化收益率:   {summary['annual_return_pct']:+.2f}%")
    print(f"  最大回撤:     {summary['max_drawdown_pct']:.2f}%")
    print(f"  总交易次数:   {summary['total_trades']}")
    print(f"  胜率:         {summary['win_rate_pct']:.1f}%")
    print(f"  盈亏比:       {summary['profit_factor']:.2f}")
    print(f"  总手续费:     ${summary['total_fees']:,.2f}")
    print(f"  总滑点成本:   ${summary['total_slippage']:,.2f}")
    print(f"  总资金费率:   ${summary['total_funding']:,.2f}")

    # 输出信号统计
    if all_signals:
        print(f"\n📈 信号统计:")
        by_dir = {}
        for s in all_signals:
            by_dir.setdefault(s.direction, []).append(s)
        for d, sigs in by_dir.items():
            print(f"  {d}: {len(sigs)} 个信号")
            for s in sorted(sigs, key=lambda x: x.strength, reverse=True)[:3]:
                print(f"    {s.time} {s.notes} 强度={s.strength} 趋势={s.trend} 关键位={s.at_key_level}")

    # ── 保存结果到 DB ──
    from naked_kline_db import save_run, _default_db_path

    results_dir = args.results_dir or os.path.join(os.path.dirname(__file__), 'data', 'backtests')
    os.makedirs(results_dir, exist_ok=True)
    db_path = os.path.join(results_dir, 'naked_kline_backtest.db')

    run_meta = {
        'run_time': datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
        'runner': args.runner,
        'host': socket.gethostname(),
        'start_date': args.start,
        'end_date': args.end,
        'initial_capital': args.capital,
        'leverage': args.leverage,
        'risk_per_trade': args.risk,
        'max_hold_days': args.max_hold,
        'data_bars': len(df),
    }

    equity_curve = [
        {'date': h['time'][:10], 'total': round(h['total'], 2), 'price': round(h['price'], 2)}
        for h in engine.history
    ]

    # 信号转为 dict 列表
    signal_dicts = [s.to_dict() for s in all_signals]

    run_id = save_run(
        db_path=db_path,
        run_meta=run_meta,
        summary=summary,
        book_analysis=book_analysis,
        daily_records=engine.daily_records,
        trades=engine.trades,
        signals=signal_dicts,
        equity_curve=equity_curve,
    )

    print(f"\n💾 结果已写入 DB: {db_path} (run_id={run_id})")

    # 同时输出 JSON (兼容旧页面)
    result = {
        'run_meta': run_meta,
        'summary': summary,
        'book_analysis': book_analysis,
        'pattern_stats': pattern_stats,
        'signal_summary': {
            'total_signals': len(all_signals),
            'by_pattern': {},
            'by_direction': {},
        },
        'daily_records': engine.daily_records,
        'trade_details': engine.trades,
        'equity_curve': equity_curve,
    }
    for s in all_signals:
        result['signal_summary']['by_pattern'].setdefault(s.pattern, 0)
        result['signal_summary']['by_pattern'][s.pattern] += 1
        result['signal_summary']['by_direction'].setdefault(s.direction, 0)
        result['signal_summary']['by_direction'][s.direction] += 1

    json_path = os.path.join(results_dir, 'naked_kline_backtest_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"   JSON 备份: {json_path}")
    return result


if __name__ == '__main__':
    main()
