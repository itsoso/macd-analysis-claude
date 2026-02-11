"""
合约交易策略 — 允许做空, 双向操作
模拟币安USDT永续合约交易规则:
- 保证金模式: 逐仓(Isolated)
- 杠杆: 可配置 (1x ~ 10x)
- 手续费: Maker 0.02%, Taker 0.05%
- 资金费率: 每8小时收取, 约 ±0.01%
- 强制平仓: 保证金率 < 维持保证金率(5%) 时强平
- 支持同时持有多空仓位(对冲模式)

策略变体:
M: 现货+空头对冲 — 持有现货ETH, 下跌信号时开空对冲
N: 纯合约双向 — 只用合约, 顶背离做空, 底背驰做多
O: 现货+合约增强 — 现货为基础, 合约双向增强收益
P: 自适应杠杆 — 信号强度决定杠杆倍数和仓位方向
Q: 量价+合约最优 — 基于量价确认策略(J)的合约增强版
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer
from strategy_enhanced import (
    analyze_signals_enhanced, get_realtime_indicators,
    get_signal_at, DEFAULT_SIG, fetch_all_data
)


# ======================================================
#   Futures Position Tracker
# ======================================================
class FuturesPosition:
    """合约仓位管理"""

    def __init__(self, direction, entry_price, quantity, leverage, margin):
        self.direction = direction  # 'long' or 'short'
        self.entry_price = entry_price
        self.quantity = quantity
        self.leverage = leverage
        self.margin = margin  # 投入的保证金(USDT)
        self.unrealized_pnl = 0

    def calc_pnl(self, current_price):
        """计算未实现盈亏"""
        if self.direction == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        return self.unrealized_pnl

    def is_liquidated(self, current_price, maintenance_rate=0.05):
        """检查是否触发强平"""
        pnl = self.calc_pnl(current_price)
        # 保证金 + 未实现盈亏 < 维持保证金
        remaining = self.margin + pnl
        maintenance = self.margin * maintenance_rate
        return remaining < maintenance

    def liquidation_price(self, maintenance_rate=0.05):
        """计算强平价格"""
        loss_limit = self.margin * (1 - maintenance_rate)
        if self.direction == 'long':
            return self.entry_price - loss_limit / self.quantity
        else:
            return self.entry_price + loss_limit / self.quantity


class FuturesEngine:
    """合约+现货混合回测引擎"""

    TAKER_FEE = 0.0005   # 0.05% taker
    MAKER_FEE = 0.0002   # 0.02% maker
    FUNDING_RATE = 0.0001  # 0.01% per 8h
    SLIPPAGE = 0.001      # 0.1% 滑点(含市场冲击)

    def __init__(self, name, initial_usdt=100000, initial_eth_value=100000,
                 max_leverage=3, use_spot=True):
        self.name = name
        self.initial_usdt = initial_usdt
        self.initial_eth_value = initial_eth_value
        self.initial_total = initial_usdt + initial_eth_value
        self.max_leverage = max_leverage
        self.use_spot = use_spot

        self.usdt = initial_usdt
        self.spot_eth = 0  # 现货ETH持仓
        self.futures_long = None   # 多头合约仓位
        self.futures_short = None  # 空头合约仓位
        self.frozen_margin = 0     # 冻结保证金

        self.trades = []
        self.history = []
        self.funding_counter = 0

        # 风控: 最大合约保证金 = 初始总资产的10%
        self.max_margin_total = self.initial_total * 0.10
        # 单笔最大保证金(初始资产的5%)
        self.max_single_margin = self.initial_total * 0.05
        # 累计使用保证金(限制总吞吐量, 初始的150%)
        self.lifetime_margin_used = 0
        self.max_lifetime_margin = self.initial_total * 1.5

    def available_usdt(self):
        """可用USDT(排除冻结保证金)"""
        return self.usdt - self.frozen_margin

    def available_margin(self):
        """可用于开合约的保证金(受总敞口/终身上限约束)"""
        avail_usdt = self.available_usdt()
        remaining_cap = self.max_margin_total - self.frozen_margin
        lifetime_remain = self.max_lifetime_margin - self.lifetime_margin_used
        return max(0, min(avail_usdt * 0.5, remaining_cap, self.max_single_margin, lifetime_remain))

    def total_value(self, price):
        """计算总资产价值"""
        spot_val = self.spot_eth * price
        futures_pnl = 0
        if self.futures_long:
            futures_pnl += self.futures_long.calc_pnl(price)
        if self.futures_short:
            futures_pnl += self.futures_short.calc_pnl(price)
        return self.usdt + spot_val + futures_pnl

    def _record_trade(self, dt, price, action, direction, quantity, value,
                      fee, leverage, reason):
        self.trades.append({
            'time': dt.isoformat(),
            'action': action,
            'direction': direction,
            'price': round(price, 2),
            'quantity': round(quantity, 4),
            'value': round(value, 2),
            'fee': round(fee, 2),
            'leverage': leverage,
            'usdt_after': round(self.usdt, 2),
            'spot_eth': round(self.spot_eth, 4),
            'frozen_margin': round(self.frozen_margin, 2),
            'total': round(self.total_value(price), 2),
            'reason': reason,
            'long_pos': bool(self.futures_long),
            'short_pos': bool(self.futures_short),
        })

    # === 现货操作 ===
    def spot_buy(self, price, dt, usdt_amount, reason):
        """现货买入ETH"""
        avail = self.available_usdt()
        invest = min(usdt_amount, avail)
        if invest < 200:
            return
        actual_p = price * (1 + self.SLIPPAGE)
        fee = invest * self.TAKER_FEE
        qty = (invest - fee) / actual_p
        self.usdt -= invest
        self.spot_eth += qty
        self._record_trade(dt, price, 'SPOT_BUY', 'long', qty, invest, fee, 1, reason)

    def spot_sell(self, price, dt, ratio, reason):
        """现货卖出ETH"""
        qty = self.spot_eth * ratio
        if qty * price < 200:
            return
        actual_p = price * (1 - self.SLIPPAGE)
        revenue = qty * actual_p
        fee = revenue * self.TAKER_FEE
        self.usdt += revenue - fee
        self.spot_eth -= qty
        self._record_trade(dt, price, 'SPOT_SELL', 'long', qty, revenue, fee, 1, reason)

    # === 合约操作 ===
    def open_long(self, price, dt, margin, leverage, reason):
        """开多仓"""
        if self.futures_long:
            return  # 已有多仓
        avail = self.available_margin()
        margin = min(margin, avail)
        if margin < 200:
            return
        actual_p = price * (1 + self.SLIPPAGE)
        notional = margin * leverage
        qty = notional / actual_p
        fee = notional * self.TAKER_FEE
        self.usdt -= fee
        self.frozen_margin += margin
        self.lifetime_margin_used += margin
        self.futures_long = FuturesPosition('long', actual_p, qty, leverage, margin)
        self._record_trade(dt, price, 'OPEN_LONG', 'long', qty, notional, fee,
                           leverage, reason)

    def close_long(self, price, dt, reason):
        """平多仓"""
        if not self.futures_long:
            return
        pos = self.futures_long
        actual_p = price * (1 - self.SLIPPAGE)
        pnl = (actual_p - pos.entry_price) * pos.quantity
        notional = actual_p * pos.quantity
        fee = notional * self.TAKER_FEE
        self.usdt += pos.margin + pnl - fee
        self.frozen_margin -= pos.margin
        self._record_trade(dt, price, 'CLOSE_LONG', 'long', pos.quantity,
                           notional, fee, pos.leverage,
                           f"{reason} PnL={pnl:+.0f}")
        self.futures_long = None

    def open_short(self, price, dt, margin, leverage, reason):
        """开空仓"""
        if self.futures_short:
            return  # 已有空仓
        avail = self.available_margin()
        margin = min(margin, avail)
        if margin < 200:
            return
        actual_p = price * (1 - self.SLIPPAGE)
        notional = margin * leverage
        qty = notional / actual_p
        fee = notional * self.TAKER_FEE
        self.usdt -= fee
        self.frozen_margin += margin
        self.lifetime_margin_used += margin
        self.futures_short = FuturesPosition('short', actual_p, qty, leverage, margin)
        self._record_trade(dt, price, 'OPEN_SHORT', 'short', qty, notional, fee,
                           leverage, reason)

    def close_short(self, price, dt, reason):
        """平空仓"""
        if not self.futures_short:
            return
        pos = self.futures_short
        actual_p = price * (1 + self.SLIPPAGE)
        pnl = (pos.entry_price - actual_p) * pos.quantity
        notional = actual_p * pos.quantity
        fee = notional * self.TAKER_FEE
        self.usdt += pos.margin + pnl - fee
        self.frozen_margin -= pos.margin
        self._record_trade(dt, price, 'CLOSE_SHORT', 'short', pos.quantity,
                           notional, fee, pos.leverage,
                           f"{reason} PnL={pnl:+.0f}")
        self.futures_short = None

    def check_liquidation(self, price, dt):
        """检查强平"""
        if self.futures_long and self.futures_long.is_liquidated(price):
            loss = self.futures_long.margin
            self.frozen_margin -= loss
            self._record_trade(dt, price, 'LIQUIDATED', 'long',
                               self.futures_long.quantity, 0, 0,
                               self.futures_long.leverage, f"多仓强平,损失{loss:.0f}")
            self.futures_long = None

        if self.futures_short and self.futures_short.is_liquidated(price):
            loss = self.futures_short.margin
            self.frozen_margin -= loss
            self._record_trade(dt, price, 'LIQUIDATED', 'short',
                               self.futures_short.quantity, 0, 0,
                               self.futures_short.leverage, f"空仓强平,损失{loss:.0f}")
            self.futures_short = None

    def charge_funding(self, price, dt):
        """收取资金费率(每8小时)"""
        self.funding_counter += 1
        if self.funding_counter % 8 != 0:
            return
        if self.futures_long:
            cost = self.futures_long.quantity * price * self.FUNDING_RATE
            self.usdt -= cost
        if self.futures_short:
            # 空头收取资金费(通常多头付给空头)
            income = self.futures_short.quantity * price * self.FUNDING_RATE
            self.usdt += income

    def record_history(self, dt, price):
        total = self.total_value(price)
        spot_val = self.spot_eth * price
        long_pnl = self.futures_long.calc_pnl(price) if self.futures_long else 0
        short_pnl = self.futures_short.calc_pnl(price) if self.futures_short else 0
        self.history.append({
            'time': dt.isoformat(),
            'total': round(total, 2),
            'usdt': round(self.usdt, 2),
            'spot_eth_value': round(spot_val, 2),
            'long_pnl': round(long_pnl, 2),
            'short_pnl': round(short_pnl, 2),
            'frozen_margin': round(self.frozen_margin, 2),
            'eth_price': round(price, 2),
        })

    def get_result(self, main_df):
        first_price = main_df['close'].iloc[0]
        last_price = main_df['close'].iloc[-1]
        final_total = self.total_value(last_price)
        initial_total = self.initial_usdt + self.initial_eth_value
        bh_eth = self.initial_eth_value / first_price
        bh_total = self.initial_usdt + bh_eth * last_price

        # Max drawdown
        totals = [h['total'] for h in self.history]
        peak = totals[0] if totals else initial_total
        max_dd = 0
        for t in totals:
            peak = max(peak, t)
            dd = (t - peak) / peak
            max_dd = min(max_dd, dd)

        return {
            'name': self.name,
            'initial_total': initial_total,
            'final_total': round(final_total, 2),
            'strategy_return': round((final_total - initial_total) / initial_total * 100, 2),
            'buy_hold_return': round((bh_total - initial_total) / initial_total * 100, 2),
            'alpha': round((final_total - initial_total) / initial_total * 100 -
                           (bh_total - initial_total) / initial_total * 100, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'total_trades': len(self.trades),
            'liquidations': sum(1 for t in self.trades if t['action'] == 'LIQUIDATED'),
            'final_usdt': round(self.usdt, 2),
            'final_spot_eth': round(self.spot_eth, 4),
            'has_long': bool(self.futures_long),
            'has_short': bool(self.futures_short),
            'trades': self.trades,
            'history': self.history,
        }


# ======================================================
#   策略实现
# ======================================================

def build_sell_buy_scores(sig, ind, main_df, idx, is_downtrend):
    """通用多维度卖出/买入评分 (复用量价确认+全指标融合逻辑)"""
    price = main_df['close'].iloc[idx]
    k_val = ind.get('K')
    rsi = ind.get('RSI6')
    cci = ind.get('CCI')

    # 量能
    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()
    avg_vol = vol_ma20.iloc[idx] if idx >= 20 else None
    vol5 = vol_ma5.iloc[idx] if idx >= 5 else None
    vol_ratio = main_df['volume'].iloc[idx] / avg_vol if avg_vol and avg_vol > 0 else 1
    is_shrinking = vol5 < avg_vol * 0.6 if avg_vol and vol5 else False
    price_change_5 = (price - main_df['close'].iloc[max(0, idx - 5)]) / main_df['close'].iloc[max(0, idx - 5)] if idx >= 5 else 0
    price_up_vol_down = price_change_5 > 0.01 and is_shrinking

    # === 卖出评分 ===
    sell_score = 0
    sell_parts = []

    if sig['top'] >= 15:
        sell_score += min(sig['top'] / 3, 10)
        sell_parts.append(f"背离={sig['top']:.0f}")
    if sig.get('separated_top', 0) >= 1 or sig.get('dif_top_div', 0) >= 1:
        sell_score += 8; sell_parts.append("MACD背离")
    if sig.get('area_top_div', 0) >= 1:
        sell_score += 4; sell_parts.append("面积背离")
    if sig.get('sep_divs_top', 0) >= 1 and sig.get('zero_returns_top', 0) >= 1:
        sell_score += 10; sell_parts.append("隔堆+零轴")
    elif sig.get('sep_divs_top', 0) >= 2:
        sell_score += 12; sell_parts.append("双隔堆")
    if sig.get('exhaust_sell'):
        sell_score += 15; sell_parts.append("背驰")
    if k_val and k_val > 75:
        sell_score += 5; sell_parts.append(f"KDJ={k_val:.0f}")
    if rsi and rsi > 65:
        sell_score += 4; sell_parts.append(f"RSI={rsi:.0f}")
    if cci and cci > 100:
        sell_score += 4; sell_parts.append(f"CCI={cci:.0f}")
    if price_up_vol_down or sig.get('vol_price_up_down', 0) >= 1:
        sell_score += 5; sell_parts.append("价升量减")
    if is_downtrend:
        sell_score *= 1.3; sell_parts.append("↓趋势")
    if sig.get('last_cross') == 'death':
        sell_score += 5; sell_parts.append("死叉")

    # === 买入评分 ===
    buy_score = 0
    buy_parts = []

    if sig['bottom'] >= 30:
        buy_score += min(sig['bottom'] / 5, 8)
        buy_parts.append(f"底背离={sig['bottom']:.0f}")
    if sig.get('separated_bottom', 0) >= 1 or sig.get('dif_bottom_div', 0) >= 1:
        buy_score += 6; buy_parts.append("MACD底背离")
    if sig.get('sep_divs_bottom', 0) >= 2 and sig.get('zero_returns_bottom', 0) >= 2:
        buy_score += 15; buy_parts.append("背驰条件")
    if sig.get('exhaust_buy'):
        buy_score += 12; buy_parts.append("底背驰")
    if k_val and k_val < 25:
        buy_score += 5; buy_parts.append(f"KDJ={k_val:.0f}")
    if rsi and rsi < 35:
        buy_score += 4; buy_parts.append(f"RSI={rsi:.0f}")
    if vol_ratio < 0.3 or sig.get('vol_ground', 0) >= 1:
        buy_score += 5; buy_parts.append("地量")
    if sig.get('last_cross') == 'golden':
        buy_score += 4; buy_parts.append("金叉")
    if not is_downtrend:
        buy_score *= 1.3; buy_parts.append("↑趋势")
    else:
        buy_score *= 0.7

    return sell_score, sell_parts, buy_score, buy_parts


def run_strategy_M(data, signals_all):
    """策略M: 现货+空头对冲
    持有现货ETH, 顶背离时开空头对冲, 底信号时平空"""
    eng = FuturesEngine("M: 现货+空头对冲", max_leverage=2, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()
    cooldown = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val; break

        sig = _merge_signals(signals_all, dt)
        sell_s, sell_p, buy_s, buy_p = build_sell_buy_scores(sig, ind, main_df, idx, is_downtrend)

        if cooldown == 0:
            # 顶信号 → 开空对冲
            if sell_s >= 15 and not eng.futures_short:
                spot_val = eng.spot_eth * price
                # 对冲比例随信号强度增加
                if sell_s >= 35:
                    hedge_ratio = 0.8  # 强信号: 对冲80%现货
                elif sell_s >= 25:
                    hedge_ratio = 0.5
                else:
                    hedge_ratio = 0.3
                margin = spot_val * hedge_ratio / 2  # 2x杠杆
                eng.open_short(price, dt, margin, 2,
                    f"空头对冲{hedge_ratio*100:.0f}% S={sell_s:.0f} {','.join(sell_p[:3])}")
                cooldown = 4

            # 底信号 → 平空
            elif buy_s >= 18 and eng.futures_short:
                eng.close_short(price, dt,
                    f"平空 B={buy_s:.0f} {','.join(buy_p[:3])}")
                cooldown = 4

            # 超强底信号 → 平空+现货加仓
            if buy_s >= 30 and not eng.futures_short:
                invest = eng.available_usdt() * 0.2
                eng.spot_buy(price, dt, invest,
                    f"加仓 B={buy_s:.0f} {','.join(buy_p[:3])}")
                cooldown = 8

            # 空仓止盈: 盈利>保证金50%
            if eng.futures_short:
                pnl = eng.futures_short.calc_pnl(price)
                if pnl > eng.futures_short.margin * 0.5:
                    eng.close_short(price, dt, f"空仓止盈 PnL={pnl:+.0f}")
                    cooldown = 4

        if idx % 4 == 0:
            eng.record_history(dt, price)

    # 平掉所有合约仓位
    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short:
        eng.close_short(last_price, last_dt, "结束平空")
    if eng.futures_long:
        eng.close_long(last_price, last_dt, "结束平多")

    return eng.get_result(main_df)


def run_strategy_N(data, signals_all):
    """策略N: 纯合约双向
    只用合约, 不持有现货. 顶背离做空, 底背驰做多"""
    eng = FuturesEngine("N: 纯合约双向", max_leverage=3, use_spot=False)
    # 不买现货, 全部资金用于合约
    eng.usdt = eng.initial_usdt + eng.initial_eth_value  # 200000 USDT
    main_df = data['1h']
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()
    cooldown = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val; break

        sig = _merge_signals(signals_all, dt)
        sell_s, sell_p, buy_s, buy_p = build_sell_buy_scores(sig, ind, main_df, idx, is_downtrend)

        if cooldown == 0:
            avail_m = eng.available_margin()

            # 做空信号
            if sell_s >= 18:
                # 先平多仓
                if eng.futures_long:
                    eng.close_long(price, dt, f"平多转空 S={sell_s:.0f}")

                if not eng.futures_short:
                    if sell_s >= 35:
                        margin = avail_m * 0.5; lev = 3
                    elif sell_s >= 25:
                        margin = avail_m * 0.4; lev = 2
                    else:
                        margin = avail_m * 0.25; lev = 2
                    eng.open_short(price, dt, margin, lev,
                        f"做空 S={sell_s:.0f} {','.join(sell_p[:3])}")
                    cooldown = 4

            # 做多信号
            elif buy_s >= 22:
                # 先平空仓
                if eng.futures_short:
                    eng.close_short(price, dt, f"平空转多 B={buy_s:.0f}")

                if not eng.futures_long:
                    avail_m2 = eng.available_margin()
                    if buy_s >= 35:
                        margin = avail_m2 * 0.4; lev = 3
                    elif buy_s >= 28:
                        margin = avail_m2 * 0.3; lev = 2
                    else:
                        margin = avail_m2 * 0.15; lev = 2
                    eng.open_long(price, dt, margin, lev,
                        f"做多 B={buy_s:.0f} {','.join(buy_p[:3])}")
                    cooldown = 6

            # 止盈/止损
            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.6:
                    eng.close_short(price, dt, f"空仓止盈{pnl_r*100:.0f}%")
                    cooldown = 3
                elif pnl_r < -0.4:
                    eng.close_short(price, dt, f"空仓止损{pnl_r*100:.0f}%")
                    cooldown = 6

            if eng.futures_long:
                pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
                if pnl_r > 0.6:
                    eng.close_long(price, dt, f"多仓止盈{pnl_r*100:.0f}%")
                    cooldown = 3
                elif pnl_r < -0.4:
                    eng.close_long(price, dt, f"多仓止损{pnl_r*100:.0f}%")
                    cooldown = 6

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_price, last_dt, "结束平多")
    return eng.get_result(main_df)


def run_strategy_O(data, signals_all):
    """策略O: 现货+合约增强
    现货ETH为基础, 合约双向增强收益:
    - 下跌: 现货减仓 + 开空 (双重获利)
    - 上涨: 现货持仓 + 开多 (杠杆增强)"""
    eng = FuturesEngine("O: 现货+合约增强", max_leverage=2, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val; break

        sig = _merge_signals(signals_all, dt)
        sell_s, sell_p, buy_s, buy_p = build_sell_buy_scores(sig, ind, main_df, idx, is_downtrend)

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            # === 看空操作: 现货减仓 + 开空 ===
            if sell_s >= 12:
                # 1) 现货渐进减仓
                if eth_r > min_eth_ratio + 0.05:
                    if sell_s >= 30:
                        sr = 0.35
                    elif sell_s >= 20:
                        sr = 0.2
                    else:
                        sr = 0.12
                    available = eth_r - min_eth_ratio
                    sr = min(sr, available * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"减仓 S={sell_s:.0f} {','.join(sell_p[:2])}")

                # 2) 开空头合约(只在较强信号时)
                if sell_s >= 22 and not eng.futures_short:
                    am = eng.available_margin()
                    margin = am * 0.3
                    lev = 2
                    if sell_s >= 35: margin = am * 0.5
                    eng.open_short(price, dt, margin, lev,
                        f"开空 S={sell_s:.0f} {','.join(sell_p[:2])}")
                cooldown = 4

            # === 看多操作: 平空 + 现货加仓 + (可选)开多 ===
            if buy_s >= 18:
                # 1) 平空
                if eng.futures_short:
                    eng.close_short(price, dt,
                        f"平空 B={buy_s:.0f} {','.join(buy_p[:2])}")

                # 2) 现货加仓
                if buy_s >= 28 and eth_r < 0.5:
                    invest = eng.available_usdt() * 0.2
                    eng.spot_buy(price, dt, invest,
                        f"加仓 B={buy_s:.0f} {','.join(buy_p[:2])}")

                # 3) 开多合约(只在强信号+上升趋势)
                if buy_s >= 35 and not is_downtrend and not eng.futures_long:
                    margin = eng.available_margin() * 0.25
                    eng.open_long(price, dt, margin, 2,
                        f"开多 B={buy_s:.0f} {','.join(buy_p[:2])}")
                cooldown = 6

            # 合约止盈/止损
            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.5:
                    eng.close_short(price, dt, f"空仓止盈{pnl_r*100:.0f}%")
                    cooldown = 3
                elif pnl_r < -0.35:
                    eng.close_short(price, dt, f"空仓止损{pnl_r*100:.0f}%")
                    cooldown = 4

            if eng.futures_long:
                pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
                if pnl_r > 0.5:
                    eng.close_long(price, dt, f"多仓止盈{pnl_r*100:.0f}%")
                    cooldown = 3
                elif pnl_r < -0.35:
                    eng.close_long(price, dt, f"多仓止损{pnl_r*100:.0f}%")
                    cooldown = 4

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_price, last_dt, "结束平多")
    return eng.get_result(main_df)


def run_strategy_P(data, signals_all):
    """策略P: 自适应杠杆
    信号强度决定方向和杠杆倍数:
    - 弱信号: 1x现货操作
    - 中等信号: 2x合约
    - 强信号: 3x合约
    严格止损保护"""
    eng = FuturesEngine("P: 自适应杠杆", max_leverage=3, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val; break

        sig = _merge_signals(signals_all, dt)
        sell_s, sell_p, buy_s, buy_p = build_sell_buy_scores(sig, ind, main_df, idx, is_downtrend)

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            # === 自适应做空 ===
            if sell_s >= 12:
                # 现货减仓(总是做)
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.15 + sell_s * 0.005, 0.5)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"减仓 S={sell_s:.0f}")

                # 合约做空(信号强度决定杠杆)
                if sell_s >= 20 and not eng.futures_short:
                    am = eng.available_margin()
                    if sell_s >= 40:
                        lev = 3; margin_r = 0.5
                    elif sell_s >= 30:
                        lev = 2; margin_r = 0.4
                    else:
                        lev = 2; margin_r = 0.2
                    margin = am * margin_r
                    eng.open_short(price, dt, margin, lev,
                        f"做空{lev}x S={sell_s:.0f} {','.join(sell_p[:2])}")
                cooldown = 4

            # === 自适应做多 ===
            if buy_s >= 18:
                if eng.futures_short:
                    eng.close_short(price, dt, f"平空 B={buy_s:.0f}")

                if buy_s >= 30 and not eng.futures_long:
                    am = eng.available_margin()
                    if buy_s >= 40:
                        lev = 3; margin_r = 0.4
                    elif buy_s >= 35:
                        lev = 2; margin_r = 0.3
                    else:
                        lev = 2; margin_r = 0.15
                    margin = am * margin_r
                    eng.open_long(price, dt, margin, lev,
                        f"做多{lev}x B={buy_s:.0f} {','.join(buy_p[:2])}")

                if buy_s >= 28 and eth_r < 0.45:
                    invest = eng.available_usdt() * 0.15
                    eng.spot_buy(price, dt, invest,
                        f"加仓 B={buy_s:.0f}")
                cooldown = 6

            # 止盈止损
            for pos_type in ['futures_short', 'futures_long']:
                pos = getattr(eng, pos_type)
                if pos:
                    pnl_r = pos.calc_pnl(price) / pos.margin
                    if pnl_r > 0.4 + 0.1 * pos.leverage:
                        if pos_type == 'futures_short':
                            eng.close_short(price, dt, f"止盈{pnl_r*100:.0f}%")
                        else:
                            eng.close_long(price, dt, f"止盈{pnl_r*100:.0f}%")
                        cooldown = 3
                    elif pnl_r < -(0.25 + 0.05 * pos.leverage):
                        if pos_type == 'futures_short':
                            eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                        else:
                            eng.close_long(price, dt, f"止损{pnl_r*100:.0f}%")
                        cooldown = 5

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_price, last_dt, "结束平多")
    return eng.get_result(main_df)


def run_strategy_Q(data, signals_all):
    """策略Q: 量价+合约最优
    基于之前最优的量价确认逻辑, 增加做空能力:
    - 价升量减 + 顶背离 → 减仓 + 开空
    - 地量 + 底背驰 → 平空 + 开多
    - 保守杠杆(2x), 严格风控"""
    eng = FuturesEngine("Q: 量价合约最优", max_leverage=2, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()

    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val; break

        sig = _merge_signals(signals_all, dt)

        # 量能分析
        avg_vol = vol_ma20.iloc[idx]
        vol5 = vol_ma5.iloc[idx]
        cur_vol = main_df['volume'].iloc[idx]
        vol_ratio = cur_vol / avg_vol if avg_vol and avg_vol > 0 else 1
        is_shrinking = vol5 < avg_vol * 0.6 if avg_vol and vol5 else False
        is_ground_vol = vol_ratio < 0.3
        price_change_5 = (price - main_df['close'].iloc[idx - 5]) / main_df['close'].iloc[idx - 5]
        price_up_vol_down = price_change_5 > 0.01 and is_shrinking

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            # === 看空: 量价确认+背离 ===
            sell_trigger = (sig['top'] >= 15 or sig.get('exhaust_sell') or
                            sig.get('vol_price_up_down', 0) >= 1)

            if sell_trigger:
                # 量价背离加权卖出强度
                intensity = 0
                parts = []
                if sig['top'] >= 15: intensity += sig['top'] / 5; parts.append(f"T={sig['top']:.0f}")
                if price_up_vol_down: intensity += 5; parts.append("价升量减")
                if sig.get('exhaust_sell'): intensity += 8; parts.append("背驰")
                if sig.get('vol_price_up_down', 0): intensity += 3; parts.append("VP↓")
                if is_downtrend: intensity *= 1.3; parts.append("↓")

                # 1) 现货减仓
                if eth_r > min_eth_ratio + 0.05 and intensity >= 5:
                    sr = min(0.12 + intensity * 0.01, 0.5)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr, f"减仓 {','.join(parts[:3])}")

                # 2) 开空(中等以上强度)
                if intensity >= 10 and not eng.futures_short:
                    am = eng.available_margin()
                    margin = am * min(0.15 + intensity * 0.01, 0.5)
                    eng.open_short(price, dt, margin, 2,
                        f"开空 i={intensity:.0f} {','.join(parts[:3])}")
                cooldown = 4

            # === 看多: 地量/底背驰 ===
            buy_trigger = (sig.get('exhaust_buy') or
                           (is_ground_vol and sig['bottom'] >= 30) or
                           sig['bottom'] >= 50)

            if buy_trigger:
                # 平空
                if eng.futures_short:
                    eng.close_short(price, dt, f"平空 B={sig['bottom']:.0f}")

                # 加仓
                if sig.get('exhaust_buy') and eth_r < 0.4:
                    invest = eng.available_usdt() * 0.15
                    eng.spot_buy(price, dt, invest,
                        f"加仓 背驰+{'地量' if is_ground_vol else ''}")
                    cooldown = 10
                elif sig['bottom'] >= 50 and eth_r < 0.35:
                    invest = eng.available_usdt() * 0.1
                    eng.spot_buy(price, dt, invest,
                        f"加仓 B={sig['bottom']:.0f}")
                    cooldown = 10

            # 止盈止损
            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.5:
                    eng.close_short(price, dt, f"空仓止盈{pnl_r*100:.0f}%")
                    cooldown = 3
                elif pnl_r < -0.35:
                    eng.close_short(price, dt, f"空仓止损{pnl_r*100:.0f}%")
                    cooldown = 5

            if eng.futures_long:
                pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
                if pnl_r > 0.5:
                    eng.close_long(price, dt, f"多仓止盈{pnl_r*100:.0f}%")
                    cooldown = 3
                elif pnl_r < -0.35:
                    eng.close_long(price, dt, f"多仓止损{pnl_r*100:.0f}%")
                    cooldown = 5

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_price, last_dt, "结束平多")
    return eng.get_result(main_df)


def _merge_signals(signals_all, dt):
    """合并多周期信号"""
    sig = {k: (0 if isinstance(v, (int, float)) else (False if isinstance(v, bool) else ''))
           for k, v in DEFAULT_SIG.items()}
    tw = 0
    weights = {'4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3}
    for tf, w in weights.items():
        if tf in signals_all:
            s_tf = get_signal_at(signals_all[tf], dt)
            if not s_tf: continue
            sig['top'] += s_tf.get('top', 0) * w
            sig['bottom'] += s_tf.get('bottom', 0) * w
            for k in ['exhaust_sell', 'exhaust_buy']:
                if s_tf.get(k): sig[k] = True
            for k in DEFAULT_SIG:
                if isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                    sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
                elif isinstance(DEFAULT_SIG[k], str) and s_tf.get(k):
                    sig[k] = s_tf[k]
            tw += w
    if tw > 0: sig['top'] /= tw; sig['bottom'] /= tw
    return sig


# ======================================================
#   现货对照组(无合约)
# ======================================================
def run_baseline_spot(data, signals_all):
    """对照: 纯现货量价确认策略(J)"""
    eng = FuturesEngine("对照: 纯现货(J策略)", use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()
    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val; break

        sig = _merge_signals(signals_all, dt)
        avg_vol = vol_ma20.iloc[idx]
        vol5 = vol_ma5.iloc[idx]
        is_shrinking = vol5 < avg_vol * 0.6 if avg_vol and vol5 else False
        price_change_5 = (price - main_df['close'].iloc[idx - 5]) / main_df['close'].iloc[idx - 5]
        price_up_vol_down = price_change_5 > 0.01 and is_shrinking

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            sell_trigger = (sig['top'] >= 15 or sig.get('exhaust_sell') or
                            sig.get('vol_price_up_down', 0) >= 1)
            if sell_trigger and eth_r > min_eth_ratio + 0.05:
                intensity = 0
                if sig['top'] >= 15: intensity += sig['top'] / 5
                if price_up_vol_down: intensity += 5
                if sig.get('exhaust_sell'): intensity += 8
                if is_downtrend: intensity *= 1.3

                sr = min(0.12 + intensity * 0.01, 0.5)
                sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 T={sig['top']:.0f}")
                    cooldown = 4

            buy_trigger = sig.get('exhaust_buy') or sig['bottom'] >= 50
            if buy_trigger and eth_r < 0.4:
                invest = eng.available_usdt() * 0.15
                eng.spot_buy(price, dt, invest, f"加仓 B={sig['bottom']:.0f}")
                cooldown = 10

        if idx % 4 == 0:
            eng.record_history(dt, price)

    return eng.get_result(main_df)


# ======================================================
def run_all():
    data = fetch_all_data()

    print("\n计算各周期增强信号...")
    signal_windows = {'1h': 168, '2h': 150, '4h': 120, '6h': 100, '8h': 90}
    signals_all = {}
    for tf, df in data.items():
        w = signal_windows.get(tf, 120)
        if len(df) > w:
            signals_all[tf] = analyze_signals_enhanced(df, w)
            print(f"  {tf}: {len(signals_all[tf])} 个信号点")

    strategies = [
        ("对照", run_baseline_spot),
        ("M", run_strategy_M),
        ("N", run_strategy_N),
        ("O", run_strategy_O),
        ("P", run_strategy_P),
        ("Q", run_strategy_Q),
    ]

    print(f"\n{'='*110}")
    print("  运行 {0} 种合约策略...".format(len(strategies)))
    print(f"  初始: 100,000 USDT + 价值100,000 USDT的ETH")
    print(f"  合约规则: 逐仓/Taker 0.05%/资金费率 0.01%每8h/维持保证金5%")
    print(f"{'='*110}")

    results = []
    for name, func in strategies:
        print(f"\n>>> 策略 {name}...")
        r = func(data, signals_all)
        results.append(r)
        liq_str = f" 强平:{r['liquidations']}次" if r['liquidations'] > 0 else ""
        print(f"    收益: {r['strategy_return']:+.2f}% | 超额: {r['alpha']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔{liq_str} | "
              f"资产: ${r['final_total']:,.0f}")

    print(f"\n\n{'='*120}")
    print("                      合约策略排名 (按超额收益)")
    print(f"{'='*120}")
    fmt = "{:>3} {:<34} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>12}"
    print(fmt.format("#", "策略", "策略收益", "买入持有", "超额收益", "最大回撤",
                      "交易数", "强平", "最终资产"))
    print("-" * 120)
    for rank, r in enumerate(sorted(results, key=lambda x: x['alpha'], reverse=True), 1):
        star = " ★" if rank == 1 else ""
        print(fmt.format(
            rank, r['name'] + star,
            f"{r['strategy_return']:+.2f}%",
            f"{r['buy_hold_return']:+.2f}%",
            f"{r['alpha']:+.2f}%",
            f"{r['max_drawdown']:.2f}%",
            str(r['total_trades']),
            str(r['liquidations']),
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 120)

    output = {
        'futures_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results],
        'best_strategy': sorted(results, key=lambda x: x['alpha'], reverse=True)[0]['name'],
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_futures_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {output_path}")
    return output


if __name__ == '__main__':
    run_all()
