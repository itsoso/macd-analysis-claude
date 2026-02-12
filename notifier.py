"""
通知系统 - Telegram Bot 推送交易通知
支持: 交易通知 / 风险告警 / 每日总结 / 错误报告
"""

import json
import os
import time
import traceback
from datetime import datetime
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class TelegramNotifier:
    """Telegram Bot 通知器"""

    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str = "", chat_id: str = "",
                 enabled: bool = False):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self._last_send_time = 0
        self._min_interval = 1.0  # 最小发送间隔(秒)，防止频率限制
        self._error_count = 0
        self._max_errors = 10     # 连续错误超过此数则禁用

    def _send(self, text: str, parse_mode: str = "HTML",
              disable_notification: bool = False) -> bool:
        """发送消息到 Telegram"""
        if not self.enabled or not HAS_REQUESTS:
            return False

        if self._error_count >= self._max_errors:
            return False

        # 频率限制
        now = time.time()
        elapsed = now - self._last_send_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            url = self.TELEGRAM_API.format(token=self.bot_token, method="sendMessage")
            # Telegram 消息最大 4096 字符
            if len(text) > 4000:
                text = text[:3997] + "..."

            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }, timeout=10)

            self._last_send_time = time.time()

            if resp.status_code == 200:
                self._error_count = 0
                return True
            else:
                self._error_count += 1
                return False

        except Exception:
            self._error_count += 1
            return False

    # ============================================================
    # 交易通知
    # ============================================================
    def notify_trade(self, action: str, symbol: str, side: str,
                     price: float, qty: float, margin: float = 0,
                     leverage: int = 0, fee: float = 0,
                     pnl: float = 0, reason: str = ""):
        """交易执行通知"""
        # 根据 action 选择 emoji
        emoji_map = {
            "OPEN_LONG": "🟢", "OPEN_SHORT": "🔴",
            "CLOSE_LONG": "💰", "CLOSE_SHORT": "💰",
            "PARTIAL_TP": "✂️", "STOP_LOSS": "🛑",
            "LIQUIDATION": "💀", "PAPER_TRADE": "📝",
        }
        emoji = emoji_map.get(action, "📊")
        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else ""

        text = (
            f"{emoji} <b>{action}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"交易对: <code>{symbol}</code>\n"
            f"方向: <b>{side}</b>\n"
            f"价格: <code>${price:.2f}</code>\n"
            f"数量: <code>{qty:.4f}</code>\n"
        )
        if margin > 0:
            text += f"保证金: <code>${margin:.2f}</code>\n"
        if leverage > 0:
            text += f"杠杆: <code>{leverage}x</code>\n"
        if fee > 0:
            text += f"手续费: <code>${fee:.2f}</code>\n"
        if pnl != 0:
            text += f"盈亏: <code>${pnl:+.2f}</code> {pnl_emoji}\n"
        if reason:
            text += f"原因: {reason}\n"
        text += f"时间: <code>{datetime.now():%Y-%m-%d %H:%M:%S}</code>"

        self._send(text)

    # ============================================================
    # 风险告警
    # ============================================================
    def notify_risk(self, event_type: str, message: str,
                    current_value: float = 0, threshold: float = 0,
                    action: str = ""):
        """风险事件告警 - 高优先级"""
        severity_map = {
            "LIQUIDATION": "🚨🚨🚨",
            "KILL_SWITCH": "🚨🚨🚨",
            "CIRCUIT_BREAKER": "🚨🚨",
            "MAX_LOSS_DAILY": "🚨🚨",
            "MAX_LOSS_WEEKLY": "🚨🚨",
            "CONSECUTIVE_LOSS": "⚠️⚠️",
            "DRAWDOWN_ALERT": "⚠️⚠️",
            "MARGIN_WARNING": "⚠️",
            "STOP_LOSS": "⚠️",
            "SLIPPAGE_HIGH": "⚠️",
        }
        emoji = severity_map.get(event_type, "⚠️")

        text = (
            f"{emoji} <b>风险告警: {event_type}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"详情: {message}\n"
            f"当前值: <code>{current_value:.4f}</code>\n"
            f"阈值: <code>{threshold:.4f}</code>\n"
        )
        if action:
            text += f"<b>执行动作: {action}</b>\n"
        text += f"时间: <code>{datetime.now():%Y-%m-%d %H:%M:%S}</code>"

        self._send(text, disable_notification=False)

    # ============================================================
    # 每日总结
    # ============================================================
    def notify_daily_summary(self, date: str, equity: float,
                             daily_pnl: float, daily_return: float,
                             trades_count: int, wins: int, losses: int,
                             max_drawdown: float = 0,
                             positions: list = None,
                             extra: dict = None):
        """每日交易总结"""
        pnl_emoji = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "➡️"

        text = (
            f"📋 <b>每日总结 {date}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"总权益: <code>${equity:.2f}</code>\n"
            f"日盈亏: <code>${daily_pnl:+.2f}</code> "
            f"(<code>{daily_return:+.2%}</code>) {pnl_emoji}\n"
            f"交易次数: <code>{trades_count}</code>\n"
            f"胜/负: <code>{wins}/{losses}</code>"
        )
        if trades_count > 0:
            text += f" (胜率: <code>{wins / trades_count:.0%}</code>)"
        text += "\n"

        if max_drawdown > 0:
            text += f"最大回撤: <code>{max_drawdown:.2%}</code>\n"

        if positions:
            text += "\n<b>当前持仓:</b>\n"
            for p in positions:
                text += (
                    f"  {p['side']}: 入场 ${p['entry_price']:.2f} "
                    f"浮盈 ${p['pnl']:+.2f}\n"
                )

        if extra:
            text += "\n"
            for k, v in extra.items():
                text += f"{k}: <code>{v}</code>\n"

        self._send(text, disable_notification=True)

    # ============================================================
    # 系统通知
    # ============================================================
    def notify_system(self, event: str, message: str):
        """系统事件通知 (启动/停止/错误)"""
        emoji_map = {
            "START": "🟢", "STOP": "🔴", "ERROR": "❌",
            "RESTART": "🔄", "CONFIG_CHANGE": "⚙️",
        }
        emoji = emoji_map.get(event, "ℹ️")

        text = (
            f"{emoji} <b>系统: {event}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{message}\n"
            f"时间: <code>{datetime.now():%Y-%m-%d %H:%M:%S}</code>"
        )
        self._send(text)

    def notify_error(self, error: Exception, context: str = ""):
        """错误通知"""
        tb = traceback.format_exc()
        text = (
            f"❌ <b>错误报告</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"上下文: {context}\n"
            f"错误: <code>{type(error).__name__}: {str(error)[:500]}</code>\n"
            f"堆栈:\n<pre>{tb[-500:]}</pre>\n"
            f"时间: <code>{datetime.now():%Y-%m-%d %H:%M:%S}</code>"
        )
        self._send(text)

    # ============================================================
    # 信号通知 (可选，可能频繁)
    # ============================================================
    def notify_signal(self, sell_score: float, buy_score: float,
                      action: str, price: float, symbol: str = "ETHUSDT"):
        """信号通知 (仅在高分信号时发送)"""
        if action == "HOLD":
            return  # HOLD 不通知

        text = (
            f"📡 <b>信号: {action}</b>\n"
            f"交易对: <code>{symbol}</code> | "
            f"价格: <code>${price:.2f}</code>\n"
            f"SS={sell_score:.1f} BS={buy_score:.1f}"
        )
        self._send(text, disable_notification=True)

    def test_connection(self) -> bool:
        """测试 Telegram 连接"""
        if not self.enabled:
            print("[Telegram] 未启用")
            return False

        result = self._send(
            "🔗 <b>连接测试成功</b>\n"
            f"MACD Analysis 实盘系统\n"
            f"时间: <code>{datetime.now():%Y-%m-%d %H:%M:%S}</code>"
        )
        if result:
            print("[Telegram] 连接测试成功 ✓")
        else:
            print("[Telegram] 连接测试失败 ✗")
        return result


class DummyNotifier:
    """空通知器 - 当 Telegram 未配置时使用"""

    def notify_trade(self, *args, **kwargs): pass
    def notify_risk(self, *args, **kwargs): pass
    def notify_daily_summary(self, *args, **kwargs): pass
    def notify_system(self, *args, **kwargs): pass
    def notify_error(self, *args, **kwargs): pass
    def notify_signal(self, *args, **kwargs): pass
    def test_connection(self) -> bool: return True


def create_notifier(telegram_config) -> object:
    """根据配置创建通知器"""
    if telegram_config.enabled:
        return TelegramNotifier(
            bot_token=telegram_config.bot_token,
            chat_id=telegram_config.chat_id,
            enabled=True
        )
    return DummyNotifier()
