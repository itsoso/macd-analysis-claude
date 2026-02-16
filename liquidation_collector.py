"""
Phase 3c: 清算流数据采集

通过 Binance WebSocket 实时采集强制清算数据 (<symbol>@forceOrder),
聚合到 1h 级别供策略使用。

数据字段:
- liq_notional_1h: 1h 内清算总名义价值 (USDT)
- liq_imbalance_1h: 1h 内清算多空失衡 ((long_liq - short_liq) / total_liq)
- liq_count_1h: 1h 内清算次数

存储: data/liquidations/ 目录, Parquet 格式

使用方式:
    # 启动采集 (后台运行)
    python liquidation_collector.py --symbol ETHUSDT

    # 查看已采集数据
    python liquidation_collector.py --symbol ETHUSDT --show

    # 导出指定时间段
    python liquidation_collector.py --symbol ETHUSDT --export --start 2026-01-01 --end 2026-02-01
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# 数据存储目录
_LIQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'liquidations')


class LiquidationCollector:
    """
    币安永续合约清算流采集器。

    通过 WebSocket 订阅 <symbol>@forceOrder 流:
    - 实时采集每一笔强制清算事件
    - 聚合到 1h 级别 (notional, imbalance, count)
    - 定期持久化到 Parquet 文件
    """

    def __init__(self, symbol='ETHUSDT', flush_interval_sec=300):
        """
        Parameters
        ----------
        symbol : str
            交易对 (默认 ETHUSDT)
        flush_interval_sec : int
            持久化间隔 (秒, 默认 300)
        """
        self.symbol = symbol.upper()
        self.flush_interval = flush_interval_sec

        # 缓冲区: {hour_key: {long_notional, short_notional, count}}
        self._buffer = defaultdict(lambda: {
            'long_notional': 0.0,
            'short_notional': 0.0,
            'count': 0,
            'long_count': 0,
            'short_count': 0,
            'max_single': 0.0,  # 最大单笔清算
        })
        self._lock = threading.Lock()
        self._running = False
        self._ws = None
        self._last_flush = time.time()

        os.makedirs(_LIQ_DIR, exist_ok=True)

    def _on_message(self, message):
        """处理 WebSocket 消息。"""
        try:
            data = json.loads(message)
            order = data.get('o', {})

            side = order.get('S', '')  # SELL = long被清算, BUY = short被清算
            price = float(order.get('p', 0))
            qty = float(order.get('q', 0))
            notional = price * qty

            now = datetime.utcnow()
            hour_key = now.strftime('%Y-%m-%d %H:00:00')

            with self._lock:
                buf = self._buffer[hour_key]
                buf['count'] += 1

                if side == 'SELL':
                    # SELL = 强制卖出 = 多头被清算
                    buf['long_notional'] += notional
                    buf['long_count'] += 1
                elif side == 'BUY':
                    # BUY = 强制买入 = 空头被清算
                    buf['short_notional'] += notional
                    buf['short_count'] += 1

                buf['max_single'] = max(buf['max_single'], notional)

            logger.debug(f"LIQ: {side} {qty:.4f} @ {price:.2f} = ${notional:.0f}")

        except Exception as e:
            logger.warning(f"LIQ parse error: {e}")

    def _flush_to_disk(self):
        """将缓冲区持久化到 Parquet 文件。"""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available, skipping flush")
            return

        with self._lock:
            if not self._buffer:
                return
            records = []
            for hour_key, buf in sorted(self._buffer.items()):
                total = buf['long_notional'] + buf['short_notional']
                imbalance = 0.0
                if total > 0:
                    imbalance = (buf['long_notional'] - buf['short_notional']) / total
                records.append({
                    'timestamp': hour_key,
                    'long_notional': round(buf['long_notional'], 2),
                    'short_notional': round(buf['short_notional'], 2),
                    'total_notional': round(total, 2),
                    'imbalance': round(imbalance, 6),
                    'count': buf['count'],
                    'long_count': buf['long_count'],
                    'short_count': buf['short_count'],
                    'max_single': round(buf['max_single'], 2),
                })

        if not records:
            return

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # 追加写入 Parquet
        filepath = os.path.join(_LIQ_DIR, f'{self.symbol}_liquidations.parquet')
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            # 合并, 新数据覆盖旧数据 (同一小时)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_parquet(filepath)
        else:
            df.to_parquet(filepath)

        logger.info(f"LIQ flush: {len(records)} hours → {filepath}")
        self._last_flush = time.time()

    def start(self, blocking=True):
        """
        启动 WebSocket 采集。

        Parameters
        ----------
        blocking : bool
            是否阻塞 (默认 True, 适合独立运行)
        """
        try:
            import websocket
        except ImportError:
            print("需要安装 websocket-client: pip install websocket-client")
            return

        self._running = True
        stream = f"{self.symbol.lower()}@forceOrder"
        url = f"wss://fstream.binance.com/ws/{stream}"

        logger.info(f"LIQ collector starting: {url}")

        def on_open(ws):
            logger.info(f"LIQ WebSocket connected: {stream}")

        def on_close(ws, close_code, close_msg):
            logger.info(f"LIQ WebSocket closed: {close_code} {close_msg}")

        def on_error(ws, error):
            logger.error(f"LIQ WebSocket error: {error}")

        def on_message(ws, message):
            self._on_message(message)
            # 定期 flush
            if time.time() - self._last_flush > self.flush_interval:
                self._flush_to_disk()

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_close=on_close,
            on_error=on_error,
            on_message=on_message,
        )

        if blocking:
            # 注册信号处理, 优雅退出时 flush
            def _signal_handler(sig, frame):
                logger.info("LIQ collector stopping...")
                self._running = False
                self._flush_to_disk()
                if self._ws:
                    self._ws.close()
                sys.exit(0)

            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)

            # 自动重连循环
            while self._running:
                try:
                    self._ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.error(f"LIQ WebSocket crashed: {e}")
                if self._running:
                    logger.info("LIQ reconnecting in 5s...")
                    time.sleep(5)
        else:
            thread = threading.Thread(target=self._ws.run_forever,
                                      kwargs={'ping_interval': 30, 'ping_timeout': 10},
                                      daemon=True)
            thread.start()

    def stop(self):
        """停止采集并 flush。"""
        self._running = False
        self._flush_to_disk()
        if self._ws:
            self._ws.close()

    @staticmethod
    def load_data(symbol='ETHUSDT', start=None, end=None):
        """
        加载已采集的清算数据。

        Parameters
        ----------
        symbol : str
            交易对
        start : str or datetime
            起始时间
        end : str or datetime
            结束时间

        Returns
        -------
        pd.DataFrame or None
        """
        import pandas as pd
        filepath = os.path.join(_LIQ_DIR, f'{symbol.upper()}_liquidations.parquet')
        if not os.path.exists(filepath):
            return None

        df = pd.read_parquet(filepath)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='清算流数据采集')
    parser.add_argument('--symbol', default='ETHUSDT', help='交易对')
    parser.add_argument('--show', action='store_true', help='查看已采集数据')
    parser.add_argument('--export', action='store_true', help='导出数据')
    parser.add_argument('--start', type=str, default=None, help='起始时间')
    parser.add_argument('--end', type=str, default=None, help='结束时间')
    parser.add_argument('--flush-interval', type=int, default=300,
                        help='持久化间隔 (秒)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    if args.show or args.export:
        df = LiquidationCollector.load_data(args.symbol, args.start, args.end)
        if df is None or len(df) == 0:
            print(f"无数据: {args.symbol}")
            return
        print(f"\n{args.symbol} 清算数据: {len(df)} 小时")
        print(f"时间范围: {df.index[0]} → {df.index[-1]}")
        print(f"\n最近 24h:")
        print(df.tail(24).to_string())
        if args.export:
            out = os.path.join(_LIQ_DIR, f'{args.symbol}_liquidations.csv')
            df.to_csv(out)
            print(f"\n已导出: {out}")
    else:
        print(f"启动清算流采集: {args.symbol}")
        print("按 Ctrl+C 停止")
        collector = LiquidationCollector(args.symbol,
                                          flush_interval_sec=args.flush_interval)
        collector.start(blocking=True)


if __name__ == '__main__':
    main()
