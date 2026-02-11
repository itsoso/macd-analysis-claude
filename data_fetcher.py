"""
数据获取模块
支持通过 akshare 获取A股/港股数据, 也支持从CSV文件加载
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_stock_data(symbol: str,
                     start_date: str = None,
                     end_date: str = None,
                     period: str = 'daily',
                     adjust: str = 'qfq') -> pd.DataFrame:
    """
    获取股票历史行情数据

    参数:
        symbol: 股票代码 (如 '000001' 或 'sh600519')
        start_date: 开始日期 'YYYY-MM-DD' (默认1年前)
        end_date: 结束日期 'YYYY-MM-DD' (默认今天)
        period: 周期 'daily'/'weekly'/'monthly'
        adjust: 复权方式 'qfq'(前复权) / 'hfq'(后复权) / '' (不复权)
                书中特别说明: 所有K线图表都系前复权(qfq)

    返回: DataFrame, 包含 open, high, low, close, volume 列, 以日期为索引
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    else:
        start_date = start_date.replace('-', '')

    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    else:
        end_date = end_date.replace('-', '')

    try:
        import akshare as ak

        # 判断是否是指数
        if symbol.startswith('sh') or symbol.startswith('sz'):
            # 带前缀的代码
            clean_symbol = symbol[2:]
        else:
            clean_symbol = symbol

        # 尝试获取A股日线数据
        try:
            df = ak.stock_zh_a_hist(
                symbol=clean_symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
        except Exception:
            # 如果失败, 尝试获取指数数据
            df = ak.stock_zh_index_daily(symbol=f"sh{clean_symbol}")
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]

        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
        }
        df = df.rename(columns=column_mapping)

        # 确保必要列存在
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # 尝试小写匹配
                for c in df.columns:
                    if c.lower() == col:
                        df = df.rename(columns={c: col})
                        break

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df.sort_index()

        print(f"成功获取 {symbol} 数据: {len(df)} 条记录 "
              f"({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")

        return df

    except ImportError:
        print("提示: 未安装akshare, 请运行 pip install akshare")
        print("将使用模拟数据进行演示...")
        return generate_demo_data()
    except Exception as e:
        print(f"获取数据失败: {e}")
        print("将使用模拟数据进行演示...")
        return generate_demo_data()


def load_csv_data(filepath: str,
                  date_col: str = 'date',
                  encoding: str = 'utf-8') -> pd.DataFrame:
    """
    从CSV文件加载股票数据

    参数:
        filepath: CSV文件路径
        date_col: 日期列名
        encoding: 文件编码

    返回: 标准化的DataFrame
    """
    df = pd.read_csv(filepath, encoding=encoding)

    # 尝试识别日期列
    date_candidates = ['date', 'Date', '日期', 'datetime', 'time', 'trade_date']
    for col in date_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break

    # 标准化列名
    col_map = {
        'Open': 'open', '开盘价': 'open', '开盘': 'open',
        'High': 'high', '最高价': 'high', '最高': 'high',
        'Low': 'low', '最低价': 'low', '最低': 'low',
        'Close': 'close', '收盘价': 'close', '收盘': 'close',
        'Volume': 'volume', '成交量': 'volume', 'vol': 'volume',
    }
    df = df.rename(columns=col_map)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.sort_index()

    print(f"从CSV加载数据: {len(df)} 条记录")
    return df


def generate_demo_data(days: int = 300) -> pd.DataFrame:
    """
    生成模拟数据用于演示
    包含: 上涨段 -> 顶背离 -> 下跌段 -> 底背离 -> 反弹
    """
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')

    # 生成一个含有明显背离特征的价格序列
    # 阶段1: 上涨 (0-80)
    # 阶段2: 继续上涨但动能减弱(背离) (80-130)
    # 阶段3: 下跌 (130-200)
    # 阶段4: 继续下跌但动能减弱(底背离) (200-260)
    # 阶段5: 反弹 (260-300)

    price = np.zeros(days)
    price[0] = 10.0

    for i in range(1, days):
        noise = np.random.randn() * 0.02

        if i < 80:
            drift = 0.003 + noise
        elif i < 110:
            drift = 0.002 + noise  # 动能减弱
        elif i < 130:
            drift = 0.001 + noise  # 更弱, 形成顶背离
        elif i < 160:
            drift = -0.004 + noise  # 下跌
        elif i < 200:
            drift = -0.003 + noise  # 继续下跌
        elif i < 240:
            drift = -0.001 + noise  # 下跌减弱
        elif i < 260:
            drift = -0.0005 + noise  # 底背离
        else:
            drift = 0.003 + noise  # 反弹

        price[i] = price[i - 1] * (1 + drift)
        price[i] = max(price[i], 1.0)

    # 生成OHLCV
    high = price * (1 + np.abs(np.random.randn(days)) * 0.01)
    low = price * (1 - np.abs(np.random.randn(days)) * 0.01)
    open_price = price * (1 + np.random.randn(days) * 0.005)

    # 成交量: 上涨放量, 顶部缩量, 下跌缩量, 底部地量
    base_volume = 1000000
    volume = np.zeros(days)
    for i in range(days):
        if i < 80:
            volume[i] = base_volume * (1 + i / 80)
        elif i < 130:
            volume[i] = base_volume * (2 - (i - 80) / 50 * 0.8)  # 缩量
        elif i < 200:
            volume[i] = base_volume * (1.2 - (i - 130) / 70 * 0.5)
        elif i < 260:
            volume[i] = base_volume * 0.3  # 地量
        else:
            volume[i] = base_volume * (0.5 + (i - 260) / 40 * 1.5)  # 放量反弹

        volume[i] *= (1 + np.random.randn() * 0.3)
        volume[i] = max(volume[i], 100000)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume.astype(int)
    }, index=dates)

    # 确保high >= max(open, close) 和 low <= min(open, close)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    print(f"已生成模拟数据: {len(df)} 条记录 (含上涨-背离-下跌-底背离-反弹)")
    return df
