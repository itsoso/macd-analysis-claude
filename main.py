"""
《背离技术分析》代码实现 - 主程序入口
基于 江南小隐 著 《背离技术分析》

使用方法:
    1. 分析指定股票:
       python main.py --symbol 000001 --start 2024-01-01

    2. 使用模拟数据演示:
       python main.py --demo

    3. 从CSV加载:
       python main.py --csv data.csv

    4. 分析并保存图表:
       python main.py --symbol 600519 --save chart.png
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_stock_data, load_csv_data, generate_demo_data
from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer
from visualization import plot_comprehensive_analysis


def run_analysis(df, title='背离技术分析', save_path=None):
    """执行完整的背离分析流程"""

    print("\n" + "=" * 70)
    print(f"  开始分析: {title}")
    print(f"  数据范围: {df.index[0]} ~ {df.index[-1]} ({len(df)}条)")
    print("=" * 70)

    # 步骤1: 计算所有技术指标
    print("\n[1/4] 计算技术指标 (MACD, KDJ, CCI, RSI, 均线)...")
    df = add_all_indicators(df)

    # 步骤2: 执行综合背离分析
    print("[2/4] 执行综合背离分析...")
    analyzer = ComprehensiveAnalyzer(df)
    results = analyzer.analyze_all()

    # 步骤3: 输出分析报告
    print("[3/4] 生成分析报告...\n")
    analyzer.print_report(results)

    # 步骤4: 生成可视化图表
    print("\n[4/4] 生成可视化图表...")
    chart_path = save_path or 'divergence_analysis.png'
    plot_comprehensive_analysis(df, results, title=title, save_path=chart_path)

    return df, results


def main():
    parser = argparse.ArgumentParser(
        description='《背离技术分析》代码实现 - 综合背离分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --demo                          # 使用模拟数据演示
  python main.py --symbol 000001                 # 分析平安银行
  python main.py --symbol 600519 --start 2024-01-01  # 分析贵州茅台
  python main.py --csv data.csv --save output.png    # 从CSV分析并保存图表

分析模块说明 (基于《背离技术分析》全书):
  第二章: 几何形态背离 - 趋势K线/幅度/时间度/均线/均线面积背离
  第三章: MACD指标背离 - 黄白线/彩柱线长度(当堆/邻堆/隔堆)/面积背离
  第四章: 背离与背驰   - 卖点用背离, 买点用背驰
  第五章: 其他背离     - KDJ/CCI/RSI/量价背离
        """
    )

    parser.add_argument('--symbol', type=str, help='股票代码 (如 000001, 600519)')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--period', type=str, default='daily',
                        choices=['daily', 'weekly', 'monthly'],
                        help='K线周期 (default: daily)')
    parser.add_argument('--csv', type=str, help='从CSV文件加载数据')
    parser.add_argument('--demo', action='store_true', help='使用模拟数据演示')
    parser.add_argument('--save', type=str, help='图表保存路径')

    args = parser.parse_args()

    # 获取数据
    if args.demo:
        print("使用模拟数据进行演示分析...")
        df = generate_demo_data(300)
        title = '背离技术分析 - 模拟数据演示'
    elif args.csv:
        print(f"从CSV加载数据: {args.csv}")
        df = load_csv_data(args.csv)
        title = f'背离技术分析 - {args.csv}'
    elif args.symbol:
        print(f"获取 {args.symbol} 的行情数据...")
        df = fetch_stock_data(args.symbol, args.start, args.end, args.period)
        title = f'背离技术分析 - {args.symbol}'
    else:
        print("未指定数据源, 使用模拟数据进行演示...")
        print("提示: 使用 --help 查看所有选项\n")
        df = generate_demo_data(300)
        title = '背离技术分析 - 模拟数据演示'

    if df is None or len(df) < 50:
        print("错误: 数据不足, 至少需要50条记录")
        sys.exit(1)

    # 执行分析
    run_analysis(df, title=title, save_path=args.save)
    print("\n分析完成!")


if __name__ == '__main__':
    main()
