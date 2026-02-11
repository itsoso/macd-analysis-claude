"""
Flask Web 应用
展示: 书籍大纲 + 代码实现说明 + ETH/USDT 回测结果
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify

app = Flask(__name__)

BACKTEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'backtest_result.json')


def load_backtest_data():
    """加载回测结果"""
    if os.path.exists(BACKTEST_FILE):
        with open(BACKTEST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/backtest')
def api_backtest():
    """返回回测数据"""
    data = load_backtest_data()
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到回测数据, 请先运行 python backtest.py'}), 404


@app.route('/api/run_backtest')
def api_run_backtest():
    """触发回测"""
    try:
        from backtest import run_eth_backtest
        report = run_eth_backtest(days=30)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  《背离技术分析》Web 展示平台")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
