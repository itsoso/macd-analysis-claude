"""
Flask Web 应用
展示: 书籍大纲 + 代码实现说明 + ETH/USDT 多周期回测结果对比 + 策略优化对比
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_FILE = os.path.join(BASE_DIR, 'backtest_result.json')
BACKTEST_MULTI_FILE = os.path.join(BASE_DIR, 'backtest_multi.json')
GLOBAL_STRATEGY_FILE = os.path.join(BASE_DIR, 'global_strategy_result.json')
STRATEGY_COMPARE_FILE = os.path.join(BASE_DIR, 'strategy_compare_result.json')
STRATEGY_OPTIMIZE_FILE = os.path.join(BASE_DIR, 'strategy_optimize_result.json')
STRATEGY_ENHANCED_FILE = os.path.join(BASE_DIR, 'strategy_enhanced_result.json')


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/backtest')
def api_backtest():
    """返回单周期回测数据 (兼容旧逻辑)"""
    data = load_json(BACKTEST_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到回测数据'}), 404


@app.route('/api/backtest_multi')
def api_backtest_multi():
    """返回全部12周期回测数据"""
    data = load_json(BACKTEST_MULTI_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到多周期回测数据, 请先生成'}), 404


@app.route('/api/global_strategy')
def api_global_strategy():
    """返回全局多周期融合策略结果"""
    data = load_json(GLOBAL_STRATEGY_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到全局策略数据, 请先运行 python global_strategy.py'}), 404


@app.route('/api/strategy_compare')
def api_strategy_compare():
    """返回6种策略变体对比数据"""
    data = load_json(STRATEGY_COMPARE_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到策略对比数据, 请先运行 python strategy_compare.py'}), 404


@app.route('/api/strategy_optimize')
def api_strategy_optimize():
    """返回策略优化变体对比数据"""
    data = load_json(STRATEGY_OPTIMIZE_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到优化数据, 请先运行 python strategy_optimize.py'}), 404


@app.route('/api/strategy_enhanced')
def api_strategy_enhanced():
    """返回深度指标增强策略结果"""
    data = load_json(STRATEGY_ENHANCED_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到增强策略数据, 请先运行 python strategy_enhanced.py'}), 404


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  《背离技术分析》Web 展示平台")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
