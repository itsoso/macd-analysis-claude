"""
Flask Web 应用
展示: 书籍大纲 + 代码实现说明 + ETH/USDT 多周期回测结果对比 + 策略优化对比
"""

import glob
import json
import os
import subprocess
import sys
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'macd-analysis-secret-key-2026-change-in-production')

# ======================================================
#   用户认证
# ======================================================
USERS = {
    'admin': generate_password_hash('system123'),
}


def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@app.before_request
def check_auth():
    """全局认证检查 - 除登录页和静态资源外，所有请求都需要认证"""
    # 允许访问登录页面和静态资源
    allowed_endpoints = ('login', 'static')
    if request.endpoint in allowed_endpoints:
        return
    # 未登录
    if not session.get('logged_in'):
        # API 请求返回 JSON 401（而非 HTML 重定向，避免前端 JSON 解析失败）
        if request.path.startswith('/api/'):
            return jsonify({"success": False, "error": "未登录", "login_required": True}), 401
        return redirect(url_for('login', next=request.url))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username in USERS and check_password_hash(USERS[username], password):
            session['logged_in'] = True
            session['username'] = username
            session.permanent = True
            app.permanent_session_lifetime = __import__('datetime').timedelta(days=7)
            next_url = request.args.get('next') or request.form.get('next') or url_for('page_overview')
            return redirect(next_url)
        else:
            error = '用户名或密码错误'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """登出"""
    session.clear()
    return redirect(url_for('login'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_FILE = os.path.join(BASE_DIR, 'backtest_result.json')
BACKTEST_MULTI_FILE = os.path.join(BASE_DIR, 'backtest_multi.json')
GLOBAL_STRATEGY_FILE = os.path.join(BASE_DIR, 'global_strategy_result.json')
STRATEGY_COMPARE_FILE = os.path.join(BASE_DIR, 'strategy_compare_result.json')
STRATEGY_OPTIMIZE_FILE = os.path.join(BASE_DIR, 'strategy_optimize_result.json')
STRATEGY_ENHANCED_FILE = os.path.join(BASE_DIR, 'strategy_enhanced_result.json')
STRATEGY_FUTURES_FILE = os.path.join(BASE_DIR, 'strategy_futures_result.json')
STRATEGY_FUTURES_V2_FILE = os.path.join(BASE_DIR, 'strategy_futures_v2_result.json')
STRATEGY_FUTURES_V3_FILE = os.path.join(BASE_DIR, 'strategy_futures_v3_result.json')
STRATEGY_FUTURES_V4_FILE = os.path.join(BASE_DIR, 'strategy_futures_v4_result.json')
STRATEGY_FUTURES_V5_FILE = os.path.join(BASE_DIR, 'strategy_futures_v5_result.json')
STRATEGY_FUTURES_FINAL_FILE = os.path.join(BASE_DIR, 'strategy_futures_final_result.json')
TIMEFRAME_ANALYSIS_FILE = os.path.join(BASE_DIR, 'timeframe_analysis_result.json')
STRATEGY_15M_FILE = os.path.join(BASE_DIR, 'strategy_15m_result.json')
MA_STRATEGY_FILE = os.path.join(BASE_DIR, 'ma_strategy_result.json')
COMBINED_STRATEGY_FILE = os.path.join(BASE_DIR, 'combined_strategy_result.json')
CANDLESTICK_FILE = os.path.join(BASE_DIR, 'candlestick_result.json')
BOLLINGER_FILE = os.path.join(BASE_DIR, 'bollinger_result.json')
VOLUME_PRICE_FILE = os.path.join(BASE_DIR, 'volume_price_result.json')
FIVE_BOOK_FILE = os.path.join(BASE_DIR, 'five_book_fusion_result.json')
OPTIMIZE_SL_TP_FILE = os.path.join(BASE_DIR, 'optimize_sl_tp_result.json')
SIX_BOOK_FILE = os.path.join(BASE_DIR, 'six_book_fusion_result.json')
KDJ_FILE = os.path.join(BASE_DIR, 'kdj_result.json')
OPTIMIZE_SIX_BOOK_FILE = os.path.join(BASE_DIR, 'optimize_six_book_result.json')
BACKTEST_30D_7D_FILE = os.path.join(BASE_DIR, 'backtest_30d_7d_result.json')
MULTI_TF_BACKTEST_30D_7D_FILE = os.path.join(BASE_DIR, 'backtest_multi_tf_30d_7d_result.json')


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ======================================================
#   独立页面路由 (每个tab一个独立页面)
# ======================================================
@app.route('/')
def page_overview():
    return render_template('page_overview.html', active_page='overview')


@app.route('/architecture')
def page_architecture():
    return render_template('page_architecture.html', active_page='architecture')


@app.route('/code/indicators')
def page_code_indicators():
    return render_template('page_code_indicators.html', active_page='code-indicators')


@app.route('/code/pattern')
def page_code_pattern():
    return render_template('page_code_pattern.html', active_page='code-pattern')


@app.route('/code/macd')
def page_code_macd():
    return render_template('page_code_macd.html', active_page='code-macd')


@app.route('/code/exhaustion')
def page_code_exhaustion():
    return render_template('page_code_exhaustion.html', active_page='code-exhaustion')


@app.route('/code/other')
def page_code_other():
    return render_template('page_code_other.html', active_page='code-other')


@app.route('/strategy/global')
def page_global_strategy():
    return render_template('page_global_strategy.html', active_page='global-strategy')


@app.route('/strategy/compare')
def page_strategy_compare():
    return render_template('page_strategy_compare.html', active_page='strategy-compare')


@app.route('/strategy/futures')
def page_strategy_futures():
    return render_template('page_strategy_futures.html', active_page='strategy-futures')


@app.route('/strategy/ma')
def page_ma_strategy():
    return render_template('page_ma_strategy.html', active_page='ma-strategy')


@app.route('/strategy/combined')
def page_combined_strategy():
    return render_template('page_combined_strategy.html', active_page='combined-strategy')


@app.route('/strategy/c6-detail')
def page_c6_detail():
    return render_template('page_c6_detail.html', active_page='c6-detail')


@app.route('/strategy/candlestick')
def page_candlestick():
    return render_template('page_candlestick.html', active_page='candlestick')


@app.route('/strategy/bollinger')
def page_bollinger():
    return render_template('page_bollinger.html', active_page='bollinger')


@app.route('/strategy/volume-price')
def page_volume_price():
    return render_template('page_volume_price.html', active_page='volume-price')


@app.route('/strategy/five-book')
def page_five_book():
    return render_template('page_five_book.html', active_page='five-book')


@app.route('/strategy/best')
def page_best_strategy():
    return render_template('page_best_strategy.html', active_page='best-strategy')


@app.route('/strategy/optimize-sl-tp')
def page_optimize_sl_tp():
    return render_template('page_optimize_sl_tp.html', active_page='optimize-sl-tp')


@app.route('/strategy/optimal')
def page_optimal_strategy():
    return render_template('page_optimal_strategy.html', active_page='optimal-strategy')


@app.route('/book/ma')
def page_book_ma():
    return render_template('page_book_ma.html', active_page='book-ma')


@app.route('/book/candlestick')
def page_book_candlestick():
    return render_template('page_book_candlestick.html', active_page='book-candlestick')


@app.route('/book/bollinger')
def page_book_bollinger():
    return render_template('page_book_bollinger.html', active_page='book-bollinger')


@app.route('/book/volume-price')
def page_book_volume_price():
    return render_template('page_book_volume_price.html', active_page='book-volume-price')


@app.route('/book/kdj')
def page_book_kdj():
    return render_template('page_book_kdj.html', active_page='book-kdj')


@app.route('/book/turtle')
def page_book_turtle():
    return render_template('page_book_turtle.html', active_page='book-turtle')


@app.route('/strategy/six-book')
def page_six_book():
    return render_template('page_six_book.html', active_page='six-book')


@app.route('/strategy/optimize-six-book')
def page_optimize_six_book():
    return render_template('page_optimize_six_book.html', active_page='optimize-six-book')


@app.route('/strategy/multi-tf-backtest')
def page_multi_tf_backtest():
    return render_template('page_multi_tf_backtest.html', active_page='multi-tf-backtest')


@app.route('/strategy/multi-tf-deep-dive')
def page_multi_tf_deep_dive():
    return render_template('page_multi_tf_deep_dive.html', active_page='multi-tf-deep-dive')


@app.route('/strategy/multi-tf-backtest-30d-7d')
def page_multi_tf_backtest_30d_7d():
    return render_template('page_multi_tf_backtest_30d_7d.html', active_page='multi-tf-backtest-30d-7d')


@app.route('/strategy/six-book-deep-dive')
def page_six_book_deep_dive():
    return render_template('page_six_book_deep_dive.html', active_page='six-book-deep-dive')


@app.route('/strategy/live-trading-guide')
def page_live_trading_guide():
    return render_template('page_live_trading_guide.html', active_page='live-trading-guide')


@app.route('/strategy/backtest-30d-7d')
def page_backtest_30d_7d():
    return render_template('page_backtest_30d_7d.html', active_page='backtest-30d-7d')


@app.route('/strategy/multi')
def page_multi_compare():
    return render_template('page_multi_compare.html', active_page='multi-compare')


@app.route('/strategy/backtest')
def page_backtest():
    return render_template('page_backtest.html', active_page='backtest')


# ======================================================
#   API 路由 (数据接口)
# ======================================================
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


@app.route('/api/strategy_futures')
def api_strategy_futures():
    """返回合约策略结果"""
    data = load_json(STRATEGY_FUTURES_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到合约策略数据, 请先运行 python strategy_futures.py'}), 404


@app.route('/api/strategy_futures_v2')
def api_strategy_futures_v2():
    """返回合约策略Phase 2进阶优化结果"""
    data = load_json(STRATEGY_FUTURES_V2_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到合约Phase2数据, 请先运行 python strategy_futures_v2.py'}), 404


@app.route('/api/strategy_futures_v3')
def api_strategy_futures_v3():
    """返回合约策略Phase 3深度优化结果"""
    data = load_json(STRATEGY_FUTURES_V3_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到Phase3数据, 请先运行 python strategy_futures_v3.py'}), 404


@app.route('/api/strategy_futures_v4')
def api_strategy_futures_v4():
    """返回合约策略Phase 4引擎修正后优化结果"""
    data = load_json(STRATEGY_FUTURES_V4_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到Phase4数据, 请先运行 python strategy_futures_v4.py'}), 404


@app.route('/api/strategy_futures_v5')
def api_strategy_futures_v5():
    """返回合约策略Phase 5终极优化结果"""
    data = load_json(STRATEGY_FUTURES_V5_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到Phase5数据, 请先运行 python strategy_futures_v5.py'}), 404


@app.route('/api/strategy_futures_final')
def api_strategy_futures_final():
    """返回合约策略Phase 6+7+8终极优化结果"""
    data = load_json(STRATEGY_FUTURES_FINAL_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到终极优化数据, 请先运行 python strategy_futures_final.py'}), 404


@app.route('/api/timeframe_analysis')
def api_timeframe_analysis():
    """返回多时间周期信号价值分析结果"""
    data = load_json(TIMEFRAME_ANALYSIS_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到时间周期分析数据, 请先运行 python strategy_timeframe_analysis.py'}), 404


@app.route('/api/strategy_15m')
def api_strategy_15m():
    """返回15分钟双向回测结果"""
    data = load_json(STRATEGY_15M_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到15m回测数据, 请先运行 python strategy_15m.py'}), 404


@app.route('/api/ma_strategy')
def api_ma_strategy():
    """返回均线技术分析回测结果"""
    data = load_json(MA_STRATEGY_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到均线策略数据, 请先运行 python ma_strategy.py'}), 404


@app.route('/api/combined_strategy')
def api_combined_strategy():
    """返回双书融合策略回测结果"""
    data = load_json(COMBINED_STRATEGY_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到融合策略数据, 请先运行 python combined_strategy.py'}), 404


@app.route('/api/candlestick')
def api_candlestick():
    """返回蜡烛图形态策略结果"""
    data = load_json(CANDLESTICK_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到蜡烛图策略数据, 请先运行 python candlestick_patterns.py'}), 404


@app.route('/api/bollinger')
def api_bollinger():
    """返回布林带策略结果"""
    data = load_json(BOLLINGER_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到布林带策略数据, 请先运行 python bollinger_strategy.py'}), 404


@app.route('/api/volume_price')
def api_volume_price():
    """返回量价分析策略结果"""
    data = load_json(VOLUME_PRICE_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到量价策略数据, 请先运行 python volume_price_strategy.py'}), 404


@app.route('/api/five_book')
def api_five_book():
    """返回五书融合策略结果"""
    data = load_json(FIVE_BOOK_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到五书融合数据, 请先运行 python five_book_fusion.py'}), 404


@app.route('/api/six_book')
def api_six_book():
    """返回六书融合策略结果"""
    data = load_json(SIX_BOOK_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到六书融合数据, 请先运行 python six_book_fusion.py'}), 404


@app.route('/api/optimize_sl_tp')
def api_optimize_sl_tp():
    """返回止盈止损优化结果"""
    data = load_json(OPTIMIZE_SL_TP_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到优化数据, 请先运行 python optimize_sl_tp.py'}), 404


@app.route('/api/optimize_six_book')
def api_optimize_six_book():
    """返回六书优化结果"""
    data = load_json(OPTIMIZE_SIX_BOOK_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到六书优化数据, 请先运行 python optimize_six_book.py'}), 404


@app.route('/api/backtest_30d_7d')
def api_backtest_30d_7d():
    """返回30天vs7天回测对比数据"""
    data = load_json(BACKTEST_30D_7D_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到回测对比数据, 请先运行 python backtest_30d_7d.py'}), 404


@app.route('/api/multi_tf_backtest_30d_7d')
def api_multi_tf_backtest_30d_7d():
    """返回多周期联合决策30天vs7天回测对比数据"""
    data = load_json(MULTI_TF_BACKTEST_30D_7D_FILE)
    if data:
        return jsonify(data)
    return jsonify({'error': '未找到多周期回测数据, 请先运行 python backtest_multi_tf_30d_7d.py'}), 404


@app.route('/api/turtle_backtest')
def api_turtle_backtest():
    """运行海龟交易策略回测"""
    try:
        days = int(request.args.get('days', 60))
        from turtle_strategy import main as turtle_main
        result = turtle_main(trade_days=days)
        if result:
            return jsonify(result)
        return jsonify({'error': '回测失败: 数据不足'}), 500
    except Exception as e:
        return jsonify({'error': f'回测异常: {str(e)}'}), 500


# ======================================================
#   实盘控制面板
# ======================================================
@app.route('/strategy/live-control')
def page_live_control():
    return render_template('page_live_control.html', active_page='live-control')


# 存储后台引擎进程信息
_engine_process = {"proc": None, "phase": None, "started_at": None}


@app.route('/api/live/generate_config', methods=['POST'])
def api_live_generate_config():
    """生成配置模板"""
    try:
        r = subprocess.run(
            [sys.executable, 'live_runner.py', '--generate-config', '-o', 'live_trading_config.json'],
            capture_output=True, text=True, timeout=15, cwd=BASE_DIR
        )
        config_path = os.path.join(BASE_DIR, 'live_trading_config.json')
        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        return jsonify({
            "success": True,
            "message": "配置模板已生成",
            "output": r.stdout + r.stderr,
            "config": config_data,
            "config_path": config_path,
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/live/save_config', methods=['POST'])
def api_live_save_config():
    """保存编辑后的配置"""
    try:
        config_data = request.get_json()
        config_path = os.path.join(BASE_DIR, 'live_trading_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        return jsonify({"success": True, "message": "配置已保存"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/live/load_config')
def api_live_load_config():
    """加载当前配置"""
    config_path = os.path.join(BASE_DIR, 'live_trading_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return jsonify({"success": True, "config": json.load(f)})
    return jsonify({"success": False, "message": "配置文件不存在，请先生成"})


@app.route('/api/live/test_signal', methods=['POST'])
def api_live_test_signal():
    """测试信号计算"""
    tf = request.json.get('timeframe', '1h') if request.is_json else '1h'
    try:
        r = subprocess.run(
            [sys.executable, 'live_runner.py', '--test-signal', '--timeframe', tf],
            capture_output=True, text=True, timeout=120, cwd=BASE_DIR
        )
        return jsonify({
            "success": r.returncode == 0,
            "output": r.stdout,
            "error": r.stderr if r.returncode != 0 else "",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "output": "", "error": "超时 (120s)"}), 504
    except Exception as e:
        return jsonify({"success": False, "output": "", "error": str(e)}), 500


@app.route('/api/live/test_signal_multi', methods=['POST'])
def api_live_test_signal_multi():
    """多时间框架并行信号检测"""
    data = request.json or {}
    timeframes = data.get('timeframes', ['15m', '30m', '1h', '4h', '8h'])

    # 校验时间框架
    valid_tfs = {'1m','3m','5m','10m','15m','30m','1h','2h','3h','4h',
                 '6h','8h','12h','16h','24h','1d'}
    timeframes = [tf for tf in timeframes if tf in valid_tfs]
    if not timeframes:
        return jsonify({"success": False, "error": "无有效的时间框架"}), 400

    tf_str = ','.join(timeframes)
    output_file = os.path.join(BASE_DIR, 'multi_signal_result.json')

    try:
        # 超时 = 300s (线上服务器配置较低, 9个周期可能需要2-4分钟)
        r = subprocess.run(
            [sys.executable, 'live_runner.py', '--test-signal-multi',
             '--timeframe', tf_str, '-o', output_file],
            capture_output=True, text=True,
            timeout=300, cwd=BASE_DIR
        )

        # 尝试读取结构化 JSON 结果
        result_data = None
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    result_data = json.load(f)
            except Exception:
                pass

        return jsonify({
            "success": r.returncode == 0,
            "output": r.stdout,
            "error": r.stderr if r.returncode != 0 else "",
            "data": result_data,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "output": "", "error": "超时 (300s)"}), 504
    except Exception as e:
        return jsonify({"success": False, "output": "", "error": str(e)}), 500


@app.route('/api/live/test_connection', methods=['POST'])
def api_live_test_connection():
    """测试 API 连接"""
    try:
        r = subprocess.run(
            [sys.executable, 'live_runner.py', '--test-connection'],
            capture_output=True, text=True, timeout=30, cwd=BASE_DIR
        )
        return jsonify({
            "success": r.returncode == 0,
            "output": r.stdout,
            "error": r.stderr if r.returncode != 0 else "",
        })
    except Exception as e:
        return jsonify({"success": False, "output": "", "error": str(e)}), 500


def _detect_engine_process():
    """通过系统进程检测引擎是否在运行"""
    try:
        r = subprocess.run(
            ['pgrep', '-af', 'live_runner.py'],
            capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2 and 'live_runner.py' in parts[1]:
                pid = int(parts[0])
                cmd = parts[1]
                # 解析 phase
                phase = 'paper'
                if '--phase' in cmd:
                    idx = cmd.split().index('--phase')
                    tokens = cmd.split()
                    if idx + 1 < len(tokens):
                        phase = tokens[idx + 1]
                return {"running": True, "pid": pid, "phase": phase, "cmd": cmd}
    except Exception:
        pass
    return {"running": False}


def _read_latest_balance_from_log():
    """从最新的日志文件读取最后一条 BALANCE 记录"""
    log_dir = os.path.join(BASE_DIR, 'logs', 'live')
    # 尝试 JSONL 文件 (更结构化)
    jsonl_files = sorted(glob.glob(os.path.join(log_dir, 'trade_*.jsonl')), reverse=True)
    if jsonl_files:
        try:
            with open(jsonl_files[0], 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 从后往前找 BALANCE 记录
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get('level') == 'BALANCE':
                        return entry.get('data', {})
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
    return None


@app.route('/api/live/status')
def api_live_status():
    """获取引擎状态 - 通过文件 + 进程检测，不依赖内存变量"""
    data_dir = os.path.join(BASE_DIR, 'data', 'live')
    result = {"engine": None, "risk": None, "performance": None, "process": None}

    # 1. 引擎状态文件
    engine_file = os.path.join(data_dir, 'engine_state.json')
    if os.path.exists(engine_file):
        try:
            with open(engine_file, 'r') as f:
                result["engine"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # 2. 风控状态文件
    risk_file = os.path.join(data_dir, 'risk_state.json')
    if os.path.exists(risk_file):
        try:
            with open(risk_file, 'r') as f:
                result["risk"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # 3. 绩效数据文件
    perf_file = os.path.join(data_dir, 'performance.json')
    if os.path.exists(perf_file):
        try:
            with open(perf_file, 'r') as f:
                result["performance"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # 4. 通过进程检测引擎运行状态 (不依赖内存)
    process_info = _detect_engine_process()
    result["process"] = process_info

    # 5. 如果没有 engine_state 文件但引擎在运行，从日志读取余额
    if result["engine"] is None and process_info.get("running"):
        balance = _read_latest_balance_from_log()
        if balance:
            result["engine"] = {
                "usdt": balance.get("usdt", 0),
                "frozen_margin": balance.get("frozen_margin", 0),
                "equity": balance.get("total_equity", 0),
                "unrealized_pnl": balance.get("unrealized_pnl", 0),
                "positions": balance.get("positions", []),
                "phase": process_info.get("phase", "unknown"),
                "source": "log",  # 标记数据来源
            }

    # 6. 也检查内存中的引擎进程 (Web 启动的)
    proc = _engine_process.get("proc")
    if proc is not None:
        poll = proc.poll()
        if poll is None and not process_info.get("running"):
            # 内存中有进程但 pgrep 没检测到 (不太可能，兜底)
            result["process"] = {
                "running": True,
                "pid": proc.pid,
                "phase": _engine_process.get("phase"),
                "started_at": _engine_process.get("started_at"),
            }

    return jsonify({"success": True, **result})


@app.route('/api/live/start', methods=['POST'])
def api_live_start():
    """启动交易引擎"""
    data = request.get_json() or {}
    phase = data.get('phase', 'paper')

    # 检查是否已有引擎运行 (通过进程检测，不仅仅看内存)
    existing = _detect_engine_process()
    if existing.get("running"):
        return jsonify({
            "success": False,
            "message": f"引擎已在运行 (PID={existing['pid']}, phase={existing.get('phase', '?')})"
        })
    proc = _engine_process.get("proc")
    if proc is not None and proc.poll() is None:
        return jsonify({
            "success": False,
            "message": f"引擎已在运行 (PID={proc.pid}, phase={_engine_process['phase']})"
        })

    valid_phases = ['paper', 'testnet', 'small_live', 'scale_up']
    if phase not in valid_phases:
        return jsonify({"success": False, "message": f"无效阶段: {phase}"})

    try:
        cmd = [sys.executable, 'live_runner.py', '--phase', phase, '-y']
        config_path = os.path.join(BASE_DIR, 'live_trading_config.json')
        if os.path.exists(config_path) and phase != 'paper':
            cmd.extend(['--config', config_path])

        log_dir = os.path.join(BASE_DIR, 'logs', 'live')
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, f'engine_{phase}.log'), 'a')

        p = subprocess.Popen(
            cmd, cwd=BASE_DIR,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,   # 脱离 gunicorn 进程树，Web 服务重启不影响引擎
        )
        _engine_process["proc"] = p
        _engine_process["phase"] = phase
        _engine_process["started_at"] = __import__('datetime').datetime.now().isoformat()

        return jsonify({
            "success": True,
            "message": f"引擎已启动 (PID={p.pid}, phase={phase})",
            "pid": p.pid,
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/live/stop', methods=['POST'])
def api_live_stop():
    """停止交易引擎 — 支持停止任何方式启动的引擎进程"""
    import signal as sig

    # 优先用内存中的 proc 对象
    proc = _engine_process.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.send_signal(sig.SIGTERM)
            proc.wait(timeout=10)
            _engine_process["proc"] = None
            return jsonify({"success": True, "message": "引擎已停止"})
        except subprocess.TimeoutExpired:
            proc.kill()
            _engine_process["proc"] = None
            return jsonify({"success": True, "message": "引擎已强制停止"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

    # proc 为空 (Web 服务重启过, 但引擎仍在运行) → 通过 pgrep 检测并发送 SIGTERM
    existing = _detect_engine_process()
    if not existing.get("running"):
        _engine_process["proc"] = None
        return jsonify({"success": False, "message": "引擎未在运行"})

    pid = existing["pid"]
    try:
        os.kill(pid, sig.SIGTERM)
        # 等待最多10秒
        import time
        for _ in range(20):
            time.sleep(0.5)
            check = _detect_engine_process()
            if not check.get("running"):
                _engine_process["proc"] = None
                return jsonify({"success": True, "message": f"引擎已停止 (PID={pid})"})
        # 超时 → 强制 kill
        os.kill(pid, sig.SIGKILL)
        _engine_process["proc"] = None
        return jsonify({"success": True, "message": f"引擎已强制停止 (PID={pid})"})
    except ProcessLookupError:
        _engine_process["proc"] = None
        return jsonify({"success": True, "message": f"引擎进程已不存在 (PID={pid})"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/live/kill_switch', methods=['POST'])
def api_live_kill_switch():
    """紧急平仓"""
    try:
        config_path = os.path.join(BASE_DIR, 'live_trading_config.json')
        cmd = [sys.executable, 'live_runner.py', '--kill-switch', '-y']
        if os.path.exists(config_path):
            cmd.extend(['--config', config_path])

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=BASE_DIR)
        return jsonify({
            "success": r.returncode == 0,
            "output": r.stdout,
            "error": r.stderr if r.returncode != 0 else "",
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/live/logs')
def api_live_logs():
    """获取最近日志"""
    log_dir = os.path.join(BASE_DIR, 'logs', 'live')
    lines = request.args.get('lines', 100, type=int)

    # 找到最新的日志文件
    log_files = sorted(glob.glob(os.path.join(log_dir, 'trade_*.log')), reverse=True)
    if not log_files:
        # 尝试引擎日志
        log_files = sorted(glob.glob(os.path.join(log_dir, 'engine_*.log')), reverse=True)

    if not log_files:
        return jsonify({"success": True, "logs": "暂无日志", "file": ""})

    try:
        with open(log_files[0], 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            recent = all_lines[-lines:]
        return jsonify({
            "success": True,
            "logs": "".join(recent),
            "file": os.path.basename(log_files[0]),
            "total_lines": len(all_lines),
        })
    except Exception as e:
        return jsonify({"success": False, "logs": str(e), "file": ""})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  技术分析 Web 展示平台")
    print("  五书融合: 背离+均线+蜡烛图+布林带+量价")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
