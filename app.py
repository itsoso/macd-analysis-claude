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

from flask import Flask, render_template, jsonify, request, redirect, url_for, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from date_range_report import load_latest_report_from_db
from web_routes import register_page_routes, register_result_api_routes

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


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/x-icon')


@app.before_request
def check_auth():
    """全局认证检查 - 除登录页和静态资源外，所有请求都需要认证"""
    # 允许访问登录页面和静态资源
    allowed_endpoints = ('login', 'static', 'favicon')
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
MULTI_TF_BACKTEST_30D_7D_FILE = os.path.join(BASE_DIR, 'data', 'backtests', 'backtest_multi_tf_30d_7d_result.json')
MULTI_TF_DATE_RANGE_DB_FILE = os.path.join(BASE_DIR, 'data', 'backtests', 'multi_tf_date_range_reports.db')
NAKED_KLINE_BACKTEST_FILE = os.path.join(BASE_DIR, 'data', 'backtests', 'naked_kline_backtest_result.json')

RESULT_FILE_PATHS = {
    'BACKTEST_FILE': BACKTEST_FILE,
    'BACKTEST_MULTI_FILE': BACKTEST_MULTI_FILE,
    'GLOBAL_STRATEGY_FILE': GLOBAL_STRATEGY_FILE,
    'STRATEGY_COMPARE_FILE': STRATEGY_COMPARE_FILE,
    'STRATEGY_OPTIMIZE_FILE': STRATEGY_OPTIMIZE_FILE,
    'STRATEGY_ENHANCED_FILE': STRATEGY_ENHANCED_FILE,
    'STRATEGY_FUTURES_FILE': STRATEGY_FUTURES_FILE,
    'STRATEGY_FUTURES_V2_FILE': STRATEGY_FUTURES_V2_FILE,
    'STRATEGY_FUTURES_V3_FILE': STRATEGY_FUTURES_V3_FILE,
    'STRATEGY_FUTURES_V4_FILE': STRATEGY_FUTURES_V4_FILE,
    'STRATEGY_FUTURES_V5_FILE': STRATEGY_FUTURES_V5_FILE,
    'STRATEGY_FUTURES_FINAL_FILE': STRATEGY_FUTURES_FINAL_FILE,
    'TIMEFRAME_ANALYSIS_FILE': TIMEFRAME_ANALYSIS_FILE,
    'STRATEGY_15M_FILE': STRATEGY_15M_FILE,
    'MA_STRATEGY_FILE': MA_STRATEGY_FILE,
    'COMBINED_STRATEGY_FILE': COMBINED_STRATEGY_FILE,
    'CANDLESTICK_FILE': CANDLESTICK_FILE,
    'BOLLINGER_FILE': BOLLINGER_FILE,
    'VOLUME_PRICE_FILE': VOLUME_PRICE_FILE,
    'FIVE_BOOK_FILE': FIVE_BOOK_FILE,
    'SIX_BOOK_FILE': SIX_BOOK_FILE,
    'OPTIMIZE_SL_TP_FILE': OPTIMIZE_SL_TP_FILE,
    'OPTIMIZE_SIX_BOOK_FILE': OPTIMIZE_SIX_BOOK_FILE,
    'BACKTEST_30D_7D_FILE': BACKTEST_30D_7D_FILE,
    'MULTI_TF_BACKTEST_30D_7D_FILE': MULTI_TF_BACKTEST_30D_7D_FILE,
    'NAKED_KLINE_BACKTEST_FILE': NAKED_KLINE_BACKTEST_FILE,
}


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ======================================================
#   页面路由与结果 API 路由（模块化注册）
# ======================================================
register_page_routes(app)
register_result_api_routes(app, load_json=load_json, result_paths=RESULT_FILE_PATHS)


@app.route('/api/multi_tf_date_range_report')
def api_multi_tf_date_range_report():
    """读取多周期固定区间回测(DB)的最新结果。"""
    data = load_latest_report_from_db(MULTI_TF_DATE_RANGE_DB_FILE)
    if data:
        return jsonify(data)
    return jsonify({
        "error": "未找到区间回测数据，请先运行 python backtest_multi_tf_date_range_db.py",
        "db_path": MULTI_TF_DATE_RANGE_DB_FILE,
    }), 404


# ── 裸K线交易法 · 逐日盈亏 (从 DB 读取) ──
NAKED_KLINE_DB_FILE = os.path.join(BASE_DIR, 'data', 'backtests', 'naked_kline_backtest.db')


@app.route('/api/naked_kline_daily')
def api_naked_kline_daily():
    """从 SQLite DB 加载裸K线策略最新回测的逐日盈亏数据。"""
    from naked_kline_db import load_latest_run
    data = load_latest_run(NAKED_KLINE_DB_FILE)
    if data:
        return jsonify(data)
    return jsonify({
        "error": "未找到裸K线回测数据，请先运行 python backtest_naked_kline.py",
    }), 404


# ── 多周期联合决策 · 逐日盈亏 (从 DB 读取) ──
MULTI_TF_DAILY_DB_FILE = os.path.join(BASE_DIR, 'data', 'backtests', 'multi_tf_daily_backtest.db')


@app.route('/api/multi_tf_daily')
def api_multi_tf_daily():
    """从 SQLite DB 加载回测数据。支持 ?run_id=N 指定版本，默认最新。"""
    from multi_tf_daily_db import load_latest_run, load_run_by_id
    run_id = request.args.get('run_id', type=int)
    if run_id:
        data = load_run_by_id(run_id, MULTI_TF_DAILY_DB_FILE)
    else:
        data = load_latest_run(MULTI_TF_DAILY_DB_FILE)
    if data:
        return jsonify(data)
    return jsonify({
        "error": "未找到多周期逐日回测数据，请先运行 python backtest_multi_tf_daily.py",
    }), 404


@app.route('/api/multi_tf_daily/runs')
def api_multi_tf_daily_runs():
    """列出所有回测版本，用于版本选择器和对比。"""
    from multi_tf_daily_db import list_runs
    runs = list_runs(MULTI_TF_DAILY_DB_FILE)
    return jsonify(runs)


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

        # 写入带时间戳的服务器端缓存 (供页面加载时恢复)
        if result_data:
            try:
                from datetime import datetime as _dt
                cache = {
                    "data": result_data,
                    "timeframes": timeframes,
                    "ts": _dt.now().isoformat(),
                    "ts_epoch": __import__('time').time(),
                }
                cache_path = os.path.join(BASE_DIR, 'multi_signal_cache.json')
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
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


@app.route('/api/live/signal_cache')
def api_live_signal_cache():
    """读取服务器端缓存的最新多周期检测结果"""
    cache_path = os.path.join(BASE_DIR, 'multi_signal_cache.json')
    if not os.path.exists(cache_path):
        return jsonify({"success": False, "message": "暂无缓存"})
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        # 计算缓存年龄
        import time as _time
        age_sec = _time.time() - cache.get("ts_epoch", 0)
        cache["age_sec"] = round(age_sec)
        cache["age_text"] = (
            f"{int(age_sec)}秒前" if age_sec < 60
            else f"{int(age_sec/60)}分钟前" if age_sec < 3600
            else f"{int(age_sec/3600)}小时前"
        )
        return jsonify({"success": True, **cache})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


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
    """通过 PID 文件 + 系统进程检测引擎是否在运行。
    优先读取 engine.pid 文件锁（比 pgrep 更可靠），
    如果 PID 文件对应进程存活则返回该进程信息。
    """
    pid_file = os.path.join(BASE_DIR, 'data', 'live', 'engine.pid')

    # 方案 1: 读取 PID 文件 + 验证进程存活
    if os.path.exists(pid_file):
        try:
            import fcntl
            f = open(pid_file, 'r')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 能拿到锁 → 说明没有引擎持有它，PID 文件是残留
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            except (IOError, OSError):
                # 拿不到锁 → 有进程正在持有 → 引擎在运行
                f.seek(0)
                pid_str = f.read().strip()
                f.close()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    # 读取 engine_state.json 获取 phase
                    phase = 'paper'
                    try:
                        state_file = os.path.join(BASE_DIR, 'data', 'live', 'engine_state.json')
                        with open(state_file, 'r') as sf:
                            state = json.load(sf)
                        phase = state.get('phase', 'paper')
                    except Exception:
                        pass
                    return {"running": True, "pid": pid, "phase": phase}
        except Exception:
            pass

    # 方案 2: 回退到 pgrep（兼容没有 PID 文件的旧版本）
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
                # 排除 --test-signal / --status 等非引擎命令
                if any(x in cmd for x in ['--test-signal', '--status',
                                           '--kill-switch', '--generate-config',
                                           '--test-connection']):
                    continue
                # 解析 phase
                phase = 'paper'
                if '--phase' in cmd:
                    tokens = cmd.split()
                    try:
                        idx = tokens.index('--phase')
                        if idx + 1 < len(tokens):
                            phase = tokens[idx + 1]
                    except ValueError:
                        pass
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
    result = {
        "engine": None,
        "risk": None,
        "performance": None,
        "performance_summary": None,
        "process": None,
    }

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
                perf_raw = json.load(f)
            result["performance"] = perf_raw

            # 统一口径: 额外提供 summary 字段，避免前端自行推导时单位不一致
            try:
                from performance_tracker import PerformanceTracker
                tracker = PerformanceTracker(
                    initial_capital=perf_raw.get("initial_capital", 0),
                    data_dir=data_dir,
                )
                summary = tracker.get_summary()
                result["performance_summary"] = summary

                # 兼容旧前端: 将常用汇总字段回填到 performance 顶层
                merged = dict(perf_raw)
                for k in [
                    "initial_capital", "current_equity", "total_return",
                    "total_return_pct", "total_pnl", "total_trades",
                    "wins", "losses", "win_rate", "win_rate_pct",
                    "max_drawdown", "max_drawdown_pct", "total_fees",
                ]:
                    if k in summary:
                        merged[k] = summary[k]
                result["performance"] = merged
            except Exception:
                pass
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


def _kill_all_engine_processes(exclude_pid=None):
    """杀死所有 live_runner.py 引擎进程（排除指定 PID）。
    用于清理因历史 bug 残留的僵尸进程。"""
    killed = []
    try:
        r = subprocess.run(
            ['pgrep', '-af', 'live_runner.py'],
            capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2 or 'live_runner.py' not in parts[1]:
                continue
            cmd = parts[1]
            # 排除非引擎命令
            if any(x in cmd for x in ['--test-signal', '--status',
                                       '--kill-switch', '--generate-config',
                                       '--test-connection']):
                continue
            pid = int(parts[0])
            if exclude_pid and pid == exclude_pid:
                continue
            try:
                os.kill(pid, __import__('signal').SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass
    except Exception:
        pass
    return killed


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
            # 同时清理可能的残留进程
            orphans = _kill_all_engine_processes()
            msg = "引擎已停止"
            if orphans:
                msg += f" (同时清理了 {len(orphans)} 个残留进程)"
            return jsonify({"success": True, "message": msg})
        except subprocess.TimeoutExpired:
            proc.kill()
            _engine_process["proc"] = None
            orphans = _kill_all_engine_processes()
            return jsonify({"success": True, "message": "引擎已强制停止"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

    # proc 为空 (Web 服务重启过, 但引擎仍在运行) → 通过 PID 文件/pgrep 检测
    existing = _detect_engine_process()
    if not existing.get("running"):
        # 兜底: 即使检测不到，也尝试清理残留进程
        orphans = _kill_all_engine_processes()
        _engine_process["proc"] = None
        if orphans:
            return jsonify({"success": True,
                            "message": f"清理了 {len(orphans)} 个残留引擎进程: {orphans}"})
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
                # 清理其他可能的残留进程
                orphans = _kill_all_engine_processes()
                msg = f"引擎已停止 (PID={pid})"
                if orphans:
                    msg += f" + 清理 {len(orphans)} 个残留进程"
                return jsonify({"success": True, "message": msg})
        # 超时 → 强制 kill 所有引擎进程
        _kill_all_engine_processes()
        _engine_process["proc"] = None
        return jsonify({"success": True, "message": f"引擎已强制停止 (PID={pid})"})
    except ProcessLookupError:
        _engine_process["proc"] = None
        _kill_all_engine_processes()
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
    # 排除引擎产生的日志/数据文件，防止 reloader 因这些文件变化而反复重启 Flask，
    # 导致 _engine_process 内存引用丢失、Web UI 误判引擎已停止。
    # 线上用 gunicorn 没有 reloader，不受此影响。
    app.run(
        debug=True,
        port=5000,
        exclude_patterns=[
            '**/logs/**',
            '**/data/**',
            '**/*.log',
            '**/*.jsonl',
            '**/*.pyc',
            '**/__pycache__/**',
            '**/backtest_*_result.json',
            '**/optimize_*_result.json',
        ],
    )
