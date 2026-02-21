"""
热点币 Web 路由 — Flask 蓝图

注册方式 (app.py 中添加):
    from hotcoin.web.routes import hotcoin_bp
    app.register_blueprint(hotcoin_bp)
"""

import os
import time
import json
from flask import Blueprint, render_template, jsonify

hotcoin_bp = Blueprint(
    "hotcoin",
    __name__,
    url_prefix="/hotcoin",
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

# 全局引用, runner 启动后注入
_runner = None
_STATUS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "hotcoin_runtime_status.json")


def set_runner(runner):
    """由 runner.py 或 app.py 注入 HotCoinRunner 实例。"""
    global _runner
    _runner = runner


def _read_runtime_status_file():
    if not os.path.exists(_STATUS_FILE):
        return None
    try:
        with open(_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _default_precheck_stats():
    return {"total": 0, "by_code": {}, "by_symbol": {}}


def _default_execution_metrics():
    return {
        "window_sec": 300,
        "order_attempts_5m": 0,
        "precheck_failures_5m": 0,
        "dedup_rejects_5m": 0,
        "order_errors_5m": 0,
        "order_success_5m": 0,
        "precheck_fail_rate_5m": 0.0,
        "order_error_rate_5m": 0.0,
    }


def _normalize_execution_metrics(raw):
    if not isinstance(raw, dict):
        return _default_execution_metrics()
    defaults = _default_execution_metrics()
    out = dict(defaults)
    out.update(raw)
    for key in ("window_sec", "order_attempts_5m", "precheck_failures_5m",
                "dedup_rejects_5m", "order_errors_5m", "order_success_5m"):
        try:
            out[key] = int(out.get(key, defaults[key]))
        except Exception:
            out[key] = defaults[key]
    for key in ("precheck_fail_rate_5m", "order_error_rate_5m"):
        try:
            out[key] = float(out.get(key, defaults[key]))
        except Exception:
            out[key] = defaults[key]
    return out


def _normalize_precheck_stats(raw):
    if not isinstance(raw, dict):
        return _default_precheck_stats()
    by_code = raw.get("by_code")
    by_symbol = raw.get("by_symbol")
    total = raw.get("total", 0)
    if not isinstance(by_code, dict):
        by_code = {}
    if not isinstance(by_symbol, dict):
        by_symbol = {}
    try:
        total = int(total)
    except Exception:
        total = sum(int(v) for v in by_code.values() if isinstance(v, (int, float)))
    return {"total": max(0, total), "by_code": by_code, "by_symbol": by_symbol}


def _read_runner_precheck_stats():
    if _runner is None:
        return None
    engine = getattr(_runner, "spot_engine", None)
    executor = getattr(engine, "executor", None) if engine is not None else None
    if executor is None or not hasattr(executor, "get_precheck_stats"):
        return None
    try:
        stats = executor.get_precheck_stats()
    except Exception:
        return None
    return _normalize_precheck_stats(stats)


def _read_runner_execution_metrics():
    if _runner is None:
        return None
    engine = getattr(_runner, "spot_engine", None)
    executor = getattr(engine, "executor", None) if engine is not None else None
    if executor is None or not hasattr(executor, "get_runtime_metrics"):
        return None
    try:
        stats = executor.get_runtime_metrics(window_sec=300)
    except Exception:
        return None
    return _normalize_execution_metrics(stats)


@hotcoin_bp.route("/")
def dashboard():
    """热点币仪表盘页面。"""
    return render_template("hotcoin_dashboard.html")


@hotcoin_bp.route("/api/status")
def api_status():
    """候选池状态 + 最近信号 (JSON)。"""
    cached = _read_runtime_status_file()
    if _runner is None:
        if cached:
            if "precheck_stats" not in cached:
                cached["precheck_stats"] = _default_precheck_stats()
            cached["execution_metrics"] = _normalize_execution_metrics(cached.get("execution_metrics"))
            cached["engine_state"] = str(cached.get("engine_state", "stopped"))
            cached["engine_state_reasons"] = cached.get("engine_state_reasons", []) or []
            cached["can_open_new_positions"] = bool(cached.get("can_open_new_positions", False))
            cached["freshness"] = cached.get("freshness", {}) or {}
            return jsonify(cached)
        return jsonify({
            "pool_size": 0,
            "candidates": [],
            "ws_connected": False,
            "paper": True,
            "execution_enabled": False,
            "anomaly_count": 0,
            "active_signals": 0,
            "positions": 0,
            "recent_anomalies": [],
            "recent_signals": [],
            "precheck_stats": _default_precheck_stats(),
            "execution_metrics": _default_execution_metrics(),
            "engine_state": "stopped",
            "engine_state_reasons": [],
            "can_open_new_positions": False,
            "freshness": {},
            "message": "热点币系统未启动",
        })

    # 若状态文件是新鲜的，优先使用文件里的实时统计（含 recent_signals / active_signals）
    if cached and isinstance(cached, dict):
        ts = float(cached.get("ts", 0) or 0)
        if ts > 0 and (time.time() - ts) < 180:
            merged = dict(cached)
            merged["paper"] = _runner.config.execution.use_paper_trading
            merged["execution_enabled"] = bool(getattr(_runner.config.execution, "enable_order_execution", False))
            merged["ws_connected"] = hasattr(_runner, "ticker_stream") and bool(_runner.ticker_stream.tickers)
            engine = getattr(_runner, "spot_engine", None)
            if engine:
                merged["positions"] = engine.num_positions
            merged["precheck_stats"] = _normalize_precheck_stats(merged.get("precheck_stats"))
            merged["execution_metrics"] = _normalize_execution_metrics(merged.get("execution_metrics"))
            merged["engine_state"] = str(merged.get("engine_state", "unknown"))
            merged["engine_state_reasons"] = merged.get("engine_state_reasons", []) or []
            merged["can_open_new_positions"] = bool(merged.get("can_open_new_positions", False))
            merged["freshness"] = merged.get("freshness", {}) or {}
            return jsonify(merged)

    pool = _runner.pool
    candidates = pool.get_top(n=20, min_score=0)

    candidate_list = []
    for c in candidates:
        candidate_list.append({
            "symbol": c.symbol,
            "heat_score": c.heat_score,
            "source": c.source,
            "status": c.status,
            "price_change_5m": c.price_change_5m,
            "price_change_24h": c.price_change_24h,
            "quote_volume_24h": c.quote_volume_24h,
            "volume_surge_ratio": c.volume_surge_ratio,
            "score_momentum": c.score_momentum,
            "score_liquidity": c.score_liquidity,
            "score_risk_penalty": c.score_risk_penalty,
            "has_listing_signal": c.has_listing_signal,
            "signal": "",  # filled from recent signals cache
        })

    engine = getattr(_runner, "spot_engine", None)
    positions = engine.num_positions if engine else 0
    execution_enabled = bool(getattr(_runner.config.execution, "enable_order_execution", False))
    cached_recent_signals = []
    cached_recent_anomalies = []
    if isinstance(cached, dict):
        ts = float(cached.get("ts", 0) or 0)
        if ts > 0 and (time.time() - ts) < 180:
            cached_recent_signals = cached.get("recent_signals", []) or []
            cached_recent_anomalies = cached.get("recent_anomalies", []) or []
    precheck_stats = _read_runner_precheck_stats() or _normalize_precheck_stats(
        cached.get("precheck_stats") if isinstance(cached, dict) else None
    )
    execution_metrics = _read_runner_execution_metrics() or _normalize_execution_metrics(
        cached.get("execution_metrics") if isinstance(cached, dict) else None
    )
    engine_state = "unknown"
    state_reasons = []
    can_open = False
    freshness = {}
    if isinstance(cached, dict):
        engine_state = str(cached.get("engine_state", "unknown"))
        state_reasons = cached.get("engine_state_reasons", []) or []
        can_open = bool(cached.get("can_open_new_positions", False))
        freshness = cached.get("freshness", {}) or {}

    return jsonify({
        "pool_size": pool.size,
        "candidates": candidate_list,
        "ws_connected": hasattr(_runner, "ticker_stream") and bool(_runner.ticker_stream.tickers),
        "paper": _runner.config.execution.use_paper_trading,
        "execution_enabled": execution_enabled,
        "anomaly_count": len([c for c in candidates if c.source in ("momentum", "mixed")]),
        "active_signals": len(cached_recent_signals),
        "positions": positions,
        "recent_anomalies": cached_recent_anomalies,
        "recent_signals": cached_recent_signals,
        "precheck_stats": precheck_stats,
        "execution_metrics": execution_metrics,
        "engine_state": engine_state,
        "engine_state_reasons": state_reasons,
        "can_open_new_positions": can_open,
        "freshness": freshness,
    })


@hotcoin_bp.route("/api/precheck_stats")
def api_precheck_stats():
    """下单预检失败统计。"""
    now = time.time()
    runner_stats = _read_runner_precheck_stats()
    if runner_stats is not None:
        out = dict(runner_stats)
        out["source"] = "runner"
        out["ts"] = now
        return jsonify(out)

    cached = _read_runtime_status_file()
    if isinstance(cached, dict):
        out = _normalize_precheck_stats(cached.get("precheck_stats"))
        out["source"] = "status_file"
        out["ts"] = float(cached.get("ts", 0) or 0)
        return jsonify(out)

    out = _default_precheck_stats()
    out["source"] = "none"
    out["ts"] = 0.0
    return jsonify(out)


@hotcoin_bp.route("/api/execution_metrics")
def api_execution_metrics():
    """执行与错误指标（5m 窗口）。"""
    now = time.time()
    metrics = _read_runner_execution_metrics()
    if metrics is not None:
        out = dict(metrics)
        out["source"] = "runner"
        out["ts"] = now
        return jsonify(out)

    cached = _read_runtime_status_file()
    if isinstance(cached, dict):
        out = _normalize_execution_metrics(cached.get("execution_metrics"))
        out["source"] = "status_file"
        out["ts"] = float(cached.get("ts", 0) or 0)
        return jsonify(out)

    out = _default_execution_metrics()
    out["source"] = "none"
    out["ts"] = 0.0
    return jsonify(out)


@hotcoin_bp.route("/api/pool")
def api_pool():
    """完整候选池数据。"""
    if _runner is None:
        return jsonify({"coins": []})
    coins = _runner.pool.get_all()
    return jsonify({
        "coins": [
            {
                "symbol": c.symbol,
                "heat_score": c.heat_score,
                "source": c.source,
                "status": c.status,
                "discovered_at": c.discovered_at,
                "price_change_5m": c.price_change_5m,
                "price_change_24h": c.price_change_24h,
                "quote_volume_24h": c.quote_volume_24h,
                "volume_surge_ratio": c.volume_surge_ratio,
            }
            for c in coins
        ]
    })
