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


def _read_runner_risk_summary():
    if _runner is None:
        return None
    engine = getattr(_runner, "spot_engine", None)
    risk = getattr(engine, "risk", None) if engine is not None else None
    if risk is None or not hasattr(risk, "get_summary"):
        return None
    try:
        out = risk.get_summary()
    except Exception:
        return None
    return out if isinstance(out, dict) else {}


def _read_runner_ws_connected():
    if _runner is None:
        return None
    stream = getattr(_runner, "ticker_stream", None)
    tickers = getattr(stream, "tickers", None) if stream is not None else None
    if isinstance(tickers, dict):
        return bool(tickers)
    return None


def _compute_health_status(payload: dict):
    checks = {
        "runner_attached": bool(payload.get("runner_attached")),
        "status_snapshot_fresh": bool(payload.get("status_snapshot_fresh", False)),
        "ws_connected": bool(payload.get("ws_connected", False)),
        "risk_halted": bool(payload.get("risk_halted", False)),
        "engine_state": str(payload.get("engine_state", "unknown")),
        "latest_ticker_age_sec": payload.get("latest_ticker_age_sec"),
        "order_errors_5m": int(payload.get("order_errors_5m", 0) or 0),
    }
    reasons = []
    severity = 0  # 0=ok,1=degraded,2=blocked,3=stopped

    if not checks["runner_attached"]:
        return "stopped", ["runner_not_attached"], checks

    if checks["risk_halted"]:
        severity = max(severity, 2)
        reasons.append("risk_halted")

    state = checks["engine_state"]
    if state == "blocked":
        severity = max(severity, 2)
        reasons.append("engine_state_blocked")
    elif state in ("degraded", "unknown"):
        severity = max(severity, 1)
        reasons.append(f"engine_state_{state}")

    age = checks["latest_ticker_age_sec"]
    if isinstance(age, (int, float)):
        if age >= 300:
            severity = max(severity, 2)
            reasons.append("ticker_stale>=300s")
        elif age >= 90:
            severity = max(severity, 1)
            reasons.append("ticker_stale>=90s")
    elif not checks["ws_connected"]:
        severity = max(severity, 1)
        reasons.append("ws_disconnected")

    if checks["order_errors_5m"] >= 10:
        severity = max(severity, 2)
        reasons.append("order_errors_5m>=10")
    elif checks["order_errors_5m"] >= 3:
        severity = max(severity, 1)
        reasons.append("order_errors_5m>=3")

    if not checks["status_snapshot_fresh"]:
        severity = max(severity, 1)
        reasons.append("status_snapshot_stale")

    status = "ok" if severity == 0 else ("degraded" if severity == 1 else "blocked")
    if not reasons:
        reasons.append("healthy")
    return status, reasons, checks


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
            cached["state_recovery_pending"] = (
                cached.get("state_recovery_pending")
                if isinstance(cached.get("state_recovery_pending"), dict)
                else None
            )
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
            "state_recovery_pending": None,
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
            merged["state_recovery_pending"] = (
                merged.get("state_recovery_pending")
                if isinstance(merged.get("state_recovery_pending"), dict)
                else None
            )
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
    state_recovery_pending = None
    can_open = False
    freshness = {}
    if isinstance(cached, dict):
        engine_state = str(cached.get("engine_state", "unknown"))
        state_reasons = cached.get("engine_state_reasons", []) or []
        pending = cached.get("state_recovery_pending")
        state_recovery_pending = pending if isinstance(pending, dict) else None
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
        "state_recovery_pending": state_recovery_pending,
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


@hotcoin_bp.route("/health")
def health():
    """热点币系统健康聚合接口。"""
    now = time.time()
    cached = _read_runtime_status_file()
    metrics = _read_runner_execution_metrics() or _normalize_execution_metrics(
        cached.get("execution_metrics") if isinstance(cached, dict) else None
    )

    runner_attached = _runner is not None
    ts = float(cached.get("ts", 0) or 0) if isinstance(cached, dict) else 0.0
    status_age_sec = (now - ts) if ts > 0 else None
    status_snapshot_fresh = bool(ts > 0 and status_age_sec is not None and status_age_sec < 180)

    ws_connected = _read_runner_ws_connected()
    if ws_connected is None and isinstance(cached, dict):
        ws_connected = bool(cached.get("ws_connected", False))
    ws_connected = bool(ws_connected)

    freshness = cached.get("freshness", {}) if isinstance(cached, dict) else {}
    if not isinstance(freshness, dict):
        freshness = {}
    latest_ticker_age_sec = freshness.get("latest_ticker_age_sec")
    try:
        if latest_ticker_age_sec is not None:
            latest_ticker_age_sec = float(latest_ticker_age_sec)
    except Exception:
        latest_ticker_age_sec = None

    engine_state = str(cached.get("engine_state", "unknown")) if isinstance(cached, dict) else "unknown"
    state_reasons = cached.get("engine_state_reasons", []) if isinstance(cached, dict) else []
    if not isinstance(state_reasons, list):
        state_reasons = []

    risk_summary = _read_runner_risk_summary()
    risk_halted = bool((risk_summary or {}).get("halted"))
    if not risk_halted and isinstance(cached, dict):
        risk_halted = bool(cached.get("risk_halted", False))

    status, reasons, checks = _compute_health_status(
        {
            "runner_attached": runner_attached,
            "status_snapshot_fresh": status_snapshot_fresh,
            "ws_connected": ws_connected,
            "risk_halted": risk_halted,
            "engine_state": engine_state,
            "latest_ticker_age_sec": latest_ticker_age_sec,
            "order_errors_5m": metrics.get("order_errors_5m", 0),
        }
    )

    out = {
        "status": status,
        "can_trade": status == "ok" and engine_state == "tradeable" and not risk_halted,
        "ts": now,
        "runner_attached": runner_attached,
        "engine_state": engine_state,
        "engine_state_reasons": state_reasons,
        "status_snapshot_fresh": status_snapshot_fresh,
        "status_snapshot_age_sec": round(status_age_sec, 2) if isinstance(status_age_sec, (int, float)) else None,
        "ws_connected": ws_connected,
        "latest_ticker_age_sec": latest_ticker_age_sec,
        "risk_halted": risk_halted,
        "execution_metrics": metrics,
        "checks": checks,
        "reasons": reasons,
    }
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


@hotcoin_bp.route("/monitor")
def monitor():
    """热点币实盘监控页面。"""
    return render_template("hotcoin_monitor.html")


@hotcoin_bp.route("/api/hot_posts")
def api_hot_posts():
    """最近社交热帖 (币安广场 + Twitter)。"""
    now = time.time()

    # 优先从 runner 直接读取
    if _runner is not None:
        posts = []
        sq = getattr(_runner, "square_monitor", None)
        tw = getattr(_runner, "twitter_monitor", None)
        if sq and hasattr(sq, "get_recent_posts"):
            try:
                posts.extend(sq.get_recent_posts(30))
            except Exception:
                pass
        if tw and hasattr(tw, "get_recent_posts"):
            try:
                posts.extend(tw.get_recent_posts(30))
            except Exception:
                pass
        posts.sort(key=lambda p: p.get("ts", 0), reverse=True)
        return jsonify({"posts": posts[:50], "ts": now, "source": "runner"})

    # Fallback: 从状态文件读取
    cached = _read_runtime_status_file()
    if isinstance(cached, dict) and "hot_posts" in cached:
        return jsonify({
            "posts": cached["hot_posts"],
            "ts": float(cached.get("ts", 0) or 0),
            "source": "status_file",
        })

    return jsonify({"posts": [], "ts": now, "source": "none"})
