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


@hotcoin_bp.route("/")
def dashboard():
    """热点币仪表盘页面。"""
    return render_template("hotcoin_dashboard.html")


@hotcoin_bp.route("/api/status")
def api_status():
    """候选池状态 + 最近信号 (JSON)。"""
    if _runner is None:
        cached = _read_runtime_status_file()
        if cached:
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
            "message": "热点币系统未启动",
        })

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

    return jsonify({
        "pool_size": pool.size,
        "candidates": candidate_list,
        "ws_connected": hasattr(_runner, "ticker_stream") and bool(_runner.ticker_stream.tickers),
        "paper": _runner.config.execution.use_paper_trading,
        "execution_enabled": execution_enabled,
        "anomaly_count": len([c for c in candidates if c.source in ("momentum", "mixed")]),
        "active_signals": 0,
        "positions": positions,
        "recent_anomalies": [],
        "recent_signals": [],
    })


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
