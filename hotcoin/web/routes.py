"""
热点币 Web 路由 — Flask 蓝图

注册方式 (app.py 中添加):
    from hotcoin.web.routes import hotcoin_bp
    app.register_blueprint(hotcoin_bp)
"""

import gzip
import os
import re
import time
import json
from flask import Blueprint, render_template, jsonify, request

hotcoin_bp = Blueprint(
    "hotcoin",
    __name__,
    url_prefix="/hotcoin",
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

# 全局引用, runner 启动后注入
_runner = None
_STATUS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "hotcoin_runtime_status.json")
_EVENTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "hotcoin_events.jsonl")
_TRADES_GLOB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_VALID_CHART_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}


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


def _sanitize_symbol(raw: str) -> str:
    symbol = str(raw or "").strip().upper()
    if not symbol or not re.fullmatch(r"[A-Z0-9]{5,20}", symbol):
        return ""
    return symbol


def _sanitize_trace_id(raw: str) -> str:
    trace_id = str(raw or "").strip()
    if not trace_id:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", trace_id):
        return ""
    return trace_id


def _parse_int_arg(name: str, default: int, min_v: int, max_v: int) -> int:
    raw = request.args.get(name, "")
    try:
        val = int(raw) if raw else int(default)
    except Exception:
        val = int(default)
    return max(min_v, min(max_v, val))


def _iter_jsonl_file(path: str):
    if not os.path.exists(path):
        return
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _event_related_to_symbol(event_type: str, payload: dict, symbol: str) -> bool:
    et = str(event_type or "")
    sym = str(payload.get("symbol", "")).upper()
    if et in ("order_attempt", "order_result"):
        return sym == symbol
    if et == "signal_snapshot":
        signals = payload.get("signals", [])
        if isinstance(signals, list):
            for s in signals:
                if isinstance(s, dict) and str(s.get("symbol", "")).upper() == symbol:
                    return True
        return False
    if et == "candidate_snapshot":
        tops = payload.get("top_symbols", [])
        if isinstance(tops, list):
            return symbol in [str(x).upper() for x in tops]
        return False
    return False


def _summarize_event(event_type: str, payload: dict, symbol: str) -> str:
    et = str(event_type or "")
    if et == "candidate_snapshot":
        return (
            f"candidate_snapshot pool={payload.get('pool_size', '-')}"
            f" candidates={payload.get('candidate_count', '-')}"
        )
    if et == "signal_snapshot":
        sigs = payload.get("signals", [])
        action = "-"
        strength = "-"
        if isinstance(sigs, list):
            for s in sigs:
                if isinstance(s, dict) and str(s.get("symbol", "")).upper() == symbol:
                    action = s.get("action", "-")
                    strength = s.get("strength", "-")
                    break
        return (
            f"signal_snapshot action={action} strength={strength}"
            f" actionable={payload.get('actionable_count', '-')}"
        )
    if et == "order_attempt":
        side = payload.get("side", "-")
        intent = payload.get("intent", "-")
        qty = payload.get("qty", payload.get("quote_amount", "-"))
        return f"order_attempt {side} {intent} qty={qty}"
    if et == "order_result":
        side = payload.get("side", "-")
        intent = payload.get("intent", "-")
        ok = bool(payload.get("ok"))
        status = payload.get("status", "-")
        price = payload.get("executed_price", payload.get("hint_price", "-"))
        return f"order_result {side} {intent} ok={ok} status={status} price={price}"
    return et


def _list_related_event_logs(base_file: str, keep_latest: int = 8):
    base_file = base_file or _EVENTS_FILE
    directory = os.path.dirname(base_file) or "."
    filename = os.path.basename(base_file)
    stem = filename[:-6] if filename.endswith(".jsonl") else filename
    out = []
    if os.path.exists(base_file):
        out.append(base_file)
    try:
        for name in os.listdir(directory):
            if not name.startswith(stem + "."):
                continue
            if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
                continue
            full = os.path.join(directory, name)
            if os.path.abspath(full) == os.path.abspath(base_file):
                continue
            out.append(full)
    except Exception:
        pass
    out.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0, reverse=True)
    return out[: max(1, int(keep_latest))]


def _load_chart_klines(symbol: str, interval: str, days: int, bars_limit: int = 800):
    from binance_fetcher import fetch_binance_klines

    df = fetch_binance_klines(symbol=symbol, interval=interval, days=days)
    if df is None or df.empty:
        return []
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(df.columns)):
        return []

    out = []
    for ts, row in df.tail(max(50, bars_limit)).iterrows():
        try:
            t_sec = int(ts.timestamp())
            out.append(
                {
                    "time": t_sec,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    return out


def _compute_bar_scores(symbol: str, interval: str, days: int):
    """计算每根 bar 的六书融合分数, 返回 {unix_ts: {ss, bs}} 。"""
    try:
        from binance_fetcher import fetch_binance_klines
        from hotcoin.engine.signal_worker import _add_hot_indicators
        from hotcoin.engine.hot_coin_params import HOT_COIN_FUSION_CONFIG
        from ma_indicators import add_moving_averages
        from signal_core import compute_signals_six, calc_fusion_score_six

        df = fetch_binance_klines(symbol=symbol, interval=interval, days=days)
        if df is None or len(df) < 20:
            return {}
        df = _add_hot_indicators(df)
        add_moving_averages(df, timeframe=interval)
        data_all = {interval: df}
        signals = compute_signals_six(df, interval, data_all, max_bars=800)

        scores = {}
        step = max(1, len(df) // 200)
        for i in range(max(0, len(df) - 200), len(df), step):
            try:
                dt = df.index[i]
                ss, bs = calc_fusion_score_six(signals, df, i, dt, HOT_COIN_FUSION_CONFIG)
                t_sec = int(dt.timestamp())
                scores[t_sec] = {"ss": round(ss, 1), "bs": round(bs, 1)}
            except Exception:
                continue
        return scores
    except Exception:
        return {}


def _load_trade_markers_from_pnl(symbol: str, since_ts: float, until_ts: float, limit: int = 200):
    import glob

    markers = []
    points = []
    pattern = os.path.join(_TRADES_GLOB_DIR, "hotcoin_trades_*.jsonl")
    files = sorted(glob.glob(pattern), reverse=True)[:14]
    for path in files:
        for obj in _iter_jsonl_file(path):
            if str(obj.get("symbol", "")).upper() != symbol:
                continue
            try:
                entry_ts = float(obj.get("entry_time", 0) or 0)
                exit_ts = float(obj.get("exit_time", 0) or 0)
                entry_price = float(obj.get("entry_price", 0) or 0)
                exit_price = float(obj.get("exit_price", 0) or 0)
            except Exception:
                continue
            if since_ts <= entry_ts <= until_ts:
                markers.append(
                    {
                        "time": int(entry_ts),
                        "position": "belowBar",
                        "color": "#00d68f",
                        "shape": "arrowUp",
                        "text": f"BUY {time.strftime('%H:%M', time.localtime(entry_ts))} @{entry_price:.6g}",
                    }
                )
                points.append(
                    {
                        "time": entry_ts,
                        "time_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry_ts)),
                        "side": "BUY",
                        "intent": "open",
                        "price": entry_price,
                        "qty": float(obj.get("qty", 0) or 0),
                        "source": "pnl",
                    }
                )
            if since_ts <= exit_ts <= until_ts:
                markers.append(
                    {
                        "time": int(exit_ts),
                        "position": "aboveBar",
                        "color": "#ff4d6a",
                        "shape": "arrowDown",
                        "text": f"SELL {time.strftime('%H:%M', time.localtime(exit_ts))} @{exit_price:.6g}",
                    }
                )
                points.append(
                    {
                        "time": exit_ts,
                        "time_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exit_ts)),
                        "side": "SELL",
                        "intent": "close",
                        "price": exit_price,
                        "qty": float(obj.get("qty", 0) or 0),
                        "source": "pnl",
                    }
                )
    points.sort(key=lambda x: float(x.get("time", 0)))
    markers.sort(key=lambda x: int(x.get("time", 0)))
    return markers[-limit:], points[-limit:]


def _load_order_markers(
    symbol: str,
    since_ts: float,
    until_ts: float,
    limit: int = 200,
    trace_id_filter: str = "",
):
    cached = _read_runtime_status_file()
    events_file = _EVENTS_FILE
    r = _get_runner()
    if r is not None and hasattr(r, "_events_file"):
        events_file = getattr(r, "_events_file") or events_file
    elif isinstance(cached, dict):
        events_file = str(cached.get("event_log_file") or events_file)

    markers = []
    points = []
    trace_events = {}
    symbol_events = []
    related_types = ("candidate_snapshot", "signal_snapshot", "order_attempt", "order_result")
    files = _list_related_event_logs(events_file, keep_latest=8)
    for path in files:
        for obj in _iter_jsonl_file(path):
            event_type = str(obj.get("event_type", ""))
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
            ts = obj.get("ts", payload.get("ts"))
            try:
                ts = float(ts)
            except Exception:
                continue
            if ts < since_ts or ts > until_ts:
                continue
            trace_id = str(obj.get("trace_id", "") or "")
            if trace_id_filter and trace_id != trace_id_filter:
                continue
            if event_type in related_types and _event_related_to_symbol(event_type, payload, symbol):
                raw_event = {
                    "ts": ts,
                    "event_type": event_type,
                    "trace_id": trace_id,
                    "payload": payload,
                }
                item = {
                    "ts": ts,
                    "time_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                    "event_type": event_type,
                    "summary": _summarize_event(event_type, payload, symbol),
                    "trace_id": trace_id,
                    "raw": raw_event,
                }
                symbol_events.append(item)
                if trace_id:
                    trace_events.setdefault(trace_id, []).append(item)

            if event_type != "order_result":
                continue
            if str(payload.get("symbol", "")).upper() != symbol:
                continue
            if bool(payload.get("ok")) is not True:
                continue

            side = str(payload.get("side", "")).upper()
            intent = str(payload.get("intent", "order"))
            try:
                price = float(
                    payload.get("executed_price", payload.get("fill_price", payload.get("hint_price", 0.0))) or 0.0
                )
            except Exception:
                price = 0.0
            try:
                qty = float(payload.get("executed_qty", payload.get("fill_qty", payload.get("qty", 0.0))) or 0.0)
            except Exception:
                qty = 0.0

            is_buy = side == "BUY"
            markers.append(
                {
                    "time": int(ts),
                    "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#00d68f" if is_buy else "#ff4d6a",
                    "shape": "arrowUp" if is_buy else "arrowDown",
                    "text": f"{side} {intent} {time.strftime('%H:%M', time.localtime(ts))}" + (
                        f" @{price:.6g}" if price > 0 else ""
                    ),
                }
            )
            points.append(
                {
                    "time": ts,
                    "time_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                    "side": side,
                    "intent": intent,
                    "price": price,
                    "qty": qty,
                    "trace_id": trace_id,
                    "source": "events",
                }
            )
    if not points:
        if trace_id_filter:
            return [], []
        return _load_trade_markers_from_pnl(symbol, since_ts, until_ts, limit=limit)

    symbol_events = sorted(symbol_events, key=lambda x: float(x.get("ts", 0)))
    for p in points:
        tid = str(p.get("trace_id", "") or "")
        evs = sorted(trace_events.get(tid, []), key=lambda x: float(x.get("ts", 0))) if tid else []
        if not evs and not trace_id_filter:
            # 向后兼容：老事件无 trace_id 时，按时间窗做符号内关联，避免“有点位无事件”。
            try:
                point_ts = float(p.get("time", 0) or 0)
            except Exception:
                point_ts = 0.0
            window_sec = 180.0
            evs = [e for e in symbol_events if abs(float(e.get("ts", 0) or 0.0) - point_ts) <= window_sec]
        p["related_events"] = [
            {
                "ts": e.get("ts"),
                "time_text": e.get("time_text"),
                "event_type": e.get("event_type"),
                "summary": e.get("summary"),
                "raw": e.get("raw"),
            }
            for e in evs[-12:]
        ]

    points.sort(key=lambda x: float(x.get("time", 0)))
    markers.sort(key=lambda x: int(x.get("time", 0)))
    return markers[-limit:], points[-limit:]


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


def _default_monitor_status():
    return {
        "square": {
            "enabled": True,
            "running": False,
            "last_error": "",
            "recent_posts": 0,
        },
        "twitter": {
            "enabled": False,
            "running": False,
            "last_error": "",
            "recent_posts": 0,
        },
    }


def _normalize_monitor_status(raw):
    out = _default_monitor_status()
    if not isinstance(raw, dict):
        return out
    for name in ("square", "twitter"):
        cur = raw.get(name)
        if isinstance(cur, dict):
            out[name].update(cur)
    return out


def _read_runner_monitor_status():
    runner = _get_runner()
    if runner is None:
        return None
    out = _default_monitor_status()
    try:
        sq = getattr(runner, "square_monitor", None)
        if sq and hasattr(sq, "status"):
            out["square"].update(sq.status() or {})
    except Exception:
        pass
    try:
        tw = getattr(runner, "twitter_monitor", None)
        if tw and hasattr(tw, "status"):
            out["twitter"].update(tw.status() or {})
    except Exception:
        pass
    return out


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
    r = _get_runner()
    if r is None:
        return None
    engine = getattr(r, "spot_engine", None)
    executor = getattr(engine, "executor", None) if engine is not None else None
    if executor is None or not hasattr(executor, "get_precheck_stats"):
        return None
    try:
        stats = executor.get_precheck_stats()
    except Exception:
        return None
    return _normalize_precheck_stats(stats)


def _read_runner_execution_metrics():
    r = _get_runner()
    if r is None:
        return None
    engine = getattr(r, "spot_engine", None)
    executor = getattr(engine, "executor", None) if engine is not None else None
    if executor is None or not hasattr(executor, "get_runtime_metrics"):
        return None
    try:
        stats = executor.get_runtime_metrics(window_sec=300)
    except Exception:
        return None
    return _normalize_execution_metrics(stats)


def _read_runner_risk_summary():
    r = _get_runner()
    if r is None:
        return None
    engine = getattr(r, "spot_engine", None)
    risk = getattr(engine, "risk", None) if engine is not None else None
    if risk is None or not hasattr(risk, "get_summary"):
        return None
    try:
        out = risk.get_summary()
    except Exception:
        return None
    return out if isinstance(out, dict) else {}


def _read_runner_ws_connected():
    r = _get_runner()
    if r is None:
        return None
    stream = getattr(r, "ticker_stream", None)
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


def _get_runner():
    """获取 runner 本地引用, 避免 TOCTOU 竞争。"""
    return _runner


@hotcoin_bp.route("/api/status")
def api_status():
    """候选池状态 + 最近信号 (JSON)。"""
    cached = _read_runtime_status_file()
    runner = _get_runner()
    if runner is None:
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
            cached["monitors"] = _normalize_monitor_status(cached.get("monitors"))
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
            "monitors": _default_monitor_status(),
            "message": "热点币系统未启动",
        })

    # 若状态文件是新鲜的，优先使用文件里的实时统计（含 recent_signals / active_signals）
    if cached and isinstance(cached, dict):
        ts = float(cached.get("ts", 0) or 0)
        if ts > 0 and (time.time() - ts) < 180:
            merged = dict(cached)
            merged["paper"] = runner.config.execution.use_paper_trading
            merged["execution_enabled"] = bool(getattr(runner.config.execution, "enable_order_execution", False))
            merged["ws_connected"] = hasattr(runner, "ticker_stream") and bool(runner.ticker_stream.tickers)
            engine = getattr(runner, "spot_engine", None)
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
            merged["monitors"] = _read_runner_monitor_status() or _normalize_monitor_status(merged.get("monitors"))
            return jsonify(merged)

    pool = runner.pool
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
            "score_announcement": c.score_announcement,
            "score_social": c.score_social,
            "score_sentiment": c.score_sentiment,
            "score_momentum": c.score_momentum,
            "score_liquidity": c.score_liquidity,
            "score_risk_penalty": c.score_risk_penalty,
            "has_listing_signal": c.has_listing_signal,
            "pump_phase": c.pump_phase,
            "pump_score": c.pump_score,
            "alert_level": c.alert_level,
            "alert_score": c.alert_score,
            "active_signals": c.active_signals,
            "active_filters": c.active_filters,
            "signal": "",  # filled from recent signals cache
        })

    engine = getattr(runner, "spot_engine", None)
    positions = engine.num_positions if engine else 0
    execution_enabled = bool(getattr(runner.config.execution, "enable_order_execution", False))
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
        "ws_connected": hasattr(runner, "ticker_stream") and bool(runner.ticker_stream.tickers),
        "paper": runner.config.execution.use_paper_trading,
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
        "monitors": _read_runner_monitor_status() or _normalize_monitor_status(
            cached.get("monitors") if isinstance(cached, dict) else None
        ),
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


@hotcoin_bp.route("/api/chart")
def api_chart():
    """热点币K线 + 买卖点（基于事件日志/交易记录）。"""
    symbol = _sanitize_symbol(request.args.get("symbol", ""))
    if not symbol:
        return jsonify({"error": "invalid_symbol", "message": "symbol 参数必填且需为有效交易对"}), 400
    interval = str(request.args.get("interval", "5m")).strip()
    if interval not in _VALID_CHART_INTERVALS:
        return jsonify(
            {
                "error": "invalid_interval",
                "message": f"interval 仅支持: {','.join(sorted(_VALID_CHART_INTERVALS))}",
            }
        ), 400
    days = _parse_int_arg("days", default=3, min_v=1, max_v=30)
    bars_limit = _parse_int_arg("bars_limit", default=800, min_v=100, max_v=2000)
    raw_trace_id = request.args.get("trace_id", "")
    trace_id_filter = _sanitize_trace_id(raw_trace_id)
    if str(raw_trace_id or "").strip() and not trace_id_filter:
        return jsonify({"error": "invalid_trace_id", "message": "trace_id 仅允许字母、数字、下划线和短横线"}), 400

    klines = _load_chart_klines(symbol=symbol, interval=interval, days=days, bars_limit=bars_limit)
    if not klines:
        return jsonify(
            {
                "symbol": symbol,
                "interval": interval,
                "days": days,
                "trace_id_filter": trace_id_filter,
                "klines": [],
                "markers": [],
                "trade_points": [],
                "ts": time.time(),
                "message": "无可用K线数据",
            }
        )

    since_ts = float(klines[0]["time"])
    until_ts = float(klines[-1]["time"]) + 3600.0
    markers, points = _load_order_markers(
        symbol=symbol,
        since_ts=since_ts,
        until_ts=until_ts,
        limit=200,
        trace_id_filter=trace_id_filter,
    )

    scores = {}
    include_scores = request.args.get("scores", "0") == "1"
    if include_scores:
        scores = _compute_bar_scores(symbol, interval, days)

    return jsonify(
        {
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "trace_id_filter": trace_id_filter,
            "klines": klines,
            "markers": markers,
            "trade_points": points,
            "scores": scores,
            "ts": time.time(),
        }
    )


@hotcoin_bp.route("/health")
def health():
    """热点币系统健康聚合接口。"""
    now = time.time()
    cached = _read_runtime_status_file()
    metrics = _read_runner_execution_metrics() or _normalize_execution_metrics(
        cached.get("execution_metrics") if isinstance(cached, dict) else None
    )

    runner_attached = _get_runner() is not None
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
    r = _get_runner()
    if r is None:
        return jsonify({"coins": []})
    coins = r.pool.get_all()
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

    r = _get_runner()
    if r is not None:
        posts = []
        sq = getattr(r, "square_monitor", None)
        tw = getattr(r, "twitter_monitor", None)
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
        monitors = _default_monitor_status()
        try:
            if sq and hasattr(sq, "status"):
                monitors["square"].update(sq.status() or {})
        except Exception:
            pass
        try:
            if tw and hasattr(tw, "status"):
                monitors["twitter"].update(tw.status() or {})
        except Exception:
            pass
        return jsonify({"posts": posts[:50], "ts": now, "source": "runner", "monitors": monitors})

    # Fallback: 从状态文件读取
    cached = _read_runtime_status_file()
    if isinstance(cached, dict) and "hot_posts" in cached:
        return jsonify({
            "posts": cached["hot_posts"],
            "ts": float(cached.get("ts", 0) or 0),
            "source": "status_file",
            "monitors": _normalize_monitor_status(cached.get("monitors")),
        })

    return jsonify({"posts": [], "ts": now, "source": "none", "monitors": _default_monitor_status()})


@hotcoin_bp.route("/api/emergency_close", methods=["POST"])
def api_emergency_close():
    """紧急全部平仓。需要 confirm=yes 参数防止误操作。"""
    r = _get_runner()
    if r is None:
        return jsonify({"error": "runner 未启动"}), 503

    body = request.get_json(silent=True) or {}
    confirm = body.get("confirm", "")
    if confirm != "yes":
        return jsonify({"error": "需要 confirm=yes 确认"}), 400

    reason = body.get("reason", "manual_emergency_close")
    try:
        result = r.spot_engine.emergency_close_all(reason=reason)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@hotcoin_bp.route("/api/balances")
def api_balances():
    """查询 Binance 现货账户余额。"""
    r = _get_runner()
    if r is None:
        return jsonify({"error": "runner 未启动"}), 503
    try:
        balances = r.spot_engine.executor.query_account_balances()
        return jsonify({"ok": True, "balances": balances, "ts": time.time()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@hotcoin_bp.route("/api/positions")
def api_positions():
    """返回活跃持仓详情, 含实时 PnL、止损止盈状态。"""
    r = _get_runner()
    if r is None:
        return jsonify({"ok": True, "positions": [], "pnl_summary": {}, "ts": time.time()})

    try:
        tickers = r.ticker_stream.tickers
        cfg = r.config.trading
        positions_out = []
        for sym, pos in list(r.spot_engine.risk.positions.items()):
            ticker = tickers.get(sym)
            cur_price = ticker.close if ticker and ticker.close > 0 else 0
            if pos.side == "BUY":
                unrealized_pnl = (cur_price - pos.entry_price) * pos.qty if cur_price > 0 else 0
                pnl_pct = (cur_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 and cur_price > 0 else 0
            else:
                unrealized_pnl = (pos.entry_price - cur_price) * pos.qty if cur_price > 0 else 0
                pnl_pct = (pos.entry_price - cur_price) / pos.entry_price if pos.entry_price > 0 and cur_price > 0 else 0

            holding_min = (time.time() - pos.entry_time) / 60
            value = cur_price * pos.qty if cur_price > 0 else pos.entry_price * pos.qty

            tp_tiers = cfg.take_profit_tiers or []
            next_tp = None
            for i, (tp_pct, exit_ratio) in enumerate(tp_tiers):
                if i >= pos.partial_exits:
                    next_tp = {"tier": i + 1, "target_pct": tp_pct, "exit_ratio": exit_ratio}
                    break
            trailing_active = pos.partial_exits >= len(tp_tiers)

            positions_out.append({
                "symbol": sym,
                "side": pos.side,
                "entry_price": round(pos.entry_price, 8),
                "current_price": round(cur_price, 8),
                "qty": round(pos.qty, 8),
                "value_usd": round(value, 2),
                "unrealized_pnl": round(unrealized_pnl, 4),
                "pnl_pct": round(pnl_pct, 4),
                "realized_pnl": round(pos.realized_pnl, 4),
                "max_pnl_pct": round(pos.max_pnl_pct, 4),
                "partial_exits": pos.partial_exits,
                "holding_min": round(holding_min, 1),
                "entry_time": pos.entry_time,
                "sl_pct": cfg.default_sl_pct,
                "next_tp": next_tp,
                "trailing_active": trailing_active,
                "trailing_stop_pct": cfg.trailing_stop_pct if trailing_active else None,
                "max_hold_minutes": cfg.max_hold_minutes,
                "time_remaining_min": round(max(0, cfg.max_hold_minutes - holding_min), 1),
            })

        pnl_summary = {}
        try:
            pnl_summary = r.spot_engine.pnl.get_summary()
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "positions": positions_out,
            "pnl_summary": pnl_summary,
            "risk_state": r.spot_engine.risk.get_summary(),
            "ts": time.time(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@hotcoin_bp.route("/api/trades")
def api_trades():
    """返回最近交易记录。"""
    r = _get_runner()
    limit = min(100, max(1, int(request.args.get("limit", 50))))

    if r is not None:
        try:
            with r.spot_engine.pnl._lock:
                trades = list(r.spot_engine.pnl._trades[-limit:])
            records = []
            for t in reversed(trades):
                records.append({
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": round(t.entry_price, 8),
                    "exit_price": round(t.exit_price, 8),
                    "qty": round(t.qty, 8),
                    "pnl": round(t.pnl, 4),
                    "pnl_pct": round(t.pnl_pct, 4),
                    "holding_min": round(t.holding_sec / 60, 1),
                    "reason": t.reason,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                })
            return jsonify({"ok": True, "trades": records, "ts": time.time()})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    trade_files = sorted(
        [f for f in os.listdir(_TRADES_GLOB_DIR)
         if f.startswith("hotcoin_trades_") and f.endswith(".jsonl")],
        reverse=True,
    ) if os.path.isdir(_TRADES_GLOB_DIR) else []
    records = []
    for fname in trade_files:
        if len(records) >= limit:
            break
        try:
            with open(os.path.join(_TRADES_GLOB_DIR, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            pass
    records.sort(key=lambda r: r.get("exit_time", 0), reverse=True)
    return jsonify({"ok": True, "trades": records[:limit], "ts": time.time()})


@hotcoin_bp.route("/api/params")
def api_params():
    """返回运行时关键参数快照 (只读)。"""
    from hotcoin.engine.hot_coin_params import (
        HOT_COIN_FUSION_CONFIG,
        HOT_COIN_INDICATOR_PARAMS,
        HOT_COIN_TIMEFRAMES,
        HOT_COIN_CONSENSUS_CONFIG,
        HOT_COIN_KLINE_PARAMS,
    )

    r = _get_runner()
    config_snapshot = {}
    if r is not None:
        cfg = r.config
        config_snapshot = {
            "execution": {
                "initial_capital": cfg.execution.initial_capital,
                "max_concurrent_positions": cfg.execution.max_concurrent_positions,
                "max_total_exposure_pct": cfg.execution.max_total_exposure_pct,
                "max_single_position_pct": cfg.execution.max_single_position_pct,
                "paper": cfg.execution.use_paper_trading,
                "enable_order_execution": cfg.execution.enable_order_execution,
            },
            "trading": {
                "min_consensus_strength": cfg.trading.min_consensus_strength,
                "default_sl_pct": cfg.trading.default_sl_pct,
                "take_profit_tiers": cfg.trading.take_profit_tiers,
                "trailing_stop_pct": cfg.trading.trailing_stop_pct,
                "max_hold_minutes": cfg.trading.max_hold_minutes,
                "signal_loop_sec": cfg.trading.signal_loop_sec,
            },
            "discovery": {
                "pool_max_size": cfg.discovery.pool_max_size,
                "pool_enter_score": cfg.discovery.pool_enter_score,
                "pool_exit_score": cfg.discovery.pool_exit_score,
                "listing_poll_sec": cfg.discovery.listing_poll_sec,
                "announcement_poll_sec": cfg.discovery.announcement_poll_sec,
            },
        }

    return jsonify({
        "ok": True,
        "fusion": HOT_COIN_FUSION_CONFIG,
        "indicators": HOT_COIN_INDICATOR_PARAMS,
        "timeframes": HOT_COIN_TIMEFRAMES,
        "consensus": HOT_COIN_CONSENSUS_CONFIG,
        "kline_params": HOT_COIN_KLINE_PARAMS,
        "config": config_snapshot,
        "ts": time.time(),
    })
