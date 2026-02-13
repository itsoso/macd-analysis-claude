"""结果文件 API 路由注册。"""

from flask import jsonify, request


_RESULT_API_SPECS = [
    ("api_backtest", "/api/backtest", "BACKTEST_FILE", "未找到回测数据"),
    ("api_backtest_multi", "/api/backtest_multi", "BACKTEST_MULTI_FILE", "未找到多周期回测数据, 请先生成"),
    ("api_global_strategy", "/api/global_strategy", "GLOBAL_STRATEGY_FILE", "未找到全局策略数据, 请先运行 python global_strategy.py"),
    ("api_strategy_compare", "/api/strategy_compare", "STRATEGY_COMPARE_FILE", "未找到策略对比数据, 请先运行 python strategy_compare.py"),
    ("api_strategy_optimize", "/api/strategy_optimize", "STRATEGY_OPTIMIZE_FILE", "未找到优化数据, 请先运行 python strategy_optimize.py"),
    ("api_strategy_enhanced", "/api/strategy_enhanced", "STRATEGY_ENHANCED_FILE", "未找到增强策略数据, 请先运行 python strategy_enhanced.py"),
    ("api_strategy_futures", "/api/strategy_futures", "STRATEGY_FUTURES_FILE", "未找到合约策略数据, 请先运行 python strategy_futures.py"),
    ("api_strategy_futures_v2", "/api/strategy_futures_v2", "STRATEGY_FUTURES_V2_FILE", "未找到合约Phase2数据, 请先运行 python strategy_futures_v2.py"),
    ("api_strategy_futures_v3", "/api/strategy_futures_v3", "STRATEGY_FUTURES_V3_FILE", "未找到Phase3数据, 请先运行 python strategy_futures_v3.py"),
    ("api_strategy_futures_v4", "/api/strategy_futures_v4", "STRATEGY_FUTURES_V4_FILE", "未找到Phase4数据, 请先运行 python strategy_futures_v4.py"),
    ("api_strategy_futures_v5", "/api/strategy_futures_v5", "STRATEGY_FUTURES_V5_FILE", "未找到Phase5数据, 请先运行 python strategy_futures_v5.py"),
    ("api_strategy_futures_final", "/api/strategy_futures_final", "STRATEGY_FUTURES_FINAL_FILE", "未找到终极优化数据, 请先运行 python strategy_futures_final.py"),
    ("api_timeframe_analysis", "/api/timeframe_analysis", "TIMEFRAME_ANALYSIS_FILE", "未找到时间周期分析数据, 请先运行 python strategy_timeframe_analysis.py"),
    ("api_strategy_15m", "/api/strategy_15m", "STRATEGY_15M_FILE", "未找到15m回测数据, 请先运行 python strategy_15m.py"),
    ("api_ma_strategy", "/api/ma_strategy", "MA_STRATEGY_FILE", "未找到均线策略数据, 请先运行 python ma_strategy.py"),
    ("api_combined_strategy", "/api/combined_strategy", "COMBINED_STRATEGY_FILE", "未找到融合策略数据, 请先运行 python combined_strategy.py"),
    ("api_candlestick", "/api/candlestick", "CANDLESTICK_FILE", "未找到蜡烛图策略数据, 请先运行 python candlestick_patterns.py"),
    ("api_bollinger", "/api/bollinger", "BOLLINGER_FILE", "未找到布林带策略数据, 请先运行 python bollinger_strategy.py"),
    ("api_volume_price", "/api/volume_price", "VOLUME_PRICE_FILE", "未找到量价策略数据, 请先运行 python volume_price_strategy.py"),
    ("api_five_book", "/api/five_book", "FIVE_BOOK_FILE", "未找到五书融合数据, 请先运行 python five_book_fusion.py"),
    ("api_six_book", "/api/six_book", "SIX_BOOK_FILE", "未找到六书融合数据, 请先运行 python six_book_fusion.py"),
    ("api_optimize_sl_tp", "/api/optimize_sl_tp", "OPTIMIZE_SL_TP_FILE", "未找到优化数据, 请先运行 python optimize_sl_tp.py"),
    ("api_optimize_six_book", "/api/optimize_six_book", "OPTIMIZE_SIX_BOOK_FILE", "未找到六书优化数据, 请先运行 python optimize_six_book.py"),
    ("api_backtest_30d_7d", "/api/backtest_30d_7d", "BACKTEST_30D_7D_FILE", "未找到回测对比数据, 请先运行 python backtest_30d_7d.py"),
    ("api_multi_tf_backtest_30d_7d", "/api/multi_tf_backtest_30d_7d", "MULTI_TF_BACKTEST_30D_7D_FILE", "未找到多周期回测数据, 请先运行 python backtest_multi_tf_30d_7d.py"),
    ("api_naked_kline_backtest", "/api/naked_kline_backtest", "NAKED_KLINE_BACKTEST_FILE", "未找到裸K线回测数据, 请先运行 python backtest_naked_kline.py"),
]


def _make_file_result_handler(load_json, path, error_msg):
    def _handler():
        data = load_json(path)
        if data:
            return jsonify(data)
        return jsonify({"error": error_msg}), 404

    return _handler


def register_result_api_routes(app, load_json, result_paths):
    """注册结果 JSON 文件 API。"""
    for endpoint, rule, path_key, error_msg in _RESULT_API_SPECS:
        handler = _make_file_result_handler(load_json, result_paths[path_key], error_msg)
        app.add_url_rule(rule, endpoint=endpoint, view_func=handler)

    @app.route("/api/turtle_backtest")
    def api_turtle_backtest():
        """运行海龟交易策略回测。"""
        try:
            days = int(request.args.get("days", 60))
            from turtle_strategy import main as turtle_main

            result = turtle_main(trade_days=days)
            if result:
                return jsonify(result)
            return jsonify({"error": "回测失败: 数据不足"}), 500
        except Exception as exc:
            return jsonify({"error": f"回测异常: {str(exc)}"}), 500
