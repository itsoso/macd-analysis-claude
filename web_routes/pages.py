"""页面路由注册。"""

from flask import render_template


def register_page_routes(app):
    """注册与静态展示相关的页面路由。"""

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
        return render_template(
            'page_multi_tf_backtest_30d_7d.html',
            active_page='multi-tf-backtest-30d-7d',
        )

    @app.route('/strategy/multi-tf-date-range-report')
    def page_multi_tf_date_range_report():
        return render_template(
            'page_multi_tf_date_range_report.html',
            active_page='multi-tf-date-range-report',
        )

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

    @app.route('/strategy/naked-kline')
    def page_naked_kline():
        return render_template('page_naked_kline.html', active_page='naked-kline')

    @app.route('/strategy/naked-kline-daily')
    def page_naked_kline_daily():
        return render_template('page_naked_kline_daily.html', active_page='naked-kline-daily')

    @app.route('/strategy/multi-tf-daily')
    def page_multi_tf_daily():
        return render_template('page_multi_tf_daily.html', active_page='multi-tf-daily')
