#!/bin/bash
#
# 部署脚本 - MACD Analysis 六书融合策略平台
# 目标: invest.executor.life (47.237.191.17:22222)
#
# 用法:
#   ./deploy.sh              # 完整部署（首次）
#   ./deploy.sh update       # 仅更新代码和重启服务
#   ./deploy.sh restart      # 仅重启服务
#   ./deploy.sh status       # 查看服务状态
#   ./deploy.sh logs         # 查看运行日志
#   ./deploy.sh ssl          # 单独配置 SSL 证书
#

set -euo pipefail

# ============================================================
# 配置变量
# ============================================================
REMOTE_HOST="47.237.191.17"
REMOTE_PORT="22222"
REMOTE_USER="root"
REMOTE_DIR="/opt/macd-analysis"
DOMAIN="invest.executor.life"
APP_PORT="5100"  # gunicorn 监听端口，避免与其他服务冲突
SERVICE_NAME="macd-analysis"
WORKERS=3        # gunicorn worker 数量

# SSH 连接参数
SSH_CMD="ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST}"
SCP_CMD="scp -P ${REMOTE_PORT} -o StrictHostKeyChecking=no"
RSYNC_CMD="rsync -avz --progress -e 'ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no'"

# 本地项目路径
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}==>${NC} ${BLUE}$1${NC}"; }

# ============================================================
# 需要同步的文件列表
# ============================================================
SYNC_FILES=(
    # 核心应用
    "app.py"
    "config.py"
    "requirements.txt"
    "gunicorn_config.py"
    # 数据获取
    "binance_fetcher.py"
    "data_fetcher.py"
    # 指标和策略
    "indicators.py"
    "ma_indicators.py"
    "kdj_strategy.py"
    "bollinger_strategy.py"
    "candlestick_patterns.py"
    "volume_price_strategy.py"
    "ma_strategy.py"
    # 融合策略
    "five_book_fusion.py"
    "six_book_fusion.py"
    "combined_strategy.py"
    "global_strategy.py"
    # 期货和优化
    "strategy_futures.py"
    "strategy_futures_final.py"
    "strategy_futures_v2.py"
    "strategy_futures_v3.py"
    "strategy_futures_v4.py"
    "strategy_futures_v5.py"
    "optimize_six_book.py"
    "optimize_sl_tp.py"
    # 回测
    "backtest.py"
    "backtest_30d_7d.py"
    "strategy_compare.py"
    "strategy_enhanced.py"
    "strategy_optimize.py"
    "strategy_15m.py"
    "strategy_timeframe_analysis.py"
    # 海龟交易策略
    "turtle_strategy.py"
    # 其他
    "main.py"
    "visualization.py"
    "test_core.py"
    "test_timeframes.py"
    # 实盘交易系统
    "live_config.py"
    "trading_logger.py"
    "notifier.py"
    "risk_manager.py"
    "order_manager.py"
    "live_signal_generator.py"
    "live_trading_engine.py"
    "live_runner.py"
    "performance_tracker.py"
)

SYNC_DIRS=(
    "templates"
    "divergence"
)

# JSON 数据文件
JSON_FILES=(
    "backtest_result.json"
    "backtest_multi.json"
    "backtest_all_intervals.json"
    "backtest_30d_7d_result.json"
    "global_strategy_result.json"
    "strategy_compare_result.json"
    "bollinger_result.json"
    "candlestick_result.json"
    "combined_strategy_result.json"
    "five_book_fusion_result.json"
    "six_book_fusion_result.json"
    "ma_strategy_result.json"
    "optimize_six_book_result.json"
    "optimize_sl_tp_result.json"
    "strategy_15m_result.json"
    "strategy_enhanced_result.json"
    "strategy_futures_result.json"
    "strategy_futures_final_result.json"
    "strategy_futures_v2_result.json"
    "strategy_futures_v3_result.json"
    "strategy_futures_v4_result.json"
    "strategy_futures_v5_result.json"
    "strategy_optimize_result.json"
    "timeframe_analysis_result.json"
    "timeframe_test_result.json"
    "volume_price_result.json"
    "turtle_result.json"
)

# ============================================================
# 功能函数
# ============================================================

check_connection() {
    log_step "检查服务器连接..."
    if $SSH_CMD "echo 'connected'" &>/dev/null; then
        log_info "服务器连接正常"
    else
        log_error "无法连接到服务器 ${REMOTE_HOST}:${REMOTE_PORT}"
        exit 1
    fi
}

sync_code() {
    log_step "同步代码到服务器..."
    
    # 创建远程目录
    $SSH_CMD "mkdir -p ${REMOTE_DIR}/templates ${REMOTE_DIR}/divergence"
    
    # 同步 Python 文件
    local file_count=0
    for f in "${SYNC_FILES[@]}"; do
        if [ -f "${LOCAL_DIR}/${f}" ]; then
            $SCP_CMD "${LOCAL_DIR}/${f}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${f}" &>/dev/null
            file_count=$((file_count + 1))
        else
            log_warn "文件不存在，跳过: ${f}"
        fi
    done
    log_info "已同步 ${file_count} 个 Python 文件"
    
    # 同步目录
    for d in "${SYNC_DIRS[@]}"; do
        if [ -d "${LOCAL_DIR}/${d}" ]; then
            eval $RSYNC_CMD --delete "'${LOCAL_DIR}/${d}/'" "'${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${d}/'" &>/dev/null
            log_info "已同步目录: ${d}/"
        fi
    done
    
    # 同步 JSON 数据文件
    local json_count=0
    for f in "${JSON_FILES[@]}"; do
        if [ -f "${LOCAL_DIR}/${f}" ]; then
            $SCP_CMD "${LOCAL_DIR}/${f}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${f}" &>/dev/null
            json_count=$((json_count + 1))
        fi
    done
    log_info "已同步 ${json_count} 个 JSON 数据文件"
}

setup_venv() {
    log_step "设置 Python 虚拟环境..."
    $SSH_CMD << 'REMOTE_SCRIPT'
cd /opt/macd-analysis

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活并安装依赖
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "依赖安装完成"
pip list --format=columns | grep -E "flask|gunicorn|pandas|numpy" || true
REMOTE_SCRIPT
    log_info "虚拟环境配置完成"
}

setup_systemd() {
    log_step "配置 systemd 服务..."
    $SSH_CMD << REMOTE_SCRIPT
cat > /etc/systemd/system/${SERVICE_NAME}.service << 'EOF'
[Unit]
Description=MACD Analysis - 六书融合策略平台
After=network.target

[Service]
Type=notify
User=root
Group=root
WorkingDirectory=${REMOTE_DIR}
Environment="PATH=${REMOTE_DIR}/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=${REMOTE_DIR}/venv/bin/gunicorn \
    --config ${REMOTE_DIR}/gunicorn_config.py \
    app:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=on-failure
RestartSec=5
KillMode=mixed
TimeoutStopSec=10

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

# 安全加固
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
echo "systemd 服务配置完成"
REMOTE_SCRIPT
    log_info "systemd 服务已创建并启用"
}

setup_nginx() {
    log_step "配置 Nginx 反向代理..."
    $SSH_CMD << REMOTE_SCRIPT
cat > /etc/nginx/sites-available/${SERVICE_NAME} << 'EOF'
# MACD Analysis - 六书融合策略平台
# Domain: ${DOMAIN}

server {
    listen 80;
    server_name ${DOMAIN};

    # 访问日志
    access_log /var/log/nginx/${SERVICE_NAME}_access.log;
    error_log  /var/log/nginx/${SERVICE_NAME}_error.log;

    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \\\$host;
        proxy_set_header X-Real-IP \\\$remote_addr;
        proxy_set_header X-Forwarded-For \\\$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \\\$scheme;

        # 超时设置
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
        proxy_send_timeout 60s;

        # 缓冲设置
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # 静态文件缓存（如果未来添加 static/ 目录）
    location /static/ {
        alias ${REMOTE_DIR}/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:${APP_PORT}/;
        access_log off;
    }
}
EOF

# 创建软链接
ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/${SERVICE_NAME}

# 测试配置
nginx -t && echo "Nginx 配置测试通过" || { echo "Nginx 配置有误!"; exit 1; }

# 重载 Nginx
systemctl reload nginx
echo "Nginx 配置已应用"
REMOTE_SCRIPT
    log_info "Nginx 反向代理配置完成"
}

setup_ssl() {
    log_step "配置 SSL 证书 (Let's Encrypt)..."
    $SSH_CMD << REMOTE_SCRIPT
# 确保 certbot 已安装
if ! command -v certbot &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq certbot python3-certbot-nginx
fi

# 申请证书
certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --email admin@executor.life --redirect

echo "SSL 证书配置完成"
REMOTE_SCRIPT
    log_info "SSL 证书已配置"
}

start_service() {
    log_step "启动服务..."
    # 确保日志目录和子目录都存在 (修复 gunicorn error.log 找不到的问题)
    $SSH_CMD "mkdir -p ${REMOTE_DIR}/logs/live ${REMOTE_DIR}/logs/test"
    $SSH_CMD << REMOTE_SCRIPT
systemctl restart ${SERVICE_NAME}
sleep 2

# 检查服务状态
if systemctl is-active --quiet ${SERVICE_NAME}; then
    echo "✅ 服务启动成功"
    systemctl status ${SERVICE_NAME} --no-pager -l | head -15
else
    echo "❌ 服务启动失败"
    journalctl -u ${SERVICE_NAME} --no-pager -n 30
    exit 1
fi

# 测试 HTTP 响应
sleep 1
HTTP_CODE=\$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${APP_PORT}/ || echo "000")
if [ "\$HTTP_CODE" = "200" ]; then
    echo "✅ HTTP 健康检查通过 (200 OK)"
else
    echo "⚠️  HTTP 响应码: \$HTTP_CODE"
fi
REMOTE_SCRIPT
    log_info "服务启动完成"
}

show_status() {
    log_step "服务状态..."
    $SSH_CMD << REMOTE_SCRIPT
echo "=== systemd 服务状态 ==="
systemctl status ${SERVICE_NAME} --no-pager -l 2>/dev/null || echo "服务未找到"

echo ""
echo "=== Nginx 状态 ==="
systemctl is-active nginx && echo "Nginx: 运行中" || echo "Nginx: 未运行"

echo ""
echo "=== 进程信息 ==="
ps aux | grep gunicorn | grep -v grep || echo "无 gunicorn 进程"

echo ""
echo "=== 端口监听 ==="
ss -tlnp | grep ${APP_PORT} || echo "端口 ${APP_PORT} 未监听"

echo ""
echo "=== 磁盘使用 ==="
du -sh ${REMOTE_DIR} 2>/dev/null || echo "项目目录不存在"

echo ""
echo "=== 域名解析测试 ==="
HTTP_CODE=\$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${APP_PORT}/ 2>/dev/null || echo "000")
echo "本地 HTTP 响应码: \$HTTP_CODE"
REMOTE_SCRIPT
}

show_logs() {
    log_step "查看最近日志..."
    $SSH_CMD "journalctl -u ${SERVICE_NAME} --no-pager -n 50 --since '1 hour ago'"
}

# ============================================================
# 完整部署流程
# ============================================================
full_deploy() {
    log_step "开始完整部署..."
    echo "============================================================"
    echo "  项目: MACD Analysis 六书融合策略平台"
    echo "  目标: ${DOMAIN} (${REMOTE_HOST}:${REMOTE_PORT})"
    echo "  路径: ${REMOTE_DIR}"
    echo "  端口: ${APP_PORT}"
    echo "============================================================"
    
    check_connection
    sync_code
    setup_venv
    setup_systemd
    setup_nginx
    start_service
    
    echo ""
    log_step "部署完成！"
    echo "============================================================"
    echo -e "  ${GREEN}HTTP 访问:${NC}  http://${DOMAIN}"
    echo ""
    echo -e "  ${YELLOW}提示:${NC} 运行以下命令配置 SSL 证书:"
    echo -e "        ${BLUE}./deploy.sh ssl${NC}"
    echo ""
    echo -e "  ${YELLOW}后续更新:${NC}"
    echo -e "        ${BLUE}./deploy.sh update${NC}   # 更新代码并重启"
    echo -e "        ${BLUE}./deploy.sh restart${NC}  # 仅重启服务"
    echo -e "        ${BLUE}./deploy.sh status${NC}   # 查看状态"
    echo -e "        ${BLUE}./deploy.sh logs${NC}     # 查看日志"
    echo "============================================================"
}

# 仅更新代码
update_deploy() {
    log_step "更新部署..."
    check_connection
    sync_code
    
    # 更新依赖（如果 requirements.txt 有变化）
    $SSH_CMD << 'REMOTE_SCRIPT'
cd /opt/macd-analysis
source venv/bin/activate
pip install -r requirements.txt -q 2>&1 | tail -3
REMOTE_SCRIPT
    
    start_service
    log_info "更新部署完成！"
}

# ============================================================
# 主入口
# ============================================================
case "${1:-}" in
    update)
        update_deploy
        ;;
    restart)
        check_connection
        start_service
        ;;
    status)
        check_connection
        show_status
        ;;
    logs)
        check_connection
        show_logs
        ;;
    ssl)
        check_connection
        setup_ssl
        ;;
    *)
        full_deploy
        ;;
esac
