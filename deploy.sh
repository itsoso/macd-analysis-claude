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

# GitHub 仓库
GITHUB_REPO="git@github.com:itsoso/macd-analysis-claude.git"
GITHUB_BRANCH="main"

# SSH 连接参数 (绕过 IdentityAgent 代理 / ProxyCommand，避免 banner exchange 超时)
SSH_OPTS="-p ${REMOTE_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityAgent=none -o ProxyCommand=none -o ConnectTimeout=15"
SSH_CMD="ssh ${SSH_OPTS} ${REMOTE_USER}@${REMOTE_HOST}"
SCP_CMD="scp -P ${REMOTE_PORT} -o StrictHostKeyChecking=no -o IdentityAgent=none -o ProxyCommand=none"

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
    log_step "通过 GitHub 同步代码到服务器..."

    $SSH_CMD << REMOTE_SCRIPT
cd ${REMOTE_DIR} 2>/dev/null || {
    echo "首次部署: clone 仓库..."
    git clone ${GITHUB_REPO} ${REMOTE_DIR}
    cd ${REMOTE_DIR}
}

# 确保是 git 仓库
if [ ! -d ".git" ]; then
    echo "目录已存在但非 git 仓库, 重新初始化..."
    git init
    git remote add origin ${GITHUB_REPO} 2>/dev/null || git remote set-url origin ${GITHUB_REPO}
fi

# 拉取最新代码
echo "拉取 ${GITHUB_BRANCH} 分支最新代码..."
git fetch origin ${GITHUB_BRANCH}
git reset --hard origin/${GITHUB_BRANCH}

echo "当前版本: \$(git log --oneline -1)"
echo "文件数量: \$(find . -name '*.py' | wc -l) 个 Python 文件"
echo "模板数量: \$(find templates/ -name '*.html' 2>/dev/null | wc -l) 个 HTML 模板"
REMOTE_SCRIPT

    log_info "代码同步完成 (via GitHub)"
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
        proxy_read_timeout 360s;
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
    log_step "检查依赖更新..."
    $SSH_CMD << 'REMOTE_SCRIPT'
cd /opt/macd-analysis
source venv/bin/activate
pip install -r requirements.txt -q 2>&1 | tail -5
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
