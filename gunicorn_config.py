"""
Gunicorn 生产环境配置
MACD Analysis 六书融合策略平台
"""

import multiprocessing
import os

# ============================================================
# 服务器配置
# ============================================================
bind = "127.0.0.1:5100"
workers = 3
worker_class = "sync"
worker_connections = 1000
timeout = 360           # 多周期信号检测可能需要 3-5 分钟
keepalive = 5
max_requests = 1000          # 每个 worker 处理 N 个请求后自动重启，防止内存泄漏
max_requests_jitter = 50     # 随机偏移，避免所有 worker 同时重启

# ============================================================
# 日志配置
# ============================================================
accesslog = "/opt/macd-analysis/logs/access.log"
errorlog = "/opt/macd-analysis/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ============================================================
# 进程管理
# ============================================================
daemon = False                # systemd 管理，不需要 daemon 模式
pidfile = "/opt/macd-analysis/gunicorn.pid"
graceful_timeout = 30
preload_app = True            # 预加载应用，减少 worker 启动时间

# ============================================================
# 钩子函数
# ============================================================
def on_starting(server):
    """服务器启动前：创建日志目录"""
    log_dir = "/opt/macd-analysis/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def post_fork(server, worker):
    """Worker 创建后"""
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def pre_exec(server):
    """服务器重启前"""
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    """服务器就绪"""
    server.log.info("Server is ready. Spawning workers")


def worker_int(worker):
    """Worker 被中断"""
    worker.log.info(f"Worker received INT or QUIT signal (pid: {worker.pid})")


def worker_abort(worker):
    """Worker 被强制终止"""
    worker.log.info(f"Worker received SIGABRT signal (pid: {worker.pid})")
