"""
配置存储模块 — 基于 SQLite 的统一配置管理

替代原来分散的 JSON 文件:
  - live_trading_config.json  → config 表 (namespace='live_trading')
  - monitor_rules.json        → config 表 (namespace='monitor_rules')

特性:
  - 所有配置以 namespace + key 的 KV 形式存储
  - API Key / Secret 等敏感字段使用 Fernet 对称加密
  - 支持 JSON 对象作为 value (自动序列化)
  - 线程安全 (每次操作独立连接)
  - 自动迁移: 首次运行时从旧 JSON 文件导入
"""

import base64
import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, Optional

# ── 加密工具 ──────────────────────────────────────────
# 使用 Fernet (AES-128-CBC + HMAC) 加密敏感字段
# 密钥来源优先级: 环境变量 CONFIG_ENCRYPT_KEY > 机器指纹派生

_ENCRYPT_KEY: Optional[bytes] = None

# 需要加密的字段名 (精确匹配)
SENSITIVE_FIELDS = frozenset({
    'api_key', 'api_secret',
    'testnet_api_key', 'testnet_api_secret',
    'bot_token',
})


def _derive_key() -> bytes:
    """派生加密密钥。优先使用环境变量，否则从机器指纹生成。"""
    env_key = os.environ.get('CONFIG_ENCRYPT_KEY', '')
    if env_key:
        # 用户提供的密钥, hash 成 32 bytes
        raw = hashlib.sha256(env_key.encode()).digest()
    else:
        # 从机器指纹派生 (hostname + uid + 固定 salt)
        import socket
        fingerprint = f"{socket.gethostname()}:{os.getuid()}:macd-analysis-salt-v1"
        raw = hashlib.sha256(fingerprint.encode()).digest()
    return base64.urlsafe_b64encode(raw)


def _get_fernet():
    """延迟初始化 Fernet 实例。"""
    global _ENCRYPT_KEY
    if _ENCRYPT_KEY is None:
        _ENCRYPT_KEY = _derive_key()
    try:
        from cryptography.fernet import Fernet
        return Fernet(_ENCRYPT_KEY)
    except ImportError:
        # cryptography 未安装, fallback 到 base64 编码 (非加密, 仅混淆)
        return None


def _encrypt(value: str) -> str:
    """加密字符串。"""
    f = _get_fernet()
    if f is not None:
        return 'enc:' + f.encrypt(value.encode()).decode()
    # Fallback: base64 混淆
    return 'b64:' + base64.b64encode(value.encode()).decode()


def _decrypt(value: str) -> str:
    """解密字符串。"""
    if value.startswith('enc:'):
        f = _get_fernet()
        if f is not None:
            return f.decrypt(value[4:].encode()).decode()
        raise ValueError("cryptography 未安装, 无法解密 Fernet 数据")
    if value.startswith('b64:'):
        return base64.b64decode(value[4:]).decode()
    # 明文 (旧数据或非敏感字段)
    return value


# ── 数据库操作 ────────────────────────────────────────

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'config.db'
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS config (
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, key)
);
"""


def _get_db_path() -> str:
    """获取数据库路径。"""
    return os.environ.get('CONFIG_DB_PATH', _DEFAULT_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接。"""
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE_SQL)
    return conn


# ── 公共 API ─────────────────────────────────────────

def get_config(namespace: str) -> Dict[str, Any]:
    """读取指定 namespace 的所有配置，返回 dict。

    敏感字段自动解密。JSON 值自动反序列化。
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT key, value FROM config WHERE namespace = ?",
            (namespace,)
        ).fetchall()
    finally:
        conn.close()

    result = {}
    for key, raw_value in rows:
        if raw_value is None:
            result[key] = None
            continue

        # 解密
        if key in SENSITIVE_FIELDS or raw_value.startswith(('enc:', 'b64:')):
            try:
                raw_value = _decrypt(raw_value)
            except Exception:
                pass  # 解密失败保留原值

        # 尝试 JSON 反序列化
        try:
            result[key] = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError):
            result[key] = raw_value

    return result


def set_config(namespace: str, data: Dict[str, Any]):
    """写入 / 更新指定 namespace 的配置。

    只更新传入的 key, 不影响其他已有 key。
    敏感字段自动加密。
    """
    conn = _get_conn()
    try:
        for key, value in data.items():
            # 序列化
            if isinstance(value, (dict, list, bool)):
                str_value = json.dumps(value, ensure_ascii=False)
            elif value is None:
                str_value = None
            else:
                str_value = str(value)

            # 加密敏感字段 (跳过模板占位符和空值)
            if (key in SENSITIVE_FIELDS
                    and str_value is not None
                    and str_value not in ('', 'YOUR_API_KEY', 'YOUR_API_SECRET',
                                          'YOUR_TESTNET_API_KEY', 'YOUR_TESTNET_API_SECRET',
                                          'YOUR_BOT_TOKEN')
                    and not str_value.startswith(('enc:', 'b64:'))):
                str_value = _encrypt(str_value)

            conn.execute(
                """INSERT INTO config (namespace, key, value, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(namespace, key)
                   DO UPDATE SET value = excluded.value,
                                 updated_at = CURRENT_TIMESTAMP""",
                (namespace, key, str_value)
            )
        conn.commit()
    finally:
        conn.close()


def delete_config(namespace: str, key: Optional[str] = None):
    """删除配置。key=None 则删除整个 namespace。"""
    conn = _get_conn()
    try:
        if key:
            conn.execute(
                "DELETE FROM config WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
        else:
            conn.execute(
                "DELETE FROM config WHERE namespace = ?",
                (namespace,)
            )
        conn.commit()
    finally:
        conn.close()


def list_namespaces() -> list:
    """列出所有 namespace。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT namespace FROM config ORDER BY namespace"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


# ── 便捷函数: 实盘配置 ──────────────────────────────

NS_LIVE_TRADING = 'live_trading'
NS_MONITOR_RULES = 'monitor_rules'
NS_HOTCOIN = 'hotcoin'


def get_live_trading_config() -> Dict[str, Any]:
    """读取实盘交易配置 (嵌套结构)。

    DB 中以扁平 KV 存储 (如 'api', 'strategy', 'telegram' 各自是一个 JSON 对象),
    读取后还原为完整的嵌套 dict。
    """
    flat = get_config(NS_LIVE_TRADING)
    if not flat:
        return {}

    # 顶层 key 直接返回 (api, telegram, strategy, risk 已是 dict)
    return flat


def set_live_trading_config(config_dict: Dict[str, Any]):
    """保存实盘交易配置。

    接收完整的嵌套 config dict, 拆分后存入 DB。
    敏感字段 (api_key 等) 从 api 子字典中提取并单独加密存储。
    """
    flat = {}

    # 提取 API 敏感字段单独存储
    api_data = config_dict.get('api', {})
    if api_data:
        for sensitive_key in ('api_key', 'api_secret',
                              'testnet_api_key', 'testnet_api_secret'):
            if sensitive_key in api_data:
                flat[sensitive_key] = api_data[sensitive_key]
        # API 子对象存储时去掉敏感字段
        api_clean = {k: v for k, v in api_data.items()
                     if k not in SENSITIVE_FIELDS}
        if api_clean:
            flat['api'] = api_clean

    # Telegram 敏感字段
    tg_data = config_dict.get('telegram', {})
    if tg_data:
        if 'bot_token' in tg_data:
            flat['bot_token'] = tg_data['bot_token']
        tg_clean = {k: v for k, v in tg_data.items()
                    if k not in SENSITIVE_FIELDS}
        if tg_clean:
            flat['telegram'] = tg_clean

    # 其他顶层 key 直接存储
    for key in ('strategy', 'risk', 'phase', 'execute_trades',
                'initial_capital', 'log_dir', 'data_dir'):
        if key in config_dict:
            flat[key] = config_dict[key]

    set_config(NS_LIVE_TRADING, flat)


def get_live_trading_config_full() -> Dict[str, Any]:
    """读取完整嵌套结构的实盘配置 (用于导出/编辑器显示)。

    将 DB 中扁平存储的敏感字段重新组装回嵌套 dict。
    """
    flat = get_config(NS_LIVE_TRADING)
    if not flat:
        return {}

    result = {}

    # 重组 API
    api_data = flat.pop('api', {}) if isinstance(flat.get('api'), dict) else {}
    for sk in ('api_key', 'api_secret', 'testnet_api_key', 'testnet_api_secret'):
        if sk in flat:
            api_data[sk] = flat.pop(sk)
    if api_data:
        result['api'] = api_data

    # 重组 Telegram
    tg_data = flat.pop('telegram', {}) if isinstance(flat.get('telegram'), dict) else {}
    if 'bot_token' in flat:
        tg_data['bot_token'] = flat.pop('bot_token')
    if tg_data:
        result['telegram'] = tg_data

    # 其他直接放入
    result.update(flat)

    return result


def get_monitor_rules() -> Dict[str, Any]:
    """读取监控规则。"""
    return get_config(NS_MONITOR_RULES)


def set_monitor_rules(rules: Dict[str, Any]):
    """保存监控规则。"""
    set_config(NS_MONITOR_RULES, rules)


def get_hotcoin_config() -> Dict[str, Any]:
    """读取热点币配置。"""
    return get_config(NS_HOTCOIN)


def set_hotcoin_config(config_dict: Dict[str, Any]):
    """保存热点币配置。"""
    if not isinstance(config_dict, dict):
        raise ValueError("hotcoin config must be a dict")
    set_config(NS_HOTCOIN, config_dict)


# ── 迁移: 从旧 JSON 文件导入 ────────────────────────

def migrate_from_json(
    config_json_path: str = '',
    monitor_json_path: str = '',
    force: bool = False,
):
    """从旧 JSON 文件迁移数据到 DB。

    Args:
        config_json_path: live_trading_config.json 路径
        monitor_json_path: monitor_rules.json 路径
        force: 即使 DB 已有数据也强制覆盖
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if not config_json_path:
        config_json_path = os.path.join(base_dir, 'live_trading_config.json')
    if not monitor_json_path:
        monitor_json_path = os.path.join(base_dir, 'data', 'live', 'monitor_rules.json')

    migrated = []

    # 迁移 live_trading_config.json
    if os.path.exists(config_json_path):
        existing = get_live_trading_config_full()
        if not existing or force:
            with open(config_json_path, 'r') as f:
                data = json.load(f)
            set_live_trading_config(data)
            migrated.append(f"live_trading: {config_json_path}")

    # 迁移 monitor_rules.json
    if os.path.exists(monitor_json_path):
        existing = get_monitor_rules()
        if not existing or force:
            with open(monitor_json_path, 'r') as f:
                data = json.load(f)
            set_monitor_rules(data)
            migrated.append(f"monitor_rules: {monitor_json_path}")

    return migrated


def ensure_migrated():
    """确保旧数据已迁移 (应用启动时调用)。

    只在 DB 为空时自动迁移, 不覆盖已有数据。
    """
    existing = get_live_trading_config_full()
    if not existing:
        results = migrate_from_json(force=False)
        if results:
            print(f"[ConfigStore] 自动迁移完成: {results}")
    return True
