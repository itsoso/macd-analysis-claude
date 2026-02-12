"""Web 路由注册模块。"""

from .pages import register_page_routes
from .result_apis import register_result_api_routes

__all__ = ["register_page_routes", "register_result_api_routes"]
