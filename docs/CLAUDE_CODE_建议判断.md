# Claude Code 建议判断摘要

## 第一层：落地提交

| 建议 | 判断 | 说明 |
|------|------|------|
| 分两个 commit（策略 vs 安全/工程） | **靠谱** | 策略变更与 infra 分离，便于回滚与 code review。 |
| requirements 加版本上限 | **部分靠谱** | 已有注释「生产建议 pip freeze」。加松散上限（如 pandas<3）可防大版本升级踩雷；过紧上限会频繁改版，生产仍建议用 pip freeze 生成锁文件。 |

## 第二层：可维护性

| 建议 | 判断 | 说明 |
|------|------|------|
| 配置单一来源（ab_test 从 StrategyConfig 生成 stable_base） | **方向靠谱，实现需谨慎** | live_config 用 `leverage`，backtest/engine 用 `lev`；还有 `sell_threshold`、`spot_cooldown` 等需与 _build_default_config 对齐。若用 asdict(StrategyConfig()) 需做 key 映射（如 lev←leverage），否则 A/B 基线会漂移。 |
| 补充测试（gate_add=35、warmup 200） | **靠谱** | 成本低，防回归；warmup 测试建议直接测 optimize_six_book 内逻辑或公式常量，避免只测手写算式。 |
| 第九轮执行+记录 | **已做** | 第九轮已接入（--live --spot），跑完结果需补入文档。 |

## 第三层：架构重构（P2）

| 建议 | 判断 | 说明 |
|------|------|------|
| 回测引擎统一（9→1） | **收益存疑** | 各 backtest_*.py 场景不同（日回测、30d/7d、区间报告），强行合并易引入回归；-60% 维护成本为估计，可作中长期目标。 |
| app.py 拆分 | **靠谱** | 1300+ 行拆 auth/routes/export 可读性明显提升。 |
| 前端 JS 抽取 | **部分靠谱** | 可维护性更好；「加载速度 +30%」依赖缓存与体积，未必达得到。 |
| Walk-Forward 4→6–8 窗口 | **靠谱** | 验证可靠性，可配置即可。 |

**结论**：第一层两 commit 建议采纳；requirements 可加松散上限。配置单一来源值得做但需处理字段名与默认值一致性。P2 架构项可排期，不必立刻上。
