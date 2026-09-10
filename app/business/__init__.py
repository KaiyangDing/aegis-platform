"""L3 业务薄片（S2；ADR-014）：租户静态配置、AgentSpec 装配、三工具、进程内模拟后端、业务话术。

依赖方向：business 可 import core / engine / domain，反向禁止（import-linter 第五条契约）；入口（routers / main）装配它。
运行时对这里一无所知——AgentSpec 是唯一注入面。
"""
