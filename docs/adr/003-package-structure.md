# ADR-003: 包结构与依赖方向

- 日期：2026-09-02
- 状态：已接受

## 背景

模板=Argus ADR-017（三域包+入口留根，其单向规则经实际依赖图验证）。v1 的分层纪律——依赖严格单向、"L1 不知道 Agent、L2 不知道客服"——需要一个框架世界的落点，否则 langchain 胶水代码会天然打穿分层。

## 决策

- `app/` = `core/`（横切件，无业务知识：config/logs/db/security/redis/breakers/limits/tokens）、`engine/`（引擎：gateway/runtime/guards/rag/fakes/utterances，子模块按里程碑立，不预建空壳）、`domain/`（数据契约与配额：models/schemas/usage/tenants）、`routers/`（HTTP 层）+ 入口留根 `main.py / worker.py / deps.py`（部署字符串 `app.main:app`、`app.worker.WorkerSettings` 永不改）。
- 依赖单向三契约，交 **import-linter** 执法（M0 起进本地门，pyproject `[tool.importlinter]`）：
  1. core 不依赖兄弟包；2. engine 只依赖 core；3. domain 只依赖 core。routers 与入口负责组装，可 import 一切。
- v1 纪律映射：gateway 不 import runtime（"L1 不知道 Agent"）；runtime 对客服业务零知识（AgentSpec 注入面，"L2 不知道客服"）。跨包部分由三契约覆盖；engine 内部（gateway↛runtime）在 M2 立子包后补第四条契约。
- 新模块归属三判据：无业务知识→core；引擎算法→engine；数据契约与配额→domain。不设 services/repositories/utils 仪式层；按域不按层。
- 预留登记：`worker.py`（M3 arq 接线）、`deps.py`（M1 起组合根：按租户装配缓存/计量实例）目前为 docstring 占位——恒空即例外，在此显式登记。

## 后果

- 分层犯规当场红：`uv run lint-imports`。
- engine 不得 import domain 的 ORM——引擎所需持久化经由 core 接口或注入达成；若 M1 计量落账实践中此规则不可守，以新 ADR 修订而非静默打穿（Argus 的对应解法=engine.retrieval 走裸 SQL 不碰 ORM）。
