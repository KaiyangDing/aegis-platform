# ADR-014: 薄 HTTP 面与 v2.0 范围

- 日期：2026-09-09
- 状态：已接受（M3.1–M3.3 落地，2026-09-09）

## 背景

M2 收官时运行时只有门面 `AgentRuntime.run / resume`，没有身份来源、没有业务工具、没有 HTTP 端点。v2.0 的范围裁定为：L1 网关 + L2 运行时 + 一层刚好能让"发消息 → 工具 → 审批挂起 → 坐席批准 → 续跑 → 回放"闭环成立的业务面，以及可复算的凭证。本 ADR 记录这层业务面的装配事实与 v2.0 未包含的范围。

探针 / 实测事实（fastapi 0.141.1、pyjwt 2.13.0）：
- FastAPI 路由级 `dependencies=[…]` 按列表序解析且先于端点参数依赖；同一可调用在一次请求内命中依赖缓存只跑一次。
- `include_router` 的结果在 `app.routes` 里是 `_IncludedRouter`（无 `path` 属性）；路径以 `app.openapi()["paths"]` 为准。
- `AgentRuntime.run` 是异步生成器：会话归属校验与 T1 CAS 在首次 `anext` 才执行；审批挂起时生成器在 `approval_requested` 之后正常结束（无 `loop_terminated`）。
- httpx `ASGITransport` 一次性收齐 `StreamingResponse` 正文；流内异常若被生成器接住则响应仍是 200。
- pyjwt `decode(algorithms=["HS256"], options={"require": [...]})` 拒绝 alg=none 与缺 exp 的票；短于 32 字节的 HS256 密钥只发告警。
- `AgentSpec` 指纹含 `tenant_config` 与 `owned_values`：按用户注入 `owned_values` 会让图缓存按用户分裂。
- `FakeReplyChatModel` 不发 tool_calls：fake 模式下 HTTP 链路只能走纯对话；审批闭环由剧本网关的测试覆盖。

## 决策

1. **身份与角色住 core**（`core/auth.py`）：HS256 短期 JWT，claims `sub / tid / role / iat / exp`；三角色 `user / operator / admin`；验签显式锁 `algorithms` 并强制 claim 清单；租户段在验签时过与网关同一条字符集规则（`^[A-Za-z0-9_-]{1,64}$`，两处常量以测试互钉）。失败分层：空钥 / 弱钥（< 32 字节）是服务端配置错误，`ValueError` fail-loud，生产环境空钥启动即炸；token 无效是客户端问题，`InvalidToken` → 401 且只回显异常类型名。无登录端点与用户目录：`scripts/mint_token.py` 签发，生产接 IdP 只换签发方。单密钥，无轮换窗。
2. **依赖序即语义**：认证依赖写入 `request.state.principal`；入站限流的身份函数 `tenant_identity` 从那里拼键 `tenant:<tid>`，读不到主体即 `RuntimeError`——依赖列表排反不许静默退化成 IP 键（ADR-010 决策 6 的执行器）。`require_roles` 是矩阵执行器：401 与 403 严格分家。
3. **业务面是独立包 `app/business`**，可 import core / engine，反向禁止（import-linter 第五条契约；`deps.py` 仍是唯一同时 import engine 与 domain 的模块，ADR-009 决策 8 不变）。租户配置是代码内静态表（`tools` 白名单、`approval_threshold`、`model_tier`、`entry_classifier`、两个预算、`monthly_token_budget`），变更入口是代码提交；`build_spec(tenant_id)` 是运行时唯一的注入面生产者，`preheat_specs()` 在 lifespan 首步把每个租户构造一遍（白名单点名不到即启动失败）；`owned_values` 留空、`memory / retrieval` 层预算显式 0。
4. **三工具与模拟后端**：`order_query`（读）/ `refund_apply`（写，`risk_policy` 阈值取租户配置、缺省 0 即任意正金额都要人批）/ `ticket_create`（写，显式豁免）。归属判定在工具内（`tenant_id` 与 `user_id` 双比对，身份全部来自运行时注入的 ctx），不存在 / 他租 / 他人三种失败逐字节同一话术。写工具把 write-ahead 的 `tool_call` 事件 id 作为 `idempotency_key` 透传后端；后端是进程内类，按键判重，同一把钥匙第二次开门返回首次结果并标 `duplicate=True`；业务拒绝以 `{"error": …}` 回填模型而不是异常。
5. **SSE 帧**：`id: <events.seq>`、`event: <事件类型>`、`data: <单行 JSON>`；同一编码器服务活流（`AgentEvent`）与回放（事实源行）。合成帧 `done / error` 不是事实源事件，不写 `id:`。可见性清单单点：终端用户不可见 `llm_call / llm_result / summary_updated / guardrail_triggered / precheck_vetoed`，坐席 / 管理员全量。
6. **`POST /v1/chat`**：依赖列表 `[认证, 租户级固定窗限流]`；准入链 401 → 429 → 403（租户未在静态表）→ 422 → 会话首见即建（`IntegrityError` 回读）、既有会话必须属于本租户且由本人创建否则 404 → 会话在 `awaiting_approval` 则 409 并附 pending 单号 → **先取首帧再返回流**（首帧前的 `SessionBusy` → 409、`ValueError` → 404；首帧之后的异常以 `error` 帧收流，只回显异常类型名，`run_state` 留 `running` 交恢复入口）。流末：见到 `loop_terminated` 合成 `done{reason}`；末事件为 `approval_requested` 合成 `done{reason: awaiting_approval, approvals}`。客户端断连不翻译为取消信号。
7. **`POST /v1/approvals/{id}`**：坐席 / 管理员 → 单不存在 404 → 单属他租 403（两种员工角色都是租户级身份）→ 入口惰性 `expire_due` → `decide` CAS（pending 且未过期才翻；False → 409 并附当前终态）→ `resume(approval_id)` 同步吸干（`SessionBusy` 与 `ValueError` 均 409）→ JSON 摘要 `done / awaiting_approval / no_op` + `reply` + `next_approval_id` + 事件类型序列。决定先落审批表终态，`resume` 只从表翻译决定（ADR-013 决策 6）。
8. **`GET /v1/sessions/{id}/events?after_seq=&limit=`**：一次性读事实源；游标 = `max(after_seq, Last-Event-ID)`，坏头按 0；终端用户只看本人会话，坐席 / 管理员看本租户任一会话，他租 404；以 `done{reason: snapshot, run_state, next_seq, count}` 收尾。无活尾轮询、无 LISTEN/NOTIFY。
9. **组合根**（`main.py` lifespan）：`settings / runtime_parts / specs` 挂 `app.state` 供路由读，路由不 import `main`；月度预算 resolver 经 `gateway_for(budget_resolver=)` 注入网关（`deps.py` 保持业务无知）；模拟后端进程内单点；关停摘下。
10. **v2.0 范围**：L1 网关（ADR-005–010）+ L2 运行时（ADR-011–013）+ 本 ADR 的薄业务面 + 凭证（`reports/`）。未包含：检索增强（RAG）与 embedding 通道、行级安全（RLS）、意图路由与 FAQ 直答、转人工三档摘要、逐 token 流式通道、活尾重订阅、用户侧撤回审批端点、批准后业务前置校验器（挂点保留）、周期任务与 worker 进程、trace 还原与指标端点、回放回归与离线评测、成本对照实验与压测、演示前端。差距表见 README。

## 实证（探针）

复现方式：`tests/core/test_auth.py`（alg=none 手工票、缺 exp、弱钥）；`tests/routers/test_chat.py::test_auth_dependency_precedes_rate_limit`（依赖列表序）与 `test_tenant_rate_limit_429_with_real_redis`；`tests/routers/test_chat.py::test_midstream_failure_yields_error_frame_not_status_code`（首帧后异常）；`tests/test_e2e_pg.py`（真 PG 的 HTTP 审批闭环）；`tests/test_main_api.py`（真 lifespan、OpenAPI 路径、fake 模式往返）；`scripts/e2e_flow.py --real`（真实上游走通审批闭环，记录 `reports/2026-09-09-e2e-real.md`）。定性结论已列于背景节；同一 system prompt 下真实模型是否调用退款工具随用户措辞而变（两次运行一拒一调），运行时的保证只覆盖"调了工具之后"的七步与审批闭环。

## 后果

- 三端点 × 三角色的矩阵写完即全部矩阵；新增端点须先在 `routers/common.py` 的状态码分工里找到位置。
- 租户配置进 spec 指纹：改配置即换图（有意）；按用户注入 `owned_values` 会打穿图缓存（本期不做）。
- 模拟后端进程内判重：重启即忘；换成持久化台账时工具面零改动。
- 会话在 `running` 因首帧后异常留下时没有周期对账，靠 `resume(approval_id=None)` 的恢复分诊（ADR-012 决策 15）。
- 静态租户表意味着新租户 = 一次提交与一次重启。
