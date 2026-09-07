# ADR-013: 人工审批加固——自建 Approvals 于 interrupt 原语

- 日期：2026-09-05
- 状态：草案（M2.7 落地前定稿；实证节随审批测试补齐）

## 背景

前作的审批是一台状态机：审批单五态（pending / approved / rejected / cancelled / expired）以 CAS 翻转，`decide` 只认 `pending AND expires_at > now`（到期 fail-closed）、`cancel` 不查过期、双坐席恰一赢家；开单 → 事件 → run_state 翻转 → 干净挂起（进程可下线）；拒绝 / 撤回 / 到期 → `cancelled` 终止且不再调 LLM；批准后执行前必须重跑业务前置校验（审批的是数小时前的参数快照）；恢复走同一单入口。本仓的框架提供 `HumanInTheLoopMiddleware` 与 `interrupt()` / checkpointer 原语，本 ADR 裁决用哪一层。

本仓 venv 探针事实（langchain 1.4.0 / langgraph 1.2.11）：
- `HumanInTheLoopMiddleware(interrupt_on={tool: True | {allowed_decisions, description, when}})` 在 after_model 内 `interrupt(HITLRequest)`；`when(ToolCallRequest) -> bool` 谓词可表达条件式闸门。
- 载荷形态 `{action_requests: [{name, args, description}], review_configs: [{action_name, allowed_decisions}]}`；resume 用 `Command(resume={"decisions": [...]})`，决定四型 approve / edit / reject / respond。
- **reject → 英文 `ToolMessage(status="error")` 且循环继续**（模型再答一轮），不是终止。
- 决定数与挂起数不匹配 → `ValueError`，线程 `next=()` 带错误任务（僵住）。
- 没有审批单、没有 TTL、没有 CAS、没有落账；resume 时 after_model **整节点重放**（钩子计数 +1），模型不重放。
- after_model 钩子按中间件列表**反序**运行：库件放在列表哪里都排不到本仓闸门之后（闸门也是 after_model）。
- `interrupt_before=["tools"]` 静态断点只提供 `next=('tools',)`，没有载荷。

## 决策

1. **不用 `HumanInTheLoopMiddleware`，自建 `Approvals` 中间件直接用 `interrupt()` 原语**。差距五条：reject 继续循环而非 `cancelled` 终止；英文 ToolMessage 不可替换；无单 / TTL / CAS / 落账；决定数错误令线程僵住；反序运行使其无法排在闸门之后。**借用其载荷形态**（`action_requests / review_configs`）与决定类型名（approve / reject）以利前端兼容。
2. **流程**：after_model 对最后一条 AIMessage 的每个 tool_call 跑 `risk_policy(args, tenant_config)`，谓词崩溃视为命中阻断（fail-closed：该调用替换为错误 ToolMessage，不进审批）；命中 → 以派生 id 幂等开单（`expires_at = now + approval_ttl_s`，数据库时钟）→ `approval_requested` 事件 → run_state 翻转 awaiting_approval → `interrupt(载荷)`。run 干净返回，进程可下线。
3. **恢复 = 同一入口**：`resume(session_id)` 即 `ainvoke(Command(resume=…), config)`，与崩溃恢复同为"从最后一个 checkpoint 之后重放"。重放进入同一 after_model 节点时读审批表终态：approved → `approval_decided` 事件 + 通行证写入 state（仅供恢复入口用）；rejected / expired / cancelled → 对应事件 + 配对 ToolMessage + `termination(cancelled)` 跳 end，**零 LLM 调用**。开单与事件 id 派生自稳定任务身份（ADR-011），节点重放不重复开单。
4. **五态 CAS 逐条平移**：`decide` 查 `status = pending AND expires_at > now()`（到期批准被拒，fail-closed）；`cancel` 只查 pending 不查过期；`expire_due` 可注入时钟；双坐席同时 decide 恰一赢家、输家得 False 并留痕。
5. **批准后前置校验挂点**位于工具执行七步的风险闸门之后、write-ahead 之前（M3 注入真实校验），否决写 `precheck_vetoed` 事件、不终止。
6. **API 层形态守卫**：决定数恒等于挂起数，形态错误在 API 层拒绝、不许触达 `Command(resume)`；审批回调只翻转表状态，事件与恢复动作统一在恢复入口完成。
7. **approvals 表**：`id`、`tenant_id`、`session_id`、`run_id`、`tool_name`、`args jsonb`、`status`、`expires_at`、`decided_by`、`decided_at`、`created_at`（细节随 M2.7 定）；`approval_ttl_s` 来自 `LoopPolicy`（租户级策略）。

## 实证（探针）

复现方式：`HumanInTheLoopMiddleware(interrupt_on={"refund": True})` 配剧本模型跑到挂起，读 `aget_state().tasks[].interrupts` 载荷；分别以 approve / reject / 决定数错误 resume，读 state 消息与 `next`；三件 after_model 中间件混排观察运行顺序。定性结论已列于背景节；自建 `Approvals` 的挂起链路事件形态、五态 CAS 矩阵、到期 fail-closed、reject → cancelled 零 LLM、重放不重复开单与"进程可下线"断言随 M2.7 补入本节。

## 后果

- 前作审批语义（五态 CAS、到期 fail-closed、reject → cancelled、计划内恢复与灾难恢复同路径、TOCTOU 挂点）全部保留；挂起与恢复的传输壳由框架 `interrupt()` + checkpointer 兑现。
- 自建件承担开单、事件、翻转、决定翻译的正确性；库件的载荷与决定名被借用，前端接口形态与官方件兼容。
- 审批中断依赖 `durability="sync"`（ADR-011）：挂起点的 checkpoint 必须已落盘，进程才可下线。
- after_model 节点重放意味着 Approvals 的每个动作都必须幂等（单、事件、翻转），这是与前作最大的实现差异。
