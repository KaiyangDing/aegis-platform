# ADR-013: 人工审批加固——自建 Approvals 于 interrupt 原语

- 日期：2026-09-05
- 状态：已接受（M2.7 定稿，2026-09-08；恢复单入口的崩溃分诊随 M2.9 同日补入）

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

自建落地前再探（2026-09-08）：
- 列表首个 after_model（循环出口节点，声明 `can_jump_to=["end"]`）里 `interrupt(载荷)` 首次 raise、run 干净返回；resume 时同节点**整体重放**、任务 id 与首次相同、`interrupt()` 原样返回 `Command(resume=…)` 的值；resume 传入的新 context 在节点内可见。返回私有通道 + 未配对调用时模型→工具边把它们送进 tools；返回配对 ToolMessage + `jump_to="end"` 时直达 after_agent、不再调模型。
- 同一节点两次 `interrupt()`：单值 resume 只满足第一个、第二个再次挂起，按 id 键控同样一次只满足一个——一个节点只放一个 interrupt。
- `stream_mode=["custom"]` 不外流 `__interrupt__`；挂起后 `aget_state().next` 指向挂起节点，`tasks[0].interrupts` 恰一个、`tasks[0].error` 为空。
- 挂起态直接喂新输入（不带 Command）：框架作废 pending 的中断、从 model 重跑，旧 AIMessage 的 tool_calls 永久悬空。
- after_model 在 `interrupt()` 之前崩溃：checkpoint 停在该节点之前，恢复重放该节点后正常挂起。

## 决策

1. **不用 `HumanInTheLoopMiddleware`，自建 `Approvals` 中间件直接用 `interrupt()` 原语**。差距五条：reject 继续循环而非 `cancelled` 终止；英文 ToolMessage 不可替换；无单 / TTL / CAS / 落账；决定数错误令线程僵住；反序运行使其无法排在闸门之后。**借用其载荷形态**（`action_requests / review_configs`）与决定类型名（approve / reject）以利前端兼容。
2. **流程**：after_model 对最后一条 AIMessage 的每个**未配对** tool_call（闸门已配对的打断 / 幻觉名不算）跑 `risk_policy(args, tenant_config)`：坏参数不进审批（没有副作用要保护，执行层回填）；谓词崩溃视为命中阻断（fail-closed：该调用配对错误 ToolMessage，不进审批）；命中 → 以派生 id 幂等开单（`uuid5(会话 id, 框架任务 id, "approval", 模型侧调用 id)`，与事件 id 同一机制、不同点名；`expires_at = now() + approval_ttl_s`，数据库时钟）→ `approval_requested{approval_id, tool_name, args, expires_at}` 事件 → run_state 翻转 awaiting_approval → **一个 interrupt 载荷装本轮全部单**（每个 action_request 多带 approval_id / expires_at 与中文说明）。run 干净返回，进程可下线。同一轮闸门之外的调用也等决定，批准后按声明序全部执行（前作挂起时弃置其后调用）。
3. **恢复 = 同一入口** `resume(session_id, approval_id=None)`（前作签名，approval_id 是调用方意图）。非 None = 计划内审批续跑：会话必须在 awaiting_approval（否则 SessionBusy：并发恢复已有赢家或会话在跑）→ 挂起点缺失（翻转之后、挂起 checkpoint 之前崩过）先以 None 重放到挂起点 → approval_id 必须属于挂起载荷 → 载荷里的审批单**全部终态**才放行（仍 pending → ValueError，等其余决定）→ T3 awaiting→running CAS（输家 SessionBusy）→ `Command(resume={"decisions": […]})` 从挂起节点重放。节点重放以**审批表终态**为准（决定形态只是唤醒信号与前端兼容）：全部批准 → `approval_decided` 事件 + 通行证 `{模型侧调用 id: approval_id}` 写入私有通道；任一拒绝 / 撤回 / 超时 → `approval_decided(approved=false)` / `approval_cancelled` / `approval_expired` + 全部未配对调用配对 ToolMessage + `termination(cancelled)` 跳 end，**零 LLM 调用**。None = 崩溃恢复（M2.9）：审批单仍 pending 的挂起是健康的（零事件、不计恢复次数）；决定已落而续跑没来得及的，与计划内续跑同一条路径。
4. **五态 CAS 逐条平移**：`decide` 查 `status = pending AND expires_at > now()`（到期批准被拒，fail-closed）；`cancel` 只查 pending 不查过期；`expire_due` 可注入时钟；双坐席同时 decide 恰一赢家、输家得 False；开单 `INSERT … ON CONFLICT (id) DO NOTHING` 并返回现状（节点重放不重复开单）。
5. **批准后前置校验挂点**位于工具执行七步的风险闸门之后、write-ahead 之前：持通行证者过 `precheck(tool_name, 参数快照)`，否决写 `precheck_vetoed{approval_id, tool_name, observation, detail}` 事件、以 observation 回填模型（detail 只进事件与日志）、不终止、不进 write-ahead；放行者 write-ahead 之后把 `tool_call` 事件 id 回填审批单（`attach_event` CAS 恰一次，批准已兑现的唯一凭证）。M3 注入真实校验。
6. **形态守卫在恢复入口**：决定由审批表终态翻译而来，数目恒等于挂起数；坐席回调只翻转表状态再调 `resume(approval_id)`；不许把外来的决定列表直接送进 `Command(resume)`。
7. **approvals 表**：`id`（派生 uuid）、`tenant_id`、`session_id`、`run_id`、`tool_name`、`args jsonb`、`status`、`operator_id`、`event_id`（回填）、`expires_at`、`decided_at`、`created_at`；`approval_ttl_s` 来自 `LoopPolicy`。
8. **无锁世界的并发续跑**：两个坐席同时 resume——都读到 awaiting 的过校验后由 T3 CAS 裁决恰一赢家；后读到 running 的直接 SessionBusy。会话锁归后续里程碑。

## 实证（探针）

复现方式：`HumanInTheLoopMiddleware(interrupt_on={"refund": True})` 配剧本模型跑到挂起，读 `aget_state().tasks[].interrupts` 载荷；分别以 approve / reject / 决定数错误 resume，读 state 消息与 `next`；三件 after_model 中间件混排观察运行顺序（2026-09-05）。自建形态：循环出口 after_model 内 `interrupt()`，以 `Command(resume=…)`、单值 / 键控多中断、配对 + 跳 end、挂起态新输入四种方式恢复，读节点内任务 id / context / 返回值与 state（2026-09-08）。定性结论已列于背景节。

审批链路（M2.7，`tests/engine/runtime/test_runtime_approvals.py` 与 `tests/domain/test_approvals.py`、真 PG 端到端在 `test_runtime_pg.py`）：
- 挂起：四事件（首事件、llm_call、llm_result、approval_requested）后干净结束，无末事件；run_state = awaiting_approval；checkpoint `next` 指向审批节点、挂起载荷带 approval_id；单据 pending；挂起期间调用未配对。
- 批准续跑：前作形态 C 的十一事件（4 + 7），通行证放行、write-ahead 后审批单 `event_id` 回填、恢复用新 run_id、seq 接续、归一化后的事件形态快照；两单一个载荷、按声明序执行；同轮闸门外的查询也等审批后执行。
- 拒绝 / 撤回 / 超时（注入时钟）：对应事件 + `cancelled` 终止零 LLM 调用、调用配对"未获人工审批"回填、不追加兜底话术；混合决定一律取消；到期后批准被拒。
- 仍 pending 不许恢复；挂起态新输入被会话互斥拒绝；重放不重复开单、事件不重复；并发恢复恰一赢家；外来单号被拒；翻转之后、挂起 checkpoint 之前崩溃可先重放回挂起再续跑；谓词崩溃 fail-closed 不开单且 run 照常；前置校验否决落事件、不执行、单据不回填；闸门终止绕过审批；取消信号在恢复段的工具检查点生效；缺审批单存取件时响亮失败。
- 五态 CAS（真 PG）：幂等开单且 TTL 不重算、双坐席恰一赢家、到期批准被拒、撤回不查过期且终态不许改写、到期扫描只碰到期 pending、注入时钟、回填恰一次、非 uuid 拒绝。

## 后果

- 前作审批语义（五态 CAS、到期 fail-closed、reject → cancelled、计划内恢复与灾难恢复同路径、TOCTOU 挂点）全部保留；挂起与恢复的传输壳由框架 `interrupt()` + checkpointer 兑现。
- 自建件承担开单、事件、翻转、决定翻译的正确性；库件的载荷与决定名被借用，前端接口形态与官方件兼容。
- 审批中断依赖 `durability="sync"`（ADR-011）：挂起点的 checkpoint 必须已落盘，进程才可下线。
- after_model 节点重放意味着 Approvals 的每个动作都必须幂等（单、事件、翻转），这是与前作最大的实现差异；节点内 `interrupt()` 之前经流写出的非事件帧会在重放时再发一次，本仓只经去重的事件外流。
- 会话互斥必须前置在起跑入口（挂起态的新输入会作废中断并留悬空调用）；无会话锁时"运行中的会话是不是死了"只能由恢复调用方断言。
- 同一轮多个调用在审批期间全部等待（前作弃置其后调用），审批后按声明序执行；`DISCARDED_NOTE` 话术暂无消费者。
