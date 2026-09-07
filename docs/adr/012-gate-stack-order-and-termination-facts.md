# ADR-012: 闸门栈序与终止事实

- 日期：2026-09-05
- 状态：草案（M2.4 落地前定稿；实证节随闸门与栈序测试补齐）

## 背景

前作的六道终止闸门（轮数 / 单步超时 / 会话预算 / 重复调用 / 协议违规 / 取消）与八值终止原因住在自研循环里，每次终止写一条 `loop_terminated` 末事件、按原因单点选兜底话术。本仓把循环骨架交给 `create_agent`，闸门只能以 `AgentMiddleware` 钩子表达；框架同时提供了现成策略件（`ModelCallLimitMiddleware`、`ToolCallLimitMiddleware`、`SummarizationMiddleware`、`PIIMiddleware`、重试 / 兜底 / 工具重试等）。本 ADR 裁决：哪些现成件用、哪些自建；自建中间件的顺序；终止事实的唯一出口；工具如何进图。

本仓 venv 探针事实（langchain 1.4.0 / langgraph 1.2.11）：
- 每个钩子是独立图节点 `"{中间件名}.{钩子}"`；`before_agent / before_model` 按列表序运行，`after_model / after_agent` **反序**，`wrap_model_call / wrap_tool_call` 列表首个最外层。
- 只有用 `hook_config(can_jump_to=[…])` 声明的钩子才获得条件边；未声明的钩子返回 `jump_to` **静默无效**（无报错）。声明后跳 `end` 直接绕过后续钩子。
- `after_model` 里 `jump_to="end"` 跳过工具节点时，state 里留下带未配对 tool_calls 的 AIMessage；模型→工具边在"全部 tool_calls 已配对 ToolMessage"时回到模型。
- `ModelCallLimitMiddleware`：超限注入英文 `AIMessage("Model call limits exceeded: …")` 并跳 end，无终止原因事实、话术不可注入、不配对悬空 tool_calls；`ToolCallLimitMiddleware` 同为英文回填。
- `SummarizationMiddleware`：`token_counter` 可注入；摘要模型被 `with_retry`（最多三次、指数抖动）包裹；产物是 `RemoveMessage(REMOVE_ALL_MESSAGES)` + 英文包装的 HumanMessage + 保留段；摘要失败直接抛出。
- `PIIMiddleware`：内建五类正则，策略 block / redact / mask / hash，没有"本人数据允许清单"。
- ToolNode：多个 tool_calls **并行**执行；同步工具函数走线程池；执行期异常默认**重抛穿出 run**；schema 错参与幻觉工具名以英文 ToolMessage 回填；`wrap_tool_call` 可以不调框架 handler、自己产出 ToolMessage。
- `wrap_model_call` 的 handler 返回完整响应，客户端流块经回调在 handler 返回之前就到达消费者：wrap 里拿不到流、也撤不回已外送 token（ADR-006 的重试落点由此维持在网关）。
- `model_settings` 的键全部作为 kwargs 传给 `bind_tools`，再落到模型 `_astream` 的具名参数——网关的档位 / deadline / session 载体由此成立。
- 私有状态通道（`Annotated[..., PrivateStateAttr]`）不出现在 `ainvoke` 输出与输入 / 输出 schema，但出现在 `aget_state()` 与 `values / updates` 流帧里；**跨 run 持久化**（同一 thread 第二次 run 从上次终值继续）；`PrivateStateAttr` 住 `langchain.agents.middleware.types`（未列入包 `__all__`）。
- `recursion_limit` 默认 25 个 superstep；每个钩子节点各算一个，多轮工具循环轻易超限。
- `StructuredTool(name, description, args_schema=<pydantic 模型>, coroutine=…)` 合法；`convert_to_openai_tool` 从工具的调用 schema 派生参数，剥掉 `title` 与顶层 `additionalProperties`，其余与模型 `model_json_schema()` 一致；coroutine 抛出的异常经图调用直接穿出 run。

## 决策

1. **栈序即语义**：`middleware = [RunEvents, Guards, AegisSummarization, Approvals, Gates, ModelCall, ToolExec]`。判据五条：① 事件写入者要么最先（首事件）要么最后（末事件）——`RunEvents` 居首（after 反序 = 最后）；② 判定类先于计数类——闸门（协议违规 / 重复调用）在审批之前，闸门终止后审批钩子不再运行（条件边绕过）；③ 压缩先于预算预检——摘要在 `before_model` 排在闸门之前；④ 出口守卫在 wrap 内、AIMessage 进 state 之前，泄漏原文不进 checkpoint；⑤ 所有可跳转钩子显式 `hook_config`，静态测试钉住列表顺序与每个节点的出边。
2. **六道闸门全部自建**于 `Gates`（before_model：取消、轮数；after_model：协议、重复）与 `ModelCall`（wrap 内：预算预检、单步超时映射）。`ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` 不用：英文话术不可注入、无终止原因事实、跳 end 时不配对悬空 tool_calls。
3. **终止事实的唯一出口**：`termination` 私有通道是唯一终止信号（wrap 内经 `ExtendedModelResponse.command` 写入，钩子内直接返回）；`loop_terminated{reason, iteration, detail, cause?}` 由 `RunEvents.after_agent` 单点写为末事件；八值终止原因逐字沿用前作，`gateway_rejected` 零话术；兜底话术按原因单点选取。
4. **跳转纪律**：任何返回 `jump_to` 的钩子必在 `hook_config(can_jump_to)` 声明；审查项 + 一条"未声明即静默无效"的回归测试防止真中间件漂移。
5. **打断、拒绝、终止一律配对 ToolMessage**：最后一条 AIMessage 的每个 tool_call 都有配对，state 里不留悬空调用（否则下一轮上游 400）。
6. **`recursion_limit` 由 `LoopPolicy` 推导**并显式传入每次调用（式子与测试随 M2.3 定）。
7. **工具进图只作载体**：`ToolDef` 转 `StructuredTool` 只带名字、说明与参数模型，框架侧 handler 是"被调用即抛 RuntimeError"的哨兵；`ToolExec.wrap_tool_call` 不调框架 handler，自己执行七步（严校验、身份注入、风险闸门、write-ahead、超时、规范化、事件）。中间件栈漏挂 `ToolExec` 时整条 run 炸，而不是工具静默绕过七步。工具执行**每 run 一把锁串行**（前作顺序执行语义；事件序确定、幂等审计可读），工具实现一律 `async def`（避免线程池）。
8. **私有通道纪律**：运行时状态通道全部 `PrivateStateAttr`；run 级计数（轮数 / 违规 / 重复 / 终止）在 `before_agent` 显式归零——通道跨 run 持久化，不归零第二轮对话一开场就撞轮数上限。`values / updates` 流帧含私有通道，对外流式通道不转发它们。
9. **现成件处置**：`SummarizationMiddleware` 接受并子类化——注入自家 token 尺、替换英文包装为中文、摘要模型不经 `with_retry`（重试权威唯一在网关，ADR-006）、摘要失败 fail-open 为提取式降级且绝不二次调 LLM、写 `summary_updated` 事件；`RemoveMessage(REMOVE_ALL)` 语义接受（原文在事件与历史 checkpoint）。`PIIMiddleware` 不用（无本人数据允许清单）；`ModelRetryMiddleware` / `ModelFallbackMiddleware` 不用（与网关重试权威相悖）；`ToolRetryMiddleware` / `ToolErrorMiddleware` 不用（写工具恒单次、结局由七步接管）；`ContextEditingMiddleware` / `TodoListMiddleware` / `LLMToolSelectorMiddleware` / `ShellToolMiddleware` 等与本仓语义无交集，不用。
10. **话术单一事实源**：L2 话术住 `engine/runtime/utterances.py`；框架英文串（限流回填、工具错误回填、摘要包装）一律不进事件与对话——对应中间件在其产生之前拦截（幻觉工具名在 after_model 先于 ToolNode 判定）。
11. **嵌套模型双发**：`stream_mode="messages"` 下网关内再调候选时每块出现两次（候选 run 与网关 run），消费方只取网关级块。

## 实证（探针）

复现方式：三个各自实现六钩子的记录型中间件组合进 `create_agent`，打印图节点与运行顺序；声明与不声明 `can_jump_to` 的 after_model 各返回 `jump_to="end"` 观察循环是否继续；`ModelCallLimitMiddleware(run_limit=2)` 与 `ToolCallLimitMiddleware` 在剧本模型下跑到超限，读 state 里的回填消息；`SummarizationMiddleware` 注入计数型 token_counter 并检查 `_summary_model` 类型与产物形态；ToolNode 下三个慢工具并行计时、同步工具记录线程、执行期异常与 schema 错参各一；`Annotated[int, PrivateStateAttr]` 通道跨两次 run 读值；`StructuredTool` 仅带 coroutine 经 `convert_to_openai_tool` 与 `create_agent` 各走一遍。定性结论已列于背景节；对应的运行时类型测试（ToolDef 到 StructuredTool 的参数同源、哨兵在图内穿出）已钉住，栈序静态测试、六道闸门触发测试与"未声明即静默"回归随 M2.4 补入本节。

模型调用中间件与图工厂（M2.3）：
- `wrap_model_call` 返回 `ExtendedModelResponse(command=Command(update=…))` 的状态更新在同一 superstep 的 after_model 里已可见，command 里的消息经 reducer 追加在模型结果之后；after_agent 返回的消息同样入 state；`stream_writer` 在 wrap 与 after_agent 内均可用。
- wrap 内读到的框架任务 id 与 before_model 节点不同；wrap 首次抛异常后从 checkpoint 恢复，模型节点重放时任务 id 与首次相同——事件 id 派生因此对模型节点同样成立。
- 最小可行 `recursion_limit` = 节点执行数 + 1：只有 before_agent / after_agent 两个钩子节点的栈，k 轮工具循环需要 2k + 4；推导式（外圈钩子节点 + max_iterations × 每轮节点数 + 终止那一遍的 before_model 节点 + 1）对该栈给出 2·max_iterations + 3，与探针值恰差一个余量，减 2 即触发 `GraphRecursionError`。
- 模型节点必须产出一条不带 tool_calls 的 AIMessage 才能让模型→工具边走向 end：模型结果为空列表时，框架回退到历史里最后一条 AIMessage，若它带已配对的 tool_calls 便会回到模型节点循环。故 wrap 内终止一律在结果里放兜底 AIMessage（零话术时为空 AIMessage）。
- 一轮多个 tool_calls 由框架并行派发，仅用互斥锁只能保证不重叠、不能保证顺序（实测出现 b 先于 a）；按声明序等待前驱完成后才执行，事件序与执行序与声明序一致。
- 以上由 `tests/engine/runtime/test_runtime_*.py` 钉住：文本 run 五事件、工具 run 九事件、四组异常映射与零话术、流级中断作废重发消耗迭代、内部异常泄漏裸穿、run 起点通道归零、第二轮 seq 接续、`recursion_limit` 紧致性、图缓存与指纹敏感性、真 PG 端到端。

## 后果

- 六道闸门与终止事实完全在本仓掌握，现成计数件的英文话术与无事实问题不进系统；代价是两件自建中间件与栈序纪律，靠静态测试维持。
- 工具执行完全绕开框架 handler：框架 ToolNode 退化为路由与幻觉名兜底，本仓承担七步的全部正确性。
- 并行工具执行关闭：多工具调用逐个执行，换取事件序确定与幂等审计可读。
- 摘要接受现成件但关闭其重试与英文包装，登记表随行。
- `PrivateStateAttr` 来自框架的次公开模块路径；升级时随探针复核。
