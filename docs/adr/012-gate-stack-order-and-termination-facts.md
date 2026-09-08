# ADR-012: 闸门栈序与终止事实

- 日期：2026-09-05
- 状态：已接受（M2.4 定稿，2026-09-07；决策 6 增补与决策 12 / 13 随 M2.5 / M2.6 同日补入）

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
- 闸门与工具落地时再探（2026-09-07）：after_model 只配对一部分 tool_calls 时，模型→工具边只把未配对的调用送进 tools，全部配对则回到 before_model 链头；同 id 的 AIMessage 经 add_messages 原位替换。两个 after_model 钩子都声明可跳转时各得一组条件边，列表靠后者先跑、它跳 end 时列表靠前者被绕过。after_model 跳 "model" 落到 before_model 链头（before_model 钩子重跑）；before_model 跳 "end" 后 after_agent 仍运行。工具任务返回 `Command(update=…)` 可写私有通道，但同一 superstep 两个任务写同一个无 reducer 通道即 `InvalidUpdateError`；tools→model 边不看 `jump_to`。在 wrap 内直接 await 工具协程时，协程内可取运行时上下文与任务 id。`SummarizationMiddleware` 的 `keep=("tokens", N)` 切点不拆 AI/Tool 对但可落在 AI/Human 之间；`RemoveMessage(REMOVE_ALL)` 只改当前 state，历史 checkpoint 仍含原文；`model.ainvoke(messages, tier=…, session_id=…, deadline_s=…)` 的 kwargs 原样到达 `_agenerate`。 列表首个 after_model 节点（循环出口）的出边是模型→工具边而非中间件默认边：返回空更新时按 state 三分（无 tool_calls → after_agent；有未配对调用 → tools；全部已配对 → before_model 链头），且该边先读 `jump_to`——这一个节点即使未声明 `can_jump_to`，跳转也生效；"未声明即静默无效"只对经中间件默认边接线的节点成立（before_agent / before_model / 非首个 after_model / after_agent，以及无工具时的 after_model）。

## 决策

1. **栈序即语义**：`middleware = [RunEvents, Guards, AegisSummarization, Approvals, Gates, ModelCall, ToolExec]`。判据五条：① 事件写入者要么最先（首事件）要么最后（末事件）——`RunEvents` 居首（after 反序 = 最后）；② 判定类先于计数类——闸门（协议违规 / 重复调用）在审批之前，闸门终止后审批钩子不再运行（条件边绕过）；③ 压缩先于预算预检——摘要在 `before_model` 排在闸门之前；④ 出口守卫在 wrap 内、AIMessage 进 state 之前，泄漏原文不进 checkpoint；⑤ 所有可跳转钩子显式 `hook_config`，静态测试钉住列表顺序与每个节点的出边。
2. **六道闸门全部自建**于 `Gates`（before_model：取消、轮数；after_model：协议、重复）与 `ModelCall`（wrap 内：预算预检、单步超时映射）。`ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` 不用：英文话术不可注入、无终止原因事实、跳 end 时不配对悬空 tool_calls。
3. **终止事实的唯一出口**：`termination` 私有通道是唯一终止信号（wrap 内经 `ExtendedModelResponse.command` 写入，钩子内直接返回）；`loop_terminated{reason, iteration, detail, cause?}` 由 `RunEvents.after_agent` 单点写为末事件；八值终止原因逐字沿用前作，`gateway_rejected` 零话术；兜底话术按原因单点选取。
4. **跳转纪律**：任何返回 `jump_to` 的钩子必在 `hook_config(can_jump_to)` 声明；审查项 + 一条"未声明即静默无效"的回归测试防止真中间件漂移。
5. **打断、拒绝、终止一律配对 ToolMessage**：最后一条 AIMessage 的每个 tool_call 都有配对，state 里不留悬空调用（否则下一轮上游 400）。
6. **`recursion_limit` 由 `LoopPolicy` 推导**并显式传入每次调用（式子与测试随 M2.3 定）。**增补（M2.4）**：推导式 = 最长路径的节点执行数 + 1，最长路径是"工具循环撞 max_iterations 由 before_model 终止"：before_agent 节点 + max_iterations × (before_model 节点 + model + after_model 节点 + tools) + 终止那一遍的 before_model 节点 + after_agent 节点 + 1；对该路径精确（余量 0），测试在该路径上钉 `-1` 即 `GraphRecursionError`。M2.3 形态的栈没有 before_model 钩子，wrap 内终止的 model 节点未被计入，该路径少算一个节点——Gates 入栈后消失。
7. **工具进图只作载体**：`ToolDef` 转 `StructuredTool` 只带名字、说明与参数模型，框架侧 handler 是"被调用即抛 RuntimeError"的哨兵；`ToolExec.wrap_tool_call` 不调框架 handler，自己执行七步（严校验、身份注入、风险闸门、write-ahead、超时、规范化、事件）。中间件栈漏挂 `ToolExec` 时整条 run 炸，而不是工具静默绕过七步。工具执行**每 run 一把锁串行**（前作顺序执行语义；事件序确定、幂等审计可读），工具实现一律 `async def`（避免线程池）。
8. **私有通道纪律**：运行时状态通道全部 `PrivateStateAttr`；run 级计数（轮数 / 违规 / 重复 / 终止）在 `before_agent` 显式归零——通道跨 run 持久化，不归零第二轮对话一开场就撞轮数上限。`values / updates` 流帧含私有通道，对外流式通道不转发它们。
9. **现成件处置**：`SummarizationMiddleware` 接受并子类化——注入自家 token 尺、替换英文包装为中文、摘要模型不经 `with_retry`（重试权威唯一在网关，ADR-006）、摘要失败 fail-open 为提取式降级且绝不二次调 LLM、写 `summary_updated` 事件；`RemoveMessage(REMOVE_ALL)` 语义接受（原文在事件与历史 checkpoint）。`PIIMiddleware` 不用（无本人数据允许清单）；`ModelRetryMiddleware` / `ModelFallbackMiddleware` 不用（与网关重试权威相悖）；`ToolRetryMiddleware` / `ToolErrorMiddleware` 不用（写工具恒单次、结局由七步接管）；`ContextEditingMiddleware` / `TodoListMiddleware` / `LLMToolSelectorMiddleware` / `ShellToolMiddleware` 等与本仓语义无交集，不用。
10. **话术单一事实源**：L2 话术住 `engine/runtime/utterances.py`；框架英文串（限流回填、工具错误回填、摘要包装）一律不进事件与对话——对应中间件在其产生之前拦截（幻觉工具名在 after_model 先于 ToolNode 判定）。
11. **嵌套模型双发**：`stream_mode="messages"` 下网关内再调候选时每块出现两次（候选 run 与网关 run），消费方只取网关级块。
12. **通道单写者纪律（M2.5 增补）**：tools 节点每个 tool_call 是一个任务，无 reducer 的私有通道同一 superstep 只许一个写者。按调用更新的账本（连败计数 / 本轮禁用 / 是否已写终止）住每 run 一份的运行时载体对象而不是通道，禁用不跨 run；取消检查点只由第一个被弃置的调用经 `Command` 写 `termination`，其余只配对 ToolMessage；tools→model 边不看 `jump_to`，终止事实由 `Gates.before_model` 翻译成跳 end。审批通行证由 Approvals 单节点写入通道，不受此限。
13. **增强层纪律（M2.5 / M2.6 增补）**：工具结果摘要与滚动摘要走同一租户网关的 fast 档（档位 / 会话 / deadline 经 `ainvoke` 的 kwargs 到达网关），内部调用打 `aegis:internal` tag 并合并框架的内部调用元数据，对外流式通道不转发；fail-open 只接网关的六类公开异常，内部家族泄漏照样裸炸（契约 C4 的分野在增强层同样成立）；注定终止的 run（已终止 / 已取消 / 轮数已满）不再压缩——增强层不为将死的 run 花钱；需要依赖的中间件经 `build_middleware(gateway, spec)` 工厂注入，`MIDDLEWARE_STACK` 类元组与工厂一一对应，由静态测试互钉。

## 实证（探针）

复现方式：三个各自实现六钩子的记录型中间件组合进 `create_agent`，打印图节点与运行顺序；声明与不声明 `can_jump_to` 的 after_model 各返回 `jump_to="end"` 观察循环是否继续；`ModelCallLimitMiddleware(run_limit=2)` 与 `ToolCallLimitMiddleware` 在剧本模型下跑到超限，读 state 里的回填消息；`SummarizationMiddleware` 注入计数型 token_counter 并检查 `_summary_model` 类型与产物形态；ToolNode 下三个慢工具并行计时、同步工具记录线程、执行期异常与 schema 错参各一；`Annotated[int, PrivateStateAttr]` 通道跨两次 run 读值；`StructuredTool` 仅带 coroutine 经 `convert_to_openai_tool` 与 `create_agent` 各走一遍。定性结论已列于背景节；对应的运行时类型测试（ToolDef 到 StructuredTool 的参数同源、哨兵在图内穿出）已钉住。

模型调用中间件与图工厂（M2.3）：
- `wrap_model_call` 返回 `ExtendedModelResponse(command=Command(update=…))` 的状态更新在同一 superstep 的 after_model 里已可见，command 里的消息经 reducer 追加在模型结果之后；after_agent 返回的消息同样入 state；`stream_writer` 在 wrap 与 after_agent 内均可用。
- wrap 内读到的框架任务 id 与 before_model 节点不同；wrap 首次抛异常后从 checkpoint 恢复，模型节点重放时任务 id 与首次相同——事件 id 派生因此对模型节点同样成立。
- 最小可行 `recursion_limit` = 节点执行数 + 1：只有 before_agent / after_agent 两个钩子节点的栈，k 轮工具循环需要 2k + 4；推导式（外圈钩子节点 + max_iterations × 每轮节点数 + 终止那一遍的 before_model 节点 + 1）对该栈给出 2·max_iterations + 3，与探针值恰差一个余量，减 2 即触发 `GraphRecursionError`。
- 模型节点必须产出一条不带 tool_calls 的 AIMessage 才能让模型→工具边走向 end：模型结果为空列表时，框架回退到历史里最后一条 AIMessage，若它带已配对的 tool_calls 便会回到模型节点循环。故 wrap 内终止一律在结果里放兜底 AIMessage（零话术时为空 AIMessage）。
- 一轮多个 tool_calls 由框架并行派发，仅用互斥锁只能保证不重叠、不能保证顺序（实测出现 b 先于 a）；按声明序等待前驱完成后才执行，事件序与执行序与声明序一致。
- 以上由 `tests/engine/runtime/test_runtime_*.py` 钉住：文本 run 五事件、工具 run 九事件、四组异常映射与零话术、流级中断作废重发消耗迭代、内部异常泄漏裸穿、run 起点通道归零、第二轮 seq 接续、`recursion_limit` 紧致性、图缓存与指纹敏感性、真 PG 端到端。

六道闸门（M2.4）：
- 部分配对的 after_model 返回只把未配对的调用送进 tools；全部配对回到 before_model 链头；两个声明可跳转的 after_model 钩子：列表靠后者先跑、跳 end 绕过靠前者；after_model 跳 "model" 落到 before_model 链头；before_model 跳 "end" 后 after_agent 仍运行。
- 含 before_model / after_model 各一个钩子节点的栈：文本路径最小 `recursion_limit` = 4k + 6（k 轮工具），"工具循环撞 max_iterations 由 before_model 终止"的最长路径恰 4M + 4 = 推导式。
- 以上由 `tests/engine/runtime/test_runtime_gates.py` 钉住：撞上限在第四次调用前终止且事件尾为工具结果 → 兜底 → 末事件；打断配对且键序抖动不重置、打断后再犯恰 15 事件 / 4 次调用 / 2 次执行；换参数重置；一轮内三个相同调用只执行两个；纠错两次仍违规终止且纠错以带标记的 user 消息注入；合法工具轮清零；一轮三个幻觉名零 tool_call 事件、三条中文回填、无框架英文串；`max_tokens` 截断按完成；起跑即取消零调用；工具里置位的取消在下一次 LLM 调用前生效；两钩子的跳转声明与条件边目标快照；未声明跳转静默无效回归；最长路径 `-1` 即 `GraphRecursionError`；八值终止原因由八个场景全部产出。

工具执行七步（M2.5）：
- wrap 直接返回 ToolMessage 或 `Command(update={"messages": […]})` 均被接受，配对与 status 保留；三调用并行派发 + 按声明序等前驱，二十次执行序全部为声明序；工具副作用之后以 BaseException 中断，checkpoint 停在 tools 之前，恢复后 wrap 再进、任务 id 与派生事件 id 与首次相同、副作用两次。
- 以上由 `tests/engine/runtime/test_runtime_tool_exec.py` 钉住：write-ahead 落盘时刻（工具执行时事实源里恰一条 tool_call 且 id 即注入的幂等键）；同任务身份二次进入不产生第二条 tool_call、下游一把钥匙；真图崩溃后恢复事件恰九条无重复、结果以原 id 闭合、去重命中的事件不外流、会话回到 idle；写超时结果不明且不进连败账、读超时记账、读退避重试 0.2 / 0.4 秒、写异常单次、超时取更严；连败两次禁用并当次宣告、成功清零、按工具按 run；超预算走 fast 档摘要且原文全量、摘要失败硬截断并留痕、无网关截断、超长摘要再截断、内部家族泄漏裸炸；闸门崩溃 fail-closed 且打码、无通行证不执行、通行证放行；取消检查点弃置剩余调用且只有第一个写终止。

上下文编译与滚动摘要（M2.6）：
- 基类摘要路径经 `RunnableRetry` 三次尝试；子类改走裸模型后一次即返；`ainvoke` 的档位 / 会话 / deadline kwargs 原样到达 `_agenerate`；第二轮的自定义帧序为 `summary_updated` 先于 `llm_call`；REMOVE_ALL 后历史 checkpoint 仍含原文；按 token 的切点不拆 AI/Tool 对但可落在 AI/Human 之间。
- 以上由 `tests/engine/runtime/test_context.py` 与 `test_runtime_summarization.py` 钉住：层序快照、system 超预算 fail-loud、当前 user 超长时历史清空原文照放、空 AIMessage 丢弃、AI 的 tool_calls 参数不折、折叠从最老起、旧轮压成 user + 终答、摘要切点前导 assistant 保留、从最新往回装、肥摘要份额一半、关层、确定性、`input_tokens_est` 按编译后 prompt；触发写事件 + 中文包装 + events 原文与 seq 不动、网关失败提取式降级且候选调用数等于网关尝试数、取消态零调用、关层零摘要。

## 后果

- 六道闸门与终止事实完全在本仓掌握，现成计数件的英文话术与无事实问题不进系统；代价是两件自建中间件与栈序纪律，靠静态测试维持。
- 工具执行完全绕开框架 handler：框架 ToolNode 退化为路由与幻觉名兜底，本仓承担七步的全部正确性。
- 并行工具执行关闭：多工具调用逐个执行，换取事件序确定与幂等审计可读。
- 摘要接受现成件但关闭其重试与英文包装，登记表随行。
- `PrivateStateAttr` 来自框架的次公开模块路径；升级时随探针复核。
- 通道单写者纪律把按调用的账本推到进程内对象：连败禁用不跨崩溃恢复（前作同款）；取消检查点的终止事实由首个被弃置的调用承担。
- 终止时 state 里多出配对的"未执行"ToolMessage 与兜底 AIMessage：下一轮 prompt 由编译器裁剪，checkpoint 里它们永久存在。
- 摘要触发按 state 全量计、编译按压缩后计，两把尺同口径不同对象；重放窗口内摘要文本可能与事件里的首次文本不同（保事实不保字节）。
