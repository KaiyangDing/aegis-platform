# ADR-006: 重试权威与三段超时

- 日期：2026-09-03
- 状态：已接受（增补 ADR-005 决策 3/4 的翻译与消毒细则）

## 背景

前作网关的受控重试有三条铁律：只重试无业务副作用的补全、只在首块之前重试、退避为指数满抖动且 429 优先服从 Retry-After；配套三段超时（connect / 首块 / 块间空闲，整流不设上限）与 deadline 向下传播。本仓改用 langchain-openai 调上游后，重试可能发生在四个地方：openai SDK 自带重试、`Runnable.with_retry`/`with_fallbacks`、agent middleware、网关自己的 `_astream`。四者只能有一个是权威，否则退避相乘、Retry-After 被放大、deadline 不可见。

同时框架带来两个前作没有的计时器：openai SDK 的 `max_retries` 退避睡眠（发生在我们任何计时器之内、不读 deadline、只在 `openai` logger 的 INFO 级留痕而不进网关日志），以及 langchain-openai 的 `stream_chunk_timeout`（默认 120s 静默启用，仅 async）。前作的"终止哨兵见证"（未见 `[DONE]` 即截断）在 SDK 层不可见，截断检测需另立判据。

本仓 venv 探针事实（langchain-core 1.6.1 / langchain-openai 1.6.0 / openai 3.7.0，零网络，`httpx2.MockTransport` 注入）：
- `Runnable.with_retry` 不重试流；`with_fallbacks` 流式只在首块前切换、无退避、不读 Retry-After；已经回调外送的 token 不可撤回（ADR-005 已引）。agent middleware 的模型调用挂点（langchain 1.3.14 源码，本仓未安装）拿到的是完整响应而非流句柄，待 M2 复核，本决策不依赖该结论。
- `stream_chunk_timeout` 对**每一次** `__anext__` 的等待计时，包括响应头之后的首个数据块（触发时 `chunks_received=0`）；流中触发时 `chunks_received≥1`。它自响应头之后起算，网关首块计时器自开流起算，等值时网关先触发；网关侧可用 `asyncio.Timeout.expired()` 区分是自家计时器还是框架计时器。`None`/0 关闭，负值回退默认值。
- 流中（响应头之后）发生的 `httpx2.ReadTimeout`/`RemoteProtocolError` **不被 SDK 包装**，以裸 httpx2 异常到达消费者；流内 `error` 事件经 langchain-openai 透传为无状态码的裸 `openai.APIError`（消息含 "exceeds the context window" 时被重包为 `ContextOverflowError` 子类，仍无状态码）；只有连接阶段（`send` 内）的传输异常才被包成 `APITimeoutError`/`APIConnectionError`。
- 首块等待被 `asyncio.timeout` 切断时，取消沿候选流的各层 `async with` 同步展开，底层 httpx2 响应流的 `aclose()` 在异常抛出前已被调用；之后再对候选流 `aclose()` 是 no-op。消费者拿到首块后 `aclose()`：候选外层生成器同步关闭，底层响应流在**下一个事件循环周期**由 asyncio 事件循环的异步生成器终结钩子释放（框架内层生成器不随 `async for` 退出而关闭，引用计数归零即调度 `aclose`，不依赖循环回收器）。
- 完整流的 `finish_reason` 落在某一块的 `response_metadata["finish_reason"]`（可与内容同块）；正文在首块之后中途结束时流**静默结束**，框架补一个空的 `chunk_position="last"` 块；正文一块都没有时框架抛 `ValueError("No generation chunks were returned")`；`finish_reason` 在场而 usage 缺席时框架照常结束、不抛错。
- 只含 `reasoning_content` 的 delta（思考型模型的思考流）被框架吐成 `content=""` 且无附加字段的空块，思考文本被丢弃；这些空块会喂饱任何以"收到一块"为判据的首块计时器。

## 决策

1. **重试权威唯一在网关 `_astream` 的首块窗口**（`engine/gateway/resilience.py` 的 `complete_with_retry`）。候选 `ChatOpenAI` 一律 `max_retries=0`（审查清单项，候选工厂单测钉住）；不用 `with_retry`/`with_fallbacks`；middleware 不承担重试。
2. **三条铁律与退避口径原样保留**：白名单 = 429 / 超时 / 5xx（`RateLimitedError`/`ProviderTimeoutError`/`ProviderServerError`）；Retry-After 优先、封顶 `max_backoff`、不套抖动，负值归零、非有限值视为未给；否则 `uniform(0, min(base·2^(n-1), max))`。尝试次数、单候选墙钟预算 `total_timeout`（含闸内排队与退避）、deadline 三重预算；预算耗尽**裸抛真实死因，不造新异常**。**首块 = 首个可见块**（content / tool_call_chunks / finish_reason / usage 任一在场）：只含 reasoning delta 的空块不满足首块窗、也不向下游透传，思考期的挂起照样被切断。首块流出后零重试，任何错误翻译后原样上抛，由候选环包成 `GatewayStreamInterrupted`。
3. **三段超时归属**：connect 归候选工厂的 `httpx2.Timeout(connect=5)`；首块归网关 `asyncio.timeout(min(first_chunk_timeout, deadline 剩余))`；块间空闲**双闸并存**——httpx2 `read=30` 守字节级静默（死连接），`stream_chunk_timeout=60` 守已解析块级静默（连接活着但模型不吐块，字节级闸看不见的形态），取值必须 ≥ 首块窗口（测试钉住），使框架计时器永不抢首块窗；若因配置错误抢窗，网关按 `expired()` 判定不冒充自家首块超时，交翻译表处理。整流不设上限。
4. **翻译落点在 `complete_with_retry`**：候选抛出的一切异常先过 `classify`，再判可重试；首块前与首块后同一张表；未知异常裸抛。流中裸 httpx2 传输异常沿用 ADR-005 决策 3 既有规则（本探针实证其必要性）。翻译表增补两条：框架零块 `ValueError` 按消息前缀识别为零块流→`ProviderServerError`（可重试；端到端测试钉住该消息）；langchain 的 `ContextOverflowError`（400 变体与流内 error 事件变体）→`BadRequestError`（上下文超长是确定性拒绝，不是上游故障）。
5. **截断检测判据 = `finish_reason` 是否到场**：流耗尽而无任何块携带 `finish_reason` → `ProviderServerError`（"流被截断"），在流尾抛出，候选环按流级中断处置并计熔断账；`finish_reason` 在场而 usage 缺席视为完整（计量侧以 `usage_missing` 标记，归后续的租户缓存与计量组合根 ADR）。
6. **出站闸按尝试计、在首块计时器之外**：`complete_with_retry` 每次尝试先过 `acquire(max_wait)` 缝（M1.3 默认放行，M1.4b 接限时取令牌真件）；`max_wait` = min(deadline 剩余, `total_timeout` 剩余) − `min_attempt_budget`，闸的等待既不可能吃掉首块预算、也不可能越过单候选墙钟预算；`max_wait ≤ 0` 表示只试一次不排队。排不上：首次尝试抛 `OutboundGateTimeout`（`RateLimitedError` 子类；候选环按 429 待遇换路、不进熔断账，且不作为终局死因——与前作 `wait_take` 拒绝时不写 `last_error` 同义），重试尝试裸抛上一次的真实死因。闸自身抛出的异常是编程错误，不翻译不重试。
7. **deadline 载体**：网关内部以绝对单调钟时刻显式传参；开首次尝试前的预检归候选环（`deadline 剩余 < min_attempt_budget` 即停止换路），`complete_with_retry` 负责重试前预检与首块窗取小。调用方如何表达 deadline（`RunnableConfig` 元数据或 kwargs）归后续的租户缓存与计量组合根 ADR。
8. **消毒增补（ADR-005 决策 4）**：`AuthError` 不把 SDK 异常挂进 `__cause__`，且翻译后的异常在 `except` 块之外抛出（解释器不会再挂 `__context__`）——401 响应体回显密钥原文，traceback 渲染（日志 `exc_info`）会绕过源头打码；其余翻译保留 `__cause__` 链。
9. `include_response_headers` 保持默认 False：Retry-After 只在 429 异常对象的 `response.headers` 上读取，成功路径不需要响应头。

## 实证（探针）

复现方式：`ChatOpenAI(api_key="fake", base_url="http://test/v1", max_retries=0, stream_usage=True, stream_chunk_timeout=…, http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))`，handler 返回 `httpx2.Response(200, stream=<自定义 AsyncByteStream>)`，以延迟/抛错/截断正文/reasoning delta 构造各形态，零网络。定性结论：
- `stream_chunk_timeout` 对响应头之后的首个数据块同样计时；网关首块计时器与它取小者先触发，`asyncio.Timeout.expired()` 可辨。
- 流中传输异常与流内 error 事件不经 SDK 包装到达消费者（上下文超长的流内事件被 langchain-openai 重包为无状态码的 `ContextOverflowError` 子类）；连接阶段异常才被包装。
- 首块窗切断与任务取消均同步释放底层响应流；消费者弃流后底层释放发生在下一个事件循环周期。
- 截断流静默结束且末块为框架补的空块；零块流抛 `ValueError`；`finish_reason` 是 SDK 层可见的稳定完整性见证（`[DONE]` 被 SDK 消费不可见、usage 块可缺席）。
- 只含 reasoning delta 的块以空 content 到达，会满足"收到一块"式的首块判据。

## 后果

- 双重重试防线维持为纪律保证（ADR-005 后果）；本决策把重试落点、三段超时归属、翻译落点三件事收进同一个模块，候选环（后续 ADR）只处理换路与记账。
- `stream_chunk_timeout` 从"框架默认"改为"显式接受并约束取值"，登记表相应更新；块间空闲存在两个计时器，各守一段，需在运维文档中说明。
- `total_timeout` 的口径从前作的"重试环内耗时"改为"单候选墙钟（含闸内排队与退避）"；闸等待上限随之受它约束。
- 首块判据改为"首个可见块"后，网关会丢弃首个可见块之前的空块；思考型模型即使误开思考，首块计时器度量的仍是到首个可见 token 的时间，隐藏思考 token 的计费不受本决策影响。
- 消费者弃流后底层连接释放晚一个循环周期，对连接池占用无实际影响，但"簿记在流尾"的负向测试须以"让出一个循环周期后"为断言时机。
- 零块流的识别依赖框架异常消息前缀，属显式接受的框架耦合，由端到端测试钉住；框架升级时该测试先红。
- 出站闸缝的语义（按尝试计、deadline 与 total_timeout 反推等待上限、排不上按 429 待遇且不作终局死因）成为 M1.4b 接真件的契约。
