# ADR-005: 网关落点与六类异常契约

- 日期：2026-09-02
- 状态：已接受（增补 ADR-002 依赖基线）

## 背景

前作网关以自研 HTTP 适配器加统一协议（`LLMRequest`/`LLMChunk`）对上层提供流式补全；其六类两级异常契约是运行时层降级与恢复语义的地基。本仓改用 langchain-openai 调上游后，两件事必须先定：网关在框架世界的形态（agent 直接持有 `ChatOpenAI`、agent middleware、还是自建 chat model），以及异常翻译的输入端从 HTTP 状态码换成 SDK 异常对象之后契约如何保持。

本仓 venv 探针事实（langchain-core 1.6.1 / langchain-openai 1.6.0 / openai 3.7.0，零网络）：
- `Runnable.with_retry` 不重试流；`with_fallbacks` 流式只在首块前切换、无退避、不读 Retry-After；已经回调外送的 token 不可撤回。"只在首块前重试"需要持有流句柄，任何整调用挂点都表达不了。
- openai 3.7 基于 httpx2；SDK 把状态码 400/401/403/404/409/422/429 映射到各自子类，402/408 落 `APIStatusError` 基类，**大于等于 500 全部归 `InternalServerError`（含 501）**。`APITimeoutError`/`APIConnectionError` 无 `response`/`status_code`；流内 error 事件是无状态码的裸 `APIError`；流被截断不抛异常。
- 经 `ChatOpenAI` 抛出的超时异常在 `__cause__` 链中完整保留 `httpx2.PoolTimeout`/`ReadTimeout`/`ConnectTimeout`；连接类错误落 `APIConnectionError`。
- `BaseChatModel` 最小子类只实现 `_astream`/`_agenerate` 即可 `ainvoke`/`astream`；`bind_tools` 返回绑定视图、原实例不变、kwargs 到达 `_astream`；`cache=False` 随绑定视图传递。
- `str(e)` 形如 `Error code: 401 - {完整响应体}`，上游回显的密钥片段原样进消息。

## 决策

1. **网关 = `BaseChatModel` 子类**（`app/engine/gateway`）。`_astream` 为主实现，`_agenerate` 聚合流，同步路径抛 `NotImplementedError`；`bind_tools` 返回 `self.bind(tools=..., tool_choice=...)`；类级 `cache=False`。档位路由、受控重试、fallback、熔断、缓存、计量全部在子类内部完成，agent 层对网关内部零知识。
2. **六类两级契约原样保留**：请求级可降级四类 `GatewayExhausted`/`BudgetExceeded`/`TenantQuotaExceeded`/`GatewayOverloadedError`，确定性拒绝 `GatewayRejected`，流级 `GatewayStreamInterrupted`；内部 `ProviderError` 家族（`RateLimitedError`/`ProviderTimeoutError`/`ProviderServerError`/`BadRequestError`/`AuthError`）永不穿出网关。`GatewayOverloadedError` 的生产者 = 超时异常的 `__cause__` 链含 `httpx2.PoolTimeout`。
3. **翻译表 `classify(provider, exc)`**：`openai.APIStatusError` 按 status_code 分段——429→RateLimited（Retry-After 从 `response.headers` 读，`retry-after-ms` 优先，秒/HTTP-date 双格式，解析失败为 None）；401/403→Auth；408→Timeout；**501→BadRequest**（在 5xx 规则之前特判）；其余 5xx→Server；其余 4xx→BadRequest。`APITimeoutError`→Timeout 或 Overloaded；`APIConnectionError`→Server；裸 `APIError`→Server（流内错误）；`StreamChunkTimeoutError` 与其它 `TimeoutError`→Timeout；httpx2 传输异常按同一规则。**未知异常返回 None，由调用方裸抛**：把编程错误伪装成上游故障会藏起 bug。
4. **消毒在源头**：`sanitize_error_text` 先打码（`sk-` 前缀密钥模式）后截断（200 字）；异常消息由结构化字段拼装，响应体片段只作消毒后附注；配置密钥用 `SecretStr`。
5. **单一事实源**：网关中文话术集中在 `engine/gateway/utterances.py`；token 尺 `core/tokens.py` 自家实现（CJK 基本区 1 字≈1 token，其余 4 字符≈1 token 向上取整），不用框架估算器（按 4 字符/token 对 CJK 系统性低估）。
6. **依赖基线增补**：`openai`、`httpx2` 由传递依赖改为显式依赖（生产代码直接 import）；`httpx` 0.28 仅供测试 `ASGITransport`。测试拦截上游的主路 = `httpx2.MockTransport` 注入 `http_async_client`（respx 默认只拦 httpx 0.28 的 httpcore）。第二供应商 Anthropic 不纳入本仓范围，需要时以 `langchain-anthropic` 另立 ADR。

## 实证（探针）

复现方式：`ChatOpenAI(api_key="fake", base_url="http://test/v1", max_retries=0, stream_usage=True, http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))`，handler 返回任意状态码/响应体或直接 `raise httpx2.PoolTimeout(...)`，零网络。定性结论：
- 状态码到 SDK 异常类的映射以 `openai.OpenAI(...)._make_status_error_from_response` 为准，测试造异常走同一入口，与生产路径同一张表。
- `__cause__` 链可辨识本地连接池排队与真超时，六类中的 Overloaded 在框架世界有生产者。
- 401 响应体中的密钥片段会原样出现在 SDK 异常消息里，经翻译表后被打码。
- `StreamChunkTimeoutError` 从 `langchain_openai` 顶层导入，是 `TimeoutError` 子类，不属 openai 异常族。

## 后果

- 契约边界从数据类型（`LLMChunk`）改锚在 chat model 行为上；运行时层直接消费 langchain 消息类型；缓存持久化值将使用自家最小 schema（后续 ADR）。
- 前作以 `[DONE]` 哨兵见证流完整性；SDK 层截断流静默结束，截断检测另由重试与超时的 ADR 裁决。
- 双重重试防线从"不引 SDK 的结构保证"退化为纪律：`max_retries=0` 进审查清单并由候选工厂单测钉住。
- 六类中 `TenantQuotaExceeded` 的生产者由限流与配额的 ADR 裁决；`BudgetExceeded` 沿用月度与单请求两处。
