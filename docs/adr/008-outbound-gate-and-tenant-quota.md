# ADR-008: 出站闸与租户配额事实源

- 日期：2026-09-03
- 状态：已接受（入站限流原语由后续 ADR 单独裁决）

## 背景

前作出站限流是 Redis Lua 令牌桶（租户 + 供应商两维、多副本精确、按候选访问计次而不计 token），候选环内每站 `wait_take(max_wait)`，桶按补给模型预判预算内等不到即刻放弃，排不上即跳过候选；租户维配额耗尽抛 `TenantQuotaExceeded`。本仓用 langchain-core 的 `InMemoryRateLimiter` 替代出站桶，须裁决其约束与承担者，并定下租户配额的事实源与 `TenantQuotaExceeded` 的生产者。

本仓 venv 探针事实（langchain-core 1.6.1）：
- `InMemoryRateLimiter` 进程内、只按次数计、桶初始为空（首次取令牌需等一个补给周期）、`aacquire(blocking=True)` 无上限等待且按 `check_every_n_seconds` 轮询、无等待预估接口；补给按 `elapsed × rps` 以整枚令牌为步进、封顶桶容量，发放要求 `available_tokens ≥ 1`；`available_tokens`/`max_bucket_size`/`requests_per_second`/`last` 为公开属性。
- 挂在候选 `rate_limiter=` 上时，取令牌发生在 `BaseChatModel.astream` 内部、位于网关首块计时器之内（本仓探针）：本地排队会被翻译成首块超时并进熔断账。
- 参考实现（Argus）中 PostgreSQL 配额是事后闸：两条普通 SELECT，无预扣、无行锁；前作月度预算闸同形（每次 DB 端聚合已用量、fail-open）。

## 决策

1. **出站闸 = 每 provider 一个 `InMemoryRateLimiter`，由网关显式限时取令牌**（`engine/gateway/outbound.py`）：不挂候选 `rate_limiter=`；候选环把 `acquire(provider, max_wait)` 注入 `complete_with_retry` 的 acquire 缝，在首块计时器之外、按尝试调用（ADR-006 决策 6）。等待上限 = min(闸自有 `max_wait`, deadline/total_timeout 反推值)；先非阻塞试取，再按桶的补给模型**预判**下一枚令牌何时够——预算内等不到即刻返回 False（前作 `wait_take` 语义，不白烧预算）；等得到则自建轮询等待、到期前至少再试一次（不受库阻塞取令牌的轮询粒度地板约束）。False → 候选环换路、不进熔断账、不作终局死因。
2. **四项约束记账**：进程内（多副本口径 = 全局配额 / 副本数，由配置承担）；按次数不按 token；初始空桶 → 预填 `available_tokens = burst` 恢复前作的冷启动满桶语义；库无 `max_wait`/不换路/无等待预估 → 自建等待预算与预判。桶容量必须 ≥ 1（库发放要求整枚令牌，小于 1 的桶会永久拒绝），构造期与配置双重校验。参数起点沿用前作口径（每 provider 8 次/秒、桶 16、最长排队 10s），进配置。
3. **租户配额事实源 = PostgreSQL 账本，事后闸口径**：月度预算闸每次 DB 端聚合已用量、与账本同一事实源、fail-open，超限产 `BudgetExceeded`（沿用前作话术）；不做预扣、不加行锁。
4. **`TenantQuotaExceeded` 的生产者 = 网关内每租户一个进程内桶**（方案 A′）：`InMemoryRateLimiter` 按 tenant_id 惰性创建、有界 LRU、限时取令牌，排不上产 `TenantQuotaExceeded`；它是速率闸不是配额，位于候选环之外（换供应商换不掉租户身份），非 HTTP 调用方（worker、脚本）同样受节流。多副本下为近似（全局 / 副本数），记账。实现随组合根一步落地。
5. **入站限流原语不在本 ADR 范围**：其实现（fastapi-limiter 固定窗口 + 自建作用域依赖）依赖本仓对框架依赖注入形态的探针复核，随对应步骤另立 ADR。

## 实证（探针）

复现方式：`InMemoryRateLimiter(requests_per_second, max_bucket_size, check_every_n_seconds)` 直接驱动 `aacquire`；挂候选与不挂候选各一次，外层 `asyncio.timeout` 包 `anext(candidate.astream(...))`，`httpx2.MockTransport` 即时应答，零网络。定性结论：
- 挂候选时首块计时器在设定阈值处触发而传输层即时应答——等待来自库内取令牌；不挂候选即时到首块。
- 库的桶初始为空、无等待预估、阻塞取令牌以固定周期轮询；预填 `available_tokens` 后开局可连续取 `burst` 枚；补给以整枚令牌为步进。预判与轮询等待的行为由出站闸单测钉住。

## 后果

- 前作"租户 + 供应商两维、多副本精确"的 Lua 桶在本仓为降级近似，记入差距记账表；精确的多副本出站配额若成为需求，应回到 Redis 侧实现。
- 出站闸的等待预算与首块预算解耦：本地排队既不冤枉供应商，也不吃掉首块窗口；预判让闸满的候选立刻换路而不是等满上限。
- `TenantQuotaExceeded` 保留生产者（A′），六类契约在框架世界各有产生条件。
