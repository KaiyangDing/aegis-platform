"""L1 网关中文话术单一事实源。

异常消息与告警文案只在这里定义，其它模块只引用常量；带 {} 占位符的用 str.format 填充。
按消费步骤分组（M1.1 只用到"异常翻译"一组，其余为后续步骤预留的同一份事实源）。
登记例外（计划 §7）：domain/usage.py 的价目表告警串、core/limits.py 的 429 串、core/config.py 的 prod 禁注入串
——分层契约不许 domain/core 反向 import engine。
"""

# --- 异常翻译（M1.1 errors.classify） ---
HTTP_STATUS = "HTTP {status}: {snippet}"
CONTEXT_OVERFLOW = "上下文超长: {detail}"
STREAM_ERROR_EVENT = "流内错误 {code}: {detail}"
POOL_TIMEOUT = "本地连接池排队超时: {detail}"
TIMEOUT = "超时: {detail}"
CONNECT_FAILED = "连接失败: {detail}"
KEY_MASK = "sk-***"

# --- 候选与路由（M1.2） ---
ROUTE_ENTRY_INVALID = "路由配置非法: {tier} -> {entry!r}"
ROUTE_CHAIN_EMPTY = "档位 {tier} 的候选链为空"
ROUTE_TIERS_MISSING = "路由配置缺少档位: {missing}（fast/standard/strong 必须齐全）"
API_KEY_MISSING = "API key 未配置（检查 .env 的 DASHSCOPE_API_KEY）"

# --- 受控重试与超时（M1.3） ---
FIRST_CHUNK_TIMEOUT = "首块超时 >{wait:.1f}s（上游挂起）"
STREAM_TRUNCATED = "流被截断：未收到终止哨兵"
STREAM_EMPTY = "流为空：上游一块都没吐"
OUTBOUND_GATE_TIMEOUT = "出站闸排队超时（等待 {wait:.1f}s 未取得令牌）"
RETRY_POLICY_INVALID = "RetryPolicy 参数非法: {field}"

# --- 候选环与终局（M1.4a） ---
SYNC_NOT_SUPPORTED = "网关只支持异步调用（ainvoke/astream）"
ROUTE_MISSING = "档位 {tier} 没有配置任何候选（检查 MODEL_ROUTES）"
STREAM_INTERRUPTED = "流中断于 {provider}:{model}"
FIRST_CHUNK_BUDGET_EXHAUSTED = "档位 {tier} 首块预算 {deadline_s}s 耗尽（候选链未走完）"
ALL_REJECTED = "档位 {tier} 全部候选均被确定性拒绝——检查 API key 配置与请求转换"
ALL_UNAVAILABLE = "档位 {tier} 的所有候选均不可用"
FAULT_ERROR = "故障注入（error）"
FAULT_HANG = "故障注入（hang 兜底，正常不应到达）"
FAULT_MIDSTREAM = "故障注入（midstream：首块后断流）"

# --- 预算与配额（M1.5c） ---
BUDGET_MONTHLY = "租户 {tenant_id} 本月已用 {spent} token，预算 {budget}"
BUDGET_REQUEST = "单请求估算 {estimated} token，超过预算 {budget}（估算口径 ADR-005）"
TENANT_QUOTA = "租户 {tenant_id} 出站配额耗尽"
TENANT_ID_INVALID = "tenant_id 非法：只允许 [A-Za-z0-9_-]{{1,64}}"

# --- 告警文案（logger.warning 的 event） ---
LOG_METER_WRITE_FAILED = "计量写入失败（对账脚本会暴露此缺口）"
LOG_BUDGET_READ_FAILED = "预算读取失败，本次放行（fail-open）"
LOG_CACHE_DEGRADED = "精确缓存不可用，降级为直通"
LOG_CACHE_RECOVERED = "精确缓存恢复，切回缓存路径"
LOG_CACHE_DIRTY = "缓存条目损坏，已删除并按 miss 处理"
CACHE_PARAMS_INVALID = "缓存参数非法：ttl_seconds > 0，probe_interval > 0"
LOG_BREAKER_STATE_CHANGED = "熔断器状态变化"
OUTBOUND_LIMITER_INVALID = (
    "出站闸参数非法：rate/burst 须 > 0，max_wait 须 ≥ 0，max_keys 须 ≥ 1"
)
BREAKER_POLICY_INVALID = "熔断参数非法：fail_max ≥ 1，四个时长为有限正数，fail_window > reset_timeout + probe_ttl"
LOG_BREAKER_DEGRADED = "熔断存储不可用，降级为进程内状态机（粘滞 probe_interval；全集群单探针在降级期失效）"
LOG_BREAKER_RECOVERED = "熔断存储恢复，切回共享态"
