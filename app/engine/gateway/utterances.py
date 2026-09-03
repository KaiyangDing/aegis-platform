"""L1 网关中文话术单一事实源。

异常消息与告警文案只在这里定义，其它模块只引用常量；带 {} 占位符的用 str.format 填充。
按消费步骤分组（M1.1 只用到"异常翻译"一组，其余为后续步骤预留的同一份事实源）。
"""

# --- 异常翻译（M1.1 errors.classify） ---
HTTP_STATUS = "HTTP {status}: {snippet}"
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

# --- 候选环与终局（M1.4a） ---
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
LOG_PRICE_MISSING = "模型不在价目表中，本行成本记 0（请尽快补配置）"
LOG_METER_WRITE_FAILED = "计量写入失败（对账脚本会暴露此缺口）"
LOG_BUDGET_READ_FAILED = "预算读取失败，本次放行（fail-open）"
LOG_CACHE_DEGRADED = "精确缓存不可用，降级为直通"
LOG_CACHE_RECOVERED = "精确缓存恢复，切回缓存路径"
LOG_BREAKER_STATE_CHANGED = "熔断器状态变化"
