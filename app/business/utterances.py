"""L3 业务话术单一事实源（登记例外：业务层自持，不进 engine/runtime/utterances——那是运行时的话术表）。

进模型上下文（system 模板、工具回填）与进用户面的文本只在这里定义；改 SYSTEM_PROMPT_TEMPLATE = 改图指纹
（spec_fingerprint 含 system_prompt，改一字即换图）。
"""

SYSTEM_PROMPT_TEMPLATE = (
    "你是「{tenant_name}」的智能客服助手。\n"
    "规则：\n"
    "1. 用中文简洁、礼貌地回答，一次只处理一个诉求；\n"
    "2. 涉及订单、退款等用户个人数据时，必须先调用相应工具查询，绝不凭记忆或猜测回答；"
    "执行退款前先查询订单确认状态与金额；\n"
    "3. 工具给不出依据的事实（品牌、活动、价格、期限时效、网址与联系方式等），一律明确告知用户"
    "「没有找到相关信息」，禁止凭常识或行业惯例推测作答；需要人工跟进时用工单工具转人工；\n"
    "4. 操作被拒绝或需要人工审批时，如实转达系统给出的说明，不要擅自重试。"
)
"""system 模板：平台规则 + 租户名，约 180 CJK ≈ 180 token，远低于 system_budget=1500（编译器 fail-loud 层）——
装配测试钉住估算上界防模板膨胀。UNTRUSTED_NOTICE 由编译器恒拼，模板不重复声明。"""

DENIED_TEXT = "订单不存在或无权操作"
"""越权三路统一话术：不存在 / 他租 / 他人三种失败不区分——泄露"存在但无权"就是泄露他人订单号的有效性。"""

REFUND_NOT_POSITIVE = "退款金额必须大于 0"
REFUND_ALREADY = "该订单已退款，不能重复退款"
REFUND_OVER_PAID = "退款金额超过订单实付金额"
"""业务拒绝三条：以 {"error": …} 回填模型（不是异常，不进连败账、不禁用工具）。"""

# --- HTTP 层（routers 消费的 detail 文本；状态码分工见 routers/common.py） ---
TENANT_NOT_ENABLED = "该租户未开通服务"
SESSION_NOT_FOUND = "会话不存在"
SESSION_BUSY = "会话正忙：上一次处理尚未结束"
SESSION_AWAITING_APPROVAL = "会话正等待人工审批，请在审批完成后再发送消息"
APPROVAL_NOT_FOUND = "审批单不存在"
APPROVAL_FOREIGN_TENANT = "无权处理其他租户的审批单"
APPROVAL_NOT_PENDING = "审批单已非待审状态（已决定、已撤回或已过期）"
RUN_FAILED = "本次处理中断，请稍后重试或联系人工客服"
