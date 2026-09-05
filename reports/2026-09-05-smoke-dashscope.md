# DashScope 联网冒烟报告（2026-09-05 23:47 UTC）

- 命令：`uv run python scripts/smoke_dashscope.py`
- 环境：openai 3.7.0 / langchain-openai 1.6.0 / httpx2 2.12.0；fake 模式关；端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 口径：每候选一次流式调用，`max_tokens=16`，提示词“只回复两个字：收到”；思考形态用 `qwen-plus` + `enable_thinking=true, thinking_budget=64, max_tokens=32`；成本 = 真实 usage × 配置价目表（元/千 token），缺 usage 按预扣计
- 预算护栏：写死 ¥0.050000；本次实付 **¥0.000390**
- 分账：本脚本不接账本（meter=None），账本里没有本次调用；fake 记账与真实花费分离

## 原始 SSE 三验

| 模型 | 变体 | HTTP | 块数 | 首块 s | usage 块位置 | usage 块 choices 空 | finish_reason（位置） | [DONE] | delta 键 | 思考字符 | tokens 入/出 | 花费 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-flash | enable_thinking=false | 200 | 4 | 1.25 | 第 4/4 块 | 是 | stop（第 3 块） | 是 | content, role | 0 | 14/1 | ¥0.000004 | 回复“收到” |
| qwen-turbo | enable_thinking=false | 200 | 4 | 0.37 | 第 4/4 块 | 是 | stop（第 3 块） | 是 | content, role | 0 | 18/1 | ¥0.000006 | 回复“收到” |
| qwen-plus | enable_thinking=false | 200 | 4 | 0.47 | 第 4/4 块 | 是 | stop（第 3 块） | 是 | content, role | 0 | 14/1 | ¥0.000013 | 回复“收到” |
| qwen3.7-max | enable_thinking=false | 200 | 4 | 0.76 | 第 4/4 块 | 是 | stop（第 3 块） | 是 | content, role | 0 | 17/1 | ¥0.000240 | 回复“收到” |
| qwen-plus | enable_thinking=true | 200 | 17 | 0.43 | 第 17/17 块 | 是 | stop（第 16 块） | 是 | content, reasoning_content, role | 90 | 14/56 | ¥0.000123 | 回复“收到” |

## 网关路径（fast 档，fake 关、缓存关、账本关）

- 块数 4，首块 0.93s，回复“收到”
- 末块 usage_metadata：`{'input_tokens': 14, 'output_tokens': 1, 'total_tokens': 15, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}`
- finish_reason：`stop`；上游 model_name：``
- 花费 ¥0.000004

## 结论

- 入池三验：5/5 个调用 HTTP 200 且流完整；逐行看上表的 usage 位置、finish_reason 位置与 [DONE] 列。
- 本报告的数字只用于 README 数字表“真实上游冒烟花费”一行；定性结论进 ADR 实证节。
