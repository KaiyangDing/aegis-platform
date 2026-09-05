# reports/

对外引用的数字只来自根 README 的数字表，数字表的每一行回指本目录中的一份报告。本目录存放三类文件：

- 三门运行记录：`YYYY-MM-DD-gates.md`，记录 `ruff` / `lint-imports` / `pytest --collect-only -q` 的命令与输出摘要。
- 探针日志：`YYYY-MM-DD-probe-<主题>.md`，对应 ADR 实证节的复现步骤，附原始输出。
- 联网冒烟报告：`YYYY-MM-DD-smoke-<供应商>.md`，真实上游一次运行的命令、预算护栏、花费与核对结论；fake 记账与真实花费分账，真实花费只出现在这里。

每份报告必须写明：日期、命令、口径（测什么、怎么算）、环境（依赖版本、是否 fake 模式）、结果。没有报告的数字不进 README。
