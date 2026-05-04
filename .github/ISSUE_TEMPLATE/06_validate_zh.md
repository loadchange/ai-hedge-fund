---
name: 🧪 信号验证（CPCV / PBO）
about: 评估 quant signal 在历史数据上是否过拟合（无 LLM 成本，速度快）
title: "信号验证："
labels: ["bot-validate", "lang-zh"]
---

> 这个模板带 `bot-validate` 标签，机器人会调用 `python -m src.validation.cli evaluate`。
> **完全不调用 LLM** —— 纯量化算法，几秒到几十秒出结果。

**信号 key（必填）**

逗号分隔。CPCV 是日频滚动评估器，只支持**技术信号**：
`trend`, `mean_reversion`, `momentum`, `volatility`, `stat_arb`。

基本面信号（`value`, `quality`, `earnings_surprise`）只在财报时点
更新，需要不同的评估器 —— 这里会直接报错并给出建议。

例如：`momentum, mean_reversion, trend`

**股票代码（必填）**

逗号分隔，例如 `AAPL,MSFT,NVDA`。

**时间范围（推荐填写）**

- 开始日期（YYYY-MM-DD）：
- 结束日期（YYYY-MM-DD）：

建议至少 18 个月，让 momentum 信号的 126 天滚动窗口有足够历史。

**CPCV 参数（可选）**

- n-splits（默认 8）：
- n-test-splits（默认 2）：
- rolling-window（默认 60，建议改成 180 让 momentum 等 6 月窗口信号能跑）：

**其他备注**

写下你想关注的角度。报告会包含 IS Sharpe / OOS Sharpe / PBO（过拟合概率）/ Deflated Sharpe Ratio。
