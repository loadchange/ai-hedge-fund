---
name: 📈 历史回测（中文）
about: 让机器人对指定股票做一次回测（调用 src/backtester.py）
title: "回测："
labels: ["bot-backtester", "lang-zh"]
---

> 这个模板带 `bot-backtester` 标签，机器人会自动解析正文里的参数并调用 `uv run python src/backtester.py`。回复语言会跟随 issue 语言。

**股票代码（必填）**

例如 `AAPL,MSFT`、`600519.SS`、`9988.HK`。

**回测时间范围（推荐填写）**

- 开始日期（YYYY-MM-DD）：
- 结束日期（YYYY-MM-DD）：

**分析师**

填 `all` 启用全部分析师，或指定子集（逗号分隔），例如：
`warren_buffett, duan_yongping, charlie_munger`

**其他备注**

回测时长越大、分析师越多，调用成本和耗时越高，请合理设置区间。
