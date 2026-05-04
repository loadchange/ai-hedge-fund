---
name: 🤖 Ticker 分析（中文）
about: 让机器人对指定股票做一次完整的多 agent 分析（调用 src/main.py）
title: "Ticker 分析："
labels: ["bot-ticker", "lang-zh"]
---

> 这个模板带 `bot-ticker` 标签，机器人会自动解析正文里的参数并调用 `uv run python src/main.py`。回复语言会跟随 issue 语言（中文 issue 返回中文，英文 issue 返回英文）。

**股票代码（必填）**

例如 `AAPL,MSFT`、`600519.SS`、`9988.HK`。也接受 6 位纯数字（A 股自动识别上证/深证）。

**时间范围（可选）**

- 开始日期（YYYY-MM-DD）：
- 结束日期（YYYY-MM-DD）：

**分析师**

填 `all` 启用全部分析师，或指定子集（逗号分隔），例如：
`warren_buffett, duan_yongping, charlie_munger`

可用 key 列表见 [README](../../#agent-roster) 或 `src/utils/analysts.py`。

**其他备注**

想看的角度、行业背景，或希望机器人特别留意的点。
