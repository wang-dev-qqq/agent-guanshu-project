# 核验 v2 Prompt（deepseek-flash，temperature=0，response_format=json_object）

来源：`核验_v2_全量.py` 中的 PROMPT 变量。占位符 {quote} = 公司原文陈述（来自语义声明_v2.csv 的 quote 列，经 quote_verified 校验），{fact} = 21 项财务事实清单（F1–F21，由数据表_全部.csv 现场构建）。

## 事实清单 F1–F21 构造规则（build_fact_sheet 函数）

F1 营业收入；F2 营收同比；F3 毛利率；F4 净利率；F5 经营现金流/营收；F6 应收账款/营收；F7 净利润；F8 资产负债率；F9 其他应收款/总资产；F10 行业增速；F11 审计意见编码；F12 历史涉财问询函计数；F13 新能源收入占比；F14 货币资金；F15 短期借款；F16 有息负债（四项相加）；F17 货币资金/总资产；F18 有息负债/总资产；F19 应收账款同比；F20 净利润同比；F21 经营现金流同比。
取数口径：t 年当期值 + t−1 年同比；事实表缺失的项不进入清单。

## Prompt 原文

```
你是审计师，核对公司年报 MD&A 原文陈述与财务报表事实是否一致。

【公司原文陈述】
{quote}

【财务报表事实】
{fact}

判断该陈述与事实的关系（四选一）：
- contradicted：陈述的断言与事实直接矛盾（如称"现金流充裕/稳健"但经营现金流为负；称"回款良好"但应收暴增）；
- misleading：陈述把变化归因于单一原因，但事实显示另有同等或更重要的原因，或回避了事实中明显可见的重大不利因素（如把现金流大降仅归因于补贴未到，而应收账款同样暴增；称"延长优质客户付款期"却无账龄佐证）；
- supported：陈述与事实一致，归因合理；
- unverifiable：材料不足。

只输出 JSON：{{"verdict":"contradicted|misleading|supported|unverifiable","reason":"一句话","evidence_fact_ids":["F1"]}}
```

## 调用与裁定聚合

- 每个公司年度取 score 最高的 3 条声明（TOP_N=3），逐条调用；
- 裁定四分类：contradicted / misleading / supported / unverifiable；
- 公司年度级聚合（核验v2_汇总.csv）：problem_rate = (contradicted + misleading)/n；contradiction_rate = contradicted/n；coverage = (contradicted + misleading + supported)/n；
- 同一脚本（核验_v2_全量.py）同时产出逐条明细（核验v2_结果.csv）与公司年度汇总（核验v2_汇总.csv），无单独汇总脚本；
- 断点续跑：已存在的公司年从汇总表跳过，逐家增量保存。
