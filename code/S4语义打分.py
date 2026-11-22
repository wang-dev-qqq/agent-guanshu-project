# -*- coding: utf-8 -*-
"""
S4 LLM 语义打分 v2（claim 层结构化输出）
========================================
输入：MD_A_正例.json + MD_A_反例.json
输出：语义评分.csv（三维度分） + 语义声明.csv（逐条 claim，供核验智能体）

每块除三个维度总分外，还输出逐条"声明"（claim）：
  dimension / claim_id / claim(可核查断言概括) / quote(逐字原文) / location / score
后续核验智能体取每公司年度最高的 2-3 条有效声明，与财务/行业/监管事实交叉核验。

用法：
    export DEEPSEEK_API_KEY="sk-xxx"
    python3 S4_LLM语义打分.py
"""

import os
import json
import time
import re
import pandas as pd
from openai import OpenAI

# ==================== 本地运行配置 ====================
# 1) 自动切到"脚本所在文件夹"，保证能读到同目录下的 MD_A_*.json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ==================== 配置 ====================
# 2) 密钥从环境变量 DEEPSEEK_API_KEY 读取，仓库内不含密钥明文
import os
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
assert API_KEY, "请先设置环境变量：export DEEPSEEK_API_KEY='你的key'"
BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("LLM_MODEL", "deepseek-flash")
CHUNK_SIZE = 6000
REPEATS = 1                # 正式打分设 3~5
MAX_RETRY = 5
DELAY = 1.5
TEST_LIMIT = None
OUTPUT_SCORE = "data/语义评分_v2.csv"
OUTPUT_CLAIM = "data/语义声明_v2.csv"

# ==================== 评分量表 ====================
RUBRIC = """
你是资深财务舞弊审计专家。你的任务是从上市公司年报"管理层讨论与分析（MD&A）"中识别与财务舞弊相关的红旗信号，而不是评价文字写得好不好。请对给定文本按以下三个维度打分（每个维度 0-10 分，允许 0.5 步进）。

【评分总原则】
1. 只有出现明确的舞弊红旗才给 5 分及以上；正常、合规的披露给 0-4 分。
2. 分数要拉开差距，不要都挤在 4-5 分。
3. 评分依据必须是文本中的具体表述，不要臆测。

【维度一：风险掩饰度 risk_concealment】
衡量公司对真实风险的披露是否充分、是否回避已发生的负面事实。
- 0-2 充分披露：负面风险有量化金额/比例/影响，且披露了应对措施；
- 3-4 模板化：风险照搬格式条款，无量化、无针对性；
- 5-6 避重就轻：重大风险（债务、回款、诉讼、政策）一句话带过、无量化；
- 7-8 明显回避：已发生的问询函、诉讼、债务逾期、大客户流失、担保代偿等只字未提或轻描淡写；
- 9-10 粉饰太平：否认客观风险，或声称"无重大影响""风险可控"却无任何数据支撑。

【维度二：资金与债务异常信号 financial_anomaly】
衡量文字是否透露资金链紧张、资金被占用、存贷双高、利润无现金支撑等舞弊高危迹象。
- 0-2 无异常：资金、债务、现金流描述与经营情况匹配，披露清晰；
- 3-4 常规：无明显资金异常信号；
- 5-6 有迹象：出现"高存款与高借款并存""资金被关联方占用""应收账款回款周期异常拉长"等迹象但未给出解释；
- 7-8 明显异常：文字暗示资金链紧张、债务逾期风险、利润与经营现金流明显背离、大额其他应收款缺乏合理解释；
- 9-10 高度吻合：文字与"存贷双高、资金占用、利润无现金支撑"等舞弊标志直接吻合，或对此给出牵强、自相矛盾的解释。

【维度三：未来画饼度 future_promises】
衡量未来规划是否脱离公司实际的资金、技术、产能约束。
- 0-2 务实：含具体产能、时间表、资金来源、可行性论证；
- 3-4 常规展望：泛泛而谈行业前景；
- 5-6 宏大叙事：大规模扩产但缺资金安排、技术来源或可行性论证；
- 7-8 脱离约束：规模明显超出经营现金流与融资能力，或与行业产能过剩脱节；
- 9-10 空想承诺：历史承诺未兑现又提更宏大承诺，或以未来叙事回应当期亏损/问询。

【输出格式】只输出一个 JSON 对象，不要解释、不要 markdown 代码块：
{
  "dimension_scores": {
    "risk_concealment": 6.5,
    "financial_anomaly": 7.0,
    "future_promises": 3.0
  },
  "claims": [
    {"claim_id": "1", "dimension": "financial_anomaly",
     "claim": "公司称货币资金充裕但短期借款同步大幅增加（可核查断言概括）",
     "quote": "逐字原文",
     "location": "经营情况讨论与分析·资金",
     "score": 7.0}
  ]
}
【硬规则】
1. 每个维度 score>=5 时，claims 必须给出至少 1 条（最多 3 条）支撑声明；score<5 的维度 claims 可为空数组。
2. claim 是"可核查的断言概括"；quote 必须是给定文本的连续子串，逐字摘录不得改写；location 写清所在章节或段落。
3. quote 校验失败的声明将被弃用，所以宁可少而准。
4. claim 必须能被财务/行业数据核验：优先选取涉及营收、利润、毛利率、现金流、负债、借款、应收/其他应收、存货、产能、订单等具体数字或趋势的断言；避免"市占率第一""技术领先""品牌知名"等无法用报表数字核验的定性断言。
"""

SYSTEM_PROMPT = "你是资深财务舞弊审计专家，识别MD&A中的财务舞弊红旗信号并输出结构化JSON。"


# ==================== 工具函数 ====================
def chunk_text(text, size=CHUNK_SIZE):
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks, cur = [], ''
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = (cur + '\n' + p).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def normalize(s):
    return re.sub(r'\s+', '', s)


def call_score(client, chunk):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RUBRIC + "\n\n【待评分 MD&A 文本】\n" + chunk},
    ]
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0,
                response_format={"type": "json_object"}, max_tokens=16000)
            raw = (resp.choices[0].message.content or '').strip()
            if not raw:
                raise ValueError('模型返回空内容')
            return json.loads(raw)
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=0, max_tokens=16000)
        raw = resp.choices[0].message.content or ''
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f'降级解析失败: {raw[:120]}')
    except Exception as e:
        raise RuntimeError(f'{last_err}; 降级失败: {e}')


def score_one_mda(client, text):
    """返回 (维度分 dict, claims list)。claims 带 quote_verified 标记。"""
    chunks = chunk_text(text)
    dims = ('risk_concealment', 'financial_anomaly', 'future_promises')
    dim_acc = {d: [] for d in dims}
    claims = []
    ev_fail = ev_total = 0

    for ci, chunk in enumerate(chunks):
        if ci > 0:
            time.sleep(DELAY)
        try:
            r = call_score(client, chunk)
        except Exception as e:
            print(f"    !! 块{ci} 打分失败: {e}")
            for d in dims:
                dim_acc[d].append(float('nan'))
            continue

        ds = r.get('dimension_scores', {})
        for d in dims:
            dim_acc[d].append(float(ds.get(d, {}).get('score', ds.get(d, 0))
                                    if isinstance(ds.get(d), dict) else ds.get(d, 0)))

        norm_chunk = normalize(chunk)
        for c in r.get('claims', []) or []:
            q = c.get('quote', '') or ''
            verified = (normalize(q) in norm_chunk) and len(q.strip()) >= 4
            if not verified:
                ev_fail += 1
            ev_total += 1
            claims.append({
                'chunk_id': ci,
                'claim_id': c.get('claim_id', str(len(claims) + 1)),
                'dimension': c.get('dimension', ''),
                'claim': c.get('claim', ''),
                'quote': q,
                'location': c.get('location', ''),
                'score': float(c.get('score', 0)),
                'quote_verified': verified,
            })

    out = {'n_chunks': len(chunks)}
    for d in dims:
        vals = [v for v in dim_acc[d] if not (isinstance(v, float) and v != v)]
        out[d + '_mean'] = round(sum(vals) / len(vals), 3) if vals else float('nan')
        out[d + '_max'] = round(max(vals), 3) if vals else float('nan')
    out['evidence_fail_rate'] = round(ev_fail / ev_total, 4) if ev_total else 0
    out['n_claims'] = len(claims)
    out['n_verified_claims'] = sum(1 for c in claims if c['quote_verified'])
    out['composite'] = round(sum(v for k, v in out.items()
                                  if k.endswith('_mean') and isinstance(v, float) and v == v) / 3, 3)
    return out, claims


# ==================== 主流程 ====================
def main():
    if not API_KEY:
        print("请先设置环境变量：export DEEPSEEK_API_KEY='sk-xxx'")
        return
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    recs = []
    for fname in ('data/MD_A_正例.json', 'data/MD_A_反例.json'):
        recs += json.load(open(fname, encoding='utf-8'))
    if TEST_LIMIT:
        recs = recs[:TEST_LIMIT]

    # ---- 断点续跑：读已有结果，跳过已完成的 firm-year ----
    done = set()
    old_score = pd.DataFrame()
    old_claim = pd.DataFrame()
    if os.path.exists(OUTPUT_SCORE) and os.path.getsize(OUTPUT_SCORE) > 0:
        try:
            old_score = pd.read_csv(OUTPUT_SCORE, dtype=str, keep_default_na=False)
            done = set(zip(old_score['firm_code'], old_score['year'].astype(str)))
        except Exception as e:
            print(f'读取已有评分失败（当作无）: {e}')
    if os.path.exists(OUTPUT_CLAIM) and os.path.getsize(OUTPUT_CLAIM) > 0:
        try:
            old_claim = pd.read_csv(OUTPUT_CLAIM, dtype=str, keep_default_na=False)
        except Exception:
            old_claim = pd.DataFrame()

    todo = [r for r in recs if (str(r['firm_code']), str(r['year'])) not in done]
    batch = int(os.environ.get('S4_BATCH', '0'))
    if batch > 0:
        todo = todo[:batch]
    print(f'共 {len(recs)} 份，已完成 {len(done)}，本轮待打 {len(todo)}')

    rows = old_score.to_dict('records') if not old_score.empty else []
    claim_rows = old_claim.to_dict('records') if not old_claim.empty else []
    new_done = 0
    for i, r in enumerate(todo, 1):
        t0 = time.time()
        try:
            s, claims = score_one_mda(client, r['text'])
        except Exception as e:
            print(f"!! {r['firm_name']} {r['year']} 整体失败: {e}", flush=True)
            continue
        row = {'firm_code': str(r['firm_code']), 'firm_name': r['firm_name'],
               'year': str(r['year']), 'label': str(r['label'])}
        row.update(s)
        rows.append(row)
        for c in claims:
            cr = {'firm_code': str(r['firm_code']), 'firm_name': r['firm_name'],
                  'year': str(r['year']), 'label': str(r['label'])}
            cr.update(c)
            claim_rows.append(cr)
        # 每样本落盘一次，中断不丢进度
        pd.DataFrame(rows).to_csv(OUTPUT_SCORE, index=False, encoding='utf-8-sig')
        pd.DataFrame(claim_rows).to_csv(OUTPUT_CLAIM, index=False, encoding='utf-8-sig')
        new_done += 1
        print(f"[{new_done}/{len(todo)}] {r['firm_name']} {r['year']} "
              f"综合={s['composite']} 声明={s['n_claims']}条(有效{s['n_verified_claims']}) "
              f"({time.time()-t0:.1f}s)", flush=True)

    print(f'\n本轮新增 {new_done} 份；累计 {len(rows)} 份、{len(claim_rows)} 条声明')
    print(f'已写 {OUTPUT_SCORE} 和 {OUTPUT_CLAIM}')


if __name__ == '__main__':
    main()
