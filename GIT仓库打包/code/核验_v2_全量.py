# -*- coding: utf-8 -*-
"""核验 v2 全量：核对公司原文陈述 vs 事实，四分类裁定（含 misleading），断点续跑"""
import os, json, time
import numpy as np, pandas as pd
from openai import OpenAI

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
assert API_KEY, "请先设置环境变量：export DEEPSEEK_API_KEY='你的key'"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
TOP_N = 3
DELAY = 1.0
OUT_R, OUT_S = 'data/核验v2_结果.csv', 'data/核验v2_汇总.csv'

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def build_fact_sheet(data, firm, year):
    r = data[(data['firm_code'] == firm) & (data['year'].astype(int) == int(year))]
    if r.empty: return None
    r = r.iloc[0]
    prev = data[(data['firm_code'] == firm) & (data['year'].astype(int) == int(year) - 1)]
    def num(v):
        try:
            f = float(v); return f if f == f else np.nan
        except (ValueError, TypeError): return np.nan
    rev, cog = num(r['rev']), num(r['cog'])
    ni, cfo = num(r['ni']), num(r['cfo'])
    ta, tl = num(r['ta']), num(r['tl'])
    ar, oth = num(r['ar']), num(r['oth_ar'])
    cash = num(r['cash'])
    stb, ltb, bond, ncl = num(r['st_borrow']), num(r['lt_borrow']), num(r['bond_pay']), num(r['ncl_1y'])
    dp = [stb, ltb, bond, ncl]
    debt_int = np.nan if all(x != x for x in dp) else sum(0 if x != x else x for x in dp)
    facts = []
    def add(fid, txt): facts.append((fid, txt))
    add('F1', f"营业收入={rev:,.0f}元")
    if not prev.empty:
        rev0 = num(prev.iloc[0]['rev'])
        if rev0 == rev0 and rev0 != 0: add('F2', f"营业收入同比={rev/rev0-1:+.1%}")
    if rev == rev and rev != 0:
        add('F3', f"毛利率={(rev-cog)/rev:.1%}"); add('F4', f"净利率={ni/rev:.1%}")
        add('F5', f"经营现金流/营业收入={cfo/rev:.1%}"); add('F6', f"应收账款/营业收入={ar/rev:.1%}")
    add('F7', f"净利润={ni:,.0f}元")
    if ta == ta and ta != 0:
        add('F8', f"资产负债率={tl/ta:.1%}"); add('F9', f"其他应收款/总资产={oth/ta:.1%}")
    add('F10', f"行业增速={num(r['industry_growth']):.1%}")
    add('F11', f"审计意见编码={num(r['audit_opinion'])}（0标准/1强调/2保留/3无法/4否定）")
    add('F12', f"历史涉财问询函={num(r['inquiry_count_hist'])}份")
    if num(r['new_energy_rev_share']) == num(r['new_energy_rev_share']):
        add('F13', f"新能源收入占比={num(r['new_energy_rev_share']):.1%}")
    if cash == cash: add('F14', f"货币资金={cash:,.0f}元")
    if stb == stb: add('F15', f"短期借款={stb:,.0f}元")
    if debt_int == debt_int: add('F16', f"有息负债={debt_int:,.0f}元")
    if ta == ta and ta != 0:
        if cash == cash: add('F17', f"货币资金/总资产={cash/ta:.1%}")
        if debt_int == debt_int: add('F18', f"有息负债/总资产={debt_int/ta:.1%}")
    if not prev.empty:
        ar0 = num(prev.iloc[0]['ar']); ni0 = num(prev.iloc[0]['ni']); cfo0 = num(prev.iloc[0]['cfo'])
        if ar0 == ar0 and ar0 != 0: add('F19', f"应收账款同比={ar/ar0-1:+.1%}")
        if ni0 == ni0 and ni0 != 0: add('F20', f"净利润同比={ni/ni0-1:+.1%}")
        if cfo0 == cfo0 and cfo0 != 0: add('F21', f"经营现金流同比={cfo/cfo0-1:+.1%}")
    return facts

PROMPT = """你是审计师，核对公司年报 MD&A 原文陈述与财务报表事实是否一致。

【公司原文陈述】
{quote}

【财务报表事实】
{fact}

判断该陈述与事实的关系（四选一）：
- contradicted：陈述的断言与事实直接矛盾（如称"现金流充裕/稳健"但经营现金流为负；称"回款良好"但应收暴增）；
- misleading：陈述把变化归因于单一原因，但事实显示另有同等或更重要的原因，或回避了事实中明显可见的重大不利因素（如把现金流大降仅归因于补贴未到，而应收账款同样暴增；称"延长优质客户付款期"却无账龄佐证）；
- supported：陈述与事实一致，归因合理；
- unverifiable：材料不足。

只输出 JSON：{{"verdict":"contradicted|misleading|supported|unverifiable","reason":"一句话","evidence_fact_ids":["F1"]}}"""

def verify(quote, fact_lines):
    fact = '\n'.join(f"- {f}: {t}" for f, t in fact_lines)
    msgs = [{"role": "user", "content": PROMPT.format(quote=quote, fact=fact)}]
    for _ in range(3):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0,
                                                  response_format={"type": "json_object"}, max_tokens=16000)
            raw = (resp.choices[0].message.content or '').strip()
            if raw: return json.loads(raw)
        except Exception:
            time.sleep(2)
    return {"verdict": "unverifiable", "reason": "调用失败", "evidence_fact_ids": []}

def main():
    claims = pd.read_csv('data/语义声明_v2.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    claims = claims[claims['quote_verified'] == 'True'].copy()
    claims['score'] = pd.to_numeric(claims['score'], errors='coerce')
    data = pd.read_csv('data/数据表_全部.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    data.columns = [c.split('（')[0] for c in data.columns]

    firms = claims[['firm_code', 'firm_name', 'year', 'label']].drop_duplicates().values.tolist()
    # 断点续跑
    done = set()
    res_rows, sum_rows = [], []
    if os.path.exists(OUT_S):
        old = pd.read_csv(OUT_S, dtype=str, keep_default_na=False)
        done = set(zip(old['firm_code'], old['year']))
        sum_rows = old.to_dict('records')
    if os.path.exists(OUT_R):
        res_rows = pd.read_csv(OUT_R, dtype=str, keep_default_na=False).to_dict('records')
    firms = [f for f in firms if (f[0], f[2]) not in done]
    print(f'已完成 {len(done)}，本轮 {len(firms)} 家')

    for i, (firm, name, year, label) in enumerate(firms, 1):
        facts = build_fact_sheet(data, firm, year)
        if not facts: continue
        sub = claims[(claims['firm_code'] == firm) & (claims['year'] == year)].sort_values('score', ascending=False).head(TOP_N)
        c = m = s = u = 0
        for _, r in sub.iterrows():
            v = verify(r['quote'], facts)
            vd = v.get('verdict', 'unverifiable')
            if vd == 'contradicted': c += 1
            elif vd == 'misleading': m += 1
            elif vd == 'supported': s += 1
            else: u += 1
            res_rows.append({'firm_code': firm, 'firm_name': name, 'year': year, 'label': label,
                             'dimension': r['dimension'], 'score': r['score'], 'verdict': vd,
                             'reason': v.get('reason', ''), 'quote': r['quote'][:120]})
            time.sleep(DELAY)
        n = len(sub)
        sum_rows.append({'firm_code': firm, 'firm_name': name, 'year': year, 'label': label,
                         'n_claims': n, 'contradicted': c, 'misleading': m, 'supported': s, 'unverifiable': u,
                         'problem_rate': round((c + m) / n, 4) if n else np.nan,
                         'contradiction_rate': round(c / n, 4) if n else np.nan,
                         'coverage': round((c + m + s) / n, 4) if n else np.nan})
        print(f'[{i}/{len(firms)}] {name} {year} 矛盾{c} 误导{m} 支持{s} 不可核{u} 问题率={(c+m)/n if n else 0:.2f}', flush=True)
        pd.DataFrame(res_rows).to_csv(OUT_R, index=False, encoding='utf-8-sig')
        pd.DataFrame(sum_rows).to_csv(OUT_S, index=False, encoding='utf-8-sig')

    sm = pd.read_csv(OUT_S, dtype=str, keep_default_na=False)
    def auc(col):
        x = pd.to_numeric(sm[col], errors='coerce'); y = (sm['label'] == '1').astype(int)
        d = pd.DataFrame({'s': x, 'y': y}).dropna()
        p, nn = d[d.y == 1]['s'].values, d[d.y == 0]['s'].values
        return round(sum((1 if a > b else (0.5 if a == b else 0)) for a in p for b in nn) / (len(p) * len(nn)), 3)
    print('\n=== 核验v2 全量区分度 ===')
    for col in ['problem_rate', 'contradiction_rate', 'coverage']:
        y1 = pd.to_numeric(sm[sm['label'] == '1'][col], errors='coerce').mean()
        y0 = pd.to_numeric(sm[sm['label'] == '0'][col], errors='coerce').mean()
        print(f'  {col:20s} AUC={auc(col):.3f}  舞弊={y1:.3f} 对照={y0:.3f}')

if __name__ == '__main__':
    main()
