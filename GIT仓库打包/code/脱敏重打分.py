# -*- coding: utf-8 -*-
"""脱敏重打分：对16篇脱敏MD&A用S4 v2.1同一prompt重打分，验证LLM不是靠认出公司名打分
输出：脱敏重打分_结果.csv（每篇三维mean/max）+ 与原版对比 + 16篇AUC"""
import os, time, json, importlib.util
import numpy as np, pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# 复用 S4 v2.1 的 prompt / 分块 / 打分函数
spec = importlib.util.spec_from_file_location('s4', 'code/S4语义打分.py')
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)

from openai import OpenAI
client = OpenAI(api_key=s4.API_KEY, base_url=s4.BASE_URL)

OUT = '脱敏重打分_结果.csv'
MAP = '脱敏盲评包/_样本对应表_仅队长.csv'
DIMS = ('risk_concealment', 'financial_anomaly', 'future_promises')

def main():
    mp = pd.read_csv(MAP, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    done = {}
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype=str, keep_default_na=False)
        done = {r['sample_id']: r.to_dict() for _, r in old.iterrows()}
    rows = list(done.values())

    for _, m in mp.iterrows():
        sid = m['sample_id']
        if sid in done:
            print(f'跳过已完成 {sid}')
            continue
        path = f'脱敏盲评包/{sid}.txt'
        text = open(path, encoding='utf-8').read()
        print(f'打分 {sid}（{len(text)}字）...', flush=True)
        try:
            scores, claims = s4.score_one_mda(client, text)
        except Exception as e:
            print(f'  !! {sid} 失败: {e}')
            continue
        row = {'sample_id': sid, 'firm_name': m['firm_name'], 'year': m['year'], 'label': m['label'],
               'n_chunks': scores.get('n_chunks')}
        for d in DIMS:
            row[f'{d}_mean'] = scores.get(f'{d}_mean')
            row[f'{d}_max'] = scores.get(f'{d}_max')
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
        print(f"  完成 {sid}: " + ' '.join(f"{d}={row[f'{d}_max']}" for d in DIMS), flush=True)

    # ===== 对比分析 =====
    de = pd.read_csv(OUT, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    for c in de.columns:
        if c.endswith(('_mean', '_max')): de[c] = pd.to_numeric(de[c], errors='coerce')
    orig = pd.read_csv('data/语义评分_v2.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    for c in orig.columns:
        if c.endswith(('_mean', '_max')): orig[c] = pd.to_numeric(orig[c], errors='coerce')
    mg = de.merge(orig, on=['firm_name', 'year'], suffixes=('_脱敏', '_原版'))

    def auc(s, lab):
        d = pd.DataFrame({'s': pd.to_numeric(s, errors='coerce'), 'y': (pd.Series(lab).astype(str) == '1').astype(int)}).dropna()
        p, n = d[d.y == 1]['s'].values, d[d.y == 0]['s'].values
        if len(p) == 0 or len(n) == 0: return float('nan')
        return round(sum((1 if a > b else (0.5 if a == b else 0)) for a in p for b in n) / (len(p) * len(n)), 3)

    from scipy import stats
    print('\n=== 脱敏前后对比（16篇：5舞弊+11对照）===')
    print(f"{'维度':24s} {'脱敏AUC':>8s} {'原版AUC':>8s} {'Spearman':>9s}")
    for d in DIMS:
        for agg in ['max', 'mean']:
            col = f'{d}_{agg}'
            rho = stats.spearmanr(mg[f'{col}_脱敏'], mg[f'{col}_原版'])[0]
            print(f"{col:24s} {auc(de[col], de['label']):8.3f} {auc(mg[f'{col}_原版'], mg['label']):8.3f} {rho:9.3f}")
    mg.to_csv('脱敏重打分_对比.csv', index=False, encoding='utf-8-sig')
    print('\n逐样本对比已存 脱敏重打分_对比.csv')

if __name__ == '__main__':
    main()
