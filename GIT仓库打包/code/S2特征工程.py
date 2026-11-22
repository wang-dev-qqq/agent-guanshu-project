# -*- coding: utf-8 -*-
"""
S2 特征工程：Beneish M-score 及衍生特征
========================================
输入：数据表_全部.csv（145 行：33 正例 + 67 反例 + 45 基准年）
输出：特征表.csv（100 个样本行：33 正例 + 67 反例）

说明：
- 金额单位为元，读入后转为 float；'NA' 与空串一律转为 NaN（含义：没拿到/不可比）。
- t-1 通过「同 firm_code、年份-1」匹配基准年行或上一期样本行。
- Beneish M-score 系数采用 Beneish (1999) 原版 8 变量模型。
- LVGI 因数据只有负债合计(tl)，采用「总负债/总资产」比率指数（常见适配）。
"""

import pandas as pd
import numpy as np

# ============ 1. 读数据 ============
df = pd.read_csv('data/数据表_全部.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)

# 列名映射：中文列名 -> 英文短名
rename = {
    'firm_code（股票代码）': 'firm_code',
    'firm_name（公司简称）': 'firm_name',
    'year（会计年度）': 'year',
    'industry（细分行业）': 'industry',
    'new_energy_rev_share（新能源业务收入占比）': 'ne_share',
    'rev_reversal_amt（分部收入冲回金额）': 'rev_rev',
    'label（标签1坏0好NA基数年）': 'label',
    'label_source（处罚文书编号）': 'label_source',
    'rev（营业收入）': 'rev',
    'cog（营业成本）': 'cog',
    'ni（净利润）': 'ni',
    'sell_exp（销售费用）': 'sell_exp',
    'adm_exp（管理费用）': 'adm_exp',
    'ar（应收账款净额）': 'ar',
    'ca（流动资产合计）': 'ca',
    'ppe（固定资产净额）': 'ppe',
    'ta（资产总计）': 'ta',
    'tl（负债合计）': 'tl',
    'cash（货币资金）': 'cash',
    'st_borrow（短期借款）': 'st_borrow',
    'lt_borrow（长期借款）': 'lt_borrow',
    'bond_pay（应付债券）': 'bond_pay',
    'ncl_1y（一年内到期非流动负债）': 'ncl_1y',
    'oth_ar（其他应收款）': 'oth_ar',
    'cfo（经营现金流净额）': 'cfo',
    'capex（购建固定资产等支付现金）': 'capex',
    'cip（在建工程可选）': 'cip',
    'dep（当年折旧可选）': 'dep',
    'audit_opinion（审计意见0至4）': 'audit',
    'inquiry_count_hist（历史涉财问询函计数）': 'inquiry',
    'industry_growth（行业增速小数）': 'ind_growth',
    'note（备注）': 'note',
}
df = df.rename(columns=rename)
df['year'] = df['year'].astype(int)

# 数值列转 float（'NA'/' ' 自动变 NaN）
num_cols = ['ne_share', 'rev_rev', 'rev', 'cog', 'ni', 'sell_exp', 'adm_exp',
            'ar', 'ca', 'ppe', 'ta', 'tl', 'cash', 'st_borrow', 'lt_borrow',
            'bond_pay', 'ncl_1y', 'oth_ar', 'cfo', 'capex', 'cip', 'dep',
            'audit', 'inquiry', 'ind_growth']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ============ 2. 派生中间量 ============
df['sga'] = df['sell_exp'] + df['adm_exp']                      # 销管费用合计
df['debt_int'] = (df['st_borrow'] + df['lt_borrow'] +           # 有息负债（分项缺失按0）
                  df['bond_pay'] + df['ncl_1y']).fillna(0)
df['gross_margin'] = (df['rev'] - df['cog']) / df['rev']        # 毛利率

# ============ 3. 构建 t-1（上期）数据 ============
prev_cols = ['rev', 'cog', 'ni', 'sga', 'ar', 'ca', 'ppe', 'ta', 'tl',
             'dep', 'ne_share', 'debt_int']
prev = df[['firm_code', 'year'] + prev_cols].copy()
prev['year'] = prev['year'] + 1                                  # 上期行对齐到本期年
prev = prev.rename(columns={c: c + '_prev' for c in prev_cols})

merged = df.merge(prev, on=['firm_code', 'year'], how='left')

# ============ 4. 只看样本行（label=0/1）============
sample = merged[merged['label'].isin(['0', '1'])].copy()
sample['label'] = sample['label'].astype(int)                   # 1=舞弊 0=对照

# ============ 5. Beneish 8 指数 ============
def safe_div(a, b):
    """除零/无效返回 NaN"""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = a / b
    return r.replace([np.inf, -np.inf], np.nan)

sample['DSRI'] = safe_div(sample['ar'] / sample['rev'],
                          sample['ar_prev'] / sample['rev_prev'])
sample['GMI'] = safe_div((sample['rev_prev'] - sample['cog_prev']) / sample['rev_prev'],
                         (sample['rev'] - sample['cog']) / sample['rev'])
sample['AQI'] = safe_div(1 - (sample['ca'] + sample['ppe']) / sample['ta'],
                         1 - (sample['ca_prev'] + sample['ppe_prev']) / sample['ta_prev'])
sample['SGI'] = safe_div(sample['rev'], sample['rev_prev'])
sample['DEPI'] = safe_div(sample['dep_prev'] / (sample['dep_prev'] + sample['ppe_prev']),
                          sample['dep'] / (sample['dep'] + sample['ppe']))
sample['SGAI'] = safe_div(sample['sga'] / sample['rev'],
                          sample['sga_prev'] / sample['rev_prev'])
sample['LVGI'] = safe_div(sample['tl'] / sample['ta'],
                          sample['tl_prev'] / sample['ta_prev'])
sample['TATA'] = safe_div(sample['ni'] - sample['cfo'], sample['ta'])

# M-score（Beneish 1999 原版系数）
sample['M_score'] = (-4.84
                     + 0.920 * sample['DSRI']
                     + 0.528 * sample['GMI']
                     + 0.404 * sample['AQI']
                     + 0.892 * sample['SGI']
                     + 0.115 * sample['DEPI']
                     - 0.172 * sample['SGAI']
                     + 4.679 * sample['TATA']
                     - 0.327 * sample['LVGI'])

# 红旗计数（各指数 > 1 记 1；TATA 取 >0）
flags = pd.DataFrame({
    'flag_DSRI': (sample['DSRI'] > 1).astype(float),
    'flag_GMI': (sample['GMI'] > 1).astype(float),
    'flag_AQI': (sample['AQI'] > 1).astype(float),
    'flag_SGI': (sample['SGI'] > 1).astype(float),
    'flag_DEPI': (sample['DEPI'] > 1).astype(float),
    'flag_SGAI': (sample['SGAI'] > 1).astype(float),
    'flag_LVGI': (sample['LVGI'] > 1).astype(float),
    'flag_TATA': (sample['TATA'] > 0).astype(float),
})
sample = pd.concat([sample, flags], axis=1)          # 把 8 个红旗列并入 sample
sample['n_flags'] = flags.sum(axis=1)

# ============ 6. 衍生特征 ============
sample['ni_margin'] = safe_div(sample['ni'], sample['rev'])                       # 净利率
sample['ar_rev'] = safe_div(sample['ar'], sample['rev'])                          # 应收/收入
sample['debt_ratio'] = safe_div(sample['tl'], sample['ta'])                       # 资产负债率
sample['cfo_rev'] = safe_div(sample['cfo'], sample['rev'])                        # 经营现金流/收入
sample['accruals_ta'] = safe_div(sample['ni'] - sample['cfo'], sample['ta'])      # 应计/总资产(=TATA)
sample['capex_rev'] = safe_div(sample['capex'], sample['rev'])                    # 资本开支/收入
sample['oth_ar_ta'] = safe_div(sample['oth_ar'], sample['ta'])                    # 其他应收款/总资产
sample['rev_growth'] = sample['rev'] / sample['rev_prev'] - 1                     # 收入增速
sample['ta_growth'] = sample['ta'] / sample['ta_prev'] - 1                        # 资产增速
sample['tl_growth'] = sample['tl'] / sample['tl_prev'] - 1                        # 负债增速
sample['ne_share_yoy'] = sample['ne_share'] / sample['ne_share_prev'] - 1         # 新能源占比同比
sample['rev_rev_ratio'] = safe_div(sample['rev_rev'], sample['rev'])              # 分部收入冲回占比

# ---- 方案 12 财务特征补齐 ----
# 其他应收款占比 = oth_ar_ta（上面已算）；资本开支强度 = capex_rev（上面已算）
# 存贷双高：货币资金/TA 与 有息负债/TA 均高于同行业中位数 → 1，否则 0；原料缺失 → NA
sample['cash_ta'] = safe_div(sample['cash'], sample['ta'])
sample['debt_ta'] = safe_div(sample['debt_int'], sample['ta'])
cash_med = sample.groupby('industry')['cash_ta'].transform('median')
debt_med = sample.groupby('industry')['debt_ta'].transform('median')
sample['cundun'] = ((sample['cash_ta'] > cash_med) & (sample['debt_ta'] > debt_med)).astype(float)
sample.loc[sample['cash_ta'].isna() | sample['debt_ta'].isna(), 'cundun'] = np.nan
# 营收与行业背离 = 收入同比增速 - 行业增速
sample['rev_indiv'] = sample['rev_growth'] - sample['ind_growth']

# ============ 7. 输出列 ============
id_cols = ['firm_code', 'firm_name', 'year', 'industry', 'label', 'label_source']
beneish_cols = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M_score', 'n_flags']
flag_cols = [c for c in flags.columns]
derived_cols = ['sga', 'debt_int', 'gross_margin', 'ni_margin', 'ar_rev', 'debt_ratio',
                'cfo_rev', 'accruals_ta', 'capex_rev', 'oth_ar_ta',
                'rev_growth', 'ta_growth', 'tl_growth', 'ne_share', 'ne_share_yoy',
                'rev_rev_ratio', 'cash_ta', 'debt_ta', 'cundun', 'rev_indiv']
signal_cols = ['audit', 'inquiry', 'ind_growth']

# 方案 12 财务特征（财务智能体输入）：
# DSRI GMI AQI SGI DEPI SGAI LVGI TATA | cundun(存贷双高) oth_ar_ta(其他应收占比) rev_indiv(营收行业背离) capex_rev(资本开支强度)
FIN12 = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA',
         'cundun', 'oth_ar_ta', 'rev_indiv', 'capex_rev']

out_cols = id_cols + beneish_cols + flag_cols + derived_cols + signal_cols
out = sample[out_cols].sort_values(['label', 'firm_code', 'year']).reset_index(drop=True)

out.to_csv('data/特征表.csv', index=False, encoding='utf-8-sig')

# 单独存一份 12 财务特征（财务智能体输入，含标识列）
out[['firm_code', 'firm_name', 'year', 'industry', 'label'] + FIN12].to_csv('data/财务特征12.csv', index=False, encoding='utf-8-sig')

# ============ 8. 摘要 ============
print(f'输出特征表：{out.shape[0]} 行 × {out.shape[1]} 列')
print(f'标签分布：舞弊(1)={int((out["label"]==1).sum())}，对照(0)={int((out["label"]==0).sum())}')
print()
print('--- Beneish 8 指数 NaN 数（不可比/缺 t-1，属预期）---')
for c in beneish_cols[:8]:
    print(f'  {c}: {int(out[c].isna().sum())} 个 NaN')
print()
print('--- M_score 分组统计 ---')
g = out.groupby('label')['M_score']
print(g.agg(['count', 'mean', 'median', 'min', 'max']).round(3).to_string())
print()
print('--- 红旗计数 n_flags 分组分布 ---')
print(out.groupby('label')['n_flags'].value_counts().unstack(fill_value=0).to_string())
print()
print('--- 12 财务特征 NaN 数 ---')
for c in FIN12:
    print(f'  {c}: {int(out[c].isna().sum())} 个 NaN')
