# -*- coding: utf-8 -*-
"""S6 SHAP 解释：全局重要性、典型样本个体解释、资金异常×TATA 交互
主模型 = 19维（12财务+3语义+2核验+2外部），全样本拟合后解释。"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.makedirs('figs', exist_ok=True)

FIN12 = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','LVGI','TATA','cundun','oth_ar_ta','rev_indiv','capex_rev']
SEM3 = ['risk_concealment_max','financial_anomaly_max','future_promises_max']
EXT2 = ['audit','inquiry']
VER2 = ['problem_rate','coverage']
COLS = FIN12 + SEM3 + VER2 + EXT2

fin = pd.read_csv('data/财务特征12.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
sem = pd.read_csv('data/语义评分_v2.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
ext = pd.read_csv('data/特征表.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)[['firm_code','year','audit','inquiry']]
ver = pd.read_csv('data/核验v2_汇总.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)[['firm_code','year']+VER2]
df = fin.merge(sem[['firm_code','year']+SEM3], on=['firm_code','year'], how='left') \
        .merge(ext, on=['firm_code','year'], how='left') \
        .merge(ver, on=['firm_code','year'], how='left')
for c in COLS: df[c] = pd.to_numeric(df[c], errors='coerce')
y = df['label'].astype(int).values

imp = SimpleImputer(strategy='median')
X = pd.DataFrame(imp.fit_transform(df[COLS]), columns=COLS)

model = xgb.XGBClassifier(max_depth=2, n_estimators=150, learning_rate=0.08, min_child_weight=4,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=2, scale_pos_weight=2.0,
                          random_state=42, tree_method='hist', n_jobs=1, verbosity=0)
model.fit(X, y)
prob = model.predict_proba(X)[:, 1]
df['prob'] = prob

explainer = shap.TreeExplainer(model)
sv = explainer(X)

# 1) 全局：mean|SHAP| 条形图
plt.figure(figsize=(8, 6))
shap.summary_plot(sv, X, plot_type='bar', show=False, max_display=19)
plt.tight_layout(); plt.savefig('figs/SHAP_global_bar.png', dpi=150); plt.close()
print('已出 figs/SHAP_global_bar.png')

# 2) 全局：beeswarm
plt.figure(figsize=(8, 7))
shap.summary_plot(sv, X, show=False, max_display=19)
plt.tight_layout(); plt.savefig('figs/SHAP_beeswarm.png', dpi=150); plt.close()
print('已出 figs/SHAP_beeswarm.png')

# 3) 个体解释：真阳性 vs 假阳性
tp = df[(df['label'] == '1')].sort_values('prob', ascending=False).index[0]
fp = df[(df['label'] == '0')].sort_values('prob', ascending=False).index[0]
for tag, idx in [('TP', tp), ('FP', fp)]:
    nm = f"{df.loc[idx,'firm_code']}_{df.loc[idx,'year']}"
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(sv[idx], show=False, max_display=14)
    plt.title(f"{tag}: firm {df.loc[idx,'firm_code']} year {df.loc[idx,'year']}  prob={prob[idx]:.3f}", fontsize=10)
    plt.tight_layout(); plt.savefig(f'figs/SHAP_waterfall_{tag}_{nm}.png', dpi=150); plt.close()
    print(f'已出 figs/SHAP_waterfall_{tag}_{nm}.png')

# 4) 交互：financial_anomaly_max × TATA 依赖图
plt.figure(figsize=(7, 5))
shap.dependence_plot('financial_anomaly_max', sv.values, X, interaction_index='TATA', show=False)
plt.tight_layout(); plt.savefig('figs/SHAP_dependence_finAnomaly_TATA.png', dpi=150); plt.close()
print('已出 figs/SHAP_dependence_finAnomaly_TATA.png')

# 5) 全局重要性表（落盘供论文引用）
gi = pd.DataFrame({'feature': COLS,
                   'mean_abs_shap': np.abs(sv.values).mean(axis=0)}).sort_values('mean_abs_shap', ascending=False)
gi.to_csv('figs/SHAP_importance.csv', index=False, encoding='utf-8-sig')
print('\nSHAP 全局重要性 Top10:')
print(gi.head(10).to_string(index=False))

# 典型样本记录
df[['firm_code','firm_name','year','label','prob']].sort_values('prob', ascending=False).to_csv(
    'figs/SHAP样本预测分.csv', index=False, encoding='utf-8-sig')
print(f"\n真阳性示例: {df.loc[tp,'firm_name']} {df.loc[tp,'year']} prob={prob[tp]:.3f}")
print(f"假阳性示例: {df.loc[fp,'firm_name']} {df.loc[fp,'year']} prob={prob[fp]:.3f}")
