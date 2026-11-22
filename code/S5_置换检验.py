# -*- coding: utf-8 -*-
"""S5 置换检验：1000次标签置换，以AUC为统计量，逐组保存可断点续跑"""
import os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
N_PERM = 1000
OUT = 'results/results_perm.csv'

FIN12 = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','LVGI','TATA','cundun','oth_ar_ta','rev_indiv','capex_rev']
SEM3 = ['risk_concealment_max','financial_anomaly_max','future_promises_max']
VER2 = ['problem_rate','coverage']
EXT2 = ['audit','inquiry']

fin = pd.read_csv('data/财务特征12.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
sem = pd.read_csv('data/语义评分_v2.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
ext = pd.read_csv('data/特征表.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)[['firm_code','year','audit','inquiry']]
ver = pd.read_csv('data/核验v2_汇总.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)[['firm_code','year']+VER2]
df = fin.merge(sem[['firm_code','year']+SEM3], on=['firm_code','year'], how='left') \
        .merge(ext, on=['firm_code','year'], how='left') \
        .merge(ver, on=['firm_code','year'], how='left')
for c in FIN12+SEM3+VER2+EXT2: df[c] = pd.to_numeric(df[c], errors='coerce')
y = df['label'].astype(int).values

GROUPS = {
    'A_财务': FIN12,
    'B_语义': SEM3,
    'C_财务加语义': FIN12+SEM3,
    'E_财务加语义加外部': FIN12+SEM3+EXT2,
    'D_全模型': FIN12+SEM3+VER2+EXT2,
}

def make(seed):
    return xgb.XGBClassifier(max_depth=2, n_estimators=150, learning_rate=0.08,
                             min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=2, scale_pos_weight=2.0, random_state=seed,
                             tree_method='hist', n_jobs=1, verbosity=0)

def cv_auc(cols, yv, seeds):
    aucs = []
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(df[cols], yv):
            imp = SimpleImputer(strategy='median')
            m = make(seed)
            m.fit(imp.fit_transform(df[cols].iloc[tr]), yv[tr])
            p = m.predict_proba(imp.transform(df[cols].iloc[te]))[:, 1]
            aucs.append(roc_auc_score(yv[te], p))
    return float(np.mean(aucs))

# 断点续跑
done = {}
if os.path.exists(OUT):
    old = pd.read_csv(OUT, dtype=str)
    done = {r['group']: r.to_dict() for _, r in old.iterrows()}

rows = list(done.values())
rng = np.random.RandomState(0)
for gname, cols in GROUPS.items():
    if gname in done:
        print(f'跳过已完成 {gname}')
        continue
    t0 = time.time()
    obs = cv_auc(cols, y, range(20261116, 20261136))
    nulls = []
    for i in range(N_PERM):
        y_s = rng.permutation(y)
        nulls.append(cv_auc(cols, y_s, [20261116]))
        if (i+1) % 200 == 0:
            print(f'  {gname}: {i+1}/{N_PERM}  ({time.time()-t0:.0f}s)', flush=True)
    nulls = np.array(nulls)
    pval = (np.sum(nulls >= obs) + 1) / (N_PERM + 1)
    rows.append({'group': gname, 'obs_auc': round(obs, 4),
                 'null_mean': round(float(nulls.mean()), 4),
                 'null_std': round(float(nulls.std()), 4),
                 'perm_p': round(float(pval), 4), 'n_perm': N_PERM})
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'【{gname}】 obs={obs:.3f}  null={nulls.mean():.3f}±{nulls.std():.3f}  p={pval:.4f}  ({time.time()-t0:.0f}s)', flush=True)

print('\n全部完成：')
print(pd.DataFrame(rows).to_string(index=False))
