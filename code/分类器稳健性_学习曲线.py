"""
分类器稳健性与学习曲线检验（对应论文 4.4 节）
================================================
1) 分类器稳健性：同一 19 维特征、同一评估协议（分层5折×20种子），
   将主模型 XGBoost 替换为逻辑回归，报告交叉验证 ROC-AUC；
2) 学习曲线：训练样本比例 40%/60%/80% 下的留出折 ROC-AUC，
   检验模型处于数据不足还是过拟合区间。

输入：data/ 下四个特征文件（与 S5 相同）
输出：控制台打印 + results/分类器稳健性_学习曲线.csv
运行：在仓库根目录执行  python code/分类器稳健性_学习曲线.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SEEDS = list(range(20261116, 20261136))   # 与 S5 相同的 20 个种子

FIN12 = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA',
         'cundun', 'oth_ar_ta', 'rev_indiv', 'capex_rev']
SEM3 = ['risk_concealment_max', 'financial_anomaly_max', 'future_promises_max']
VER2 = ['problem_rate', 'coverage']
EXT2 = ['audit', 'inquiry']
COLS = FIN12 + SEM3 + VER2 + EXT2


def read(csv, cols):
    return pd.read_csv(csv, encoding='utf-8-sig', dtype=str, keep_default_na=False)[cols]


fin = read('data/财务特征12.csv', ['firm_code', 'year', 'label'] + FIN12)
sem = read('data/语义评分_v2.csv', ['firm_code', 'year'] + SEM3)
ext = read('data/特征表.csv', ['firm_code', 'year'] + EXT2)
ver = read('data/核验v2_汇总.csv', ['firm_code', 'year'] + VER2)
df = fin.merge(sem, on=['firm_code', 'year']).merge(ext, on=['firm_code', 'year']).merge(ver, on=['firm_code', 'year'])
y = df['label'].astype(int).values
X = df[COLS].apply(pd.to_numeric, errors='coerce')

print(f'样本 {len(y)}（舞弊 {int((y == 1).sum())} / 对照 {int((y == 0).sum())}），特征 {len(COLS)} 维')

# ============ 1) 分类器稳健性：逻辑回归对照 ============
lr_aucs = []
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc', StandardScaler()),
                         ('lr', LogisticRegression(max_iter=2000, class_weight='balanced',
                                                   random_state=seed))])
        pipe.fit(X.iloc[tr], y[tr])
        lr_aucs.append(roc_auc_score(y[te], pipe.predict_proba(X.iloc[te])[:, 1]))
print(f'【逻辑回归】19维全模型 ROC-AUC = {np.mean(lr_aucs):.4f} ± {np.std(lr_aucs):.4f}（{len(lr_aucs)} 折）')

# ============ 2) 学习曲线 ============
def make_xgb(seed):
    return xgb.XGBClassifier(max_depth=2, n_estimators=150, learning_rate=0.08,
                             min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=2, scale_pos_weight=2.0,
                             random_state=seed, tree_method='hist', n_jobs=1, verbosity=0)


imp = SimpleImputer(strategy='median')
Xi = imp.fit_transform(X)
rows = [{'检验': '逻辑回归稳健性', '设置': '19维全模型,5折×20种子',
         'auc_mean': round(float(np.mean(lr_aucs)), 4), 'auc_std': round(float(np.std(lr_aucs)), 4)}]
for frac in [0.4, 0.6, 0.8]:
    aucs = []
    for seed in range(20):
        sss = StratifiedShuffleSplit(1, train_size=frac, random_state=seed)
        tr, te = next(sss.split(Xi, y))
        m = make_xgb(seed)
        m.fit(Xi[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xi[te])[:, 1]))
    print(f'【学习曲线】训练 {int(frac * 100)}%（{int(frac * len(y))} 样本）→ 留出 ROC-AUC '
          f'{np.mean(aucs):.3f} ± {np.std(aucs):.3f}')
    rows.append({'检验': '学习曲线', '设置': f'训练{int(frac * 100)}%({int(frac * len(y))}样本),20次划分',
                 'auc_mean': round(float(np.mean(aucs)), 4), 'auc_std': round(float(np.std(aucs)), 4)})

os.makedirs('results', exist_ok=True)
pd.DataFrame(rows).to_csv('results/分类器稳健性_学习曲线.csv', index=False, encoding='utf-8-sig')
print('\n已输出 results/分类器稳健性_学习曲线.csv')
