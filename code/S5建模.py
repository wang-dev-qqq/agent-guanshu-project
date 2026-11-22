# -*- coding: utf-8 -*-
"""
S5 融合建模：XGBoost 消融 A/B/C/D/E
=====================================
读：财务特征12.csv + 语义评分_v2.csv + 特征表.csv(外部) + 核验v2_汇总.csv(核验)
出：results_summary.csv

五组：A 纯财务 / B 纯语义 / C 财务+语义 / D 全模型(财务+语义+核验+外部) / E 财务+语义+外部
评估：分层5折×20次重复(种子20261116-20261135)，AUC/AP/NDCG@5/NDCG@10，1000次标签置换
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

# ==================== 配置 ====================
N_PERM = 1000                     # 标签置换次数（先小值测速，定稿用1000）
SEEDS = list(range(20261116, 20261136))   # 20 个种子

FIN12 = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA',
         'cundun', 'oth_ar_ta', 'rev_indiv', 'capex_rev']
SEM3 = ['risk_concealment_max', 'financial_anomaly_max', 'future_promises_max']
VER2 = ['problem_rate', 'coverage']
EXT2 = ['audit', 'inquiry']


# ==================== 读数据 ====================
def read(csv, cols):
    return pd.read_csv(csv, encoding='utf-8-sig', dtype=str, keep_default_na=False)[cols]


fin = read('data/财务特征12.csv', ['firm_code', 'firm_name', 'year', 'industry', 'label'] + FIN12)
sem = read('data/语义评分_v2.csv', ['firm_code', 'year'] + SEM3)
ext = read('data/特征表.csv', ['firm_code', 'year', 'audit', 'inquiry'])

df = fin.merge(sem, on=['firm_code', 'year'], how='left')
df = df.merge(ext, on=['firm_code', 'year'], how='left')

HAS_VER = os.path.exists('核验v2_汇总.csv') and pd.read_csv('data/核验v2_汇总.csv', encoding='utf-8-sig').shape[0] > 0
if HAS_VER:
    ver = read('data/核验v2_汇总.csv', ['firm_code', 'year'] + VER2)
    df = df.merge(ver, on=['firm_code', 'year'], how='left')
    print('已并入核验特征')

y = df['label'].astype(int).values
for c in FIN12 + SEM3 + VER2 + EXT2:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

print(f'样本 {len(y)}（舞弊 {int((y == 1).sum())} / 对照 {int((y == 0).sum())}）')
print(f'各特征缺失数：')
print(df[FIN12 + SEM3 + EXT2].isna().sum()[lambda s: s > 0].to_string())

GROUPS = {
    'A_财务': FIN12,
    'B_语义': SEM3,
    'C_财务加语义': FIN12 + SEM3,
    'D_全模型': FIN12 + SEM3 + VER2 + EXT2,
    'E_财务加语义加外部': FIN12 + SEM3 + EXT2,
}
if not HAS_VER:
    GROUPS.pop('D_全模型')
    print('!! 未找到核验v2_汇总.csv，跳过 D_全模型')


# ==================== 评估工具 ====================
def ndcg_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    dcg = sum((2 ** y_true[order[i]] - 1) / np.log2(i + 2) for i in range(min(k, len(order))))
    ideal = np.argsort(-y_true)
    idcg = sum((2 ** y_true[ideal[i]] - 1) / np.log2(i + 2) for i in range(min(k, len(ideal))))
    return dcg / idcg if idcg > 0 else 0.0


def make_model(seed):
    return xgb.XGBClassifier(
        max_depth=2, n_estimators=150, learning_rate=0.08,
        min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2, scale_pos_weight=2.0,
        random_state=seed, tree_method='hist', n_jobs=1, verbosity=0)


def evaluate(X, y, cols):
    X = X[cols]
    aucs, aps, n5, n10 = [], [], [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            imp = SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(X.iloc[tr])
            Xte = imp.transform(X.iloc[te])
            m = make_model(seed)
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
            n5.append(ndcg_at_k(y[te], p, 5))
            n10.append(ndcg_at_k(y[te], p, 10))
    return (np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps),
            np.mean(n5), np.std(n5), np.mean(n10), np.std(n10))


def perm_test(X, y, cols, obs_auc):
    """标签置换检验，以 AUC 为统计量，单次5折"""
    X = X[cols]
    rng = np.random.RandomState(0)
    nulls = []
    for _ in range(N_PERM):
        y_shuf = rng.permutation(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261116)
        aucs = []
        for tr, te in skf.split(X, y_shuf):
            imp = SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(X.iloc[tr])
            Xte = imp.transform(X.iloc[te])
            m = make_model(20261116)
            m.fit(Xtr, y_shuf[tr])
            p = m.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y_shuf[te], p))
        nulls.append(np.mean(aucs))
    nulls = np.array(nulls)
    pval = (np.sum(nulls >= obs_auc) + 1) / (N_PERM + 1)
    return pval, nulls


# ==================== 逐组评估 ====================
rows = []
print('\n' + '=' * 70)
for gname, cols in GROUPS.items():
    auc, auc_s, ap, ap_s, n5, n5_s, n10, n10_s = evaluate(df, y, cols)
    pval = np.nan
    if N_PERM > 0:
        pval, _ = perm_test(df, y, cols, auc)
    print(f'【{gname}】({len(cols)}特征)  AUC={auc:.3f}±{auc_s:.3f}  '
          f'AP={ap:.3f}±{ap_s:.3f}  NDCG@5={n5:.3f}  NDCG@10={n10:.3f}  p={pval:.4f}')
    rows.append({
        'group': gname, 'n_features': len(cols),
        'auc_mean': round(auc, 4), 'auc_std': round(auc_s, 4),
        'ap_mean': round(ap, 4), 'ap_std': round(ap_s, 4),
        'ndcg5_mean': round(n5, 4), 'ndcg5_std': round(n5_s, 4),
        'ndcg10_mean': round(n10, 4), 'ndcg10_std': round(n10_s, 4),
        'perm_p_auc': round(pval, 4),
    })

out = pd.DataFrame(rows)
out.to_csv('results/results_summary.csv', index=False, encoding='utf-8-sig')
print('\n已输出 results_summary.csv')
print(out.to_string(index=False))
