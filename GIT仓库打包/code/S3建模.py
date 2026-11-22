# -*- coding: utf-8 -*-
"""
S3 建模基线：逻辑回归 + 随机森林 + 交叉验证
============================================
输入：特征表.csv（100 样本 × 43 特征）
输出：控制台打印各特征组合的 AUC / F1 / 精确率 / 召回率

说明：
- 小样本（33 正例 + 67 反例）且不均衡，用 5 折分层重复交叉验证（10 次重复）求均值±标准差。
- 缺失值用中位数填充；逻辑回归前标准化。
- 对比多组特征：①原版 M_score 单一指标；②Beneish 8 指数重校准；③+衍生财务；④+监管/概念信号。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# ============ 1. 读特征 ============
df = pd.read_csv('data/特征表.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
y = df['label'].astype(int).values
for c in df.columns:
    if c not in ('firm_code', 'firm_name', 'industry', 'label_source'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ============ 2. 特征分组 ============
BENEISH = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
DERIVED = ['gross_margin', 'ni_margin', 'ar_rev', 'debt_ratio', 'cfo_rev',
           'capex_rev', 'oth_ar_ta', 'rev_growth', 'ta_growth', 'tl_growth']
SIGNALS = ['audit', 'inquiry', 'ne_share', 'ne_share_yoy', 'rev_rev_ratio', 'ind_growth']

FEATURE_SETS = {
    '①原版M_score(单指标)': ['M_score'],
    '②Beneish 8 重校准': BENEISH,
    '③+衍生财务': BENEISH + DERIVED,
    '④+监管概念信号(全量)': BENEISH + DERIVED + SIGNALS,
}

# ============ 3. 模型 ============
MODELS = {
    '逻辑回归': LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5),
    '随机森林': RandomForestClassifier(n_estimators=500, max_depth=4,
                                       min_samples_leaf=3, class_weight='balanced',
                                       random_state=42),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

def evaluate(X, y, model, name):
    """返回 (auc_mean, auc_std, f1, prec, rec)"""
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('clf', model),
    ])
    aucs = []
    for train, test in cv.split(X, y):
        X_tr = X.iloc[train] if hasattr(X, 'iloc') else X[train]
        pipe.fit(X.iloc[train], y[train])
        p = pipe.predict_proba(X.iloc[test])[:, 1]
        aucs.append(roc_auc_score(y[test], p))
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)

    # 袋外预测用于 F1/prec/rec（阈值 0.5）
    y_pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               method='predict')
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    return auc_mean, auc_std, f1, prec, rec

# ============ 4. 逐组评估 ============
print('=' * 78)
print(f'样本：{len(y)}（舞弊 {int((y==1).sum())} / 对照 {int((y==0).sum())}）；5 折分层 × 10 次重复交叉验证')
print('=' * 78)
for set_name, cols in FEATURE_SETS.items():
    print(f'\n【{set_name}】({len(cols)} 个特征)')
    X = df[cols]
    for m_name, model in MODELS.items():
        auc_m, auc_s, f1, prec, rec = evaluate(X, y, model, m_name)
        print(f'  {m_name}: AUC={auc_m:.3f}±{auc_s:.3f}  F1={f1:.3f}  精确率={prec:.3f}  召回率={rec:.3f}')

# ============ 5. 全量特征下随机森林的特征重要性 ============
print('\n' + '=' * 78)
print('特征重要性（全量特征 随机森林，Mean Decrease Impurity）')
print('=' * 78)
cols = FEATURE_SETS['④+监管概念信号(全量)']
X = df[cols]
pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                 ('clf', RandomForestClassifier(n_estimators=500, max_depth=4,
                                                min_samples_leaf=3, class_weight='balanced',
                                                random_state=42))])
pipe.fit(X, y)
imp = pd.Series(pipe.named_steps['clf'].feature_importances_, index=cols).sort_values(ascending=False)
for k, v in imp.items():
    print(f'  {k}: {v:.3f}')
