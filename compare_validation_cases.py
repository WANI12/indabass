import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
TARGET = 'food_insecurity_risk'

train = pd.read_csv('Train__4_.csv')

IPC_ORDER = ['Minimal', 'Stressed', 'Crisis', 'Emergency', 'Catastrophe']
ipc_map = {p: i for i, p in enumerate(IPC_ORDER)}


def engineer(df):
    df = df.copy()
    df['ipc_phase_ord'] = df['prior_period_ipc_phase'].map(ipc_map)
    df['month_sin'] = np.sin(2 * np.pi * df['start_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['start_month'] / 12)
    df['log_population'] = np.log1p(df['population'])
    df['cereal_gap_per_capita'] = df['prior_year_cereal_gap_tonnes'] / df['population'].replace(0, np.nan)
    df['cereal_prod_per_capita'] = df['prior_year_cereal_production_tonnes'] / df['population'].replace(0, np.nan)
    df['cereal_self_sufficiency'] = df['prior_year_cereal_production_tonnes'] / (
        df['prior_year_cereal_production_tonnes'] - df['prior_year_cereal_gap_tonnes']
    ).replace(0, np.nan)
    return df

train_fe = engineer(train).sort_values(['start_year', 'start_month']).reset_index(drop=True)
train_fe = train_fe.sort_values(['county', 'start_year', 'start_month'])
train_fe['county_hist_rate'] = (
    train_fe.groupby('county')[TARGET].apply(lambda s: s.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)
train_fe = train_fe.sort_values(['start_year', 'start_month'])
train_fe['county_hist_rate'] = train_fe['county_hist_rate'].fillna(train_fe[TARGET].expanding().mean().shift())
train_fe['county_hist_rate'] = train_fe['county_hist_rate'].fillna(train_fe[TARGET].mean())

train_fe['state_hist_rate'] = (
    train_fe.groupby('state')[TARGET].apply(lambda s: s.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)
train_fe['state_hist_rate'] = train_fe['state_hist_rate'].fillna(train_fe[TARGET].expanding().mean().shift())
train_fe['state_hist_rate'] = train_fe['state_hist_rate'].fillna(train_fe[TARGET].mean())

NUMERIC_FEATURES = [
    'log_population', 'start_year', 'month_sin', 'month_cos',
    'ipc_phase_ord', 'prior_period_phase3plus_pct',
    'prior_year_cereal_production_tonnes', 'prior_year_cereal_gap_tonnes',
    'cereal_gap_per_capita', 'cereal_prod_per_capita', 'cereal_self_sufficiency',
    'county_hist_rate', 'state_hist_rate',
]
CAT_FEATURES = ['state', 'county']
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_cat = enc.fit_transform(train_fe[CAT_FEATURES])
X = pd.concat([
    train_fe[NUMERIC_FEATURES].reset_index(drop=True),
    pd.DataFrame(train_cat, columns=CAT_FEATURES),
], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
y = train_fe[TARGET].reset_index(drop=True)


def evaluate_case(label, X_tr, X_val, y_tr, y_val):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    logreg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
    logreg.fit(X_tr_s, y_tr)
    pred_lr = logreg.predict_proba(X_val_s)[:, 1]

    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        class_weight='balanced',
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
    )
    hgb.fit(X_tr, y_tr)
    pred_hgb = hgb.predict_proba(X_val)[:, 1]
    pred_ens = 0.5 * pred_lr + 0.5 * pred_hgb

    print(f'Case {label}:')
    print(f'  LogReg AUC = {roc_auc_score(y_val, pred_lr):.4f}')
    print(f'  HistGB AUC = {roc_auc_score(y_val, pred_hgb):.4f}')
    print(f'  Ensemble AUC = {roc_auc_score(y_val, pred_ens):.4f}')

# Case A: latest 15% temporal holdout
periods = train_fe[['start_year', 'start_month']].drop_duplicates().sort_values(['start_year', 'start_month'])
val_periods = periods.tail(max(3, int(len(periods) * 0.15)))
val_mask = train_fe.set_index(['start_year', 'start_month']).index.isin(
    val_periods.set_index(['start_year', 'start_month']).index
)
val_mask = pd.Series(val_mask, index=train_fe.index).reset_index(drop=True)
X_tr, X_val = X[~val_mask], X[val_mask]
y_tr, y_val = y[~val_mask], y[val_mask]
evaluate_case('A', X_tr, X_val, y_tr, y_val)

# Case B: random 20% validation split
rng = np.random.RandomState(SEED)
idx = np.arange(len(X))
rng.shuffle(idx)
cut = int(len(X) * 0.8)
tr_idx, val_idx = idx[:cut], idx[cut:]
X_tr2, X_val2 = X.iloc[tr_idx], X.iloc[val_idx]
y_tr2, y_val2 = y.iloc[tr_idx], y.iloc[val_idx]
evaluate_case('B', X_tr2, X_val2, y_tr2, y_val2)
