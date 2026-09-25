import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

SEED = 42
np.random.seed(SEED)

train = pd.read_csv("Train__4_.csv")
test = pd.read_csv("Test__1_.csv")
ss = pd.read_csv("SampleSubmission__11_.csv")

TARGET = "food_insecurity_risk"
ID_COL = "ID"

IPC_ORDER = ["Minimal", "Stressed", "Crisis", "Emergency", "Catastrophe"]
ipc_map = {p: i for i, p in enumerate(IPC_ORDER)}

def engineer(df):
    df = df.copy()
    df["ipc_phase_ord"] = df["prior_period_ipc_phase"].map(ipc_map)
    df["month_sin"] = np.sin(2 * np.pi * df["start_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["start_month"] / 12)
    df["log_population"] = np.log1p(df["population"])
    df["cereal_gap_per_capita"] = df["prior_year_cereal_gap_tonnes"] / df["population"].replace(0, np.nan)
    df["cereal_prod_per_capita"] = df["prior_year_cereal_production_tonnes"] / df["population"].replace(0, np.nan)
    df["cereal_self_sufficiency"] = df["prior_year_cereal_production_tonnes"] / (
        df["prior_year_cereal_production_tonnes"] - df["prior_year_cereal_gap_tonnes"]
    ).replace(0, np.nan)
    return df

train_fe = engineer(train).sort_values(["start_year", "start_month"]).reset_index(drop=True)
test_fe = engineer(test)

 
train_fe = train_fe.sort_values(["county", "start_year", "start_month"])
train_fe["county_hist_rate"] = (
    train_fe.groupby("county")[TARGET].apply(lambda s: s.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)
overall_expanding = train_fe.sort_values(["start_year", "start_month"])[TARGET].expanding().mean().shift()
train_fe = train_fe.sort_values(["start_year", "start_month"])
train_fe["county_hist_rate"] = train_fe["county_hist_rate"].fillna(train_fe[TARGET].expanding().mean().shift())
train_fe["county_hist_rate"] = train_fe["county_hist_rate"].fillna(train_fe[TARGET].mean())

county_rate_map = train_fe.groupby("county")[TARGET].mean()
state_rate_map = train_fe.groupby("state")[TARGET].mean()
global_rate = train_fe[TARGET].mean()
test_fe["county_hist_rate"] = test_fe["county"].map(county_rate_map)
test_fe["county_hist_rate"] = test_fe["county_hist_rate"].fillna(test_fe["state"].map(state_rate_map))
test_fe["county_hist_rate"] = test_fe["county_hist_rate"].fillna(global_rate)

 
train_fe["state_hist_rate"] = (
    train_fe.groupby("state")[TARGET].apply(lambda s: s.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)
train_fe["state_hist_rate"] = train_fe["state_hist_rate"].fillna(train_fe[TARGET].expanding().mean().shift())
train_fe["state_hist_rate"] = train_fe["state_hist_rate"].fillna(train_fe[TARGET].mean())
test_fe["state_hist_rate"] = test_fe["state"].map(state_rate_map).fillna(global_rate)

NUMERIC_FEATURES = [
    "log_population", "start_year", "month_sin", "month_cos",
    "ipc_phase_ord", "prior_period_phase3plus_pct",
    "prior_year_cereal_production_tonnes", "prior_year_cereal_gap_tonnes",
    "cereal_gap_per_capita", "cereal_prod_per_capita", "cereal_self_sufficiency",
    "county_hist_rate", "state_hist_rate",
]
CAT_FEATURES = ["state", "county"]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_cat = enc.fit_transform(train_fe[CAT_FEATURES])
test_cat = enc.transform(test_fe[CAT_FEATURES])

X = pd.concat([
    train_fe[NUMERIC_FEATURES].reset_index(drop=True),
    pd.DataFrame(train_cat, columns=CAT_FEATURES),
], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
y = train_fe[TARGET].reset_index(drop=True)

X_test = pd.concat([
    test_fe[NUMERIC_FEATURES].reset_index(drop=True),
    pd.DataFrame(test_cat, columns=CAT_FEATURES),
], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)

 
periods = train_fe[["start_year", "start_month"]].drop_duplicates().sort_values(["start_year", "start_month"])
n_val_periods = max(3, int(len(periods) * 0.15))
val_periods = periods.tail(n_val_periods)
val_mask = train_fe.set_index(["start_year", "start_month"]).index.isin(
    val_periods.set_index(["start_year", "start_month"]).index
)
val_mask = pd.Series(val_mask, index=train_fe.index).reset_index(drop=True)

X_tr, X_val = X[~val_mask], X[val_mask]
y_tr, y_val = y[~val_mask], y[val_mask]
print(f"Train rows: {len(X_tr)}, Val rows: {len(X_val)}, Val periods: {len(val_periods)}")
print(f"Val period range: {val_periods.iloc[0].to_dict()} -> {val_periods.iloc[-1].to_dict()}")

 
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)
logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
logreg.fit(X_tr_s, y_tr)
val_pred_lr = logreg.predict_proba(X_val_s)[:, 1]
print("LogReg val AUC:", round(roc_auc_score(y_val, val_pred_lr), 4))

 
hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.15,
)
hgb.fit(X_tr, y_tr)
val_pred_hgb = hgb.predict_proba(X_val)[:, 1]
print("HistGB val AUC:", round(roc_auc_score(y_val, val_pred_hgb), 4))

 
val_pred_ens = 0.5 * val_pred_lr + 0.5 * val_pred_hgb
print("Ensemble val AUC:", round(roc_auc_score(y_val, val_pred_ens), 4))

 
scores = {
    "logreg": roc_auc_score(y_val, val_pred_lr),
    "hgb": roc_auc_score(y_val, val_pred_hgb),
    "ensemble": roc_auc_score(y_val, val_pred_ens),
}
best_name = max(scores, key=scores.get)
print("Best model:", best_name, scores)

 
best_val_pred = {"logreg": val_pred_lr, "hgb": val_pred_hgb, "ensemble": val_pred_ens}[best_name]
best_thr, best_f1 = 0.5, -1
for thr in np.arange(0.1, 0.91, 0.02):
    f1 = f1_score(y_val, (best_val_pred >= thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
preds_at_thr = (best_val_pred >= best_thr).astype(int)
print(f"Best threshold: {best_thr:.2f} | F1: {best_f1:.4f} | "
      f"Precision: {precision_score(y_val, preds_at_thr):.4f} | "
      f"Recall: {recall_score(y_val, preds_at_thr):.4f}")

 
scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X)
logreg_full = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
logreg_full.fit(X_full_s, y)
test_pred_lr = logreg_full.predict_proba(scaler_full.transform(X_test))[:, 1]

hgb_full = HistGradientBoostingClassifier(
    max_iter=hgb.n_iter_, learning_rate=0.05, max_depth=4,
    l2_regularization=1.0, class_weight="balanced", random_state=SEED,
)
hgb_full.fit(X, y)
test_pred_hgb = hgb_full.predict_proba(X_test)[:, 1]

test_pred_ens = 0.5 * test_pred_lr + 0.5 * test_pred_hgb
final_test_pred = {"logreg": test_pred_lr, "hgb": test_pred_hgb, "ensemble": test_pred_ens}[best_name]

submission = ss.copy()
submission[TARGET] = final_test_pred
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv")

 
try:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(hgb, X_val, y_val, n_repeats=10, random_state=SEED, scoring="roc_auc")
    importances = pd.Series(r.importances_mean, index=X_val.columns).sort_values(ascending=False)
    print("\nPermutation importance (val AUC drop):")
    print(importances)
except Exception as e:
    print("Importance calc skipped:", e)
