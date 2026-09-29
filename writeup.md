# South Sudan Food Insecurity Forecasting — Model Write-Up

## Features used and reasoning

All features were used as provided, since the challenge rules require meaningful use of every relevant field. From the raw columns I derived:

- **`ipc_phase_ord`** — `prior_period_ipc_phase` encoded on its natural Minimal→Catastrophe order (0–4) rather than one-hot, since the phases are ordinal and a linear encoding lets the model treat "further along the scale" as meaningfully bigger.
- **`month_sin` / `month_cos`** — cyclical encoding of `start_month`, since food security in South Sudan follows a seasonal (lean-season vs. harvest) pattern that a raw month number wouldn't capture correctly at the December→January boundary.
- **`log_population`** — log-transformed `population` to reduce skew from a few very large counties (e.g. Juba).
- **`cereal_gap_per_capita` / `cereal_prod_per_capita`** — cereal figures normalized by population, since a fixed tonnage gap means something very different in a small versus large county.
- **`cereal_self_sufficiency`** — production as a share of total requirement (production ÷ (production − gap)), a more directly interpretable ratio than the raw tonnages.
- **`county_hist_rate` / `state_hist_rate`** — each county's and state's historical Crisis+ rate, computed only from *strictly prior* periods (`.shift().expanding().mean()`), giving the model a memory of chronically food-insecure areas without seeing the outcome it's predicting.
- `state` and `county` were kept (ordinal-encoded) as identifiers rather than dropped, since geography carries information beyond what the engineered features summarize.

No feature was excluded — the two cereal columns individually add only marginal signal (see below) but were retained per the challenge's instruction to make meaningful use of all provided fields, and they do modestly sharpen the tail predictions for counties with severe deficits.

## Validation approach

Because this is a forecasting task, I used a **chronological split**, not random k-fold: the 5 most recent distinct assessment periods in the training set (Dec 2023 → Apr 2025) were held out as validation, with all earlier periods used for training (2,706 train rows / 393 validation rows). This mirrors the actual train/test relationship, where test covers Sep 2025–Jul 2026, entirely after the training window. The county- and state-history features were computed with a strict "no future leakage" rule so validation AUC reflects genuine forecasting performance rather than the model seeing its own answer.

Two models were compared — Logistic Regression (class-balanced) and HistGradientBoostingClassifier — with a simple average ensemble as a third option. Validation AUC: Logistic Regression 0.980, HistGB 0.983, Ensemble 0.984. The ensemble was selected for the final submission and refit on the full training set before predicting on test.

## Decision threshold, F1, precision, recall

Scanning thresholds from 0.10 to 0.90 on the validation set, **0.58** maximized F1. At that threshold: **F1 = 0.982, precision = 0.983, recall = 0.981**. Precision and recall are close together, which suggests the model isn't systematically over- or under-flagging counties at this cutoff — a reasonable default before a responder overlays their own risk tolerance.

## Practical meaning for a humanitarian responder

The model is an early-warning signal, not a replacement for IPC analysis. In practice, it would surface counties whose predicted risk has risen since their last formal IPC classification useful for prioritizing which counties merit closer field assessment between the periodic IPC cycles. Feature importance shows `prior_period_phase3plus_pct` (the county's own last known severity) is by far the dominant driver, meaning the model is mostly extrapolating recent trajectory rather than discovering new risk from cereal or population data alone. This is a meaningful limitation: the model will be less useful for detecting a *sudden* shock (e.g. new conflict or flooding) in a county that was previously stable, since it has little independent signal beyond the last known trend and seasonal pattern. Responders should treat high-confidence predictions as a prompt to verify with current field information, not as a standalone classification — particularly for counties near the 0.58 threshold, where the model's own precision/recall trade-off shows it is least certain.
