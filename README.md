# Insurance Claims Forecast & Chat Assistant

This workspace contains:
- Data engineering notebook (`EDA_feature_eng.ipynb`) that builds features and exports `EDA_output.csv`.
- Modeling/utility module (`GenAI_backend.py`) refactored for clean import using a `DataContext` dataclass.
- Chat + tool invocation notebook (`GenAI_API.ipynb`) that connects to OpenAI and exposes forecasting, trend, and policy simulation tools to an LLM-driven assistant.

## 1. Environment Setup (Windows PowerShell)
Install Python dependencies (choose one method):
```pwsh
# Using pip
pip install pandas numpy scikit-learn xgboost torch statsmodels openai torchmetrics matplotlib seaborn

# Or using uv (recommended for speed)
uv pip install pandas numpy scikit-learn xgboost torch statsmodels openai torchmetrics matplotlib seaborn
```

Set your OpenAI API key (NEVER hard‑code secrets in notebooks or source files):
```pwsh
# Temporary for current session
$env:OPENAI_API_KEY = "sk-your-rotated-key"

# Persist for future sessions (requires new shell after)
setx OPENAI_API_KEY "sk-your-rotated-key"
```
Conda (persistent inside one environment):
```bash
conda activate your_env_name
# Correct syntax (adds to env metadata)
conda env config config vars set OPENAI_API_KEY=sk-your-rotated-key
# Reactivate for changes to take effect
conda deactivate && conda activate your_env_name
# Verify
conda env config vars list
```
Rotate any previously exposed key immediately in the OpenAI dashboard.

## 2. Data Preparation
Source raw data: `data.csv` (original NHI dataset). Run the `EDA_feature_eng.ipynb` notebook to:
- Clean year column and convert to Gregorian year.
- Derive per-person costs, per-claim costs, shares, lags, and growth rates.
- Optionally export the enriched dataset to `EDA_output.csv` (uncomment the export cell).

## 3. Backend Module (`GenAI_backend.py`)
Public API (also listed in `__all__`):
- `load_data(path) -> DataContext`
- `forecast_model_selec(context, model_name)`
- `forecast_claims(context, trained_model, ...)`
- `summarize_trends(context, window)`
- `simulate_policy_change(context, trained_model, shocks, ...)`
- Training helpers: `train_torch_mlp`, `rolling_origin_eval_multi`, `rolling_origin_eval_torch`, `rolling_origin_arima`
- Model classes: `MLPRegressor`, `MLPWrapper`

All functions now require an explicit `DataContext` (no hidden globals). Create one with:
```python
from GenAI_backend import load_data
context = load_data("EDA_output.csv")
```

### Modeling Notes
- Rolling-origin evaluation (expanding window) is used for all model types.
- Supported models: Random Forest (rf), XGBoost (xgboost), PyTorch MLP (mlp). ARIMA utilities are present but not fully wired to multi-target serving.
- `forecast_model_selec` returns `(trained_model, metrics_dict)` after evaluation; metrics are MAE/RMSE per target.
- `simulate_policy_change` applies multiplicative shocks to features (e.g. `{"insured_persons": 1.05}`) and reports absolute & percentage impact on forecasted claim amounts.

## 4. Chat Notebook (`GenAI_API.ipynb`)
Workflow inside the notebook:
1. Import and create `context` via `load_data('EDA_output.csv')`.
2. Initialize OpenAI client from environment variable.
3. Train & register models by running `init_models()` (calls `forecast_model_selec` for each model name).
4. Use `chat_step(user_text, messages)` to interact. The assistant may auto-call tools:
   - `forecast_claims`: next year forecast.
   - `summarize_trends`: multi-year trend stats.
   - `simulate_policy_change`: scenario analysis.

Example snippet outside the chat flow:
```python
from GenAI_backend import load_data, forecast_model_selec, forecast_claims
context = load_data('EDA_output.csv')
model, metrics = forecast_model_selec(context, 'xgboost')
print(metrics)
print(forecast_claims(context, trained_model=model))
```

## 5. Policy Simulation Example
```python
shocks = {"insured_persons": 1.02, "avg_cost_per_claim": 0.97}
model, _ = forecast_model_selec(context, 'xgboost')
scenario = simulate_policy_change(context=context, trained_model=model, shocks=shocks, model_name='xgboost')
print(scenario["impact"])  # per-target changes
```

## 6. Common Issues & Fixes
| Issue | Cause | Fix |
|-------|-------|-----|
| `OPENAI_API_KEY not found` | Env var missing | Set via PowerShell then restart shell. |
| `TypeError: OpenAI() takes no positional arguments` | Positional API key passed | Use `OpenAI(api_key=...)` or rely on env var. |
| `ValueError: No trained model registered` | Forgot `init_models()` | Run the initialization cell before chat. |
| CUDA not used | GPU unavailable | Falls back to CPU automatically; no action needed. |

## 7. Extending
- Add new features: modify EDA notebook and regenerate `EDA_output.csv`, then re-run model training.
- Plug additional models: wrap in `MultiOutputRegressor` if scikit-learn; follow existing pattern.
- Integrate ARIMA fully: adapt `forecast_model_selec` to return a dict of per-target ARIMA results.

## 8. Project Structure
```
Cathay Life Insurance/
├─ data.csv
├─ EDA_feature_eng.ipynb
├─ EDA_output.csv (generated)
├─ GenAI_backend.py
├─ GenAI_API.ipynb
└─ README.md
```
