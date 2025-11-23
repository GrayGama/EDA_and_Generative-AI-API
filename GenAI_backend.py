import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import statsmodels.api as sm
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

__all__ = [
    'DataContext',
    'load_data',
    'forecast_model_selec',
    'forecast_claims',
    'summarize_trends',
    'simulate_policy_change',
    'train_torch_mlp',
    'rolling_origin_eval_multi',
    'rolling_origin_eval_torch',
    'rolling_origin_arima',
    'MLPRegressor',
    'MLPWrapper',
]

# Device selection kept minimal to avoid side effects on import.
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Data context container to avoid reliance on implicit globals.
# -----------------------------------------------------------------------------
@dataclass
class DataContext:
    supervised: pd.DataFrame
    feature_cols: List[str]
    target_cols: List[str]
    X: np.ndarray
    y: np.ndarray
    years: np.ndarray

def load_data(file_name: str) -> DataContext:
    """Load CSV and build supervised learning arrays.

    Returns a DataContext with all necessary objects for downstream modeling.
    """
    df = pd.read_csv(file_name)

    feature_cols = [
        # base level
        "insured_persons",
        "inpatient_share_amount",
        "outpatient_share_amount",
        # Avg cost per claim
        "total_cost_per_person",
        "impatient_cost_per_person",
        "outpatient_cost_per_person",
        "avg_cost_per_claim",
        "avg_impatient_cost_per_claim",
        "avg_outpatient_cost_per_claim",
        # lags
        "claims_total_amt_m_lag1",
        "insured_persons_lag1",
        # growths
        "gr_claims_total_amt_m",
        "gr_insured_persons",
    ]

    target_cols = [
        "claims_total_amt_m",
        "claims_inpatient_amt_m",
        "claims_outpatient_amt_m",
    ]

    supervised = df[["year"] + feature_cols + target_cols].dropna().reset_index(drop=True)
    X = supervised[feature_cols].values
    y = supervised[target_cols].values
    years = supervised["year"].values

    return DataContext(
        supervised=supervised,
        feature_cols=feature_cols,
        target_cols=target_cols,
        X=X,
        y=y,
        years=years,
    )

def Calculate_MAE_RMSE(target_cols: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Tuple[float, float]]:
    results: Dict[str, Tuple[float, float]] = {}
    for j, name in enumerate(target_cols):
        mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
        rmse = np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
        print(f"{name:25s}  MAE = {mae:10.2f},  RMSE = {rmse:10.2f}")
        results[name] = (mae, rmse)
    return results

def rolling_origin_eval_multi(X: np.ndarray, y: np.ndarray, years: np.ndarray, model, min_train_size: int = 8):
    """"
    X: (N, d)
    y: (N, T) multi-output targets
    years: (N,)
    model: regressor supporting multi-output (or wrapped model)
    """
    preds = []
    actuals = []
    pred_years = []
    
    n_samples = X.shape[0]
    
    for split in range(min_train_size, n_samples):
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split].reshape(1, -1), y[split]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        
        preds.append(y_pred)
        actuals.append(y_test)
        pred_years.append(years[split])
    
    return model, np.array(preds), np.array(actuals), np.array(pred_years)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
### ---- For Pytorch training loop  ------- 

def train_torch_mlp(X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 200, lr: float = 1e-2, batch_size: Optional[int] = None):
    """
    Train an MLPRegressor on X_train, y_train (numpy arrays).
    Returns: trained model, scaler_X, scaler_y
    """
    import torchmetrics
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    
    if batch_size is None or batch_size > X_tensor.shape[0]:
        batch_size = X_tensor.shape[0]
        
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=True)
    
    mlp_model = MLPRegressor(input_dim=X_train.shape[1], output_dim=y_train.shape[1]).to(device)
    
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(mlp_model.parameters(),
                            lr=lr)
    # Initialize torchmetrics
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    mse_metric = torchmetrics.MeanSquaredError().to(device)
    
    
    mlp_model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        mae_metric.reset()
        mse_metric.reset()
        
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = mlp_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            
            # Update metrics
            mae_metric.update(pred, yb)
            mse_metric.update(pred, yb)
        
        # Calculate average loss for the epoch
        epoch_loss /= X_tensor.shape[0]
        
        # if epoch % 10 == 0:
        #     mae = mae_metric.compute()
        #     rmse = torch.sqrt(mse_metric.compute())
        #     print(f"Epoch {epoch:3d}/{n_epochs} - Loss: {epoch_loss:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    
    return mlp_model, scaler_X, scaler_y
    
class MLPWrapper:
    def __init__(self, model: nn.Module, scaler_X: StandardScaler, scaler_y: StandardScaler):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.inference_mode():
            y_scaled = self.model(X_tensor).cpu().numpy()
        return self.scaler_y.inverse_transform(y_scaled)

def rolling_origin_eval_torch(X: np.ndarray, y: np.ndarray, years: np.ndarray, min_train_size: int = 8, n_epochs: int = 200, lr: float = 1e-2, batch_size: Optional[int] = None):
    """"
    X: (N, d)
    y: (N, T) multi-output targets
    years: (N,)
    model: regressor supporting multi-output (or wrapped model)
    """
    preds = []
    actuals = []
    pred_years = []
    
    n_samples = X.shape[0]
    
    print("Starting Multi-Layer Perceptron training...")
    for split in range(min_train_size, n_samples):
        X_train, y_train = X[:split], y[:split]
        
        mlp_model, scaler_X, scaler_y = train_torch_mlp(X_train, y_train, 
                                                        n_epochs=n_epochs, 
                                                        lr=lr, 
                                                        batch_size=batch_size)
        
        X_test = X[split].reshape(1, -1)
        X_test_scaled = scaler_X.transform(X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test = y[split]
        
        mlp_model.eval()
        with torch.inference_mode():
            y_pred_scaled = mlp_model(X_test_tensor).cpu().numpy()
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    
        preds.append(y_pred)
        actuals.append(y_test)
        pred_years.append(years[split])
        
        # keep last model + scalers
        final_model = mlp_model
        final_scaler_X = scaler_X
        final_scaler_y = scaler_y
        
        mlp_wrapper = MLPWrapper(final_model, final_scaler_X, final_scaler_y)
    
    return mlp_wrapper, np.array(preds), np.array(actuals), np.array(pred_years)

def rolling_origin_arima(y_series: np.ndarray, years: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1), min_train_size: int = 8):
    preds, actuals, pred_years = [], [], []
    n = len(y_series)

    for split in range(min_train_size, n):

        # Train ARIMA on y[:split]
        model = sm.tsa.ARIMA(y_series[:split], order=order)
        result = model.fit()

        # Predict next year (t+1)
        pred = result.forecast(steps=1)[0]
        true = y_series[split]

        preds.append(pred)
        actuals.append(true)
        pred_years.append(years[split])

    return np.array(preds), np.array(actuals), np.array(pred_years)

def calculate_mae_rmse_single(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:25s}  MAE = {mae:10.2f},  RMSE = {rmse:10.2f}")
    return mae, rmse

def forecast_model_selec(context: DataContext, model_name: str = "xgboost"):
    """
    Select the model and run rolling-origin evaluation, print MAE/RMSE for selected targets.

    Returns
    -------
    trained_model : fitted model or None
        - 'rf' / 'xgboost' : fitted multi-output model
        - 'mlp'           : fitted torch model (as returned by rolling_origin_eval_torch)
        - 'arima'         : None for now (ARIMA serving not wired yet)
    results : dict or DataFrame
        MAE/RMSE per target.
    """
    X = context.X
    y = context.y
    years = context.years
    target_cols = context.target_cols

    if model_name == "rf":
        rf_base = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            # max_features='auto',
            random_state=42, 
        )
        trained_model, preds, actuals, pred_years = rolling_origin_eval_multi(X, y, years, rf_base, min_train_size=8)
    elif model_name == "xgboost":
        xgb_base = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
        )
        base_model = MultiOutputRegressor(xgb_base)  # MultiOutputRegressor(XGBRegressor) 
        trained_model, preds, actuals, pred_years = rolling_origin_eval_multi(X, y, years, base_model, min_train_size=8)
    elif model_name == "mlp":
        # You may prefer to re-train or store a fitted MLP
        trained_model, preds, actuals, pred_years = rolling_origin_eval_torch(X, y, years)
    elif model_name == "arima":
        # ARIMA-only forecast for total, inpatient, outpatient
        preds, actuals, years = rolling_origin_arima(
            y[:, 0], years, order=(1,1,1)
        )
        mae, rmse = calculate_mae_rmse_single("claims_total_amt_m", actuals, preds)
        results_a = {"claims_total_amt_m": (mae, rmse)}
        return None, results_a
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    results = Calculate_MAE_RMSE(target_cols, actuals, preds)
    
    return trained_model, results

def forecast_claims(context: DataContext, horizon: int = 1, model_name: Optional[str] = None, trained_model: Any = None):
    """
    Use a trained model to forecast claims_total_amt_m, claims_inpatient_amt_m,
    claims_outpatient_amt_m for the next `horizon` years from the last available year.

    For now: horizon=1 only.

    Parameters
    ----------
    horizon : int
        Number of years ahead to forecast (currently must be 1).
    model_name : str or None
        Optional explicit model name: 'xgboost', 'rf', 'mlp', 'arima'.
        If None, it will be inferred from `trained_model`.
    trained_model :
        - 'xgboost'/'rf'/'multioutput': fitted sklearn regressor with .predict(X)
        - 'mlp'                     : MLPWrapper instance exposing .predict(X)
        - 'arima'                   : dict {target_name: fitted ARIMA result object}
    """

    # --- small helper: infer model_name from the trained_model type ---
    def _infer_model_name(m):
        # ARIMA: dict of results objects
        if isinstance(m, dict):
            return "arima"

        # Try to detect sklearn / xgboost types
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            RandomForestRegressor = type("RFPlaceholder", (), {})
            MultiOutputRegressor = type("MOPPlaceholder", (), {})

        try:
            from xgboost import XGBRegressor
        except ImportError:
            XGBRegressor = type("XGBPlaceholder", (), {})

        # RF
        if isinstance(m, RandomForestRegressor):
            return "rf"

        # MultiOutputRegressor(XGBRegressor)
        if isinstance(m, MultiOutputRegressor):
            if isinstance(m.estimator, XGBRegressor):
                return "xgboost"
            # generic multi-output – you can extend this if you add more models
            return "multioutput"

        # (Optional) detect MLP by torch.nn.Module
        try:
            import torch.nn as nn
            if isinstance(m, nn.Module):
                return "mlp"
        except ImportError:
            pass

        return None

    # ----------------- basic checks -----------------
    supervised = context.supervised
    feature_cols = context.feature_cols
    target_cols = context.target_cols
    last_row = supervised.iloc[-1]
    last_year = int(last_row["year"])

    if horizon != 1:
        raise NotImplementedError("forecast_claims currently supports horizon=1 only.")

    if trained_model is None:
        raise ValueError(
            "forecast_claims expected a pre-trained model. "
            "Call forecast_model_selec() first and pass its trained_model here."
        )

    # auto-infer model_name if not provided
    if model_name is None:
        model_name = _infer_model_name(trained_model)
        if model_name is None:
            raise ValueError(
                "Could not infer model_name from trained_model. "
                "Please pass model_name explicitly (e.g. 'xgboost', 'rf', 'arima')."
            )

    # ---------- ARIMA branch ----------
    if model_name == "arima":
        # trained_model is expected to be dict: {col_name: ARIMAResults-like}
        forecasts = {}
        for col in target_cols:
            arima_result = trained_model[col]   # ARIMAResults
            pred = float(arima_result.forecast(steps=1)[-1])
            forecasts[col] = pred

        return {
            "base_year": last_year,
            "forecast_year": last_year + 1,
            "model_name": "arima",
            "forecasts": forecasts,
        }

    # ---------- ML multi-output branch ----------
    X_last = last_row[feature_cols].values.reshape(1, -1)

    if model_name in ["xgboost", "rf", "multioutput", "mlp"]:
        # MultiOutputRegressor(XGBRegressor) or RandomForestRegressor
        y_pred = trained_model.predict(X_last)[0]  # shape: (n_targets,)

    else:
        raise ValueError(f"Unknown or unsupported model_name: {model_name}")

    forecasts = {name: float(val) for name, val in zip(target_cols, y_pred)}

    return {
        "base_year": last_year,
        "forecast_year": last_year + 1,
        "model_name": model_name,
        "forecasts": forecasts,
    }

def summarize_trends(context: DataContext, window: int = 5):
    """
    Compute numeric trend summaries over the last `window` years for each target.

    Returns
    -------
    summary_dict : dict
        {
          target_name: {
            'start_year': int,
            'end_year': int,
            'start_value': float,
            'end_value': float,
            'abs_change': float,
            'pct_change': float,   # in %
            'cagr': float,         # in % per year
            'mean': float,
            'std': float,
          },
          ...
        }
    summary_df : pd.DataFrame
        Tabular version of the same information.
    """
    # Aggregate to yearly sums (or use .mean() if that makes more sense for you)
    supervised = context.supervised
    target_cols = context.target_cols
    yearly = supervised.groupby("year")[target_cols].sum().sort_index()

    if yearly.shape[0] < 2:
        raise ValueError("Not enough years of data to summarize trends.")

    # restrict to last `window` years if available
    if yearly.shape[0] > window:
        yearly_window = yearly.iloc[-window:]
    else:
        yearly_window = yearly

    years_idx = yearly_window.index.to_numpy()
    start_year = int(years_idx[0])
    end_year = int(years_idx[-1])
    n_periods = len(years_idx) - 1  # for CAGR

    summary_rows = []

    for col in target_cols:
        series = yearly_window[col].astype(float)
        start_val = float(series.iloc[0])
        end_val = float(series.iloc[-1])
        abs_change = end_val - start_val

        if start_val != 0:
            pct_change = (abs_change / start_val) * 100.0
        else:
            pct_change = float("nan")

        # CAGR: (end/start)^(1/years) - 1
        if start_val > 0 and n_periods > 0:
            cagr = ((end_val / start_val) ** (1.0 / n_periods) - 1.0) * 100.0
        else:
            cagr = float("nan")

        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1)) if len(series) > 1 else float("nan")

        row = {
            "target": col,
            "start_year": start_year,
            "end_year": end_year,
            "start_value": start_val,
            "end_value": end_val,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "cagr": cagr,
            "mean": mean_val,
            "std": std_val,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("target")
    summary_dict = {row["target"]: {k: v for k, v in row.items() if k != "target"}
                    for row in summary_rows}

    return summary_dict, summary_df

def simulate_policy_change(context: DataContext, trained_model: Any, model_name: Optional[str] = None, shocks: Optional[Dict[str, float]] = None, horizon: int = 1):
    """
    Apply multiplicative shocks to selected features and compute impact on forecasted claims.

    Parameters
    ----------
    trained_model :
        Fitted model with .predict(X) interface (RF/XGB/MultiOutputRegressor/MLPWrapper).
    model_name : str or None
        'xgboost', 'rf', 'mlp', 'multioutput', 'arima', or None to infer.
    shocks : dict or None
        Mapping feature_name -> multiplicative factor.
        Example: {'insured_persons': 1.10, 'avg_cost_per_person': 0.95}
        If None or empty dict, no changes applied.
    horizon : int
        Currently must be 1 (same limitation as forecast_claims).

    Returns
    -------
    result : dict
        {
          'base_year': int,
          'forecast_year': int,
          'model_name': str,
          'baseline_forecasts': {target: float},
          'scenario_forecasts': {target: float},
          'impact': {
             target: {
                'abs_change': float,
                'pct_change': float
             },
             ...
          },
          'applied_shocks': shocks_dict actually used,
        }
    """

    if shocks is None:
        shocks = {}

    # --- helper: reuse the same infer logic used in forecast_claims ---
    def _infer_model_name(m):
        if isinstance(m, dict):
            return "arima"

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            RandomForestRegressor = type("RFPlaceholder", (), {})
            MultiOutputRegressor = type("MOPPlaceholder", (), {})

        try:
            from xgboost import XGBRegressor
        except ImportError:
            XGBRegressor = type("XGBPlaceholder", (), {})

        if isinstance(m, RandomForestRegressor):
            return "rf"

        if isinstance(m, MultiOutputRegressor):
            if isinstance(m.estimator, XGBRegressor):
                return "xgboost"
            return "multioutput"

        # MLPWrapper
        try:
            if isinstance(m, MLPWrapper):
                return "mlp"
        except NameError:
            pass

        try:
            import torch.nn as nn
            if isinstance(m, nn.Module):
                return "mlp"
        except ImportError:
            pass

        return None

    if trained_model is None:
        raise ValueError("simulate_policy_change requires a trained_model.")

    if horizon != 1:
        raise NotImplementedError("simulate_policy_change currently supports horizon=1 only.")

    # infer model_name if needed
    if model_name is None:
        model_name = _infer_model_name(trained_model)
        if model_name is None:
            raise ValueError(
                "Could not infer model_name from trained_model. "
                "Please pass model_name explicitly (e.g. 'xgboost', 'rf', 'mlp')."
            )

    if model_name == "arima":
        raise NotImplementedError(
            "simulate_policy_change is not implemented for ARIMA-only models, "
            "since it relies on feature shocks."
        )

    # ---------- baseline forecast using existing tool ----------
    baseline = forecast_claims(
        context=context,
        horizon=horizon,
        model_name=model_name,
        trained_model=trained_model,
    )
    baseline_forecasts = baseline["forecasts"]
    base_year = baseline["base_year"]
    forecast_year = baseline["forecast_year"]

    # ---------- build scenario features ----------
    supervised = context.supervised
    feature_cols = context.feature_cols
    target_cols = context.target_cols
    last_row = supervised.iloc[-1].copy()

    applied_shocks = {}

    for feat, factor in shocks.items():
        if feat in last_row.index:
            original_val = last_row[feat]
            last_row[feat] = original_val * factor
            applied_shocks[feat] = {
                "original": float(original_val),
                "factor": float(factor),
                "new_value": float(last_row[feat]),
            }
        else:
            # feature name not found; you could also warn/log instead of ignoring.
            # For now just skip it.
            continue

    X_scenario = last_row[feature_cols].values.reshape(1, -1)

    # ---------- scenario prediction ----------
    # trained_model implements .predict for all supported model_name values
    y_pred_scenario = trained_model.predict(X_scenario)[0]  # (n_targets,)

    scenario_forecasts = {
        name: float(val) for name, val in zip(target_cols, y_pred_scenario)
    }

    # ---------- impact calculation ----------
    impact = {}
    for tgt in target_cols:
        base_val = float(baseline_forecasts[tgt])
        scen_val = float(scenario_forecasts[tgt])
        abs_change = scen_val - base_val
        if base_val != 0:
            pct_change = (abs_change / base_val) * 100.0
        else:
            pct_change = float("nan")
        impact[tgt] = {
            "abs_change": abs_change,
            "pct_change": pct_change,
        }

    result = {
        "base_year": base_year,
        "forecast_year": forecast_year,
        "model_name": model_name,
        "baseline_forecasts": baseline_forecasts,
        "scenario_forecasts": scenario_forecasts,
        "impact": impact,
        "applied_shocks": applied_shocks,
    }

    return result