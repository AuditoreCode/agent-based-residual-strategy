import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize,differential_evolution


def evolution_function(s_t,delta_p,a,b):
    return np.tanh(a*delta_p+ b*s_t)

def energy_function(delta_p,s_t,a,b):
    f = ((a*delta_p.shift(1)+ b*s_t.shift(1))).apply(math.tanh)
    return np.sum((delta_p - f)**2) 

def two_state_ising_prediction(series:pd.Series, normalization_window:int, optimisation_window:int, env_window:int):
    # normalized series
    delta_p = (series).rolling(window=normalization_window,min_periods=1).mean()/np.abs(series).rolling(window=normalization_window,min_periods=1).max()
    # investement environment
    s_t = series - series.shift(env_window) ## variation over the last env_window time steps (identically 0 if env_window=0)

    # 1 step walk forward optimization
    test_window = 1
    results= []
    bounds = [
        (-4, 4),   # borne pour a
        (-10, 10)    # borne pour b
    ]

    for i in range(optimisation_window, len(series)-test_window):
        train_delta_p = delta_p[i-optimisation_window:i]
        train_s_t = s_t[i-optimisation_window:i]
        test_delta_p = delta_p[i:i+test_window]
        test_s_t = s_t[i:i+test_window]
        res_global = differential_evolution(
        lambda x: energy_function(train_delta_p, train_s_t, x[0], x[1]),
        bounds=bounds)

        res = minimize(
        lambda x: energy_function(train_delta_p, train_s_t, x[0], x[1]),
        res_global.x,
        method="L-BFGS-B",
                bounds=bounds)

        a_opt, b_opt = res.x
        delta_p_t_pred = (evolution_function(s_t.iloc[i], delta_p.iloc[i], a_opt, b_opt))
        ser_t_pred = (delta_p_t_pred)*np.abs(series.iloc[i-normalization_window+1:i+1]).max()
        results.append([a_opt, b_opt, delta_p_t_pred, ser_t_pred])
        
    results_df = pd.DataFrame(results, columns=['a', 'b', 'normalized_series', 'series'], index=series.index[optimisation_window+test_window:])
    return results_df


import torch


def two_state_ising_prediction_gpu(
    series: pd.Series,
    normalization_window: int,
    optimisation_window: int,
    env_window: int,
    n_steps: int = 500,
    lr: float = 0.05,
    device: str | None = None,
):
    """
    Version GPU avec PyTorch.
    Optimise a_t et b_t pour chaque fenêtre rolling en parallèle.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Préparation pandas
    # -----------------------------
    series = series.astype(float)

    rolling_mean = series.rolling(
        window=normalization_window,
        min_periods=1
    ).mean()

    rolling_absmax = series.abs().rolling(
        window=normalization_window,
        min_periods=1
    ).max()

    delta_p = rolling_mean / rolling_absmax.replace(0, np.nan)

    # Environnement d'investissement
    s_t = series - series.shift(env_window)

    # Scale utilisé pour revenir à la série originale :
    # même échelle que la normalisation (max des valeurs absolues), connue à la date courante
    scale = rolling_absmax

    # -----------------------------
    # Passage en tenseurs GPU
    # -----------------------------
    delta = torch.tensor(
        delta_p.to_numpy(dtype=np.float32),
        device=device
    )

    env = torch.tensor(
        s_t.to_numpy(dtype=np.float32),
        device=device
    )

    scale_t = torch.tensor(
        scale.to_numpy(dtype=np.float32),
        device=device
    )

    N = len(series)
    L = optimisation_window - 1
    n_pred = N - optimisation_window - 1

    if n_pred <= 0:
        raise ValueError("optimisation_window est trop grand pour la taille de la série.")

    # -----------------------------
    # Construction des fenêtres
    # -----------------------------
    # Pour chaque fenêtre :
    # y_t ≈ tanh(a * delta_{t-1} + b * s_{t-1})
    delta_windows = delta.unfold(0, L, 1)
    env_windows = env.unfold(0, L, 1)

    x_delta = delta_windows[:n_pred]
    x_env = env_windows[:n_pred]
    y = delta_windows[1:n_pred + 1]

    # Gestion des NaN
    mask = (
        torch.isfinite(x_delta)
        & torch.isfinite(x_env)
        & torch.isfinite(y)
    )

    x_delta_clean = torch.nan_to_num(x_delta, nan=0.0)
    x_env_clean = torch.nan_to_num(x_env, nan=0.0)
    y_clean = torch.nan_to_num(y, nan=0.0)

    denom = mask.sum(dim=1).clamp_min(1)

    # -----------------------------
    # Paramètres optimisés sur GPU
    # -----------------------------
    # On optimise des paramètres libres, puis on les borne avec tanh.
    raw_a = torch.zeros(n_pred, device=device, requires_grad=True)
    raw_b = torch.zeros(n_pred, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([raw_a, raw_b], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()

        # bornes équivalentes :
        # a in [-4, 4]
        # b in [-10, 10]
        a = 4.0 * torch.tanh(raw_a)
        b = 10.0 * torch.tanh(raw_b)

        pred = torch.tanh(
            a[:, None] * x_delta_clean
            + b[:, None] * x_env_clean
        )

        error = torch.where(mask, y_clean - pred, torch.zeros_like(y_clean))

        # Énergie plus standard : somme des erreurs au carré
        loss_per_window = (error ** 2).sum(dim=1) / denom
        loss = loss_per_window.mean()

        loss.backward()
        optimizer.step()

    # -----------------------------
    # Prédiction
    # -----------------------------
    with torch.no_grad():
        a_opt = 4.0 * torch.tanh(raw_a)
        b_opt = 10.0 * torch.tanh(raw_b)

        current_delta = delta[optimisation_window:N - 1]
        current_env = env[optimisation_window:N - 1]
        current_scale = scale_t[optimisation_window:N - 1]

        valid_current = (
            torch.isfinite(current_delta)
            & torch.isfinite(current_env)
            & torch.isfinite(current_scale)
        )

        current_delta_clean = torch.nan_to_num(current_delta, nan=0.0)
        current_env_clean = torch.nan_to_num(current_env, nan=0.0)

        normalized_pred = torch.tanh(
            a_opt * current_delta_clean
            + b_opt * current_env_clean
        )

        series_pred = normalized_pred * current_scale

        normalized_pred = torch.where(
            valid_current,
            normalized_pred,
            torch.tensor(float("nan"), device=device)
        )

        series_pred = torch.where(
            valid_current,
            series_pred,
            torch.tensor(float("nan"), device=device)
        )

    results_df = pd.DataFrame(
        {
            "a": a_opt.detach().cpu().numpy(),
            "b": b_opt.detach().cpu().numpy(),
            "normalized_series": normalized_pred.detach().cpu().numpy(),
            "series": series_pred.detach().cpu().numpy(),
        },
        index=series.index[optimisation_window + 1:]
    )

    return results_df

    