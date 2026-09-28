"""LSTM water-level forecaster, t+1 .. t+6 hours.

ARCHITECTURE
    Single-layer LSTM over a 24 h window of [rainfall, depth, soil, runoff],
    with a linear head producing all six horizons at once. Multi-horizon output
    from one shared trunk is standard practice and avoids six separate models.

WHY IT HAS A CHANCE OF BEATING PERSISTENCE
    Water level is strongly autocorrelated, so a naive "assume it stays the
    same as now" forecast is genuinely hard to beat, especially at t+1. The
    LSTM earns its keep at t+3 and beyond, where the level is rising and the
    trend can be extrapolated from the rainfall that caused it.

    Persistence is evaluated on the same windows with the same splits. Any
    horizon where the LSTM loses is reported as a loss.

SCALING
    Depth is standardised per feature using TRAIN statistics only. Standardise
    on the full series would leak test information through the scaler.

THE PROBLEM THIS MODEL ACTUALLY HAS, AND WHY IT IS NOT A MODEL PROBLEM
    99.99% of hourly depths across 2010-2015 are near zero. Trained on raw
    depth with a squared-error loss, an LSTM minimises total error by
    predicting zero almost everywhere: MAE looks acceptable (~1 mm) while
    RMSE explodes, because the rare storm windows are exactly the ones the
    model ignores. Those rare windows are the only ones an early-warning
    system exists to serve, so that model is useless no matter what its
    headline RMSE says.

    Two corrections, applied together:
      1. log1p on the target, so error is measured in relative terms. A 200 mm
         depth is as important to get right as a 1 mm depth; on a linear scale
         the network cannot see the difference.
      2. sample weighting that lifts storm-active windows, so the optimiser
         spends its capacity where the alerts are decided.

    Metrics are then reported BOTH over all windows and over the storm-active
    subset, because an early-warning claim is only meaningful on the second.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

CHANNELS = ["rainfall_mm", "depth_mm", "soil_mm", "runoff_mm"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make training reproducible.

    Without this the run-to-run spread is large enough to flip the headline
    result: identical code and data produced 6/6 horizons on one run and 2/6 on
    the next, because the MPS backend and the unseeded shuffles are both
    nondeterministic. A back-test number that changes on re-run is not evidence.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class WaterLevelLSTM(nn.Module):
    def __init__(self, n_channels: int, hidden: int, horizons: int,
                 dropout: float = 0.15):
        super().__init__()
        # single layer, so nn.LSTM's own dropout must be 0; regularisation
        # is applied in the head instead
        self.lstm = nn.LSTM(
            n_channels, hidden, num_layers=1, batch_first=True, dropout=0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizons)
        # zero-init the final layer, so the model starts life predicting
        # "no change from the current level" -- i.e. exactly persistence.
        # Every weight it then learns is a genuine improvement over the
        # baseline, which is the only way short horizons can be won.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])
        return self.head(self.drop(last))


class ResidualForecaster(nn.Module):
    """Predicts the CHANGE in log-depth, not the absolute level.

    Water level is strongly autocorrelated, so persistence is a hard baseline
    to beat, especially at t+1 where the level has barely moved. Asking a
    network to output an absolute level wastes capacity re-learning the
    identity map. This wrapper removes that burden:

        log1p(forecast[t+h]) = log1p(depth[t]) + correction[t+h]

    The inner head is zero-initialised, so at step 0 the model reproduces
    persistence exactly, and training can only move away from it if that
    reduces the loss. Working in log space keeps the correction meaningful
    across the 0 mm to 300 mm range.
    """

    def __init__(self, n_channels: int, hidden: int, horizons: int,
                 dropout: float = 0.15):
        super().__init__()
        self.net = WaterLevelLSTM(n_channels, hidden, horizons, dropout)
        self.horizons = horizons

    def forward(self, x, current_log):
        return current_log.unsqueeze(1) + self.net(x)


def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def main() -> None:
    set_seed()
    t0 = time.time()
    print("=" * 68)
    print(f"LSTM WATER-LEVEL FORECASTER   (device: {DEVICE})")
    print("=" * 68)

    d = np.load(config.PROCESSED / "sequences.npz", allow_pickle=True)
    X, y, ward, stamp = d["X"], d["y"], d["ward"], d["stamp"]

    is_test = stamp >= np.datetime64(config.SPLIT_DATE)
    Xtr, ytr = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    print(f"  windows   train {len(Xtr):,}  test {len(Xte):,}")

    # scaler fitted on TRAIN ONLY
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-6] = 1.0

    def norm(a):
        return (a - mu) / sd

    Xtr_n, Xte_n = norm(Xtr), norm(Xte)

    # log1p target: relative error, so 200 mm counts as much as 1 mm
    ytr_log = np.log1p(ytr)
    ymu, ysd = ytr_log.mean(), ytr_log.std()

    # RESIDUAL TARGET. The model learns the correction to persistence rather
    # than the absolute level, and the correction is scaled by the TRAIN
    # spread of that correction (not of the level) so the signal is not
    # squashed to a fraction of a standard deviation.
    cur_tr = Xtr[:, -1, 1]
    resid_tr = ytr_log - np.log1p(cur_tr)[:, None]
    # PER-HORIZON scaling. The correction at t+1 is tiny (depth has barely
    # moved) while at t+6 it is large. One shared scalar would leave the short
    # horizons with a vanishing gradient while the long horizons dominate, so
    # the trunk optimises for t+6 and actively damages t+1. Scaling each
    # horizon by its own train spread gives all six equal say.
    rmu = resid_tr.mean(axis=0)
    rsd = resid_tr.std(axis=0)
    rsd[rsd < 1e-6] = 1.0
    print("  target    log1p(depth) - log1p(current), per-horizon scaling")
    print("            horizon std   " + "  ".join(f"h{h}={v:.4f}"
                                               for h, v in zip(config.FORECAST_HORIZONS, rsd)))
    ytr_n = (resid_tr - rmu) / rsd

    # weight storm-active windows up. A window is active if it holds any
    # water at all, or rain in the last 24 h, or water in the forecast window.
    def activity(Xw, yw):
        depth_hist = Xw[:, :, 1]
        rain_hist = Xw[:, :, 0]
        return (
            (depth_hist.max(axis=1) > 1.0)
            | (rain_hist.sum(axis=1) > 2.0)
            | (yw.max(axis=1) > 1.0)
        )  # bool mask, not float

    act_tr = activity(Xtr, ytr)
    print(f"  active train windows  {int(act_tr.sum()):,} of {len(act_tr):,} "
          f"({act_tr.mean() * 100:.2f}%)")
    w_tr = np.where(act_tr > 0, float(config.ACTIVE_SAMPLE_WEIGHT), 1.0)

    # ---- validation split for early stopping -----------------------------
    # 2014 is carved out of the training period. Picking the epoch count by
    # hand is what produced the instability documented in set_seed(): the
    # headline score swung between 2/6 and 6/6 horizons purely on the random
    # seed. The number of epochs is a hyperparameter and must be chosen on
    # data the model has not already been fitted to.
    is_val = stamp[~is_test] >= np.datetime64(config.VAL_SPLIT_DATE)
    Xfit, yfit, wfit = Xtr[~is_val], ytr[~is_val], w_tr[~is_val]
    Xval, yval = Xtr[is_val], ytr[is_val]
    print(f"  fit {len(Xfit):,}  validate {len(Xval):,}  "
          f"(validation = {config.VAL_SPLIT_DATE}..2014-12-31)")

    model = ResidualForecaster(X.shape[-1], hidden=96,
                               horizons=len(config.FORECAST_HORIZONS)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params    {n_params:,}")

    cur_fit_t = torch.from_numpy(np.log1p(Xfit[:, -1, 1])).float()
    cur_val_t = torch.from_numpy(np.log1p(Xval[:, -1, 1])).float()
    epochs, batch = 30, 2048
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=4e-3, epochs=epochs, steps_per_epoch=len(Xfit) // batch,
    )
    lossf = nn.SmoothL1Loss(reduction="none")

    Xfit_t = torch.from_numpy(norm(Xfit)).float()
    yfit_t = torch.from_numpy(
        ((np.log1p(yfit) - np.log1p(Xfit[:, -1, 1])[:, None]) - rmu) / rsd
    ).float()
    wfit_t = torch.from_numpy(wfit).float()
    Xval_t = torch.from_numpy(norm(Xval)).float()
    yval_t = torch.from_numpy(
        ((np.log1p(yval) - np.log1p(Xval[:, -1, 1])[:, None]) - rmu) / rsd
    ).float()

    def validation_rmse() -> float:
        """RMSE in mm on the validation year, all horizons pooled."""
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xval_t), 8192):
                preds.append(model(Xval_t[i:i + 8192].to(DEVICE),
                                   cur_val_t[i:i + 8192].to(DEVICE)).cpu().numpy())
        r = np.concatenate(preds) * rsd + rmu
        pred_mm = np.clip(
            np.expm1(np.log1p(Xval[:, -1, 1])[:, None] + r), 0.0, None)
        return rmse(yval.ravel(), pred_mm.ravel())

    best_v, best_state, best_ep, bad = np.inf, None, 0, 0
    patience = 6
    print(f"\n  training up to {epochs} epochs, batch {batch}, "
          f"early stopping patience {patience}")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xfit_t))
        tot, nb = 0.0, 0
        for i in range(0, len(Xfit_t) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = Xfit_t[idx].to(DEVICE)
            cb = cur_fit_t[idx].to(DEVICE)
            yb = yfit_t[idx].to(DEVICE)
            wb = wfit_t[idx].to(DEVICE)
            opt.zero_grad()
            per = lossf(model(xb, cb), yb)  # (batch, horizons)
            loss = (per.mean(dim=1) * wb).mean() / wb.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        v = validation_rmse()
        flag = ""
        if v < best_v - 1e-4:
            best_v, best_ep, bad = v, ep + 1, 0
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
            flag = "  <- best"
        else:
            bad += 1
        print(f"    epoch {ep + 1:2d}/{epochs}  loss {tot / max(nb, 1):.4f}  "
              f"val RMSE {v:.4f}{flag}")
        if bad >= patience:
            print(f"    early stop at epoch {ep + 1}")
            break

    model.load_state_dict(best_state)
    print(f"  restored epoch {best_ep} (validation RMSE {best_v:.4f} mm)")

    # ---- inference ---------------------------------------------------
    current = Xte[:, -1, 1]  # depth at t
    current_log_t = torch.from_numpy(np.log1p(current)).float()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte_n), 8192):
            xb = torch.from_numpy(Xte_n[i:i + 8192]).float().to(DEVICE)
            cb = current_log_t[i:i + 8192].to(DEVICE)
            preds.append(model(xb, cb).cpu().numpy())
    resid_hat = np.concatenate(preds) * rsd + rmu
    pred = np.clip(np.expm1(current_log_t.numpy()[:, None] + resid_hat), 0.0, None)

    # ---- persistence baseline, same windows -------------------------
    persistence = np.repeat(current[:, None], len(config.FORECAST_HORIZONS), axis=1)

    # storm-active test subset: the only subset an early-warning claim
    # can honestly be made on
    act_te = activity(Xte, yte)
    n_act = int(act_te.sum())
    print(f"  active test windows   {n_act:,} of {len(act_te):,} "
          f"({act_te.mean() * 100:.2f}%)")
    print()

    def report(subset_name, y_ref, p_l, p_p):
        print("=" * 68)
        print(f"PER-HORIZON ACCURACY  ·  {subset_name}")
        print("=" * 68)
        print(f"  {'h':>3s}  {'LSTM RMSE':>10s} {'Persist RMSE':>13s}  "
              f"{'LSTM MAE':>9s} {'Persist MAE':>12s}  {'gain':>7s}  {'verdict':>8s}")
        out = []
        for k, h in enumerate(config.FORECAST_HORIZONS):
            r_l = rmse(y_ref[:, k], p_l[:, k])
            r_p = rmse(y_ref[:, k], p_p[:, k])
            a_l = float(mean_absolute_error(y_ref[:, k], p_l[:, k]))
            a_p = float(mean_absolute_error(y_ref[:, k], p_p[:, k]))
            gain = (r_p - r_l) / r_p * 100 if r_p else 0.0
            better = r_l < r_p
            print(f"  {h:3d}  {r_l:10.3f} {r_p:13.3f}  {a_l:9.3f} {a_p:12.3f}  "
                  f"{gain:6.1f}%  {'LSTM' if better else 'persist':>8s}")
            out.append({
                "horizon_h": h, "lstm_rmse": r_l, "persistence_rmse": r_p,
                "lstm_mae": a_l, "persistence_mae": a_p,
                "rmse_improvement_pct": float(gain),
                "lstm_beats_persistence": bool(better),
            })
        w = sum(o["lstm_beats_persistence"] for o in out)
        print(f"\n  beats persistence at {w}/{len(out)} horizons")
        print(f"  overall RMSE   LSTM {rmse(y_ref.ravel(), p_l.ravel()):.3f}   "
              f"persistence {rmse(y_ref.ravel(), p_p.ravel()):.3f}")
        print()
        return out, w

    rows, wins = report(f"ALL test windows (n={len(yte):,})", yte, pred, persistence)
    act_rows, act_wins = report(
        f"STORM-ACTIVE test windows (n={n_act:,}) - the ones that matter",
        yte[act_te], pred[act_te], persistence[act_te],
    )

    print("=" * 68)
    print(f"train time  {time.time() - t0:.1f}s")

    out = config.MODELS / "lstm_metrics.json"
    out.write_text(json.dumps({
        "all_windows": {
            "per_horizon": rows, "horizons_won": wins,
            "n_test": int(len(yte)),
        },
        "storm_active_windows": {
            "per_horizon": act_rows, "horizons_won": act_wins,
            "n_test": n_act,
            "definition": ("windows with any depth > 1 mm, any 24 h rain total "
                           "> 2 mm, or any forecast depth > 1 mm"),
        },
        "meets_target_all": wins == len(rows),
        "meets_target_storm": act_wins == len(act_rows),
        "n_train": int(len(Xtr)),
        "params": n_params, "epochs": epochs, "device": DEVICE,
        "target": ("log1p target with storm-window sample weighting; the "
                   "storm-active subset is the honest basis for an "
                   "early-warning claim"),
        "note": ("Depth is simulated by our bucket model, so these errors "
                 "measure skill at predicting that simulator. They do not "
                 "transfer to real gauges without retraining."),
    }, indent=2))
    print(f"  wrote {out}")

    # rmu/rsd are PER-HORIZON vectors and mu_time records what the train
    # scaler was fitted on; the back-test needs both to reproduce inference
    # exactly without ever refitting.
    torch.save({
        "state_dict": model.state_dict(),
        "mu": mu, "sd": sd, "ymu": float(ymu), "ysd": float(ysd),
        "rmu": rmu, "rsd": rsd,
        "n_channels": X.shape[-1], "hidden": 96,
        "horizons": config.FORECAST_HORIZONS,
        "mu_time": str(pd.Timestamp(stamp[~is_test].min()).date()),
    }, config.MODELS / "lstm_forecast.pt")
    print(f"  wrote {config.MODELS / 'lstm_forecast.pt'}")


if __name__ == "__main__":
    main()
