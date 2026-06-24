"""
GP Surrogate Analysis — CAFM Pedestrian Simulation
===================================================
Dewa's analysis script for partial-dependence GP sweeps.

Usage:
    python gp_cafm_analysis.py

Adjust CONFIG block below to change dataset path, sweep param, target, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, RationalQuadratic, ExpSineSquared,
    WhiteKernel, ConstantKernel as C
)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_path" : r"D:\KIT\Datasets\cafm_lhs_300.npz",
    "n_points"     : 50,          # how many rows to use from dataset
    "sweep_param"  : "density",   # which input to sweep (keep others frozen)
    "freeze_mode"  : "random",      # "mean" or "random"
    "target_idx"   : 0,           # 0=mean_crossing_time, 1=flow_rate, etc.
    "test_size"    : 0.2,
    "random_state" : 42,
    "alpha"        : 1e-6,        # GP noise regularisation (jitter)
    "n_restarts"   : 5,           # optimiser restarts for kernel hyperparams
}

PARAM_NAMES  = ["v0_mean", "tau", "A", "R_safety", "density", "flow_ratio"]
PARAM_BOUNDS = {
    "v0_mean"   : (0.80, 1.60),
    "tau"       : (0.30, 0.80),
    "A"         : (10.0, 40.0),
    "R_safety"  : (0.80, 1.60),
    "density"   : (0.10, 1.00),
    "flow_ratio": (0.00, 0.50),
}
OUTPUT_NAMES = [
    "mean_crossing_time",
    "flow_rate",
    "order_param_mean",
    "mean_speed",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & SWEEP CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg):
    data    = np.load(cfg["dataset_path"])
    X       = data["X"][: cfg["n_points"]]
    Y       = data["Y"][: cfg["n_points"]]
    y_col   = Y[:, cfg["target_idx"]]
    return X, y_col


def build_sweep(X, cfg, rng=None):
    """
    Build X_sweep: all rows identical except the sweep column varies
    across the observed range of that parameter.

    freeze_mode = "mean"   → freeze at column means
    freeze_mode = "random" → freeze at a random point within PARAM_BOUNDS
    """
    sweep_idx = PARAM_NAMES.index(cfg["sweep_param"])

    if cfg["freeze_mode"] == "mean":
        freeze_point = np.mean(X, axis=0)
    else:
        rng = rng or np.random.default_rng(cfg["random_state"])
        freeze_point = np.array([
            rng.uniform(*PARAM_BOUNDS[p]) for p in PARAM_NAMES
        ])

    print(f"\n[FREEZE POINT — mode={cfg['freeze_mode']}]")
    for i, name in enumerate(PARAM_NAMES):
        marker = " ← sweep" if i == sweep_idx else ""
        print(f"  {name:12s}: {freeze_point[i]:.4f}{marker}")

    X_sweep = np.tile(freeze_point, (cfg["n_points"], 1))
    # replace sweep column with actual observed values (sorted)
    sweep_vals = np.sort(X[:, sweep_idx])
    X_sweep[:, sweep_idx] = sweep_vals

    return X_sweep, sweep_vals, freeze_point


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_kernels(alpha_noise=None):
    """
    Returns dict of kernel_name → (kernel_object, gp_kwargs).
    If alpha_noise is given, adds a WhiteKernel for explicit noise modelling.
    """
    noise_k = WhiteKernel(noise_level=alpha_noise) if alpha_noise else 0.0

    kernels = {
        "Matern-2.5": (
            C(1.0) * Matern(length_scale=1.0, nu=2.5) + noise_k,
            {"n_restarts_optimizer": CONFIG["n_restarts"]}
        ),
        "Matern-1.5": (
            C(1.0) * Matern(length_scale=1.0, nu=1.5) + noise_k,
            {"n_restarts_optimizer": CONFIG["n_restarts"]}
        ),
        "RBF": (
            C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-10, 15)) + noise_k,
            {"n_restarts_optimizer": CONFIG["n_restarts"]}
        ),
        "RationalQuadratic": (
            C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + noise_k,
            {"n_restarts_optimizer": CONFIG["n_restarts"]}
        ),
    }
    return kernels


# ─────────────────────────────────────────────────────────────────────────────
# GP FITTING & PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def fit_gp(X_train_s, y_train_s, kernel, gp_kwargs, alpha=1e-6):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, **gp_kwargs)
    gp.fit(X_train_s, y_train_s)
    print(f"  Optimised kernel : {gp.kernel_}")
    print(f"  Log-marginal-lik : {gp.log_marginal_likelihood_value_:.4f}")
    return gp


def predict_and_invert(gp, X_s, scaler_y):
    mean_s, std_s = gp.predict(X_s, return_std=True)
    mean_r = scaler_y.inverse_transform(mean_s.reshape(-1, 1)).ravel()
    std_r  = std_s * scaler_y.scale_[0]
    return mean_r, std_r


# ─────────────────────────────────────────────────────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(y_true, y_pred, kernel_name):
    """
    Returns dict with residuals, statistics, and fitted normal params.
    """
    errors = y_true - y_pred
    mu, sigma = np.mean(errors), np.std(errors)
    # Shapiro-Wilk normality test
    stat, p_val = stats.shapiro(errors) if len(errors) <= 5000 else (np.nan, np.nan)

    result = {
        "kernel"        : kernel_name,
        "errors"        : errors,
        "mean"          : mu,
        "std"           : sigma,
        "variance"      : sigma**2,
        "rmse"          : np.sqrt(np.mean(errors**2)),
        "mae"           : np.mean(np.abs(errors)),
        "shapiro_stat"  : stat,
        "shapiro_p"     : p_val,
        "norm_fit_mu"   : mu,
        "norm_fit_sig"  : sigma,
    }
    print(f"\n  [Error Analysis — {kernel_name}]")
    print(f"    mean(error)  = {mu:.4f}")
    print(f"    std(error)   = {sigma:.4f}")
    print(f"    variance     = {sigma**2:.4f}")
    print(f"    RMSE         = {result['rmse']:.4f}")
    print(f"    MAE          = {result['mae']:.4f}")
    print(f"    Shapiro-Wilk : stat={stat:.4f}, p={p_val:.4f}"
          if not np.isnan(stat) else "    Shapiro-Wilk : N/A (too many samples)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Matern-2.5"        : "#4C9BE8",
    "Matern-1.5"        : "#E8834C",
    "RBF"               : "#4CE87A",
    "RationalQuadratic" : "#C84CE8",
}


def plot_all(sweep_vals, y_actual, results_dict, cfg):
    """
    One big figure:
      Row 1: GP fit + CI for each kernel (side-by-side)
      Row 2: Residuals plot per kernel
      Row 3: Error distribution (histogram + fitted normal) per kernel
    """
    kernels = list(results_dict.keys())
    n_k     = len(kernels)
    target  = OUTPUT_NAMES[cfg["target_idx"]]
    sweep   = cfg["sweep_param"]

    fig = plt.figure(figsize=(6 * n_k, 16))
    fig.suptitle(
        f"GP Surrogate Analysis  |  sweep: {sweep}  →  {target}\n"
        f"freeze mode: {cfg['freeze_mode']}  |  N={cfg['n_points']}",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(3, n_k, figure=fig, hspace=0.45, wspace=0.35)

    for col, kname in enumerate(kernels):
        r    = results_dict[kname]
        color = COLORS.get(kname, "#888")

        # ── Row 0: GP fit ──
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(sweep_vals, y_actual, "k--", lw=1.5, label="Actual", zorder=3)
        ax0.scatter(sweep_vals, y_actual, color="k", s=18, zorder=4)
        ax0.plot(sweep_vals, r["mean"], color=color, lw=2, label="GP mean")
        ax0.fill_between(
            sweep_vals,
            r["mean"] - 1.96 * r["std"],
            r["mean"] + 1.96 * r["std"],
            color=color, alpha=0.25, label="95% CI"
        )
        ax0.set_title(kname, fontsize=11, fontweight="bold")
        ax0.set_xlabel(sweep)
        ax0.set_ylabel(target)
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)

        # ── Row 1: Residuals ──
        ax1 = fig.add_subplot(gs[1, col])
        ea   = r["error_analysis"]
        errs = ea["errors"]
        # here x-axis = sweep vals at TEST points; we use index for simplicity
        ax1.axhline(0, color="k", lw=1, linestyle="--")
        ax1.axhline( ea["std"], color=color, lw=1, linestyle=":", alpha=0.7, label="+1σ")
        ax1.axhline(-ea["std"], color=color, lw=1, linestyle=":", alpha=0.7, label="-1σ")
        ax1.scatter(range(len(errs)), errs, color=color, s=20, zorder=3)
        ax1.set_title(f"Residuals  (RMSE={ea['rmse']:.4f})", fontsize=10)
        ax1.set_xlabel("test sample index")
        ax1.set_ylabel("error (actual − pred)")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # ── Row 2: Error distribution ──
        ax2 = fig.add_subplot(gs[2, col])
        ax2.hist(errs, bins=max(5, len(errs)//3), color=color,
                 alpha=0.55, density=True, edgecolor="white", label="Residuals")
        x_fit = np.linspace(errs.min(), errs.max(), 200)
        ax2.plot(x_fit, stats.norm.pdf(x_fit, ea["norm_fit_mu"], ea["norm_fit_sig"]),
                 "k-", lw=2, label="Fitted N(μ,σ)")
        ax2.axvline(ea["mean"], color="red", lw=1.5, linestyle="--",
                    label=f"μ={ea['mean']:.4f}")
        ax2.set_title(
            f"Error dist  μ={ea['mean']:.4f}  σ²={ea['variance']:.4f}", fontsize=10
        )
        ax2.set_xlabel("error")
        ax2.set_ylabel("density")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gp_analysis_output.png", dpi=150, bbox_inches="tight")
    print("\n[Saved] gp_analysis_output.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    rng = np.random.default_rng(cfg["random_state"])

    print("=" * 60)
    print(" GP CAFM Surrogate Analysis")
    print("=" * 60)
    print(f"Dataset  : {cfg['dataset_path']}")
    print(f"N points : {cfg['n_points']}")
    print(f"Target   : {OUTPUT_NAMES[cfg['target_idx']]}")
    print(f"Sweep    : {cfg['sweep_param']}")
    print(f"Freeze   : {cfg['freeze_mode']}")

    # 1. Load
    X, y = load_data(cfg)
    print(f"\nX shape: {X.shape}  |  y shape: {y.shape}")
    print(f"y range : [{y.min():.4f}, {y.max():.4f}]")

    # 2. Build sweep
    X_sweep, sweep_vals, freeze_point = build_sweep(X, cfg, rng)

    # 3. Scale
    scale_x = StandardScaler()
    scale_y = StandardScaler()

    x_tr_raw, x_te_raw, y_tr, y_te = train_test_split(
        X_sweep, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )
    x_tr_s = scale_x.fit_transform(x_tr_raw)
    y_tr_s = scale_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    x_te_s = scale_x.transform(x_te_raw)

    # actual values for test set (for error analysis)
    y_te_actual = y_te

    # 4. Fit all kernels
    # NOTE: set alpha_noise=None to disable WhiteKernel (pure GP jitter only)
    # set alpha_noise=e.g. 0.01 to add explicit noise kernel
    kernels_dict = get_kernels(alpha_noise=None)

    results = {}
    for kname, (kernel, gp_kwargs) in kernels_dict.items():
        print(f"\n{'─'*50}")
        print(f"[Kernel] {kname}")
        gp = fit_gp(x_tr_s, y_tr_s, kernel, gp_kwargs, alpha=cfg["alpha"])

        # predict on FULL sweep (for plot)
        X_sweep_s        = scale_x.transform(X_sweep)
        mean_full, std_full = predict_and_invert(gp, X_sweep_s, scale_y)

        # predict on TEST set (for error analysis)
        mean_te, std_te  = predict_and_invert(gp, x_te_s, scale_y)
        ea               = error_analysis(y_te_actual, mean_te, kname)

        results[kname] = {
            "gp"           : gp,
            "mean"         : mean_full,
            "std"          : std_full,
            "mean_test"    : mean_te,
            "std_test"     : std_te,
            "error_analysis": ea,
        }

    # 5. Summary table
    print(f"\n{'='*60}")
    print(f"{'Kernel':<22} {'RMSE':>8} {'MAE':>8} {'μ_err':>8} {'σ²_err':>10}")
    print(f"{'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for kname, r in results.items():
        ea = r["error_analysis"]
        print(f"{kname:<22} {ea['rmse']:>8.4f} {ea['mae']:>8.4f} "
              f"{ea['mean']:>8.4f} {ea['variance']:>10.6f}")

    # 6. Plot
    plot_all(sweep_vals, y, results, cfg)


if __name__ == "__main__":
    main()
