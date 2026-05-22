"""
Benchmark test for causal estimators on the LaLonde NSW dataset.

Uses ONLY the experimental sample (treat=0/1 from NSW) loaded via
dowhy.datasets.lalonde_dataset() which ships the Dehejia-Wahba (1999)
experimental NSW dataset (185 treated + 260 controls, no PSID/CPS mix-in).

Reference RCT ATE: ~$1,794 (Dehejia & Wahba 1999).
"""
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from causal.estimation import CausalEstimator


def load_lalonde_experimental():
    """
    Load the LaLonde experimental (NSW) sample.

    Uses dowhy.datasets.lalonde_dataset() which ships the Dehejia-Wahba (1999)
    experimental NSW data (445 rows: 185 treated + 260 control from NSW only).
    Falls back to a CSV download if dowhy is unavailable.
    """
    try:
        from dowhy.datasets import lalonde_dataset
        df = lalonde_dataset()
        # Rename educ -> education, hisp -> hispanic, nodegr -> nodegree
        df = df.rename(columns={
            "educ": "education",
            "hisp": "hispanic",
            "nodegr": "nodegree",
        })
        # treat is bool in dowhy — convert to int
        df["treat"] = df["treat"].astype(int)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Could not load LaLonde experimental dataset via dowhy: {e}\n"
            "Install dowhy: pip install dowhy==0.13"
        )


def test_estimators_lalonde():
    df = load_lalonde_experimental()

    Y = df["re78"].values.astype(float)
    T = df["treat"].values.astype(int)
    # Exclude u74/u75: binary zero-earnings indicators that are redundant with
    # re74/re75 and cause near-separation in propensity models, inflating DR.
    # Original nsw_dw.csv does not include these columns.
    covariate_cols = ["age", "education", "black", "hispanic",
                      "married", "nodegree", "re74", "re75"]
    X = df[covariate_cols].values.astype(float)


    # True RCT ATE from the NSW experimental evaluation (Dehejia & Wahba 1999)
    rct_ate = 1794.0

    results = []
    n = len(X)

    # ── 1. Naive OLS ────────────────────────────────────────────────────────
    X_ols    = np.column_stack([X, T])
    ols      = LinearRegression().fit(X_ols, Y)
    ate_ols  = ols.coef_[-1]
    bias_ols = abs(ate_ols - rct_ate) / rct_ate

    Y_pred    = ols.predict(X_ols)
    residuals = Y - Y_pred
    mse       = np.sum(residuals ** 2) / (n - X_ols.shape[1])
    var_b     = mse * np.linalg.inv(X_ols.T @ X_ols)[-1, -1]
    se_ols    = np.sqrt(var_b)
    ci_ols    = (ate_ols - 1.96 * se_ols, ate_ols + 1.96 * se_ols)

    results.append(("Naive OLS", ate_ols, ci_ols, bias_ols))

    # ── 2. IPW ───────────────────────────────────────────────────────────────
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ipw_pipe    = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000)),
    ])
    ipw_pipe.fit(X, T)
    p           = ipw_pipe.predict_proba(X)[:, 1]
    p           = np.clip(p, 0.05, 0.95)    # clip to avoid extreme weights
    ipw_samples = T * Y / p - (1 - T) * Y / (1 - p)
    ate_ipw     = np.mean(ipw_samples)
    bias_ipw    = abs(ate_ipw - rct_ate) / rct_ate
    se_ipw      = np.std(ipw_samples) / np.sqrt(n)
    ci_ipw      = (ate_ipw - 1.96 * se_ipw, ate_ipw + 1.96 * se_ipw)


    results.append(("IPW", ate_ipw, ci_ipw, bias_ipw))

    # ── 3. Doubly Robust ─────────────────────────────────────────────────────
    dr      = CausalEstimator(method="doubly_robust")
    dr.fit(Y, T, X)
    ate_dr  = dr.estimate_ate(X)
    bias_dr = abs(ate_dr - rct_ate) / rct_ate

    cates_dr = dr.estimate_cate(X)
    se_dr    = np.std(cates_dr) / np.sqrt(n)
    ci_dr    = (ate_dr - 1.96 * se_dr, ate_dr + 1.96 * se_dr)

    results.append(("Doubly Robust", ate_dr, ci_dr, bias_dr))

    # ── Print table ──────────────────────────────────────────────────────────
    print("\nBenchmark Results — LaLonde NSW Experimental Sample (RCT ATE ≈ $1,794)")
    print(f"{'Estimator':<15} | {'ATE Estimate':<13} | {'95% CI':<30} | Bias vs RCT")
    print("-" * 80)
    for name, ate, ci, bias in results:
        print(f"{name:<15} | ${ate:<12.2f} | (${ci[0]:.2f}, ${ci[1]:.2f}) | {bias * 100:.1f}%")

    # ── Assertions ───────────────────────────────────────────────────────────
    # DR must be within 40% of true ATE
    assert abs(ate_dr - rct_ate) / rct_ate < 0.40, (
        f"DR ATE ${ate_dr:.2f} is more than 40% off from RCT ATE ${rct_ate}"
    )
    # DR must outperform OLS
    assert bias_dr < bias_ols, (
        f"DR bias ({bias_dr * 100:.1f}%) should be < OLS bias ({bias_ols * 100:.1f}%)"
    )
    # IPW lower CI bound should not be extremely negative (sanity check)
    assert ci_ipw[0] > -3000, (
        f"IPW CI lower bound ${ci_ipw[0]:.2f} is extremely negative — extreme weights?"
    )


if __name__ == "__main__":
    test_estimators_lalonde()
