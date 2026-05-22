"""
Test NOTEARS DAG structure recovery on a synthetic linear Gaussian SCM.

Settings:
  n = 3000 samples, d = 5 nodes, 4 true edges (fan-out structure)
  noise std = 1.0, edge weights = 1.5
  threshold = 0.2 for binarizing the estimated adjacency matrix

DAG structure (fan-out, avoids collinearity from chain compounding):
  X0 → X1
  X0 → X2
  X1 → X3
  X2 → X4
"""
import numpy as np
from causal.discovery import notears_linear


def shd(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    """Structural Hamming Distance — counts missing, extra, and reversed edges."""
    est  = (np.abs(est_adj)  > 0.2).astype(int)
    true = (np.abs(true_adj) > 0.0).astype(int)
    extra     = np.sum((est  - true).clip(min=0))   # false positives
    missing   = np.sum((true - est ).clip(min=0))   # false negatives
    reversed_ = np.sum((est * true.T) * (1 - true)) # wrong direction
    return int(extra + missing + reversed_)


def test_notears_dag_recovery():
    np.random.seed(42)
    n, d = 3000, 5   # 3000 samples for reliable recovery

    # True DAG: fan-out structure (avoids chain collinearity)
    #   X0 → X1, X0 → X2, X1 → X3, X2 → X4
    # W_true[i, j] = weight of edge i → j (lower-triangular for valid DAG)
    true_W = np.zeros((d, d))
    true_W[0, 1] = 1.5   # X0 → X1
    true_W[0, 2] = 1.5   # X0 → X2
    true_W[1, 3] = 1.5   # X1 → X3
    true_W[2, 4] = 1.5   # X2 → X4

    # Generate data from the linear Gaussian SCM in topological order
    X = np.zeros((n, d))
    X[:, 0] = np.random.normal(0, 1.0, n)   # root: no parents
    X[:, 1] = true_W[0, 1] * X[:, 0] + np.random.normal(0, 1.0, n)
    X[:, 2] = true_W[0, 2] * X[:, 0] + np.random.normal(0, 1.0, n)
    X[:, 3] = true_W[1, 3] * X[:, 1] + np.random.normal(0, 1.0, n)
    X[:, 4] = true_W[2, 4] * X[:, 2] + np.random.normal(0, 1.0, n)

    # Normalize so NOTEARS works on comparable scales
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Run NOTEARS
    W_est = notears_linear(
        X_norm,
        lambda1=0.01,
        loss_type="l2",
        w_threshold=0.2,
        max_iter=200,
    )

    W_true = (true_W != 0).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    shd_val    = shd(W_true, W_est)
    est_binary = (np.abs(W_est) > 0.2).astype(int)

    TP = np.sum((est_binary == 1) & (W_true == 1))
    FP = np.sum((est_binary == 1) & (W_true == 0))
    FN = np.sum((est_binary == 0) & (W_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # Acyclicity constraint h(W) = tr(e^{W∘W}) − d (should be ≈ 0)
    WW = W_est * W_est
    from scipy.linalg import expm
    E   = expm(WW)
    h_W = np.trace(E) - d

    print("\n### DAG Structure Recovery (Synthetic, n=3000, d=5 nodes, 4 true edges)")
    print("| Metric                      | Value   |")
    print("|-----------------------------|---------|")
    print(f"| Structural Hamming Distance | {shd_val}       |")
    print(f"| Precision                   | {precision:.2f}    |")
    print(f"| Recall                      | {recall:.2f}    |")
    print(f"| Acyclicity h(W)             | {h_W:.2e} |")
    print("\nTrue W:")
    print(W_true)
    print("\nEst W:")
    print(np.round(W_est, 3))
    print("\nEst Binary:")
    print(est_binary)

    # ── Assertions ───────────────────────────────────────────────────────────
    assert shd_val <= 3, (
        f"SHD = {shd_val} > 3. NOTEARS is not recovering the structure well."
    )
    assert precision >= 0.5, (
        f"Precision = {precision:.2f} < 0.5. Too many false positive edges."
    )
    assert recall >= 0.5, (
        f"Recall = {recall:.2f} < 0.5. Too many true edges are missed."
    )


if __name__ == "__main__":
    test_notears_dag_recovery()
