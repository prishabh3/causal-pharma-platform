import pytest
import numpy as np
from causal.discovery import notears_linear

def shd(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    # Binarize estimated adjacency at threshold 0.3
    est = (np.abs(est_adj) > 0.3).astype(int)
    true = (np.abs(true_adj) > 0.0).astype(int)
    extra = np.sum((est - true).clip(min=0))   # false positives
    missing = np.sum((true - est).clip(min=0)) # false negatives
    # reversals: edges present but in wrong direction
    reversed_ = np.sum((est * true.T) * (1 - true))
    return int(extra + missing + reversed_)

def test_notears_dag_recovery():
    rng = np.random.default_rng(42)
    n = 1000
    d = 5
    
    # Generate synthetic DAG: 0 -> 1 -> 2 -> 3 -> 4
    X = np.zeros((n, d))
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = 2.0 * X[:, 0] + rng.normal(size=n)
    X[:, 2] = -1.5 * X[:, 1] + rng.normal(size=n)
    X[:, 3] = 1.0 * X[:, 2] + rng.normal(size=n)
    X[:, 4] = -2.0 * X[:, 3] + rng.normal(size=n)
    
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    W_est = notears_linear(X_norm, lambda1=0.1, loss_type="l2", w_threshold=0.3)
    
    W_true = np.zeros((d, d), dtype=int)
    W_true[0, 1] = 1
    W_true[1, 2] = 1
    W_true[2, 3] = 1
    W_true[3, 4] = 1
    
    # Metrics
    shd_val = shd(W_true, W_est)
    
    est_binary = (np.abs(W_est) > 0.3).astype(int)
    
    TP = np.sum((est_binary == 1) & (W_true == 1))
    FP = np.sum((est_binary == 1) & (W_true == 0))
    FN = np.sum((est_binary == 0) & (W_true == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    WW = (W_est * W_est)
    from scipy.linalg import expm
    E = expm(WW)
    h_W = np.trace(E) - d
    
    print("\n### DAG Structure Recovery (Synthetic, n=1000, d=5 nodes, 4 true edges)")
    print("| Metric                      | Value   |")
    print("|-----------------------------|---------|")
    print(f"| Structural Hamming Distance | {shd_val}       |")
    print(f"| Precision                   | {precision:.2f}    |")
    print(f"| Recall                      | {recall:.2f}    |")
    print(f"| Acyclicity h(W)             | {h_W:.2e} |")

if __name__ == "__main__":
    test_notears_dag_recovery()
