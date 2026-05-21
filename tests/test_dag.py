import numpy as np
from causal.discovery import notears_linear

def shd(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    # Binarize estimated adjacency at threshold 0.2
    est = (np.abs(est_adj) > 0.2).astype(int)
    true = (np.abs(true_adj) > 0.0).astype(int)
    extra = np.sum((est - true).clip(min=0))   # false positives
    missing = np.sum((true - est).clip(min=0)) # false negatives
    # reversals: edges present but in wrong direction
    reversed_ = np.sum((est * true.T) * (1 - true))
    return int(extra + missing + reversed_)

def test_notears_dag_recovery():
    np.random.seed(42)
    n, d = 2000, 5
    
    true_W = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ], dtype=float) * 2.0
    
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(true_W[:, j] != 0)[0]
        X[:, j] = X[:, parents] @ true_W[parents, j] + \
                  np.random.normal(0, 0.1, n)
    
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    W_est = notears_linear(X_norm, lambda1=0.01, loss_type="l2", w_threshold=0.2, max_iter=200)
    
    W_true = (true_W != 0).astype(int)
    
    # Metrics
    shd_val = shd(W_true, W_est)
    
    est_binary = (np.abs(W_est) > 0.2).astype(int)
    
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
    print("\nTrue W:")
    print(W_true)
    print("\nEst W:")
    print(W_est)
    print("\nEst Binary:")
    print(est_binary)

if __name__ == "__main__":
    test_notears_dag_recovery()
