import pytest
import numpy as np
from causal.discovery import notears_linear

def test_notears_dag_recovery():
    # Generate synthetic DAG: 0 -> 1 -> 2 -> 3 -> 4
    rng = np.random.default_rng(42)
    n = 1000
    d = 5
    
    X = np.zeros((n, d))
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = 2.0 * X[:, 0] + rng.normal(size=n)
    X[:, 2] = -1.5 * X[:, 1] + rng.normal(size=n)
    X[:, 3] = 1.0 * X[:, 2] + rng.normal(size=n)
    X[:, 4] = -2.0 * X[:, 3] + rng.normal(size=n)
    
    # Normalize
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Run NOTEARS
    W_est = notears_linear(X_norm, lambda1=0.1, loss_type="l2", w_threshold=0.3)
    W_binary = (np.abs(W_est) > 0).astype(int)
    
    # True Adjacency Matrix
    W_true = np.zeros((d, d), dtype=int)
    W_true[0, 1] = 1
    W_true[1, 2] = 1
    W_true[2, 3] = 1
    W_true[3, 4] = 1
    
    # Calculate Structural Hamming Distance (SHD)
    # SHD = missing edges + extra edges + reversed edges
    # Standard approximation: 
    shd = 0
    for i in range(d):
        for j in range(d):
            if W_true[i, j] == 1 and W_binary[i, j] == 0:
                # missing or reversed
                if W_binary[j, i] == 1:
                    # reversed (we only count once, so we don't penalize the extra edge below)
                    shd += 1
                else:
                    shd += 1
            elif W_true[i, j] == 0 and W_binary[i, j] == 1:
                # extra edge, but check if it was counted as reversed
                if W_true[j, i] == 0:
                    shd += 1
                    
    # False edge weights > 0.5
    false_edges_high_weight = 0
    for i in range(d):
        for j in range(d):
            if W_true[i, j] == 0 and abs(W_est[i, j]) > 0.5:
                # Check it's not a reversed true edge
                if W_true[j, i] == 0:
                    false_edges_high_weight += 1
                
    # Calculate Precision and Recall
    true_edges = np.sum(W_true)
    pred_edges = np.sum(W_binary)
    correct_edges = 0
    for i in range(d):
        for j in range(d):
            if W_true[i, j] == 1 and W_binary[i, j] == 1:
                correct_edges += 1
                
    precision = correct_edges / pred_edges if pred_edges > 0 else 0
    recall = correct_edges / true_edges if true_edges > 0 else 0
    
    # Calculate h(W) acyclicity
    WW = (W_est * W_est)
    from scipy.linalg import expm
    E = expm(WW)
    h_W = np.trace(E) - d
    
    print("\nBenchmark Results")
    print(f"{'Metric':<30} | {'Value'}")
    print(f"{'Structural Hamming Distance':<30} | {shd}")
    print(f"{'Precision (edge detection)':<30} | {precision:.2f}")
    print(f"{'Recall (edge detection)':<30} | {recall:.2f}")
    print(f"{'Acyclicity constraint h(W)':<30} | {h_W:.2e}")
    
    assert shd <= 3, f"SHD {shd} exceeds tolerance of 3"
    assert false_edges_high_weight == 0, f"Found {false_edges_high_weight} false edges with weight > 0.5"

if __name__ == "__main__":
    test_notears_dag_recovery()
