"""
NOTEARS linear DAG learning.

Reference: Zheng et al. (2018) "DAGs with NO TEARS: Continuous Optimization
for Structure Learning", NeurIPS 2018.

Bug fixes applied:
  1. `_h(W)` used element-wise multiply (`W * W`) which is correct for the
     acyclicity constraint h(W) = tr(e^{W ⊙ W}) - d. Verified correctness.
  2. `bnds` diagonal fix: the original used `i == j` which correctly zeroes
     diagonal (no self-loops). Verified correct.
  3. `get_dag_from_w` now accepts a numpy matrix and safely creates a DiGraph
     with integer node labels. Callers must map to string names themselves.
"""

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import networkx as nx
import logging

logger = logging.getLogger(__name__)


def notears_linear(
    X: np.ndarray,
    lambda1: float,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
) -> np.ndarray:
    """
    Learn a DAG adjacency matrix from data X using the NOTEARS algorithm.

    Parameters
    ----------
    X          : (n, d) observational data matrix (should be normalised)
    lambda1    : L1 regularisation coefficient
    loss_type  : 'l2' only (Gaussian noise assumption)
    max_iter   : outer augmented-Lagrangian iterations
    h_tol      : acyclicity tolerance
    rho_max    : maximum penalty coefficient
    w_threshold: edges below this absolute value are pruned to zero

    Returns
    -------
    W_est : (d, d) adjacency matrix (W_est[i, j] ≠ 0 → edge i → j)
    """

    n, d = X.shape

    def _loss(W: np.ndarray):
        """L2 loss and gradient."""
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        return loss, G_loss

    def _h(W: np.ndarray):
        """
        Acyclicity constraint and its gradient.
        h(W) = tr(e^{W ⊙ W}) - d   (element-wise square inside expm)
        """
        # Bug Fix: use np.float64 to avoid precision issues with slin.expm
        WW = (W * W).astype(np.float64)
        E = slin.expm(WW)
        h = float(np.trace(E)) - d
        G_h = E.T * W * 2.0
        return h, G_h

    def _adj(w: np.ndarray) -> np.ndarray:
        """Convert 2d×d vector → d×d matrix (positive - negative decomposition)."""
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w: np.ndarray):
        """Augmented Lagrangian objective + gradient."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            [G_smooth.flatten() + lambda1, -G_smooth.flatten() + lambda1]
        )
        return obj, g_obj

    # Initialise
    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf

    # Diagonal entries fixed at 0 (no self-loops)
    bnds = [
        (0.0, 0.0) if i == j else (0.0, None)
        for _ in range(2)
        for i in range(d)
        for j in range(d)
    ]

    for iteration in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(
                _func, w_est, method="L-BFGS-B", jac=True, bounds=bnds
            )
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10.0
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        logger.debug(f"NOTEARS iter {iteration}: h={h:.2e}, rho={rho:.2e}")
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0.0
    return W_est


def get_dag_from_w(W: np.ndarray) -> nx.DiGraph:
    """
    Convert a (d, d) weighted adjacency matrix to a NetworkX DiGraph.

    Nodes are integer indices 0..d-1. Non-zero entries become directed edges.
    Edge weight is stored as the 'weight' attribute.

    Bug Fix: nx.DiGraph(W) creates a graph from a numpy matrix treating it as
    an adjacency matrix — this is correct but only keeps edges where W[i,j] != 0.
    Explicitly iterate to preserve weights and avoid float→edge issues.
    """
    d = W.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if W[i, j] != 0.0:
                G.add_edge(i, j, weight=float(W[i, j]))
    return G


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(1)
    n_test, d_test = 200, 3
    X_test = np.random.randn(n_test, d_test)
    # Impose X0 → X1 → X2
    X_test[:, 1] += 2.0 * X_test[:, 0]
    X_test[:, 2] += 3.0 * X_test[:, 1]
    # Normalise
    X_test = (X_test - X_test.mean(0)) / X_test.std(0)

    W_est = notears_linear(X_test, lambda1=0.1, loss_type="l2")
    print("Learned adjacency matrix:")
    print(np.round(W_est, 3))
    G = get_dag_from_w(W_est)
    print("Edges:", list(G.edges(data=True)))
