import numpy as np

class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        """
        LinUCB contextual bandit.
        n_arms: number of possible actions (treatments)
        d: feature dimension
        alpha: exploration parameter
        """
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        
        # Initialize A and b arrays
        # A_a = I_d for each arm a
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        # b_a = 0 for each arm a
        self.b = [np.zeros((self.d, 1)) for _ in range(self.n_arms)]
        
    def select_arm(self, x):
        """
        Select best arm based on context x (shape: (d,))
        """
        x = x.reshape(-1, 1)
        p = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            
            # Confidence bound
            cb = self.alpha * np.sqrt(x.T @ A_inv @ x)[0, 0]
            
            # Expected reward
            expected_reward = (theta_a.T @ x)[0, 0]
            p[a] = expected_reward + cb
            
        return np.argmax(p)
        
    def update(self, chosen_arm, x, reward):
        """
        Update the model after observing reward.
        """
        x = x.reshape(-1, 1)
        self.A[chosen_arm] += x @ x.T
        self.b[chosen_arm] += reward * x

if __name__ == '__main__':
    # Test LinUCB
    bandit = LinUCB(n_arms=2, d=3, alpha=1.0)
    context = np.array([1.0, 0.5, -0.2])
    arm = bandit.select_arm(context)
    bandit.update(arm, context, reward=1.0)
    print("Selected arm:", arm)
