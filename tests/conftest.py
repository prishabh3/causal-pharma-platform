import pytest
import numpy as np

@pytest.fixture
def sample_patient_dict():
    return {
        "age": 65.0,
        "blood_pressure": 120.0,
        "comorbidities": 2.0
    }

@pytest.fixture
def synthetic_data():
    """Generates a synthetic dataset with X, T, Y (n=500)."""
    rng = np.random.default_rng(42)
    n = 500
    
    age = rng.normal(65, 12, n)
    bp = rng.normal(120, 20, n)
    comorb = rng.poisson(2, n).astype(float)
    
    X = np.column_stack([age, bp, comorb])
    
    logit_p = -2.0 + 0.05*(age-65) - 0.02*(bp-120) + 0.5*comorb
    p_t = np.clip(1.0 / (1.0 + np.exp(-logit_p)), 0.01, 0.99)
    T = rng.binomial(1, p_t)
    
    true_cate = 5.0 + 0.2*(age-65) - 1.5*comorb
    Y0 = 50.0 - 0.3*(age-65) + 0.1*(bp-120) - 2.0*comorb
    Y = Y0 + T * true_cate + rng.normal(0, 5, n)
    
    return X, T, Y
