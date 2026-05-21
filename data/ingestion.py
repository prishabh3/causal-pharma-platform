"""
Data ingestion pipeline for causal inference datasets.

Supports:
  - LaLonde (1986) observational study dataset
  - Synthetic MIMIC-III-inspired EHR data

Note: Dask was removed due to a compatibility crash (dask 2024.2.0 +
pandas 2.2.0 + Python 3.11). All ingestion now uses Pandas directly.
The causal ML engine operates on NumPy arrays so this is transparent.
"""

import os
import urllib.request
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve the data/raw directory relative to THIS file — never relative to CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW_DIR = os.path.join(_THIS_DIR, "raw")

# Multiple mirrors for the LaLonde dataset
_LALONDE_URLS = [
    "https://raw.githubusercontent.com/gsbDBI/ds-blog/master/posts/lalonde/lalonde.csv",
    "https://raw.githubusercontent.com/gsbDBI/ExperimentData/master/lalonde/lalonde.csv",
]


def load_lalonde() -> pd.DataFrame:
    """
    Load the LaLonde (1986) observational dataset.
    Dataset is cached locally on first download.

    Returns
    -------
    pd.DataFrame
    """
    os.makedirs(_RAW_DIR, exist_ok=True)
    file_path = os.path.join(_RAW_DIR, "lalonde.csv")

    if not os.path.exists(file_path):
        downloaded = False
        for url in _LALONDE_URLS:
            try:
                logger.info(f"Downloading LaLonde dataset from {url} ...")
                urllib.request.urlretrieve(url, file_path)
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
        if not downloaded:
            raise RuntimeError(
                "Could not download the LaLonde dataset. "
                "Please place lalonde.csv in data/raw/ manually."
            )

    pdf = pd.read_csv(file_path)
    if "Unnamed: 0" in pdf.columns:
        pdf = pdf.drop(columns=["Unnamed: 0"])

    logger.info(f"LaLonde dataset loaded: {pdf.shape}")
    return pdf


def generate_mimic_synthetic(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic EHR data inspired by MIMIC-III.

    Causal structure:
        Age, BP, Comorbidities → Treatment (propensity model)
        Age, BP, Comorbidities + Treatment × CATE → Outcome

    Propensity scores are clipped to [0.01, 0.99] to ensure overlap
    and prevent degenerate IPW weights in doubly-robust estimators.

    Parameters
    ----------
    n_samples : int

    Returns
    -------
    pd.DataFrame with columns:
        age, blood_pressure, comorbidities, treatment, outcome, true_cate
    """
    rng = np.random.default_rng(42)

    # Confounders
    age = rng.normal(65, 12, n_samples)
    bp = rng.normal(120, 20, n_samples)
    comorbidities = rng.poisson(2, n_samples).astype(float)

    # Propensity model
    logit_p = -2.0 + 0.05 * (age - 65) - 0.02 * (bp - 120) + 0.5 * comorbidities
    p_t = np.clip(1.0 / (1.0 + np.exp(-logit_p)), 0.01, 0.99)
    T = rng.binomial(1, p_t).astype(int)

    # Heterogeneous treatment effect
    true_cate = 5.0 + 0.2 * (age - 65) - 1.5 * comorbidities

    # Outcome
    Y0 = 50.0 - 0.3 * (age - 65) + 0.1 * (bp - 120) - 2.0 * comorbidities
    Y = Y0 + T * true_cate + rng.normal(0, 5, n_samples)

    pdf = pd.DataFrame({
        "age": age,
        "blood_pressure": bp,
        "comorbidities": comorbidities,
        "treatment": T,
        "outcome": Y,
        "true_cate": true_cate,
    })

    logger.info(f"Generated synthetic MIMIC dataset: {pdf.shape}")
    return pdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_mimic_synthetic(n_samples=500)
    print(df.describe())
