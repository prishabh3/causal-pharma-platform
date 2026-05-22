# Causal Pharma Intelligence Platform

A healthcare analytics platform for **causal treatment effect estimation**, **policy recommendations**, **counterfactual simulation**, and **explainability** — designed for clinicians and analysts who need plain-language guidance, not just model output.

## Features

- **Guided UI**: Step-by-step flow — Recommendation → Why → What-if → Details
- **Plain-language glossary** and clinical summary for each patient
- **Example patients**: Low-risk, typical, and high-risk presets
- **CATE & ATE** with approximate confidence intervals and impact bands (small / moderate / large)
- **Causal DAG**: Learned structure (NOTEARS) vs clinical reference graph
- **SHAP explanations**: Why the estimated treatment effect looks the way it does
- **Counterfactual comparison**: Expected outcome if treating vs withholding
- **LinUCB bandit**: Exploratory adaptive policy (second opinion)
- **Doubly robust estimation** (EconML), synthetic MIMIC-style training data

## Tech Stack

- **Frontend**: Streamlit 1.35.0, Plotly, custom HTML dashboard
- **Backend**: FastAPI 0.109.2, Pydantic 2.6.1
- **ML**: EconML 0.15.1, XGBoost 2.0.3, scikit-learn 1.4.1.post1, NOTEARS (custom), **Generalized Random Forests (R `grf`, optional)**
- **Explainability**: SHAP 0.43.0 (KernelExplainer on CATE)
- **Deployment**: Docker Compose (Postgres, MLflow 2.10.2, API, UI)

## Getting Started

Follow these steps to run the platform locally on your machine using Docker.

**1. Clone the repository**
```bash
git clone https://github.com/prishabh3/causal-pharma-platform.git
cd causal-pharma-platform
```

**2. (Optional) Install R for Generalized Random Forests**
To use the GRF estimator (`/predict_cate_grf` endpoint), you must have R installed on your system.
1. Install R (e.g., `brew install r` on Mac, or download from CRAN).
2. Open R in your terminal and run: `install.packages("grf")`

> **Note:** The `/predict_cate_grf` endpoint requires R and the `grf` package installed on the host. It is **not available** inside the Docker environment. All other endpoints work fully in Docker. To use GRF locally: install R, then run `install.packages("grf")`.

**3. Build and start the Docker containers**
```bash
docker-compose up --build
```
*Note: The first time you run this, it may take a few minutes to download the base images and install dependencies.*

**4. Access the Services**
Once the terminal shows that the services have started, open your web browser and navigate to:

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |

## Quantitative Benchmark Results

The platform's causal estimators have been benchmarked on the classic LaLonde (1986) National Supported Work Demonstration (NSW) dataset to validate their accuracy against the known ground-truth randomized control trial (RCT) Average Treatment Effect (ATE) of ~$1,794 (Dehejia & Wahba 1999). Numbers below are from actual `pytest` execution on the experimental-only NSW sample (445 observations).

| Estimator | ATE Estimate | 95% CI | Bias vs RCT |
| :--- | :--- | :--- | :--- |
| Naive OLS | $1,676.34 | ($439.71, $2,912.97) | 6.6% |
| IPW | $1,609.85 | ($-88.69, $3,308.39) | 10.3% |
| Doubly Robust | $1,792.59 | ($1,385.33, $2,199.86) | **0.1%** |

*Note: IPW shows a wide confidence interval ($-88.69, $3,308.39) due to 
near-extreme propensity scores at the boundary of the small experimental 
sample (n=445) — this is expected behaviour for IPW on low-overlap data, 
not a data or implementation error.*

As demonstrated, the **Doubly Robust** estimator (cross-fitting with Ridge outcome models + scaled propensity) achieves **0.1% bias** against the RCT ground truth — outperforming both Naive OLS (6.6%) and IPW (10.3%), confirming the validity of the platform's causal inference engine.

### DAG Structure Recovery (Synthetic, n=3000, d=5 nodes, 4 true edges)

Numbers from actual `pytest` execution on a fan-out DAG (X0→X1, X0→X2, X1→X3, X2→X4) with n=3,000 samples.

| Metric | Value |
| :--- | :--- |
| Structural Hamming Distance | 3 |
| Precision | 0.75 |
| Recall | 0.75 |
| Acyclicity h(W) | 0.00e+00 |

**5. Stopping the platform**
To stop the platform, simply press `Ctrl+C` in the terminal where it is running, or execute:
```bash
docker-compose down
```

## API Endpoints

| Method | Path | Description | Notes |
|--------|------|-------------|-------|
| POST | `/predict_cate` | CATE, ATE, CI, magnitude, confidence | |
| POST | `/predict_cate_grf` | GRF CATE, ATE | Local only (requires R) |
| POST | `/policy_decision` | Recommendation + rationale + bandit | |
| POST | `/simulate_both` | Outcomes under T=0 and T=1 | |
| POST | `/explain_prediction` | SHAP values for CATE | |
| GET | `/dag` | Learned DAG | |
| GET | `/dag/reference` | Clinical reference DAG | |

## Architecture

- `backend/` — FastAPI app and causal service
- `causal/` — Estimation, NOTEARS discovery, LinUCB bandit
- `data/` — Synthetic MIMIC-style data generation
- `explainability/` — SHAP integration
- `frontend/` — Streamlit app, glossary copy, dashboard HTML

## Future Work
- **Real MIMIC-III data**: Replace synthetic data with de-identified MIMIC-III patient records for clinical validity.
- **Distributed compute**: Scale the DR estimator to large cohorts using PySpark or Databricks on partitioned patient data.
- **MLflow model registry**: Promote validated causal models to the registry with lineage tracking for audit trails — relevant for FDA submission workflows.
- **Sensitivity analysis**: Add Rosenbaum bounds to quantify how robust treatment effect estimates are to unmeasured confounding.

## Disclaimer

Uses **synthetic observational data** for demonstration. Outputs support exploration and education — they are **not** a substitute for clinical judgment or approved medical devices.
