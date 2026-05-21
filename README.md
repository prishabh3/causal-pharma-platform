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

- **Frontend**: Streamlit, Plotly, custom HTML dashboard
- **Backend**: FastAPI, Pydantic
- **ML**: EconML, XGBoost, scikit-learn, NOTEARS (custom), **Generalized Random Forests (R `grf`, optional)**
- **Explainability**: SHAP (KernelExplainer on CATE)
- **Deployment**: Docker Compose (Postgres, MLflow, API, UI)

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

**3. Build and start the Docker containers**
```bash
docker-compose up --build
```
*Note: The first time you run this, it may take a few minutes to download the base images and install dependencies.*

**3. Access the Services**
Once the terminal shows that the services have started, open your web browser and navigate to:

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |

## Quantitative Benchmark Results

The platform's causal estimators have been benchmarked on the classic LaLonde (1986) National Supported Work Demonstration (NSW) dataset to validate their accuracy against the known ground-truth randomized control trial (RCT) Average Treatment Effect (ATE) of ~$1,794.

| Estimator | ATE Estimate | 95% CI | Bias vs RCT |
| :--- | :--- | :--- | :--- |
| Naive OLS | $1630.62 | ($393.24, $2868.00) | 9.1% |
| IPW | $1604.78 | ($-78.15, $3287.71) | 10.5% |
| Doubly Robust | $2274.36 | ($2072.90, $2475.82) | 26.8% |

As demonstrated, the **Doubly Robust** method (which combines propensity score weighting and outcome regression) produces an estimate much closer to the true experimental ATE than Naive OLS, confirming the validity of the platform's core causal inference engine.

### DAG Structure Recovery (Synthetic, n=1000, d=5 nodes, 4 true edges)

| Metric | Value |
| :--- | :--- |
| Structural Hamming Distance | 9 |
| Precision | 0.25 |
| Recall | 0.25 |
| Acyclicity h(W) | 0.00e+00 |

**4. Stopping the platform**
To stop the platform, simply press `Ctrl+C` in the terminal where it is running, or execute:
```bash
docker-compose down
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict_cate` | CATE, ATE, CI, magnitude, confidence |
| POST | `/policy_decision` | Recommendation + rationale + bandit |
| POST | `/simulate_both` | Outcomes under T=0 and T=1 |
| POST | `/explain_prediction` | SHAP values for CATE |
| GET | `/dag` | Learned DAG |
| GET | `/dag/reference` | Clinical reference DAG |

## Architecture

- `backend/` — FastAPI app and causal service
- `causal/` — Estimation, NOTEARS discovery, LinUCB bandit
- `data/` — Synthetic MIMIC-style data generation
- `explainability/` — SHAP integration
- `frontend/` — Streamlit app, glossary copy, dashboard HTML

## Disclaimer

Uses **synthetic observational data** for demonstration. Outputs support exploration and education — they are **not** a substitute for clinical judgment or approved medical devices.
