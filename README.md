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
- **ML**: EconML, XGBoost, scikit-learn, NOTEARS (custom)
- **Explainability**: SHAP (KernelExplainer on CATE)
- **Deployment**: Docker Compose (Postgres, MLflow, API, UI)

## Getting Started

Follow these steps to run the platform locally on your machine using Docker.

**1. Clone the repository**
```bash
git clone https://github.com/prishabh3/casual-pharma-platform.git
cd casual-pharma-platform
```

**2. Build and start the Docker containers**
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
