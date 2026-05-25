# Causal Pharma Intelligence Platform

A tool for clinicians and analysts that estimates, for a given patient, how much a specific treatment is expected to help (or harm) them — not just on average across a population, but for that individual. It explains its reasoning, simulates counterfactual scenarios, and provides a second-opinion recommendation from a contextual bandit policy.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Architecture](#architecture)
3. [How the Pieces Fit Together](#how-the-pieces-fit-together)
4. [Quick Start](#quick-start)
5. [Feature Walkthrough](#feature-walkthrough)
6. [API Reference](#api-reference)
7. [Database Layout](#database-layout)
8. [Key Design Decisions](#key-design-decisions)
9. [Testing](#testing)
10. [Project Structure](#project-structure)
11. [Common Issues](#common-issues)

---

## Tech Stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| Frontend | Streamlit | 1.35.0 | Browser UI — step-by-step patient analysis flow |
| Frontend | Plotly | (via Streamlit) | Interactive charts for SHAP and DAG visualizations |
| API | FastAPI | 0.109.2 | REST API between UI and ML engine |
| API | Uvicorn | 0.27.1 | ASGI server for FastAPI |
| API | Pydantic | 2.6.1 | Request/response validation and schema generation |
| ML — Estimation | EconML | 0.15.1 | LinearDML, CausalForestDML, TLearner, XLearner |
| ML — Estimation | scikit-learn | 1.4.1 | Ridge, LogisticRegression, KFold, StandardScaler |
| ML — Estimation | XGBoost | 2.0.3 | Outcome and nuisance models inside R-Learner / meta-learners |
| ML — DAG | NetworkX | 3.3 | Directed graph representation of learned causal structure |
| ML — GRF (optional) | R `grf` package | — | Generalized Random Forest CATE estimator (local only, needs R) |
| ML — GRF (optional) | rpy2 | — | Python ↔ R bridge for GRF estimator |
| Explainability | SHAP | 0.43.0 | KernelExplainer on `model.effect()` for feature attribution |
| Data | DoWhy | 0.13 | Ships the LaLonde NSW experimental dataset for benchmarking |
| Data | NumPy / Pandas / SciPy | 1.26.4 / 2.2.0 / 1.12.0 | Numerical computation and data handling |
| Experiment tracking | MLflow | 2.10.2 | Run metadata storage (backed by Postgres) |
| Database | PostgreSQL | 15 | MLflow backend store |
| Infrastructure | Docker Compose | — | Orchestrates all four services |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's Browser                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Streamlit Frontend  :8501                       │
│  Patient input form → calls backend endpoints → renders results │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST/GET
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend  :8000                         │
│                                                                  │
│  routes.py          models.py (Pydantic schemas)                │
│       │                                                          │
│       ▼                                                          │
│  CausalService  (singleton, initialised at startup)             │
│  ├── CausalEstimator  (CustomDRLearner by default)              │
│  ├── NOTEARS linear DAG discovery                               │
│  ├── LinUCB contextual bandit (warm-started on 500 samples)     │
│  └── CausalExplainer  (SHAP KernelExplainer, lazy-init)        │
└────────────┬────────────────────────────┬────────────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐    ┌───────────────────────────────────┐
│  PostgreSQL  :5432      │    │  MLflow Server  :5001             │
│  (MLflow metadata only) │◄───│  --backend-store-uri postgres://… │
└────────────────────────┘    └───────────────────────────────────┘
```

**Service startup order:** PostgreSQL starts first. MLflow and the backend both wait for Postgres to pass its healthcheck. The frontend waits for the backend's healthcheck (`GET /` returns 200). The backend's healthcheck has a 90-second start period because `CausalService.__init__()` trains models synchronously on boot.

---

## How the Pieces Fit Together

**At startup**, the backend instantiates a single `CausalService` object. This object immediately generates 2,000 synthetic patient records (or processes real MIMIC-III files if `DATA_SOURCE=mimic`) and fits three things: the doubly-robust CATE estimator, the NOTEARS causal DAG, and warms up the LinUCB bandit on the first 500 training rows. The SHAP explainer is lazy-initialized on the first `/explain_prediction` call because it is slow to construct.

**When a user submits a patient**, the Streamlit UI sends a POST request to the relevant endpoint with a JSON body containing one or more `PatientFeatures` objects (age, blood pressure, comorbidities). The backend's `_prepare_features()` method starts with a row of training-set medians for all model columns, then overwrites whichever columns the UI actually provided. This lets the three-column UI work against a 15-column MIMIC model without breaking anything.

**For CATE estimation**, `CausalEstimator.estimate_cate(X)` calls `model.effect(X)` — the `effect` method is the EconML standard interface. The result is a per-patient float: positive means the treatment is expected to help, negative means harm. The confidence interval is computed as `cate ± 1.96 * training_cate_std / sqrt(2000)` — a heuristic, not a bootstrap. Magnitude is bucketed as small (<2 units), moderate (2–5), or large (≥5).

**For policy decisions**, the recommendation is simply `1 if cate > 0 else 0`. Alongside this, the LinUCB bandit selects an arm (0 or 1) based on the upper confidence bound of its learned reward model for each arm. The bandit recommendation can differ from the CATE-based recommendation; that disagreement is the point — it provides an exploratory second opinion.

**For counterfactual simulation**, outcomes under T=0 and T=1 are computed as `Y_mean + T * cate`. This uses the training outcome mean as a baseline and adds or subtracts the individual CATE.

**For SHAP explanations**, `CausalExplainer` wraps `model.effect` as a black box and runs SHAP's `KernelExplainer` against a 150-sample background (summarized to 10 k-means clusters). Each SHAP value tells you how much a particular feature (age, blood pressure, comorbidities) pushed the estimated treatment effect up or down relative to the baseline.

**For DAG discovery**, NOTEARS runs at startup on the full feature matrix including treatment and outcome. The result is compared in the UI against a manually specified clinical reference DAG to highlight where the learned structure agrees or disagrees with domain knowledge.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- ~4 GB free RAM (model training + MLflow)

### Steps

**1. Clone and configure environment variables**

```bash
git clone <repo-url>
cd causal-pharma-intelligence-platform
cp .env.example .env
```

Open `.env` and set a real password:

```
POSTGRES_USER=causal_user
POSTGRES_PASSWORD=change_me_now
POSTGRES_DB=causal_db
DATABASE_URL=postgresql://causal_user:change_me_now@db:5432/causal_db
MLFLOW_TRACKING_URI=http://mlflow:5000
BACKEND_URL=http://backend:8000
```

**2. Build and start all services**

```bash
docker-compose up --build
```

The first build takes 3–5 minutes (downloading images, installing Python packages). On subsequent runs it is faster due to Docker layer caching.

**3. Wait for the backend to finish training**

Watch the backend logs. You will see:

```
INFO  Fitting initial causal model on synthetic data...
INFO  Model fitting complete.
INFO  GRF model fitting complete.   ← or a warning if R is not available
```

This takes 30–90 seconds. Until it completes, API calls will return zeros.

**4. Open the services**

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI interactive docs | http://localhost:8000/docs |
| MLflow experiment tracker | http://localhost:5001 |

**5. Stop**

```bash
docker-compose down          # keeps postgres_data volume
docker-compose down -v       # also deletes the database volume
```

### Running locally without Docker

```bash
# Install production deps
pip install -r requirements.txt

# Install dev deps (pytest)
pip install -r requirements-dev.txt

# Start the API (trains models at startup, takes ~60s)
uvicorn backend.main:app --reload

# In a second terminal, start the UI
streamlit run frontend/app.py
```

Set environment variables for local dev:

```bash
export DATABASE_URL=postgresql://causal_user:password@localhost:5432/causal_db
export MLFLOW_TRACKING_URI=http://localhost:5001
export BACKEND_URL=http://localhost:8000
export DATA_SOURCE=synthetic   # or "mimic" if you have MIMIC-III files
```

### Switching to MIMIC-III data

Place the following raw CSV files in `data/mimic_raw/`:

```
ADMISSIONS.csv
PATIENTS.csv
PRESCRIPTIONS.csv
DIAGNOSES_ICD.csv
LABEVENTS.csv
```

Then set `DATA_SOURCE=mimic` in your environment (or `.env`) and restart the backend. The pipeline extracts: patient age, gender, admission type, top-6 ICD-9 codes, five lab values (creatinine, lactate, WBC, glucose, bilirubin), treatment (vasopressor administration), and outcome (in-hospital mortality).

### Using the GRF estimator (optional)

The `/predict_cate_grf` endpoint requires R installed on the host and is not available inside Docker.

```bash
# macOS
brew install r

# Then in an R session:
install.packages("grf")

# Then install the rpy2 Python bridge
pip install rpy2
```

---

## Feature Walkthrough

This describes the main user journey through the Streamlit UI.

**Step 1 — Enter patient data**

The left panel contains three sliders: age, blood pressure (systolic mmHg), and number of comorbidities. These map directly to the `PatientFeatures` schema sent to the API. You can also choose one of three preset patients (low-risk, typical, high-risk) to populate the sliders.

**Step 2 — Get a recommendation**

Click the "Analyze" button. The UI sends a POST to `/policy_decision` and displays:
- The primary recommendation: **Treat** or **Withhold**
- The rationale string (e.g., "Estimated benefit is +3.21 outcome units. Treatment is expected to help this patient.")
- The bandit's independent recommendation and whether it agrees

**Step 3 — Understand why (SHAP)**

The "Why?" tab sends a POST to `/explain_prediction`. It shows a bar chart of SHAP values — the contribution of each feature to the estimated treatment effect. A positive bar means that feature is pushing the model toward a higher treatment benefit; negative means toward lower benefit.

**Step 4 — Simulate counterfactuals**

The "What-if?" tab sends a POST to `/simulate_both`. It shows two numbers side by side: expected patient outcome if treated (T=1) versus if not treated (T=0), along with the implied treatment effect.

**Step 5 — Inspect the causal graph**

The "DAG" tab fetches `/dag` (learned structure) and `/dag/reference` (clinical prior). Both are rendered as interactive network graphs using Plotly. Discrepancies between them are where the model disagrees with clinical prior knowledge.

---

## API Reference

All endpoints are served at `http://localhost:8000`. Interactive documentation is at `/docs`. No authentication is required.

### Input schema

Every POST endpoint accepts a `features` array of patient objects:

```json
{
  "features": [
    {
      "age": 65.0,
      "blood_pressure": 140.0,
      "comorbidities": 3
    }
  ]
}
```

Field constraints: `age` ∈ [0, 120], `blood_pressure` ∈ [50, 250], `comorbidities` ∈ [0, 20]. All three are required.

---

### `POST /predict_cate`

Estimates the Conditional Average Treatment Effect (CATE) for each patient using the doubly-robust estimator.

| Response field | Description |
|---|---|
| `cate_estimates` | Per-patient CATE: positive = treatment helps, negative = treatment harms |
| `ate_estimate` | Average CATE across submitted patients |
| `ci_lower`, `ci_upper` | Approximate 95% confidence interval (see [Key Design Decisions](#key-design-decisions)) |
| `magnitude` | `"small"`, `"moderate"`, or `"large"` bucket for the absolute CATE |
| `confidence_pct` | Integer 55–95: heuristic confidence score derived from CI width vs CATE size |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict_cate \
  -H "Content-Type: application/json" \
  -d '{"features": [{"age": 72.0, "blood_pressure": 155.0, "comorbidities": 4}]}'
```

**Example response:**
```json
{
  "cate_estimates": [6.41],
  "ate_estimate": 6.41,
  "ci_lower": [6.16],
  "ci_upper": [6.66],
  "magnitude": ["large"],
  "confidence_pct": [91]
}
```

---

### `POST /predict_cate_grf`

Same response schema as `/predict_cate` but uses the R `grf::causal_forest()` estimator instead of the Python doubly-robust model. Raises HTTP 500 if R and rpy2 are not installed.

---

### `POST /policy_decision`

Returns a treatment recommendation per patient.

| Response field | Description |
|---|---|
| `recommended_treatment` | `1` = treat, `0` = withhold. Derived from sign of CATE. |
| `rationale` | Plain-English string explaining the recommendation, including the CATE estimate |
| `bandit_recommendation` | Arm selected by the LinUCB bandit (independent of CATE sign) |

**Example request:**
```bash
curl -X POST http://localhost:8000/policy_decision \
  -H "Content-Type: application/json" \
  -d '{"features": [{"age": 65.0, "blood_pressure": 120.0, "comorbidities": 2}]}'
```

**Example response:**
```json
{
  "recommended_treatment": [1],
  "rationale": ["Estimated benefit is positive (+5.03 outcome units). Treatment is expected to help this patient."],
  "bandit_recommendation": [1]
}
```

---

### `POST /simulate_both`

Returns expected patient outcome under both treatment arms.

| Response field | Description |
|---|---|
| `outcome_if_withhold` | Expected outcome if T=0. Computed as `Y_mean + 0 * cate`. |
| `outcome_if_treat` | Expected outcome if T=1. Computed as `Y_mean + 1 * cate`. |
| `treatment_effect` | Difference: same as CATE from `/predict_cate`. |

**Example request:**
```bash
curl -X POST http://localhost:8000/simulate_both \
  -H "Content-Type: application/json" \
  -d '{"features": [{"age": 65.0, "blood_pressure": 120.0, "comorbidities": 2}]}'
```

---

### `POST /simulate_intervention`

Returns expected outcome for a single specified treatment value.

**Additional required field:** `"treatment_value": 0` or `"treatment_value": 1`

**Example request:**
```bash
curl -X POST http://localhost:8000/simulate_intervention \
  -H "Content-Type: application/json" \
  -d '{"features": [{"age": 65.0, "blood_pressure": 120.0, "comorbidities": 2}], "treatment_value": 1}'
```

---

### `POST /explain_prediction`

Returns SHAP feature attributions for the CATE prediction.

| Response field | Description |
|---|---|
| `feature_names` | Internal column names (e.g. `"age"`, `"blood_pressure"`, `"comorbidities"`) |
| `feature_labels` | Human-readable labels (e.g. `"Age"`, `"Blood Pressure"`, `"Comorbidities"`) |
| `shap_values` | List of lists — shape `(n_patients, n_features)`. Each value is the contribution of that feature to the CATE prediction relative to the base value. |
| `base_value` | The SHAP explainer's expected CATE (average CATE over the background dataset). |

**Example request:**
```bash
curl -X POST http://localhost:8000/explain_prediction \
  -H "Content-Type: application/json" \
  -d '{"features": [{"age": 80.0, "blood_pressure": 160.0, "comorbidities": 5}]}'
```

---

### `GET /dag`

Returns the causal graph learned by NOTEARS from training data.

**Example response:**
```json
{
  "nodes": ["age", "blood_pressure", "comorbidities", "treatment", "outcome"],
  "edges": [
    {"source": "age", "target": "outcome"},
    {"source": "comorbidities", "target": "treatment"}
  ]
}
```

---

### `GET /dag/reference`

Returns the manually specified clinical prior DAG (not learned from data). In synthetic mode this is: age, blood pressure, and comorbidities each cause treatment and outcome; treatment causes outcome.

---

### `GET /health`

Returns `{"status": "ok", "version": "1.1.0", "endpoints": [...]}`. Used as the Docker healthcheck.

---

## Database Layout

PostgreSQL is used exclusively by MLflow. The API and causal engine do not write patient data to any database — everything is computed in memory from models trained at startup.

| Table | Owner | What it stores |
|---|---|---|
| `experiments` | MLflow | One row per MLflow experiment name |
| `runs` | MLflow | One row per training run with metadata (start time, status, params) |
| `params` | MLflow | Key-value pairs logged per run (model hyperparameters) |
| `metrics` | MLflow | Numeric metrics logged per run (ATE, training loss, etc.) |
| `tags` | MLflow | String tags per run |
| `artifacts` | MLflow | Paths to artifact files stored in `/mlflow/artifacts` inside the container |

MLflow creates and manages these tables automatically on first startup via Alembic migrations.

**There are no patient tables.** The synthetic training data is regenerated from a fixed random seed (`numpy.random.default_rng(42)`) every time the backend starts.

---

## Key Design Decisions

### 1. Doubly Robust estimation with cross-fitting

The default estimator is `CustomDRLearner` in [causal/estimation.py](causal/estimation.py). It is a cross-fitting doubly-robust (DR) estimator rather than a naive IPW or a single-model outcome regression.

DR has a "double protection" property: it gives a consistent estimate of the ATE if *either* the propensity model (P(T=1|X)) *or* the outcome models (E[Y|T=t, X]) are correctly specified — not both. Cross-fitting (5-fold KFold) means the outcome models are trained on held-out folds, which prevents the overfitting bias that would otherwise inflate CATE estimates.

The estimator computes augmented IPW scores:

```
dr_i = (mu1 - mu0) + T*(Y - mu1)/e - (1-T)*(Y - mu0)/(1-e)
```

These are then regressed on X using a Ridge model to produce individual CATE predictions.

Propensity scores are clipped to [0.05, 0.95] to prevent extreme IPW weights, which would cause variance explosion on patients who were very likely or very unlikely to receive treatment.

**Validation:** On the LaLonde NSW experimental dataset (445 patients, known RCT ATE ≈ $1,794), the DR estimator achieves 0.1% bias vs 6.6% for naive OLS and 10.3% for IPW.

---

### 2. Confidence intervals are heuristic, not bootstrap

The CI formula in `predict_cate_with_meta` is:

```python
margin = max(1.96 * self._cate_std / (2000 ** 0.5), 0.25)
ci_lower = cate - margin
ci_upper = cate + margin
```

`self._cate_std` is the standard deviation of CATE predictions on the training set. This is a constant margin applied to every patient — it does not reflect individual-level uncertainty. A proper CI would require bootstrapping or conformal prediction sets. The 0.25 floor prevents the margin from collapsing to zero for near-zero CATEs.

---

### 3. Feature alignment via training medians

When the UI sends three features but the model was trained on 15 (MIMIC mode), `_prepare_features()` fills missing columns with the training-set median:

```python
X = np.tile(self._X_medians, (len(features_df), 1))
for i, col in enumerate(self._X_cols):
    if col in features_df.columns:
        X[:, i] = features_df[col].values
```

This allows a simple three-feature UI to query a richer model without breaking the estimator. The median-filling is intentional: medians are robust to outliers and represent a "typical" patient on the unobserved dimensions.

---

### 4. LinUCB contextual bandit as a second opinion

`LinUCB` in [causal/bandits.py](causal/bandits.py) maintains a separate reward model per arm (treat / withhold). For each arm `a` and context `x`:

```
UCB_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
```

The second term is the uncertainty bonus: it increases for contexts that are far from previously seen data. `alpha=1.0` balances exploration vs exploitation.

The bandit is warm-started on 500 synthetic training trajectories at startup. Its recommendation can differ from the CATE-based recommendation because it models *reward* (a scaled outcome) rather than *effect size*. The disagreement is informative: if CATE says treat but the bandit is uncertain, that signals a patient in a region with limited evidence.

---

### 5. NOTEARS for causal structure learning

`notears_linear` in [causal/discovery.py](causal/discovery.py) reformulates DAG learning as a continuous optimization problem. The key idea is the acyclicity constraint:

```
h(W) = tr(e^{W ⊙ W}) - d = 0  iff W is a DAG
```

Where `⊙` is elementwise multiplication and `e^{}` is the matrix exponential. NOTEARS minimizes the L2 reconstruction loss plus an L1 penalty (lambda1=0.1) subject to `h(W) → 0` using an augmented Lagrangian method.

The learned adjacency matrix is thresholded at 0.3: edges with absolute weight below this are pruned. Node indices in the adjacency matrix map to the ordered column list `[...feature cols..., "treatment", "outcome"]`.

**Validation:** On a 5-node fan-out DAG with 3,000 samples, NOTEARS achieves SHD=3, precision=0.75, recall=0.75.

---

### 6. SHAP on `model.effect()` directly

Most SHAP tutorials explain output model predictions. Here, `CausalExplainer` wraps `model.effect` — the function that maps X → CATE — not the outcome model. This means SHAP values answer "what is driving the *treatment effect estimate* up or down" rather than "what is driving the *outcome* up or down." The distinction matters: high age might predict worse outcomes regardless of treatment, but a SHAP value on CATE tells you whether age is associated with *greater* or *lesser* treatment benefit.

`KernelExplainer` is model-agnostic and works on any function. The background dataset is reduced to 10 k-means cluster centers for speed. `nsamples=80` per call controls the tradeoff between SHAP accuracy and latency.

---

## Testing

### Run the full test suite

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Run a specific test file

```bash
pytest tests/test_estimators.py -v -s   # -s shows benchmark table output
pytest tests/test_dag.py -v -s          # -s shows structure recovery metrics
```

### What each test covers

**`tests/test_api.py`** — FastAPI integration tests using `TestClient` (no network required).

| Test | What it checks |
|---|---|
| `test_predict_cate_valid` | `/predict_cate` returns 200 with `cate_estimates` and `ci_lower` keys |
| `test_policy_decision_valid` | `/policy_decision` returns 200 with `recommended_treatment` key |
| `test_get_dag` | `/dag` returns 200 with `nodes` and `edges` keys |
| `test_invalid_payload` | `/predict_cate` with missing fields returns 422 Unprocessable Entity |

**`tests/test_estimators.py`** — Benchmark on the LaLonde NSW experimental dataset (real observational data with known RCT ground truth).

| Assertion | Threshold |
|---|---|
| DR ATE bias vs RCT ATE ($1,794) | < 40% |
| DR bias < OLS bias | must hold |
| IPW lower CI bound | > -$3,000 (sanity check for extreme weights) |

**`tests/test_dag.py`** — NOTEARS structure recovery on a synthetic 5-node fan-out DAG (3,000 samples, 4 true edges).

| Assertion | Threshold |
|---|---|
| Structural Hamming Distance | ≤ 3 |
| Precision | ≥ 0.5 |
| Recall | ≥ 0.5 |

### Coverage report

```bash
pytest tests/ --cov=backend --cov=causal --cov=data --cov=explainability --cov-report=term-missing
```

---

## Project Structure

```
.
├── backend/
│   ├── api/
│   │   └── routes.py           # FastAPI route handlers — thin wrappers, delegate to CausalService
│   ├── services/
│   │   └── causal_service.py   # All business logic: model training, inference, DAG, bandit
│   ├── models.py               # Pydantic schemas for every request and response
│   ├── main.py                 # FastAPI app, CORS config, /health and / endpoints
│   ├── Dockerfile              # python:3.11-slim, sets PYTHONPATH=/app
│   └── requirements.txt        # Production deps (subset of root requirements.txt)
│
├── causal/
│   ├── estimation.py           # CausalEstimator wrapper + CustomDRLearner implementation
│   ├── discovery.py            # NOTEARS linear DAG learning algorithm
│   ├── bandits.py              # LinUCB contextual bandit
│   └── grf_estimator.py        # rpy2 bridge to R's grf::causal_forest()
│
├── data/
│   ├── ingestion.py            # LaLonde dataset downloader + synthetic MIMIC generator
│   ├── mimic_pipeline.py       # Real MIMIC-III CSV → parquet cohort pipeline
│   └── raw/                    # LaLonde CSV cached here on first test run
│
├── explainability/
│   └── shap_explainer.py       # CausalExplainer: KernelExplainer on model.effect()
│
├── mlflow/
│   └── Dockerfile              # MLflow server image
│
├── tests/
│   ├── conftest.py             # Shared pytest fixtures (sample patient dict, synthetic data)
│   ├── test_api.py             # FastAPI endpoint integration tests
│   ├── test_dag.py             # NOTEARS structure recovery benchmark
│   └── test_estimators.py      # DR/OLS/IPW benchmark on LaLonde NSW data
│
├── .streamlit/
│   └── config.toml             # Streamlit theme settings
│
├── docker-compose.yml          # Wires db → mlflow → backend → frontend
├── .env.example                # Template — copy to .env and fill in passwords
├── requirements.txt            # All production Python deps with pinned versions
└── requirements-dev.txt        # pytest and pytest-cov
```

---

## Common Issues

**The backend takes 2+ minutes to become healthy**

The `CausalService.__init__()` trains the DR estimator, runs NOTEARS, and warms up the bandit synchronously at startup. On a slow machine with 2,000 training samples this takes up to 90 seconds. The Docker healthcheck has a 90-second `start_period` to account for this. Watch the backend logs for `"Model fitting complete."` before assuming something is wrong.

---

**`/predict_cate_grf` returns HTTP 500**

This endpoint requires R and the `grf` R package installed on the same machine running the backend — not inside Docker. The Docker image does not include R. Install R, run `install.packages("grf")` in an R session, install `rpy2` in Python, and run the backend locally (not via Docker).

---

**The Streamlit UI cannot connect to the backend**

Check that `BACKEND_URL` is set to `http://backend:8000` (the Docker service name), not `http://localhost:8000`. Inside the Docker network, services reach each other by service name. If running everything locally without Docker, set `BACKEND_URL=http://localhost:8000`.

---

**All API responses return zeros**

The service returns zeros when model fitting failed during startup. Check the backend container logs for an exception. The most common cause is a Postgres connection failure (the DB healthcheck sometimes takes longer than expected on first run). Restarting the backend container usually fixes it: `docker-compose restart backend`.

---

**SHAP explanation takes 30+ seconds**

`KernelExplainer` calls `model.effect()` hundreds of times per patient to compute attributions. The `nsamples=80` argument in `causal_service.explain_prediction()` controls this. Reduce it for faster (less accurate) explanations. The background dataset is already reduced to 10 k-means cluster centers; reducing it further is not recommended.

---

**Import error related to Dask**

Dask was removed from this project due to a crash with Pandas 2.2.0 + Python 3.11. All data processing uses Pandas directly. If you see a Dask import error, you have a stale branch or a local modification — check [data/ingestion.py](data/ingestion.py).

---

**MLflow shows no runs**

The platform does not automatically log to MLflow at startup. MLflow is present as infrastructure. Run logging requires explicit `mlflow.start_run()` calls, which you would add when running controlled experiments. The MLflow UI at `:5001` starts empty and ready.

---

**`pytest tests/test_estimators.py` fails with `RuntimeError: Could not load LaLonde dataset`**

This test fetches the LaLonde CSV via `dowhy.datasets.lalonde_dataset()`. If `dowhy` is not installed, run `pip install dowhy==0.13`. If network access is restricted, place `lalonde.csv` in `data/raw/` manually — the test uses DoWhy's bundled copy, so this fallback only applies if DoWhy itself is unavailable.

---

## Disclaimer

This platform uses synthetic observational data for all default operation. Outputs are for research and educational purposes only and are not a substitute for clinical judgment or approved medical devices. The treatment effect estimates, policy recommendations, and counterfactual simulations should not be used to make real clinical decisions.
