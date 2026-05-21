from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bug Fix: Use relative imports so this works both inside Docker (PYTHONPATH=/app)
# and when run directly as `uvicorn backend.main:app` from project root.
# ---------------------------------------------------------------------------
from backend.api.routes import router

app = FastAPI(
    title="Causal Pharma Intelligence Platform API",
    description="Production-grade API for Causal Inference in Healthcare",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to Causal Pharma Intelligence API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.1.0",
        "endpoints": [
            "/predict_cate",
            "/policy_decision",
            "/simulate_intervention",
            "/simulate_both",
            "/explain_prediction",
            "/dag",
            "/dag/reference",
        ],
    }


@app.on_event("startup")
async def startup_event():
    logger.info("Causal Pharma Intelligence API is starting up...")
    logger.info("Model training will happen on first import of causal_service.")
