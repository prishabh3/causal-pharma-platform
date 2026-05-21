from fastapi import APIRouter, HTTPException
from backend.models import (
    PredictCATERequest, PredictCATEResponse,
    DAGResponse,
    PolicyDecisionRequest, PolicyDecisionResponse,
    SimulateInterventionRequest, SimulateInterventionResponse,
    SimulateBothRequest, SimulateBothResponse,
    ExplainPredictionRequest, ExplainPredictionResponse,
)
from backend.services.causal_service import causal_service
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict_cate", response_model=PredictCATEResponse)
async def predict_cate(request: PredictCATERequest):
    try:
        df = pd.DataFrame([f.model_dump() for f in request.features])
        result = causal_service.predict_cate_with_meta(df)
        return PredictCATEResponse(**result)
    except Exception as e:
        logger.exception("Error in /predict_cate")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate_intervention", response_model=SimulateInterventionResponse)
async def simulate_intervention(request: SimulateInterventionRequest):
    try:
        df = pd.DataFrame([f.model_dump() for f in request.features])
        expected_outcomes = causal_service.simulate_intervention(
            df, treatment_value=request.treatment_value
        )
        return SimulateInterventionResponse(expected_outcomes=expected_outcomes)
    except Exception as e:
        logger.exception("Error in /simulate_intervention")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate_both", response_model=SimulateBothResponse)
async def simulate_both(request: SimulateBothRequest):
    try:
        df = pd.DataFrame([f.model_dump() for f in request.features])
        result = causal_service.simulate_both_outcomes(df)
        return SimulateBothResponse(**result)
    except Exception as e:
        logger.exception("Error in /simulate_both")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policy_decision", response_model=PolicyDecisionResponse)
async def policy_decision(request: PolicyDecisionRequest):
    try:
        df = pd.DataFrame([f.model_dump() for f in request.features])
        result = causal_service.policy_decision(df)
        return PolicyDecisionResponse(**result)
    except Exception as e:
        logger.exception("Error in /policy_decision")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain_prediction", response_model=ExplainPredictionResponse)
async def explain_prediction(request: ExplainPredictionRequest):
    try:
        df = pd.DataFrame([f.model_dump() for f in request.features])
        result = causal_service.explain_prediction(df)
        return ExplainPredictionResponse(**result)
    except Exception as e:
        logger.exception("Error in /explain_prediction")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dag", response_model=DAGResponse)
async def get_dag():
    try:
        nodes, edges = causal_service.get_dag_edges()
        return DAGResponse(nodes=nodes, edges=edges)
    except Exception as e:
        logger.exception("Error in /dag")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dag/reference", response_model=DAGResponse)
async def get_reference_dag():
    try:
        ref = causal_service.get_reference_dag()
        return DAGResponse(nodes=ref["nodes"], edges=ref["edges"])
    except Exception as e:
        logger.exception("Error in /dag/reference")
        raise HTTPException(status_code=500, detail=str(e))
