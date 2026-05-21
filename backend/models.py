"""
Pydantic request/response schemas for the Causal Pharma Intelligence API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    blood_pressure: float = Field(..., ge=50, le=250, description="Systolic blood pressure in mmHg")
    comorbidities: int = Field(..., ge=0, le=20, description="Number of comorbid conditions")


class PredictCATERequest(BaseModel):
    features: List[PatientFeatures]


class PredictCATEResponse(BaseModel):
    cate_estimates: List[float]
    ate_estimate: float
    ci_lower: List[float]
    ci_upper: List[float]
    magnitude: List[str]
    confidence_pct: List[int]


class SimulateInterventionRequest(BaseModel):
    features: List[PatientFeatures]
    treatment_value: int = Field(..., ge=0, le=1)


class SimulateInterventionResponse(BaseModel):
    expected_outcomes: List[float]


class SimulateBothRequest(BaseModel):
    features: List[PatientFeatures]


class SimulateBothResponse(BaseModel):
    outcome_if_withhold: List[float]
    outcome_if_treat: List[float]
    treatment_effect: List[float]


class PolicyDecisionRequest(BaseModel):
    features: List[PatientFeatures]


class PolicyDecisionResponse(BaseModel):
    recommended_treatment: List[int]
    rationale: List[str]
    bandit_recommendation: List[int]


class ExplainPredictionRequest(BaseModel):
    features: List[PatientFeatures]


class ExplainPredictionResponse(BaseModel):
    feature_names: List[str]
    feature_labels: List[str]
    shap_values: List[List[float]]
    base_value: float


class DAGResponse(BaseModel):
    nodes: List[str]
    edges: List[Dict[str, str]]
