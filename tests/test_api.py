import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_predict_cate_valid(sample_patient_dict):
    response = client.post("/predict_cate", json={"features": [sample_patient_dict]})
    assert response.status_code == 200
    data = response.json()
    assert "cate_estimates" in data
    assert "ci_lower" in data

def test_policy_decision_valid(sample_patient_dict):
    response = client.post("/policy_decision", json={"features": [sample_patient_dict]})
    assert response.status_code == 200
    data = response.json()
    assert "recommended_treatment" in data

def test_get_dag():
    response = client.get("/dag")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data

def test_invalid_payload():
    # Missing required fields
    response = client.post("/predict_cate", json={"features": [{"age": 65.0}]})
    assert response.status_code == 422
