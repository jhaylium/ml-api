from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pca_endpoint_logic():
    payload = {
        "model_config": {"desired_features": 2},
        "data": [{"f1": 1, "f2": 10, "f3": 100}, {"f1": 2, "f2": 20, "f3": 120}, {"f1": 3, "f2": 25, "f3": 110}]
    }
    response = client.post("/api/pca", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    
    # Assert structure of the response
    assert isinstance(json_response["reduced_features"], list)
    assert all(isinstance(row, list) for row in json_response["reduced_features"])
    assert all(len(row) == 2 for row in json_response["reduced_features"])
    assert len(json_response["model_loadings"]["PC1"]) == 3


def test_pca_s3_not_implemented():
    payload = {
        "model_config": {"desired_features": 2},
        "data": [{"f1": 1, "f2": 10, "f3": 100}],
        "s3_uri": "s3://bucket/path"
    }
    response = client.post("/api/pca", json=payload)
    assert response.status_code == 400
    assert "s3_uri is not implemented yet" in response.json()["detail"]


def test_linear_regression_stub():
    payload = {
        "model_config": {"target_variable_name": "y"},
        "data": [{"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 30}]
    }
    response = client.post("/api/linear_regression", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "model_predictions": [10.0, 20.0, 30.0],
        "model_performance": {"r2_score": 0.95, "mse": 2.5},
        "model_results": {"coefficients": [1.0, 2.0], "intercept": 0.5}
    }


def test_linear_regression_missing_target_col():
    payload = {
        "model_config": {"target_variable_name": "y"},
        "data": [{"x": 1}, {"x": 2}, {"x": 3}]
    }
    response = client.post("/api/linear_regression", json=payload)
    assert response.status_code == 400
    assert "Target column 'y' not found in data." in response.json()["detail"]

