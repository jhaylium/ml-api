from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pca_endpoint_stub():
    payload = {
        "model_config": {"desired_features": 2},
        "data": [{"feature1": 1.0, "feature2": 2.0}, {"feature1": 3.0, "feature2": 4.0}]
    }
    response = client.post("/api/pca", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "reduced_features": [[1.0, 0.5], [0.5, 0.1]],
        "model_loadings": {"PC1": [0.8, 0.2], "PC2": [0.2, -0.8]}
    }

