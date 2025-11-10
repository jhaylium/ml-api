from fastapi.testclient import TestClient
from moto import mock_s3
import boto3
import polars as pl
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pca_endpoint_logic():
    payload = {
        "model_conf": {"desired_features": 2},
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


def test_exclusivity_error():
    """Test that providing both data and s3_uri raises an error."""
    payload = {
        "model_conf": {"desired_features": 2},
        "data": [{"f1": 1, "f2": 10, "f3": 100}],
        "s3_uri": "s3://bucket/path"
    }
    response = client.post("/api/pca", json=payload)
    assert response.status_code == 400
    assert response.json() == {
        "status_code": 400,
        "error_type": "DataInputError",
        "detail_message": "Cannot provide both 'data' and 's3_uri'"
    }


def test_s3_header_error():
    """Test that providing s3_uri without headers raises an error."""
    payload = {
        "model_conf": {"desired_features": 2},
        "s3_uri": "s3://bucket/path"
    }
    response = client.post("/api/pca", json=payload)
    assert response.status_code == 400
    assert response.json() == {
        "status_code": 400,
        "error_type": "DataInputError",
        "detail_message": "S3 headers are required for s3_uri: X-Aws-Access-Key-Id, X-Aws-Secret-Access-Key, X-Aws-Region"
    }


@mock_s3
def test_s3_success():
    """Test successful S3 data loading with mocked S3."""
    # Create mock S3 bucket and upload parquet file
    s3_client = boto3.client('s3', region_name='us-east-1')
    bucket_name = 'test-bucket'
    key = 'test-data.parquet'
    
    # Create test data
    test_data = pl.DataFrame({
        'f1': [1, 2, 3],
        'f2': [10, 20, 25],
        'f3': [100, 120, 110]
    })
    
    # Convert to parquet bytes
    parquet_bytes = test_data.write_parquet()
    
    # Create bucket and upload
    s3_client.create_bucket(Bucket=bucket_name)
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=parquet_bytes)
    
    # Test request
    payload = {
        "model_conf": {"desired_features": 2},
        "s3_uri": f"s3://{bucket_name}/{key}"
    }
    
    headers = {
        "X-Aws-Access-Key-Id": "test-key",
        "X-Aws-Secret-Access-Key": "test-secret",
        "X-Aws-Region": "us-east-1"
    }
    
    response = client.post("/api/pca", json=payload, headers=headers)
    assert response.status_code == 200
    json_response = response.json()
    
    # Assert structure of the response
    assert isinstance(json_response["reduced_features"], list)
    assert all(isinstance(row, list) for row in json_response["reduced_features"])
    assert all(len(row) == 2 for row in json_response["reduced_features"])


def test_linear_regression_stub():
    payload = {
        "model_conf": {"target_variable_name": "y"},
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
        "model_conf": {"target_variable_name": "y"},
        "data": [{"x": 1}, {"x": 2}, {"x": 3}]
    }
    response = client.post("/api/linear_regression", json=payload)
    assert response.status_code == 400
    assert response.json() == {
        "status_code": 400,
        "error_type": "ValidationError",
        "detail_message": "Target column 'y' not found in data."
    }


def test_linear_regression_normalize():
    base_payload = {
        "model_conf": {"target_variable_name": "y"},
        "data": [
            {"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 30},
            {"x": 4, "y": 40}, {"x": 5, "y": 50}, {"x": 6, "y": 60},
            {"x": 7, "y": 70}, {"x": 8, "y": 80}, {"x": 9, "y": 90},
            {"x": 10, "y": 100}
        ]
    }
    
    # Test with normalize=False
    payload_unscaled = {**base_payload, "model_conf": {**base_payload["model_conf"], "normalize": False}}
    response_unscaled = client.post("/api/linear_regression", json=payload_unscaled)
    
    # Test with normalize=True
    payload_scaled = {**base_payload, "model_conf": {**base_payload["model_conf"], "normalize": True}}
    response_scaled = client.post("/api/linear_regression", json=payload_scaled)
    
    assert response_unscaled.status_code == 200
    assert response_scaled.status_code == 200
    
    # Assert that predictions differ when normalization is applied
    assert response_unscaled.json()["model_predictions"] != response_scaled.json()["model_predictions"]
    
    # Assert that coefficients differ when normalization is applied
    assert response_unscaled.json()["model_results"]["coefficients"] != response_scaled.json()["model_results"]["coefficients"]


def test_logistic_regression_logic():
    payload = {
        "model_conf": {"target_variable_name": "y"},
        "data": [
            {"x1": 1, "x2": 2, "y": 0},
            {"x1": 2, "x2": 3, "y": 0},
            {"x1": 3, "x2": 1, "y": 0},
            {"x1": 1, "x2": 1, "y": 0},
            {"x1": 2, "x2": 2, "y": 0},
            {"x1": 10, "x2": 1, "y": 1},
            {"x1": 11, "x2": 2, "y": 1},
            {"x1": 12, "x2": 3, "y": 1},
            {"x1": 10, "x2": 4, "y": 1},
            {"x1": 11, "x2": 5, "y": 1}
        ]
    }
    response = client.post("/api/logistic_regression", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    
    # Assert all keys in model_performance are present
    assert "accuracy" in json_response["model_performance"]
    assert "f1_score" in json_response["model_performance"]
    assert "roc_auc" in json_response["model_performance"]

