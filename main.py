from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from typing import List, Dict, Optional
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

app = FastAPI()


class PCAModelConfig(BaseModel):
    desired_features: int


class PCARequest(BaseModel):
    model_conf: PCAModelConfig
    data: List[dict]
    s3_uri: Optional[str] = None


class PCAModelLoadings(RootModel[Dict[str, List[float]]]):
    root: Dict[str, List[float]]


class PCAResponse(BaseModel):
    reduced_features: List[List[float]]
    model_loadings: PCAModelLoadings


class LinearRegressionModelConfig(BaseModel):
    target_variable_name: str
    normalize: bool = False
    force_intercept: bool = False


class LinearRegressionRequest(BaseModel):
    model_conf: LinearRegressionModelConfig
    data: List[dict]
    s3_uri: Optional[str] = None


class LinearRegressionResponse(BaseModel):
    model_predictions: List[float]
    model_performance: dict
    model_results: dict


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/pca", response_model=PCAResponse)
def pca_endpoint(request: PCARequest):
    # Input Validation: Check if s3_uri is present
    if request.s3_uri is not None:
        raise HTTPException(status_code=400, detail="s3_uri is not implemented yet")
    
    # Data Conversion: Convert request.data to Polars DataFrame
    df = pl.from_dicts(request.data)
    
    # ML Execution
    n_components = request.model_conf.desired_features
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df.to_numpy())
    
    # Response Formatting
    loadings = pca.components_
    model_loadings_dict = {}
    for i, component in enumerate(loadings):
        component_name = f"PC{i + 1}"
        model_loadings_dict[component_name] = [float(val) for val in component]
    
    reduced_features = [list(row) for row in reduced_data]
    
    return PCAResponse(
        reduced_features=reduced_features,
        model_loadings=PCAModelLoadings(root=model_loadings_dict)
    )


@app.post("/api/linear_regression", response_model=LinearRegressionResponse)
def linear_regression_endpoint(request: LinearRegressionRequest):
    # Input Validation: Check if s3_uri is present
    if request.s3_uri is not None:
        raise HTTPException(status_code=400, detail="s3_uri is not implemented yet")
    
    # Data Conversion: Convert request.data to Polars DataFrame
    df = pl.from_dicts(request.data)
    
    # Data Separation
    target_col = request.model_conf.target_variable_name
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in data.")
    
    X = df.drop(target_col)
    y = df.select(target_col)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), 
        y.to_numpy(), 
        test_size=0.2, 
        random_state=42
    )
    
    # Stubbed Return: Mock response for now
    return LinearRegressionResponse(
        model_predictions=[10.0, 20.0, 30.0],
        model_performance={"r2_score": 0.95, "mse": 2.5},
        model_results={"coefficients": [1.0, 2.0], "intercept": 0.5}
    )
