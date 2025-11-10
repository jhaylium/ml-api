from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = FastAPI()


class PCAModelConfig(BaseModel):
    desired_features: int


class PCARequest(BaseModel):
    model_config: PCAModelConfig
    data: List[dict]
    s3_uri: Optional[str] = None


class PCAModelLoadings(BaseModel):
    __root__: Dict[str, List[float]]


class PCAResponse(BaseModel):
    reduced_features: List[List[float]]
    model_loadings: PCAModelLoadings


class LinearRegressionModelConfig(BaseModel):
    target_variable_name: str
    normalize: bool = False
    force_intercept: bool = False


class LinearRegressionRequest(BaseModel):
    model_config: LinearRegressionModelConfig
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
    n_components = request.model_config.desired_features
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
        model_loadings=PCAModelLoadings(__root__=model_loadings_dict)
    )


@app.post("/api/linear_regression", response_model=LinearRegressionResponse)
def linear_regression_endpoint(request: LinearRegressionRequest):
    # Input Validation: Check if s3_uri is present
    if request.s3_uri is not None:
        raise HTTPException(status_code=400, detail="s3_uri is not implemented yet")
    
    # Data Conversion: Convert request.data to Polars DataFrame
    df = pl.from_dicts(request.data)
    
    # Data Separation
    target_col = request.model_config.target_variable_name
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
    
    # Model Config
    fit_intercept = not request.model_config.force_intercept
    
    # Build Pipeline
    pipeline_steps = []
    
    # Normalization Logic
    if request.model_config.normalize:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    # Model Step
    pipeline_steps.append(('model', LinearRegression(fit_intercept=fit_intercept)))
    
    # Model Training
    pipeline = Pipeline(steps=pipeline_steps)
    pipeline.fit(X_train, y_train)
    
    # Prediction & Metrics
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Response Formatting
    model_predictions = [float(p) for p in y_pred.flatten()]
    model_performance = {"r_squared": float(r2), "mean_squared_error": float(mse)}
    
    # Get model from pipeline to access coefficients
    model_step = pipeline.named_steps['model']
    
    # Get column names for coefficients
    X_columns = X.columns
    # Handle both 1D and 2D coefficient arrays
    coef = model_step.coef_[0] if model_step.coef_.ndim > 1 else model_step.coef_
    coefficients_dict = dict(zip(X_columns, coef))
    coefficients_dict = {k: float(v) for k, v in coefficients_dict.items()}
    
    # Handle intercept (can be scalar or array)
    intercept_value = model_step.intercept_
    if hasattr(intercept_value, '__len__') and not isinstance(intercept_value, str):
        intercept_value = intercept_value[0]
    intercept_value = float(intercept_value)
    
    model_results = {
        "intercept": intercept_value,
        "coefficients": coefficients_dict
    }
    
    return LinearRegressionResponse(
        model_predictions=model_predictions,
        model_performance=model_performance,
        model_results=model_results
    )
