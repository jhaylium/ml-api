from fastapi import FastAPI, Request, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from data_loader import get_dataframe

app = FastAPI()


# Custom Exceptions
class DataInputError(Exception):
    """Raised when there are issues with input data format or requirements."""
    pass


class ValidationError(Exception):
    """Raised when validation fails (e.g., missing required fields)."""
    pass


class RemoteAccessError(Exception):
    """Raised when there are errors accessing remote resources (e.g., S3)."""
    pass


# Error Response Model
class ErrorResponse(BaseModel):
    status_code: int
    error_type: str
    detail_message: str


# Exception Handlers
@app.exception_handler(DataInputError)
async def data_input_exception_handler(request: Request, exc: DataInputError):
    return JSONResponse(
        status_code=400,
        content={
            "status_code": 400,
            "error_type": "DataInputError",
            "detail_message": str(exc)
        }
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "status_code": 400,
            "error_type": "ValidationError",
            "detail_message": str(exc)
        }
    )


@app.exception_handler(RemoteAccessError)
async def remote_access_exception_handler(request: Request, exc: RemoteAccessError):
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "error_type": "RemoteAccessError",
            "detail_message": str(exc)
        }
    )


class PCAModelConfig(BaseModel):
    desired_features: int


class PCARequest(BaseModel):
    model_conf: PCAModelConfig
    data: Optional[List[dict]] = None
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
    model_conf: LinearRegressionModelConfig
    data: Optional[List[dict]] = None
    s3_uri: Optional[str] = None


class LinearRegressionResponse(BaseModel):
    model_predictions: List[float]
    model_performance: dict
    model_results: dict


class LogisticRegressionModelConfig(BaseModel):
    target_variable_name: str
    normalize: bool = False
    regularization_strength: float = 1.0


class LogisticRegressionRequest(BaseModel):
    model_conf: LogisticRegressionModelConfig
    data: Optional[List[dict]] = None
    s3_uri: Optional[str] = None


class LogisticRegressionResponse(BaseModel):
    model_predictions: List[float]
    model_performance: dict
    model_results: dict


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/pca", response_model=PCAResponse)
async def pca_endpoint(request: PCARequest, http_request: Request):
    # Load data from either direct data or S3
    df = await get_dataframe(request, http_request)
    
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
        model_loadings=PCAModelLoadings(__root__=model_loadings_dict)
    )


@app.post("/api/linear_regression", response_model=LinearRegressionResponse)
async def linear_regression_endpoint(request: LinearRegressionRequest, http_request: Request):
    # Load data from either direct data or S3
    df = await get_dataframe(request, http_request)
    
    # Data Separation
    target_col = request.model_conf.target_variable_name
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in data.")
    
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
    fit_intercept = not request.model_conf.force_intercept
    
    # Build Pipeline
    pipeline_steps = []
    
    # Normalization Logic
    if request.model_conf.normalize:
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


@app.post("/api/logistic_regression", response_model=LogisticRegressionResponse)
async def logistic_regression_endpoint(request: LogisticRegressionRequest, http_request: Request):
    # Load data from either direct data or S3
    df = await get_dataframe(request, http_request)
    
    # Data Separation
    target_col = request.model_conf.target_variable_name
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in data.")
    
    X = df.drop(target_col)
    y = df.select(target_col)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), 
        y.to_numpy().flatten(), 
        test_size=0.2, 
        random_state=42
    )
    
    # Model Config
    C = request.model_conf.regularization_strength
    
    # Build Pipeline
    pipeline_steps = []
    
    # Normalization Logic
    if request.model_conf.normalize:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    # Model Step
    pipeline_steps.append(('model', LogisticRegression(C=C, max_iter=1000)))
    
    # Model Training
    pipeline = Pipeline(steps=pipeline_steps)
    pipeline.fit(X_train, y_train)
    
    # Prediction & Metrics
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_class = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Response Formatting
    model_predictions = [float(p) for p in y_pred_proba]
    model_performance = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }
    
    # Get model from pipeline to access coefficients
    model_step = pipeline.named_steps['model']
    
    # Get column names for coefficients
    X_columns = X.columns
    coefficients_dict = dict(zip(X_columns, model_step.coef_[0]))
    coefficients_dict = {k: float(v) for k, v in coefficients_dict.items()}
    
    model_results = {
        "intercept": float(model_step.intercept_[0]),
        "coefficients": coefficients_dict
    }
    
    return LogisticRegressionResponse(
        model_predictions=model_predictions,
        model_performance=model_performance,
        model_results=model_results
    )
