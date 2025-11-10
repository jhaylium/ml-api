from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()


class PCAModelConfig(BaseModel):
    desired_features: int


class PCARequest(BaseModel):
    model_config: PCAModelConfig
    data: List[dict]


class PCAModelLoadings(BaseModel):
    __root__: Dict[str, List[float]]


class PCAResponse(BaseModel):
    reduced_features: List[List[float]]
    model_loadings: PCAModelLoadings


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/pca", response_model=PCAResponse)
def pca_endpoint(request: PCARequest):
    return {
        "reduced_features": [[1.0, 0.5], [0.5, 0.1]],
        "model_loadings": {"PC1": [0.8, 0.2], "PC2": [0.2, -0.8]}
    }
