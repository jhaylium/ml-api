# Uvicorn Startup Error Resolution

**Date:** 2025-11-09-23-38  
**Error:** `TypeError: To define root models, use 'pydantic.RootModel' rather than a field called '__root__'`

## Problem Diagnosis

The error occurred when trying to start the FastAPI application with uvicorn. The issue is in the `PCAModelLoadings` class definition on line 22-23 of `main.py`.

### Root Cause

The code was using the **Pydantic v1** syntax for root models:
```python
class PCAModelLoadings(BaseModel):
    __root__: Dict[str, List[float]]
```

However, the project is using **Pydantic v2** (which comes with FastAPI 0.121.1). In Pydantic v2, the `__root__` field syntax has been deprecated and replaced with `pydantic.RootModel`.

### Error Location

- **File:** `main.py`
- **Line 22-23:** Class definition using deprecated `__root__` syntax
- **Line 79:** Instantiation using `__root__` parameter

## Solution

The fix involves two changes:

1. **Import `RootModel`** from pydantic
2. **Update the class definition** to use `RootModel` instead of `BaseModel` with `__root__`
3. **Update the instantiation** to use `root` parameter instead of `__root__`

### Changes Made

1. **Updated imports** (line 2):
   ```python
   from pydantic import BaseModel, RootModel
   ```

2. **Updated class definition** (line 22-23):
   ```python
   class PCAModelLoadings(RootModel[Dict[str, List[float]]]):
       root: Dict[str, List[float]]
   ```

3. **Updated instantiation** (line 79):
   ```python
   model_loadings=PCAModelLoadings(root=model_loadings_dict)
   ```

## Verification

After these changes, the uvicorn server should start successfully. The API endpoint `/api/pca` will continue to work as expected, but now uses the correct Pydantic v2 syntax for root models.

## Additional Notes

- This is a breaking change between Pydantic v1 and v2
- `RootModel` is the recommended way to define root models in Pydantic v2
- The `root` field name replaces the old `__root__` field name

