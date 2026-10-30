from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model/model.joblib"

app = FastAPI(title="Churn Prediction API", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}") from exc


class ChurnRequest(BaseModel):
    tenant_months: Optional[int] = None
    tenure_months: Optional[int] = None
    monthly_charges: Optional[float] = None
    monthly_fee: Optional[float] = None
    contract: str = Field(..., min_length=1)
    internet_service: str = Field(..., min_length=1)
    support_calls: int = Field(..., ge=0)
    customer_id: Optional[str] = None

    @property
    def tenure(self) -> int:
        value = self.tenure_months if self.tenure_months is not None else self.tenant_months
        if value is None:
            raise ValueError("tenure_months is required")
        return int(value)

    @property
    def monthly(self) -> float:
        value = self.monthly_charges if self.monthly_charges is not None else self.monthly_fee
        if value is None:
            raise ValueError("monthly_charges is required")
        return float(value)


class ChurnResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    churn_prediction: int
    risk: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> ChurnResponse:
    try:
        tenure = request.tenure
        monthly = request.monthly
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = pd.DataFrame(
        [{
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": request.contract,
            "internet_service": request.internet_service,
            "support_calls": request.support_calls,
        }]
    )

    probability = float(model.predict_proba(record)[0, 1])
    prediction = int(probability >= 0.5)
    risk = "high" if prediction == 1 else "low"

    return ChurnResponse(
        customer_id=request.customer_id,
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk=risk,
    )


@app.post("/predict/batch")
def predict_batch(requests: list[ChurnRequest]) -> list[ChurnResponse]:
    return [predict(item) for item in requests]
