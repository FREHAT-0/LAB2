import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction_consistency():

    payload = {
        "tenure_months": 24,
        "monthly_charges": 50.0,
        "support_calls": 2,
        "contract": "Month-to-Month",
        "internet_service": "Fiber"
    }

    response1 = client.post("/predict", json=payload)
    response2 = client.post("/predict", json=payload)

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()
