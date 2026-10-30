import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_edge_case():

    payload = {
        "tenure_months": 0,
        "monthly_charges": 0.0,
        "support_calls": 0,
        "contract": "Month-to-Month",
        "internet_service": "Fiber"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200