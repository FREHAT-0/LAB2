import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_invalid_schema():

    response = client.post(
        "/predict",
        json={"wrong_field": 123}
    )

    assert response.status_code == 422