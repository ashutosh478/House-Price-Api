 
from app import app, FEATURES


def get_payload():
    return {
        "MedInc": 5.0,
        "HouseAge": 25,
        "AveRooms": 5.5,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 3,
        "Latitude": 34.05,
        "Longitude": -118.25
    }


def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_model_info():
    client = app.test_client()
    response = client.get("/model-info")
    assert response.status_code == 200
    assert set(response.get_json()["features"]) == set(FEATURES)


def test_prediction():
    client = app.test_client()
    response = client.post("/predict", json=get_payload())
    assert response.status_code == 200
    assert response.get_json()["estimated_price_usd"] > 0


def test_missing_feature():
    client = app.test_client()
    data = get_payload()
    data.pop("MedInc")

    response = client.post("/predict", json=data)
    assert response.status_code == 400


def test_invalid_value():
    client = app.test_client()
    data = get_payload()
    data["MedInc"] = "abc"

    response = client.post("/predict", json=data)
    assert response.status_code == 400
