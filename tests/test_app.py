import pytest
from app import app


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """
    Test Case 2a: Verify GET / route loads landing page with status code 200.
    """
    response = client.get("/")
    assert response.status_code == 200


def test_predict_get_route(client):
    """
    Test Case 2b: Verify GET /predict route loads home form page with status code 200.
    """
    response = client.get("/predict")
    assert response.status_code == 200


def test_predict_post_route(client):
    """
    Test Case 2c: Verify POST /predict route processes valid form data and returns 200.
    """
    form_data = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": "72",
        "writing_score": "74"
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 200
