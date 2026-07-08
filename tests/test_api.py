"""
Tests unitaires de l'API de scoring crédit.

Stratégie :
    - Aucun chargement réel du modèle, dataset ni explainer SHAP pendant les tests
    - Remplacement du modèle et de l'explainer via app.dependency_overrides (FastAPI)
    - Remplacement de get_client_features via unittest.mock.patch
    - Couverture des endpoints :
        * GET  /                            : réponse de bienvenue
        * GET  /health                      : disponibilité
        * POST /predict                     : entier seul, liste, not_found, décisions, type invalide
        * GET  /predict/{id}/explain        : client trouvé, client introuvable (404)
        * GET  /model/info                  : structure de réponse
        * GET  /stats                       : compteurs après appels /predict

Lancement :
    pytest tests/
    pytest tests/ -v
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from api.app import app, get_model, get_explainer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """
    Faux modèle LightGBM.
    predict_proba → [[0.66, 0.34]] : score 0.34 < 0.46 → 'accordé'.
    feature_name_ : 3 colonnes simulées (552 en production).
    """
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.66, 0.34]])
    m.feature_name_ = ["EXT_SOURCE_1", "EXT_SOURCE_2", "AMT_CREDIT"]
    return m


@pytest.fixture
def mock_explainer():
    """
    Faux explainer SHAP.
    shap_values → array de contributions simulées pour 3 features.
    """
    e = MagicMock()
    e.shap_values.return_value = np.array([[0.15, -0.08, 0.05]])
    return e


@pytest.fixture
def mock_features():
    """Features simulées retournées pour un client trouvé dans le dataset."""
    return {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "AMT_CREDIT": 300000.0}


@pytest.fixture
def client(mock_model, mock_explainer, mock_features):
    """
    Client HTTP de test avec mocks injectés :
        1. load_artifacts via patch               → aucun chargement réel (modèle, dataset, SHAP)
        2. get_model via dependency_overrides     → modèle simulé
        3. get_explainer via dependency_overrides → explainer SHAP simulé
        4. get_client_features via patch          → features simulées (pas de lecture CSV)

    Sans le patch de load_artifacts, TestClient déclencherait le lifespan FastAPI
    qui charge le vrai .pkl et le vrai CSV — inutile et lent pour des tests unitaires.
    """
    app.dependency_overrides[get_model]     = lambda: mock_model
    app.dependency_overrides[get_explainer] = lambda: mock_explainer
    with patch("api.model.load_artifacts"):
        with patch("api.model.get_client_features", return_value=mock_features):
            with TestClient(app) as c:
                yield c
    app.dependency_overrides.clear()


# ── Tests GET / ───────────────────────────────────────────────────────────────

def test_root(client):
    """Vérification de la réponse racine — liste des endpoints présente."""
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


# ── Tests GET /health ─────────────────────────────────────────────────────────

def test_health(client):
    """Vérification de la disponibilité de l'API."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Tests POST /predict ───────────────────────────────────────────────────────

def test_predict_single_int(client):
    """Entier unique → normalisé en liste → une prédiction, not_found vide."""
    r = client.post("/predict", json={"client_ids": 100002})
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["client_id"] == 100002
    assert 0.0 <= data["predictions"][0]["score"] <= 1.0
    assert data["not_found"] == []


def test_predict_list(client):
    """Liste de clients → autant de prédictions que d'identifiants trouvés."""
    r = client.post("/predict", json={"client_ids": [100002, 100003]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_not_found(mock_model, mock_explainer, mock_features):
    """Client absent → dans not_found, pas d'erreur 500."""
    app.dependency_overrides[get_model]    = lambda: mock_model
    app.dependency_overrides[get_explainer] = lambda: mock_explainer

    def fake_lookup(cid):
        return mock_features if cid == 100002 else None

    with patch("api.model.get_client_features", side_effect=fake_lookup):
        with TestClient(app) as c:
            r = c.post("/predict", json={"client_ids": [100002, 999999]})
    app.dependency_overrides.clear()

    data = r.json()
    assert len(data["predictions"]) == 1
    assert 999999 in data["not_found"]


def test_predict_empty_list(client):
    """Liste vide → réponse valide, predictions et not_found vides."""
    r = client.post("/predict", json={"client_ids": []})
    assert r.status_code == 200
    assert r.json() == {"predictions": [], "not_found": []}


def test_predict_refused(mock_model, mock_explainer, mock_features):
    """Score >= 0.46 → décision 'refusé'."""
    mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])
    app.dependency_overrides[get_model]    = lambda: mock_model
    app.dependency_overrides[get_explainer] = lambda: mock_explainer
    with patch("api.model.get_client_features", return_value=mock_features):
        with TestClient(app) as c:
            r = c.post("/predict", json={"client_ids": 100002})
    app.dependency_overrides.clear()
    assert r.json()["predictions"][0]["decision"] == "refusé"


def test_predict_approved(mock_model, mock_explainer, mock_features):
    """Score < 0.46 → décision 'accordé'."""
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    app.dependency_overrides[get_model]    = lambda: mock_model
    app.dependency_overrides[get_explainer] = lambda: mock_explainer
    with patch("api.model.get_client_features", return_value=mock_features):
        with TestClient(app) as c:
            r = c.post("/predict", json={"client_ids": 100002})
    app.dependency_overrides.clear()
    assert r.json()["predictions"][0]["decision"] == "accordé"


def test_predict_invalid_type(client):
    """Type non entier → rejeté par Pydantic avec code 422."""
    r = client.post("/predict", json={"client_ids": "not_an_int"})
    assert r.status_code == 422


# ── Tests POST /predict/new ───────────────────────────────────────────────────

def test_predict_new_valid(client):
    """Features fournies directement → code 200, score valide, décision présente."""
    payload = {
        "features": {
            "EXT_SOURCE_1":    0.5,
            "EXT_SOURCE_2":    0.6,
            "EXT_SOURCE_MEAN": 0.55,
            "AMT_CREDIT":      406597.5,
        }
    }
    r = client.post("/predict/new", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["score"] <= 1.0
    assert data["decision"] in ("accordé", "refusé")
    assert data["threshold"] == 0.46


def test_predict_new_with_client_id(client):
    """client_id optionnel fourni → renvoyé dans la réponse."""
    payload = {
        "client_id": 999001,
        "features": {"EXT_SOURCE_1": 0.4},
    }
    r = client.post("/predict/new", json=payload)
    assert r.status_code == 200
    assert r.json()["client_id"] == 999001


def test_predict_new_without_client_id(client):
    """Sans client_id → champ None dans la réponse, pas d'erreur."""
    r = client.post("/predict/new", json={"features": {"EXT_SOURCE_1": 0.4}})
    assert r.status_code == 200
    assert r.json()["client_id"] is None


def test_predict_new_empty_features(client):
    """Aucune feature fournie → NaN partout, LightGBM prédit quand même."""
    r = client.post("/predict/new", json={"features": {}})
    assert r.status_code == 200


def test_predict_new_invalid_type(client):
    """Valeur non numérique → rejeté par Pydantic, code 422."""
    r = client.post("/predict/new", json={"features": {"EXT_SOURCE_1": "texte"}})
    assert r.status_code == 422


# ── Tests GET /predict/{client_id}/explain ────────────────────────────────────

def test_explain_found(client):
    """Client trouvé → code 200, top_factors présent et non vide."""
    r = client.get("/predict/100002/explain")
    assert r.status_code == 200
    data = r.json()
    assert data["client_id"] == 100002
    assert "top_factors" in data
    assert len(data["top_factors"]) > 0
    # Vérification de la structure d'un facteur SHAP
    factor = data["top_factors"][0]
    assert "feature" in factor
    assert "contribution" in factor
    assert factor["direction"] in ("favorable", "défavorable")


def test_explain_not_found(mock_model, mock_explainer):
    """Client absent du dataset → code 404."""
    app.dependency_overrides[get_model]    = lambda: mock_model
    app.dependency_overrides[get_explainer] = lambda: mock_explainer
    with patch("api.model.get_client_features", return_value=None):
        with TestClient(app) as c:
            r = c.get("/predict/999999/explain")
    app.dependency_overrides.clear()
    assert r.status_code == 404


def test_explain_top_n(client):
    """Paramètre top_n respecté — nombre de facteurs retournés ≤ top_n."""
    r = client.get("/predict/100002/explain?top_n=2")
    assert r.status_code == 200
    assert len(r.json()["top_factors"]) <= 2


# ── Tests GET /model/info ─────────────────────────────────────────────────────

def test_model_info(client):
    """Structure de réponse complète — champs clés présents."""
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "model_type" in data
    assert "auc_test" in data
    assert "threshold" in data
    assert "n_features" in data


# ── Tests GET /stats ──────────────────────────────────────────────────────────

def test_stats_after_predict(client):
    """Appel /predict → compteurs /stats mis à jour."""
    client.post("/predict", json={"client_ids": 100002})
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_requests"] >= 1
    assert data["total_clients_scored"] >= 1
    assert 0.0 <= data["refusal_rate"] <= 1.0
    assert 0.0 <= data["avg_score"] <= 1.0
