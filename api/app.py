"""
API de scoring crédit — Prêt à Dépenser.

Lancement en développement :
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Documentation interactive Swagger :
    http://localhost:8000/docs

Endpoints disponibles :
    GET  /                              → informations générales sur l'API
    GET  /health                        → état du service
    POST /predict                       → score(s) de défaut pour un ou plusieurs clients
    GET  /predict/{client_id}/explain   → explication SHAP locale pour un client
    GET  /model/info                    → métadonnées du modèle en production
    GET  /stats                         → statistiques d'usage depuis le démarrage
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Query

from api.schemas import (
    PredictRequest,
    PredictionResponse,
    BatchResponse,
    NewClientRequest,
    ExplainResponse,
    ModelInfoResponse,
    StatsResponse,
)
from api import model as scoring_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application.

    Au démarrage : chargement du modèle LightGBM, du dataset et de l'explainer SHAP.
    À l'arrêt    : libération automatique par le garbage collector Python.

    Pattern asynccontextmanager — code avant yield = startup, après yield = shutdown.
    """
    scoring_model.load_artifacts()
    yield


app = FastAPI(
    title="Credit Scoring API — Prêt à Dépenser",
    description="Score de défaut de remboursement par identifiant client (SK_ID_CURR).",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Dépendances injectées ─────────────────────────────────────────────────────

def get_model():
    """
    Accesseur vers le modèle LightGBM chargé en mémoire.
    Remplaçable par un mock dans les tests via app.dependency_overrides.
    """
    return scoring_model.get_model()


def get_explainer():
    """
    Accesseur vers l'explainer SHAP chargé en mémoire.
    Remplaçable par un mock dans les tests via app.dependency_overrides.
    """
    return scoring_model.get_explainer()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Monitoring"])
def root():
    """
    Point d'entrée racine.
    Confirmation que le service est en ligne avec la liste des endpoints disponibles.
    """
    return {
        "service": "Credit Scoring API — Prêt à Dépenser",
        "version": "2.0.0",
        "endpoints": {
            "root":        "GET  /",
            "health":      "GET  /health",
            "predict":     "POST /predict",
            "predict_new": "POST /predict/new",
            "explain":     "GET  /predict/{client_id}/explain",
            "info":        "GET  /model/info",
            "stats":       "GET  /stats",
            "docs":        "GET  /docs",
        },
    }


@app.get("/health", tags=["Monitoring"])
def health():
    """
    Vérification de la disponibilité de l'API.
    Utilisé par Docker healthcheck et les pipelines CI/CD.
    """
    return {"status": "ok", "model": "lgbm_optimized"}


@app.post("/predict", response_model=BatchResponse, tags=["Prédiction"])
def predict(data: PredictRequest, model=Depends(get_model)):
    """
    Score(s) de défaut pour un ou plusieurs clients.

    Accepte un identifiant unique ou une liste :
    - `{"client_ids": 100002}`
    - `{"client_ids": [100002, 100003, 100004]}`

    - **predictions** : liste des scores pour les clients trouvés
    - **not_found**   : liste des identifiants absents du dataset
    """
    predictions = []
    not_found   = []

    for cid in data.client_ids:
        try:
            features = scoring_model.get_client_features(cid)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dataset lookup error client {cid}: {e}")

        if features is None:
            not_found.append(cid)
            continue

        try:
            result = scoring_model.predict(model, features)
            predictions.append(PredictionResponse(client_id=cid, **result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error client {cid}: {e}")

    # Mise à jour des compteurs de monitoring après chaque appel réussi
    scoring_model.update_stats(predictions, not_found)

    return BatchResponse(predictions=predictions, not_found=not_found)


@app.post("/predict/new", response_model=PredictionResponse, tags=["Prédiction"])
def predict_new(data: NewClientRequest, model=Depends(get_model)):
    """
    Score de défaut pour un nouveau client sans historique dans le dataset.

    Cas d'usage : demande de crédit en temps réel — le client n'est pas encore
    dans `train_processed_global.csv`. Les features pré-calculées en amont
    (par un pipeline ETL ou un data warehouse) sont envoyées directement.

    - Features manquantes → NaN, gérées nativement par LightGBM
    - **client_id** optionnel : numéro de dossier assigné par l'appelant
    """
    try:
        result = scoring_model.predict(model, data.features)
        prediction = PredictionResponse(client_id=data.client_id, **result)
        # Mise à jour des stats — liste d'un élément pour réutiliser update_stats
        scoring_model.update_stats([prediction], [])
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predict/{client_id}/explain",
    response_model=ExplainResponse,
    tags=["Prédiction"],
)
def explain(
    client_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Nombre de features à retourner"),
    model=Depends(get_model),
    explainer=Depends(get_explainer),
):
    """
    Explication locale SHAP pour un client — pourquoi cette décision.

    - **top_factors** : features classées par impact décroissant
    - `contribution` positive → pousse vers défaut (défavorable)
    - `contribution` négative → pousse vers remboursement (favorable)
    - **top_n** : nombre de features retournées (défaut 10, max 50)

    Obligation réglementaire en crédit : tout refus doit être explicable.
    """
    result = scoring_model.explain_client(model, explainer, client_id, top_n)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} introuvable dans le dataset.",
        )

    return ExplainResponse(**result)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info(model=Depends(get_model)):
    """
    Métadonnées du modèle actuellement en production.

    Métriques issues du notebook d'entraînement (Projet 6) :
    AUC test, coût métier, recall, précision, nombre de features.
    """
    return ModelInfoResponse(**scoring_model.get_model_info(model))


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
def stats():
    """
    Statistiques d'usage de l'API depuis le dernier démarrage.

    - **refusal_rate** : proportion de décisions "refusé"
    - **avg_score**    : score de défaut moyen des clients scorés

    Compteurs remis à zéro à chaque redémarrage — pour un monitoring persistant,
    brancher une base de données ou un outil comme Prometheus.
    """
    return StatsResponse(**scoring_model.get_stats())
