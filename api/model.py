"""
Chargement unique du modèle LightGBM, du dataset et de l'explainer SHAP au démarrage.
Réutilisation en mémoire à chaque requête — pas de rechargement à la volée.

Variable d'environnement MODEL_SOURCE :
    hf     → téléchargement depuis HF Hub (HF Spaces, défaut Docker)
    mlflow → chargement depuis MLflow Registry (développement local)
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
import shap
import duckdb
import mlflow.lightgbm
from pathlib import Path

# Chemin résolu depuis l'emplacement de ce fichier (api/model.py → racine du projet)
_ROOT         = Path(__file__).resolve().parent.parent
_DATASET_PATH = _ROOT / "data" / "processed" / "train_processed_global.csv"

# URI MLflow Registry — utilisé quand MODEL_SOURCE=mlflow
_MODEL_URI = "models:/LightGBM_credit_scoring@champion"

# Repos HF Hub — utilisés quand MODEL_SOURCE=hf
_HF_MODEL_REPO   = "Faiza93/projet8-credit-scoring"
_HF_DATASET_REPO = "Faiza93/projet8-credit-scoring-data"

# Seuil de décision métier optimisé : minimise la fonction de coût 10×FN + FP
THRESHOLD = 0.46

# Colonnes à exclure lors du lookup — non utilisées par le modèle
_COLS_EXCLUDE = {"TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"}

# Métadonnées fixes issues de l'entraînement (Projet 6 — modeling_mlflow.ipynb)
_MODEL_METADATA = {
    "model_type":             "LightGBM",
    "threshold":              THRESHOLD,
    "auc_test":               0.7893,
    "auc_oof":                0.7875,
    "business_cost_formula":  "10 × FN + FP",
    "business_cost_test":     29797,
    "training_samples":       246005,
    "test_samples":           61502,
    "recall_test":            0.6628,
    "precision_test":         0.2013,
}

# Variables globales initialisées au démarrage par load_artifacts()
_model        = None   # LGBMClassifier chargé depuis .pkl
_parquet_path = None   # Chemin local du Parquet (mode hf) — lu à la demande par DuckDB
_dataset      = None   # DataFrame indexé par SK_ID_CURR (mode mlflow local uniquement)
_explainer    = None   # shap.TreeExplainer — optimisé pour les modèles à base d'arbres

# Compteurs en mémoire — remis à zéro au redémarrage de l'API
_stats = {
    "total_requests":       0,   # nombre total d'appels à /predict
    "total_clients_scored": 0,   # clients effectivement scorés (trouvés dans le dataset)
    "total_refused":        0,   # décisions "refusé"
    "total_approved":       0,   # décisions "accordé"
    "total_not_found":      0,   # identifiants absents du dataset
    "scores":               [],  # historique des scores pour le calcul de la moyenne
}


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suppression des caractères spéciaux dans les noms de colonnes.
    Même transformation appliquée lors de l'entraînement dans le notebook.
    LightGBM rejette les caractères non alphanumériques dans les noms de colonnes.
    """
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
    return df


def load_artifacts():
    """
    Désérialisation du modèle, téléchargement du dataset et initialisation de l'explainer SHAP.
    Appel unique dans le lifespan de l'application FastAPI.

    Mode hf    : modèle .pkl + Parquet sur disque (DuckDB lit à la demande, ~300MB RAM)
    Mode mlflow: modèle MLflow Registry + CSV chargé en DataFrame (~1.5GB RAM, local uniquement)
    """
    global _model, _parquet_path, _dataset, _explainer

    model_source = os.getenv("MODEL_SOURCE", "mlflow")

    if model_source == "hf":
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id=_HF_MODEL_REPO,
            filename="lgbm_optimized.pkl",
            repo_type="model",
        )
        _model = joblib.load(model_path)

        # Parquet téléchargé sur disque (cache HF Hub) — jamais chargé en RAM
        # DuckDB lira uniquement la ligne du client demandé à chaque requête
        _parquet_path = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            filename="train_processed_global.parquet",
            repo_type="dataset",
        )

    else:
        # Développement local : MLflow Registry + CSV local
        # Path.as_uri() produit file:///C:/... sur Windows — requis par MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (_ROOT / "mlruns").as_uri()))
        _model = mlflow.lightgbm.load_model(_MODEL_URI)

        if _DATASET_PATH.exists():
            df       = pd.read_csv(_DATASET_PATH)
            df       = _clean_column_names(df)
            _dataset = df.set_index("SK_ID_CURR")

    _explainer = shap.TreeExplainer(_model)


def get_model():
    """Accesseur vers le modèle — utilisé par Depends() dans app.py."""
    return _model


def get_explainer():
    """Accesseur vers l'explainer SHAP — utilisé par Depends() dans app.py."""
    return _explainer


def get_client_features(client_id: int) -> dict | None:
    """
    Lookup des features d'un client.

    Mode hf    : requête DuckDB sur le Parquet local — une seule ligne lue sur disque.
    Mode mlflow: lookup O(1) dans le DataFrame en mémoire.
    Retourne None si le client_id est introuvable.
    """
    if _parquet_path is not None:
        # f-string pour le chemin (read_parquet ne supporte pas les paramètres positionnels)
        # client_id reste en paramètre positionnel pour éviter toute injection
        df = duckdb.execute(
            f"SELECT * FROM read_parquet('{_parquet_path}') WHERE SK_ID_CURR = ?",
            [int(client_id)],
        ).fetchdf()
        if df.empty:
            return None
        df  = _clean_column_names(df)
        row = df.iloc[0]
        return {
            col: (np.nan if pd.isna(val) else float(val))
            for col, val in row.items()
            if col not in _COLS_EXCLUDE
        }

    if _dataset is None or client_id not in _dataset.index:
        return None
    row = _dataset.loc[client_id]
    return {
        col: (None if pd.isna(val) else float(val))
        for col, val in row.items()
        if col not in _COLS_EXCLUDE
    }


def predict(model, features: dict) -> dict:
    """
    Prédiction du score de défaut à partir d'un dictionnaire de features.

    Colonnes manquantes → NaN — LightGBM gère nativement les valeurs manquantes.
    Retourne score (float), decision (str) et threshold (float).
    """
    feature_names = model.feature_name_
    row           = {col: features.get(col, np.nan) for col in feature_names}
    df            = pd.DataFrame([row])
    proba         = float(model.predict_proba(df)[0][1])
    decision      = "refusé" if proba >= THRESHOLD else "accordé"

    return {
        "score":     round(proba, 4),
        "decision":  decision,
        "threshold": THRESHOLD,
    }


def explain_client(model, explainer, client_id: int, top_n: int = 10) -> dict | None:
    """
    Explication SHAP locale pour un client — quelles features ont orienté la décision.

    Retourne None si le client_id est introuvable dans le dataset.

    Valeur SHAP positive → pousse vers défaut (défavorable pour le client).
    Valeur SHAP négative → pousse vers remboursement (favorable pour le client).

    Paramètres
    ----------
    model     : LGBMClassifier chargé en mémoire
    explainer : shap.TreeExplainer initialisé sur le modèle
    client_id : identifiant SK_ID_CURR du client
    top_n     : nombre de features à retourner (par ordre d'impact décroissant)
    """
    features = get_client_features(client_id)
    if features is None:
        return None

    # Construction du vecteur de features dans l'ordre attendu par le modèle
    row  = {col: features.get(col, np.nan) for col in model.feature_name_}
    df   = pd.DataFrame([row])

    # Calcul des valeurs SHAP — TreeExplainer est optimisé pour LightGBM
    shap_vals = explainer.shap_values(df)

    # LightGBM binaire peut retourner une liste [classe_0, classe_1] ou un array direct
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # classe positive = défaut

    # Tri par valeur absolue décroissante → features les plus impactantes en premier
    vals    = shap_vals[0]
    cols    = model.feature_name_
    factors = sorted(zip(cols, vals), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    # Prédiction associée pour contextualiser l'explication
    result = predict(model, features)

    return {
        "client_id":  client_id,
        "score":      result["score"],
        "decision":   result["decision"],
        "threshold":  result["threshold"],
        "top_factors": [
            {
                "feature":      feat,
                "contribution": round(float(val), 4),
                "direction":    "défavorable" if val > 0 else "favorable",
            }
            for feat, val in factors
        ],
    }


def update_stats(predictions: list, not_found: list):
    """
    Mise à jour des compteurs en mémoire après chaque appel à /predict.

    Appelé depuis l'endpoint /predict dans app.py après chaque réponse réussie.
    """
    _stats["total_requests"]       += 1
    _stats["total_clients_scored"] += len(predictions)
    _stats["total_not_found"]      += len(not_found)

    for p in predictions:
        if p.decision == "refusé":
            _stats["total_refused"]  += 1
        else:
            _stats["total_approved"] += 1
        _stats["scores"].append(p.score)


def get_stats() -> dict:
    """
    Calcul des statistiques agrégées depuis le démarrage de l'API.

    avg_score    : moyenne des scores de défaut prédits
    refusal_rate : proportion de décisions "refusé" parmi les clients scorés
    """
    scored    = _stats["total_clients_scored"]
    avg_score = round(float(np.mean(_stats["scores"])), 4) if _stats["scores"] else 0.0
    refusal_rate = (
        round(_stats["total_refused"] / scored, 4) if scored > 0 else 0.0
    )
    return {
        "total_requests":       _stats["total_requests"],
        "total_clients_scored": scored,
        "total_refused":        _stats["total_refused"],
        "total_approved":       _stats["total_approved"],
        "total_not_found":      _stats["total_not_found"],
        "avg_score":            avg_score,
        "refusal_rate":         refusal_rate,
    }


def get_model_info(model) -> dict:
    """
    Métadonnées du modèle en production : version, métriques d'entraînement, configuration.
    Combine les métadonnées fixes (issues du notebook) et les informations dynamiques du modèle chargé.
    """
    return {
        **_MODEL_METADATA,
        "n_features": len(model.feature_name_),
    }
