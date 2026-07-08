"""
Schémas Pydantic — contrat d'entrée/sortie de l'API.

Pydantic valide automatiquement les types à la réception de chaque requête.
Toute valeur non conforme → réponse HTTP 422 sans atteindre le modèle.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union


# ── Requête ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Corps de la requête POST /predict.

    client_ids accepte :
        - un entier unique    : {"client_ids": 100002}
        - une liste d'entiers : {"client_ids": [100002, 100003, 100004]}

    Le validateur normalise automatiquement un entier seul en liste d'un élément.
    """
    client_ids: Union[int, List[int]] = Field(
        ...,
        description="Un identifiant SK_ID_CURR ou une liste d'identifiants",
    )

    @field_validator("client_ids", mode="before")
    @classmethod
    def normalize_to_list(cls, v):
        """Conversion d'un entier unique en liste — uniformisation pour l'endpoint."""
        return [v] if isinstance(v, int) else v

    model_config = {
        "json_schema_extra": {
            "examples": {
                "client unique":     {"value": {"client_ids": 100002}},
                "plusieurs clients": {"value": {"client_ids": [100002, 100003, 100004]}},
            }
        }
    }


# ── Réponses /predict ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Prédiction pour un client unique.

    client_id : identifiant SK_ID_CURR du client scoré
    score     : probabilité de défaut entre 0.0 et 1.0
    decision  : 'accordé' si score < seuil, 'refusé' sinon
    threshold : seuil métier utilisé (0.46 — optimisé sur coût 10×FN + FP)
    """
    client_id: Optional[int] = Field(None, description="Identifiant SK_ID_CURR (None pour un nouveau client)")
    score:     float         = Field(...,  description="Probabilité de défaut (0.0 → 1.0)")
    decision:  str           = Field(...,  description="'accordé' ou 'refusé'")
    threshold: float         = Field(...,  description="Seuil de décision métier")


class BatchResponse(BaseModel):
    """
    Réponse unique pour toutes les requêtes POST /predict.

    predictions : prédictions pour les clients trouvés dans le dataset
    not_found   : identifiants absents du dataset (aucune prédiction possible)
    """
    predictions: List[PredictionResponse] = Field(..., description="Prédictions réussies")
    not_found:   List[int]                = Field(..., description="Identifiants introuvables")


# ── Requête /predict/new ─────────────────────────────────────────────────────

class NewClientRequest(BaseModel):
    """
    Corps de la requête POST /predict/new — nouveau client sans historique dans le dataset.

    Champs
    ------
    client_id : identifiant optionnel assigné par l'appelant (ex: numéro de dossier)
    features  : features pré-calculées en amont par le pipeline de feature engineering
                - colonnes manquantes → NaN (LightGBM gère nativement)
                - valeurs None acceptées (équivalent NaN)
                - valeur non numérique → erreur 422

    Cas d'usage : demande de crédit en temps réel, client sans historique chez Home Credit.
    Le feature engineering (bureau, previous_application, etc.) doit avoir été fait en amont.
    """
    client_id: Optional[int]               = Field(None, description="Identifiant optionnel du nouveau client")
    features:  Dict[str, Optional[float]]  = Field(
        ...,
        description="Features pré-calculées du client (clé = nom de colonne, valeur = float ou null)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "client_id": None,
                "features": {
                    "EXT_SOURCE_1":    0.50,
                    "EXT_SOURCE_2":    0.60,
                    "EXT_SOURCE_3":    0.45,
                    "EXT_SOURCE_MEAN": 0.52,
                    "AMT_INCOME_TOTAL": 202500.0,
                    "AMT_CREDIT":       406597.5,
                    "AMT_ANNUITY":      24700.5,
                    "DAYS_BIRTH":      -9461.0,
                    "DAYS_EMPLOYED":   -637.0,
                    "PAYMENT_RATE":     0.061,
                },
            }
        }
    }


# ── Réponse /predict/{client_id}/explain ─────────────────────────────────────

class ShapFactor(BaseModel):
    """
    Contribution d'une feature à la prédiction d'un client (valeur SHAP).

    feature      : nom de la colonne
    contribution : valeur SHAP — magnitude = importance, signe = direction
    direction    : 'défavorable' (pousse vers défaut) ou 'favorable' (pousse vers remboursement)
    """
    feature:      str   = Field(..., description="Nom de la feature")
    contribution: float = Field(..., description="Valeur SHAP (positive = défavorable)")
    direction:    str   = Field(..., description="'favorable' ou 'défavorable'")


class ExplainResponse(BaseModel):
    """
    Explication locale SHAP pour un client — pourquoi cette décision.

    top_factors : features classées par impact décroissant sur la prédiction
    """
    client_id:   int              = Field(..., description="Identifiant SK_ID_CURR")
    score:       float            = Field(..., description="Probabilité de défaut")
    decision:    str              = Field(..., description="'accordé' ou 'refusé'")
    threshold:   float            = Field(..., description="Seuil de décision métier")
    top_factors: List[ShapFactor] = Field(..., description="Features les plus impactantes")


# ── Réponse /model/info ───────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    """
    Métadonnées du modèle actuellement en production.

    Combine les informations fixes (issues du notebook d'entraînement)
    et les informations dynamiques extraites du modèle chargé en mémoire.
    """
    model_type:            str   = Field(..., description="Famille de modèle")
    n_features:            int   = Field(..., description="Nombre de features en entrée")
    threshold:             float = Field(..., description="Seuil de décision métier")
    auc_test:              float = Field(..., description="AUC sur le jeu de test")
    auc_oof:               float = Field(..., description="AUC OOF (validation croisée)")
    business_cost_formula: str   = Field(..., description="Formule du coût métier")
    business_cost_test:    int   = Field(..., description="Coût métier sur le jeu de test")
    training_samples:      int   = Field(..., description="Nombre d'exemples d'entraînement")
    test_samples:          int   = Field(..., description="Nombre d'exemples de test")
    recall_test:           float = Field(..., description="Recall sur le jeu de test")
    precision_test:        float = Field(..., description="Précision sur le jeu de test")


# ── Réponse /stats ────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    """
    Statistiques d'usage de l'API depuis le dernier démarrage.

    Compteurs remis à zéro à chaque redémarrage du service.
    Utile pour le monitoring en temps réel et la détection de dérives.
    """
    total_requests:       int   = Field(..., description="Nombre total d'appels à /predict")
    total_clients_scored: int   = Field(..., description="Clients effectivement scorés")
    total_refused:        int   = Field(..., description="Décisions 'refusé'")
    total_approved:       int   = Field(..., description="Décisions 'accordé'")
    total_not_found:      int   = Field(..., description="Identifiants introuvables")
    avg_score:            float = Field(..., description="Score de défaut moyen")
    refusal_rate:         float = Field(..., description="Taux de refus (0.0 → 1.0)")
