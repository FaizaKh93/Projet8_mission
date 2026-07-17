# Projet 8 — API de Scoring Crédit « Prêt à Dépenser »

API de scoring crédit déployée sur Render, prédisant la probabilité de défaut de
remboursement d'un client à partir d'un modèle LightGBM.

**API live** : https://projet8-credit-scoring-api.onrender.com/docs

---

## Structure du projet

```
.
├── api/
│   ├── app.py              # Endpoints FastAPI (/predict, /explain, /stats, /logs…)
│   ├── model.py            # Chargement du modèle, DuckDB, cache LRU, SHAP
│   └── logger.py           # Logging JSON structuré vers stdout
│
├── notebooks/
│   ├── drift_analysis.ipynb          # Étape 3 — détection du data drift (Evidently + KS)
│   └── performance_profiling.ipynb   # Étape 4 — profiling cProfile + optimisation cache LRU
│
├── tests/
│   └── test_api.py         # Tests unitaires FastAPI (pytest + mocks)
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # CI/CD : tests pytest + build & push image Docker (GHCR)
│
├── data/
│   └── processed/          # Parquet (ignoré par git, hébergé sur HF Hub)
│
├── models/
│   └── lgbm_optimized.pkl  # Modèle LightGBM entraîné sur 80% des clients
│
├── Dockerfile
├── pyproject.toml
└── .gitignore
```

---

## API — Endpoints principaux

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Bienvenue + liste des endpoints |
| GET | `/health` | Disponibilité de l'API |
| POST | `/predict` | Score de défaut pour un ou plusieurs clients existants |
| POST | `/predict/new` | Score pour un nouveau client (features fournies directement) |
| GET | `/predict/{id}/explain` | Explication SHAP locale (top 10 features) |
| GET | `/model/info` | Métadonnées du modèle (AUC, seuil, nombre de features) |
| GET | `/stats` | Compteurs depuis le démarrage (requêtes, taux de refus…) |
| GET | `/logs` | 100 derniers appels loggés (ordre anti-chronologique) |

Documentation interactive : `/docs` (Swagger UI)

---

## Modèle

- **Algorithme** : LightGBM (552 features, seuil de décision = 0.46)
- **Métrique** : AUC = 0.7893 — optimisée sur la fonction de coût métier `10×FN + FP`
- **Registre** : MLflow Registry (`lgbm-credit-scoring@champion`) + HF Hub (`Faiza93/projet8-credit-scoring`)
- **Données** : Parquet hébergé sur HF Hub (`Faiza93/projet8-credit-scoring-data`)

---

## Installation locale

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
git clone https://github.com/FaizaKh93/Projet8_mission.git
cd Projet8_mission
uv sync
```

### Variables d'environnement

| Variable | Valeur | Description |
|---|---|---|
| `MODEL_SOURCE` | `hf` \| `mlflow` | Source du modèle (`hf` pour Render, `mlflow` en local) |
| `MLFLOW_TRACKING_URI` | `file:///…/mlruns` | URI MLflow (mode local uniquement) |

### Lancer l'API en local

```bash
MODEL_SOURCE=mlflow uv run python -m uvicorn api.app:app --reload --port 8080
```

---

## Tests

```bash
uv run pytest tests/ -v
```

Les tests couvrent : `/predict`, `/predict/new`, `/explain`, `/model/info`, `/stats`  
avec mocks du modèle, de l'explainer SHAP et du dataset (aucun fichier réel chargé).

---

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) se déclenche sur chaque push :

1. **Tests** (`pytest`) sur toutes les branches
2. **Build & Push** image Docker vers GHCR (`ghcr.io/faizakh93/projet8-scoring-api`) uniquement sur `main`

Render se connecte à la branche `main` et redéploie automatiquement à chaque merge.

---

## Notebooks

| Notebook | Description |
|---|---|
| `drift_analysis.ipynb` | Analyse du data drift entre données de référence (80%) et production (20%) avec Evidently et test KS — 0/552 features driftées |
| `performance_profiling.ipynb` | Profiling cProfile + timeit du pipeline `/predict` — goulot identifié : DuckDB (98%). Optimisation : cache LRU ×13 en production |
