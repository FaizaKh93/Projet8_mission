# Projet 6 — Scoring Crédit « Prêt à Dépenser »

Modèle de scoring crédit pour prédire la probabilité de défaut de remboursement
des clients, en s'appuyant sur les données du challenge Kaggle
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

---

## Objectif

Construire un modèle binaire (`TARGET` : 0 = remboursé, 1 = défaut) qui minimise
une **métrique métier asymétrique** : un défaillant manqué coûte 10× plus qu'un
bon client refusé.

```
Coût métier = 10 × FN + 1 × FP
```

---

## Structure du projet

```
.
├── data/
│   ├── processed/                  # Données transformées (ignorées par git)
│   └── README_data.md              # Description des fichiers sources attendus
│
├── models/
│   ├── lr_optimized.pkl            # Logistic Regression finale (sklearn Pipeline)
│   └── lgbm_optimized.pkl          # LightGBM final
│
├── notebooks/
│   ├── data_preparation.ipynb      # Préparation et nettoyage des données
│   ├── exploratory_analysis.ipynb  # Analyse exploratoire (EDA)
│   ├── modeling_local.ipynb        # Modélisation sans MLflow (36 cellules)
│   └── modeling_mlflow.ipynb       # Modélisation avec tracking MLflow (50 cellules)
│
├── reports/
│   ├── figures/                    # Graphiques générés (ROC, SHAP, matrice…)
│
├── src/
│   ├── config.py                   # Chemins, paramètres, détection GPU/CPU
│   ├── data_loading.py             # Chargement des CSV bruts
│   ├── feature_engineering.py      # Pipeline de feature engineering par table
│   └── build_dataset.py            # Assemblage du dataset multi-tables final
│
├── .env.example                    # Variables d'environnement (copier en .env)
├── pyproject.toml                  # Dépendances (uv)
└── README.md
```

---

## Données sources

Six tables du challenge Home Credit à placer dans `data/` :

| Fichier                     | Description                                             |
|-----------------------------|---------------------------------------------------------|
| `application_train.csv`     | Données principales — une ligne par demande de crédit   |
| `bureau.csv`                | Historique des crédits externes (autres établissements) |
| `bureau_balance.csv`        | Historique mensuel des crédits bureau                   |
| `previous_application.csv`  | Anciennes demandes de crédit chez Home Credit           |
| `POS_CASH_balance.csv`      | Historique mensuel crédits POS/Cash                     |
| `installments_payments.csv` | Historique des remboursements par échéance              |
| `credit_card_balance.csv`   | Historique des cartes de crédit                         |

---

## Feature Engineering (`src/`)

Chaque table est traitée indépendamment puis jointe au niveau client (`SK_ID_CURR`) :

| Fonction                   | Table source            | Variables créées                                                                |
|----------------------------|-------------------------|---------------------------------------------------------------------------------|
| `application_train_test()` | application_train.csv   | Ratios financiers : revenu/crédit, annuité/revenu, revenu/foyer, ancienneté/âge |
| `bureau_and_balance()`     | bureau + bureau_balance | Statistiques crédits actifs/clôturés, ratios dette et retard                    |
| `previous_applications()`  | previous_application    | Statistiques demandes approuvées/refusées, ratio crédit/demande, apport         |
| `pos_cash()`               | POS_CASH_balance        | Indicateurs retards DPD, volume d'historique                                    |
| `installments_payments()`  | installments_payments   | DPD, DBD, taux respect des échéances, flag retard                               |
| `credit_card_balance()`    | credit_card_balance     | Utilisation de la limite, variabilité des soldes et paiements                   |

Le dataset final contient **~552 features** pour ~307 000 clients.

---

## Modélisation

### Méthodologie

```
X, y  ──►  train_test_split stratifié  (80 % train / 20 % test)
                    │
            X_train, y_train  ──►  Validation croisée StratifiedKFold (5 folds)
                                   Optimisation des hyperparamètres
                    │
            X_test, y_test    ──►  Évaluation finale non biaisée (jamais vue pendant l'entraînement)
```

### Étape 3 — Comparaison des modèles baseline

Trois modèles évalués en validation croisée OOF sur `X_train` :

| Modèle                          | AUC OOF |
|---------------------------------|---------|
| Dummy Classifier                | 0.50    |
| Logistic Regression (C=0.1, L2) | 0.7724  |
| LightGBM baseline               | 0.7859  |

### Étape 4 — Optimisation des hyperparamètres

- **Logistic Regression** : `GridSearchCV` sur 20 % de `X_train`
- **LightGBM** : `RandomizedSearchCV` (30 combinaisons) puis `GridSearchCV`
  (fine-tuning autour du meilleur résultat), CV avec early stopping (100 rounds)

### Résultats finaux — LightGBM optimisé

| Métrique               | Valeur                              |
|------------------------|-------------------------------------|
| **AUC Test**           | **0.7893**                          |
| AUC OOF (train)        | 0.7875                              |
| Gap OOF / Test         | −0.0018 — aucun overfitting détecté |
| Seuil optimal          | 0.46                                |
| Recall                 | 0.6628                              |
| Précision              | 0.2013                              |
| **Coût métier (test)** | **29 797**                          |

Matrice de confusion sur le jeu de test (61 502 clients, seuil = 0.46) :

|                         | Prédit Non-défaillant | Prédit Défaillant |
|-------------------------|-----------------------|-------------------|
| **Réel Non-défaillant** | 43 480 (VN)           | 13 057 (FP)       |
| **Réel Défaillant**     | 1 674 (FN)            | 3 291 (VP)        |

### Interprétabilité — SHAP

- **Globale** : beeswarm plot + bar chart groupé (colonnes one-hot regroupées
  par variable d'origine)
- **Locale** : waterfall plots pour un client non-défaillant (proba < 20 %)
  et un client défaillant (proba > 53 %)
- Calcul sur un échantillon de 5 000 clients de `X_train`

---

## Tracking MLflow

Chaque run MLflow est tracé avec les métriques OOF + test et les tags suivants :

```python
mlflow.set_tag('model_type', 'baseline' | 'optimise' | 'evaluation' | 'interpretabilite')
mlflow.set_tags({
    'split_type': 'stratified',
    'test_size':  '0.2',
    'n_train':    str(len(X_train)),
    'n_test':     str(len(X_test)),
})
```

Les modèles finaux sont enregistrés dans le **Model Registry** :
- `LR_credit_scoring`
- `LightGBM_credit_scoring`

Lancer l'interface MLflow :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Ouvrir http://localhost:5000
```

---

## Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
# Cloner le repo
git clone https://github.com/FaizaKh93/Projet6_mission.git
cd Projet6_mission

# Installer les dépendances
uv sync

# Copier et adapter le fichier d'environnement
cp .env.example .env
```

> **GPU** : le projet détecte automatiquement CUDA (LightGBM + PyTorch).
> Pour forcer le CPU, définir `DEVICE_TYPE=cpu` dans `.env`.

### Variables `.env`

| Variable               | Défaut                | Description               |
|------------------------|-----------------------|---------------------------|
| `DEVICE_TYPE`          | `auto`                | `auto` / `cuda` / `cpu`   |
| `MLFLOW_TRACKING_URI`  | `sqlite:///mlflow.db` | URI du serveur MLflow     |
| `RANDOM_STATE`         | `42`                  | Graine aléatoire          |
| `TEST_SIZE`            | `0.2`                 | Proportion du jeu de test |
| `N_FOLDS`              | `5`                   | Nombre de folds CV        |

---

## Lancer les notebooks

```bash
uv run jupyter notebook
```

Ordre d'exécution recommandé :

1. `notebooks/data_preparation.ipynb`
2. `notebooks/exploratory_analysis.ipynb`
3. `notebooks/modeling_mlflow.ipynb`

---

## Livrables

| Fichier                           | Description                                    |
|-----------------------------------|------------------------------------------------|
| `notebooks/modeling_mlflow.ipynb` | Notebook principal avec tracking MLflow        |
| `notebooks/modeling_local.ipynb`  | Version sans MLflow                            |
| `models/lgbm_optimized.pkl`       | Modèle LightGBM entraîné sur tout X            |
| `models/lr_optimized.pkl`         | Modèle Logistic Regression entraîné sur tout X |
