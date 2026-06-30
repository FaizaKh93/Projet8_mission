"""
config.py — Configuration centralisée et portable du projet.

Ce module résout automatiquement tous les chemins à partir de
l'emplacement de ce fichier, quelle que soit la machine utilisée.
Il charge également les variables d'environnement depuis un fichier
.env s'il existe à la racine du projet.

Sélection GPU / CPU
-------------------
Par défaut le GPU est utilisé s'il est disponible.
Pour forcer le CPU, définir dans .env :
    DEVICE_TYPE=cpu

Usage :
    from src.config import DATA_DIR, MODELS_DIR, MLFLOW_TRACKING_URI
    from src.config import DEVICE, LGB_DEVICE, XGB_DEVICE
"""

import os
import logging
import subprocess
from pathlib import Path

# ===========================================================================
# 1. CHARGEMENT DU FICHIER .env
# ===========================================================================
# python-dotenv lit le fichier .env à la racine du projet et injecte
# chaque ligne "CLE=valeur" dans os.environ, exactement comme si tu
# avais tapé ces variables dans ton terminal.
# Si le fichier n'existe pas ou si la librairie n'est pas installée,
# on passe silencieusement — aucune erreur levée.
try:
    from dotenv import load_dotenv

    # __file__  = chemin absolu de CE fichier (src/config.py)
    # .parent   = dossier src/
    # .parent   = racine du projet  ← c'est là qu'on cherche .env
    _env_file = Path(__file__).resolve().parent.parent / ".env"

    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"[config] Variables d'environnement chargees depuis : {_env_file}")

except ImportError:
    # python-dotenv absent → les variables devront être définies autrement
    # (manuellement dans le terminal, ou via un système CI/CD)
    pass


# ===========================================================================
# 2. CHEMINS DU PROJET (portables, indépendants de la machine)
# ===========================================================================
# Principe : on part du fichier actuel (config.py) et on remonte
# l'arborescence pour construire tous les autres chemins.
# Résultat : le projet fonctionne sur n'importe quel PC,
# peu importe où il est cloné.

# Racine du projet : src/config.py → src/ → racine/
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Données brutes (fichiers CSV sources, jamais modifiés)
DATA_DIR: Path = BASE_DIR / "data"

# Données transformées après feature engineering / nettoyage
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Modèles entraînés sauvegardés (.pkl, .json, .cbm, …)
MODELS_DIR: Path = BASE_DIR / "models"

# Rapports exportés (HTML, PDF, CSV de résultats…)
REPORTS_DIR: Path = BASE_DIR / "reports"

# Graphiques et visualisations (courbes ROC, feature importance…)
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Notebooks Jupyter
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# Fichiers de logs applicatifs
LOGS_DIR: Path = BASE_DIR / "logs"


# ===========================================================================
# 3. CONFIGURATION MLFLOW
# ===========================================================================
# MLflow enregistre les expériences, métriques et modèles.
# Par défaut on utilise une base SQLite locale (mlflow.db à la racine).
# Sur un serveur partagé, on pointe vers l'URL du serveur MLflow distant
# en définissant MLFLOW_TRACKING_URI dans .env.

# URI de la base de tracking : SQLite local par défaut
# Exemple distant → MLFLOW_TRACKING_URI=http://mon-serveur:5000
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"  # chemin absolu vers mlflow.db
)

# Nom de l'expérience MLflow (groupe d'entraînements liés)
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "home_credit_scoring"
)


# ===========================================================================
# 4. PARAMÈTRES DU MODÈLE
# ===========================================================================
# Ces valeurs pilotent l'entraînement. Elles peuvent toutes être
# surchargées depuis .env sans toucher au code.

# Graine aléatoire → reproductibilité des splits et initialisations
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))

# Proportion des données réservées au test (0.2 = 20 %)
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))

# Nombre de folds pour la validation croisée (cross-validation)
N_FOLDS: int = int(os.getenv("N_FOLDS", "5"))

# Nom de la colonne cible à prédire (0 = remboursé, 1 = défaut)
TARGET_COL: str = os.getenv("TARGET_COL", "TARGET")

# Identifiant unique client → sert de clé pour les jointures de tables
ID_COL: str = os.getenv("ID_COL", "SK_ID_CURR")


# ===========================================================================
# 5. LOGGING (journalisation)
# ===========================================================================
# Configure le format et le niveau des messages affichés dans la console
# et enregistrés dans les fichiers de log.
# Niveaux disponibles (du plus verbeux au plus silencieux) :
#   DEBUG → INFO → WARNING → ERROR → CRITICAL

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    # getattr(logging, "INFO") → logging.INFO (valeur entière 20)
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    # Format d'affichage : heure | niveau | nom du module — message
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ===========================================================================
# 6. CRÉATION AUTOMATIQUE DES DOSSIERS
# ===========================================================================
# Si un dossier n'existe pas encore (ex : premier clone du projet),
# il est créé automatiquement.
# parents=True  → crée aussi les dossiers parents manquants
# exist_ok=True → pas d'erreur si le dossier existe déjà
for _dir in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 7. SÉLECTION GPU / CPU
# ===========================================================================

def _detect_gpu_info() -> dict:
    """
    Interroge nvidia-smi pour récupérer les infos de la carte graphique.

    nvidia-smi est l'outil officiel NVIDIA installé avec les drivers.
    On l'appelle en sous-processus (comme dans un terminal) et on parse
    sa sortie CSV.

    Retourne un dict {"name", "vram_mb", "driver"} ou {} si pas de GPU.
    Ne lève jamais d'exception (machine sans GPU, driver absent, timeout…).
    """
    try:
        # Lance la commande :
        # nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode
        #            --format=csv,noheader,nounits
        # Exemple de sortie : "NVIDIA RTX A1000 6GB Laptop GPU, 6144, 528.90, Default"
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_mode",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,   # capture stdout et stderr
            text=True,             # décode en string (pas en bytes)
            timeout=5,             # abandonne si nvidia-smi met > 5 sec
        )

        if result.returncode == 0 and result.stdout.strip():
            # Découpe la ligne CSV en parties
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "name":    parts[0] if len(parts) > 0 else "unknown",
                "vram_mb": int(parts[1]) if len(parts) > 1 else 0,   # en Mo
                "driver":  parts[2] if len(parts) > 2 else "unknown",
            }

    except Exception:
        # Cas possibles : nvidia-smi absent, permission refusée, timeout…
        pass

    return {}  # Aucun GPU détecté ou erreur


def _torch_cuda_available() -> bool:
    """
    Vérifie si PyTorch peut utiliser un GPU CUDA.

    torch.cuda.is_available() retourne True uniquement si :
      - PyTorch est installé en version CUDA (ex: torch==2.4.1+cu118)
      - Le driver NVIDIA est compatible
      - Un GPU est physiquement présent

    Si PyTorch n'est pas installé du tout, retourne False sans planter.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False  # PyTorch absent → on suppose CPU


# ------------------------------------------------------------------
# Lecture de la préférence utilisateur depuis .env
# ------------------------------------------------------------------
# Valeurs acceptées (insensibles à la casse) :
#   "auto"  → GPU si disponible, sinon CPU  (défaut)
#   "cuda"  → Force GPU (erreur si CUDA absent)
#   "cpu"   → Force CPU (même si un GPU est présent)
_DEVICE_PREF: str = os.getenv("DEVICE_TYPE", "auto").lower()

# Appels aux fonctions de détection définies ci-dessus
_gpu_info: dict = _detect_gpu_info()       # infos hardware (nom, VRAM…)
_cuda_ok: bool  = _torch_cuda_available()  # CUDA utilisable par PyTorch ?

# ------------------------------------------------------------------
# Décision finale : GPU ou CPU ?
# ------------------------------------------------------------------
if _DEVICE_PREF == "cpu":
    # L'utilisateur force le CPU → on n'utilise pas le GPU
    USE_GPU: bool = False

elif _DEVICE_PREF == "cuda":
    # L'utilisateur force le GPU → on vérifie qu'il est bien disponible
    if not _cuda_ok:
        raise EnvironmentError(
            "DEVICE_TYPE=cuda demande mais CUDA n'est pas disponible. "
            "Verifiez votre installation ou mettez DEVICE_TYPE=cpu dans .env"
        )
    USE_GPU: bool = True

else:
    # Mode "auto" : on utilise le GPU s'il est disponible, sinon le CPU
    USE_GPU: bool = _cuda_ok


# ------------------------------------------------------------------
# Device PyTorch  (torch.device)
# ------------------------------------------------------------------
# torch.device("cuda") ou torch.device("cpu") est l'objet standard
# PyTorch pour indiquer où les tenseurs et modèles sont stockés.
# Exemple d'utilisation :
#   model = MonModele().to(DEVICE)
#   tensor = torch.tensor([1.0]).to(DEVICE)
try:
    import torch as _torch
    DEVICE = _torch.device("cuda" if USE_GPU else "cpu")
except ImportError:
    DEVICE = None  # PyTorch non installé → DEVICE inutilisable


# ------------------------------------------------------------------
# Device pour LightGBM
# ------------------------------------------------------------------
# Priorité : "cuda" (build custom) > "gpu" (OpenCL, build PyPI standard) > "cpu"
# Le build PyPI standard inclut le support OpenCL (device="gpu").
# Exemple d'utilisation :
#   lgb.LGBMClassifier(device=LGB_DEVICE)

def _lgbm_best_device() -> str:
    """
    Détecte le meilleur device LightGBM disponible.
    Teste "cuda" puis "gpu" (OpenCL) — retourne "cpu" si aucun ne fonctionne.
    """
    if not USE_GPU:
        return "cpu"
    try:
        import lightgbm as _lgb
        import numpy as _np
        _X = _np.random.rand(20, 2)
        _y = _np.random.randint(0, 2, 20)
        for _dev in ("cuda", "gpu"):
            try:
                _m = _lgb.LGBMClassifier(n_estimators=1, device=_dev, verbosity=-1)
                _m.fit(_X, _y)
                return _dev   # premier device qui fonctionne
            except Exception:
                continue
    except ImportError:
        pass
    return "cpu"

LGB_DEVICE: str = _lgbm_best_device()

_lgb_logger = logging.getLogger(__name__)
if USE_GPU and LGB_DEVICE == "cpu":
    _lgb_logger.warning(
        "LightGBM : aucun device GPU disponible (CUDA ni OpenCL). "
        "LGB_DEVICE force a 'cpu'."
    )
else:
    _lgb_logger.info(f"LightGBM device : {LGB_DEVICE}")


# ------------------------------------------------------------------
# Device pour XGBoost (version >= 2.0)
# ------------------------------------------------------------------
# XGBoost >= 2.0 utilise device="cuda" pour le GPU.
# Les anciennes versions utilisaient tree_method="gpu_hist".
# Exemple d'utilisation :
#   xgb.XGBClassifier(device=XGB_DEVICE)
XGB_DEVICE: str = "cuda" if USE_GPU else "cpu"


# ------------------------------------------------------------------
# Description lisible du device sélectionné
# ------------------------------------------------------------------
# Sert uniquement à l'affichage (logs, résumé de config).
if USE_GPU:
    _gpu_name = _gpu_info.get("name", "GPU inconnu")
    _vram     = _gpu_info.get("vram_mb", 0)
    _driver   = _gpu_info.get("driver", "?")
    # Exemple : "GPU - NVIDIA RTX A1000 6GB | VRAM 6.0 GB | driver 528.90"
    DEVICE_INFO: str = (
        f"GPU - {_gpu_name} | VRAM {_vram / 1024:.1f} GB | driver {_driver}"
    )
else:
    import platform
    # Exemple : "CPU - Intel64 Family 6 Model 141"
    DEVICE_INFO: str = f"CPU - {platform.processor() or platform.machine()}"


# ===========================================================================
# 8. POINT D'ENTRÉE DIRECT  (python src/config.py)
# ===========================================================================
# Ce bloc ne s'exécute QUE si on lance ce fichier directement.
# Il est ignoré quand on fait "from src.config import ...".
# Utile pour vérifier rapidement la configuration sur une nouvelle machine.
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CONFIGURATION DU PROJET")
    print("=" * 60)
    print(f"  BASE_DIR              : {BASE_DIR}")
    print(f"  DATA_DIR              : {DATA_DIR}")
    print(f"  PROCESSED_DIR         : {PROCESSED_DIR}")
    print(f"  MODELS_DIR            : {MODELS_DIR}")
    print(f"  REPORTS_DIR           : {REPORTS_DIR}")
    print(f"  LOGS_DIR              : {LOGS_DIR}")
    print()
    print(f"  MLFLOW_TRACKING_URI   : {MLFLOW_TRACKING_URI}")
    print(f"  MLFLOW_EXPERIMENT     : {MLFLOW_EXPERIMENT_NAME}")
    print()
    print(f"  RANDOM_STATE          : {RANDOM_STATE}")
    print(f"  TEST_SIZE             : {TEST_SIZE}")
    print(f"  N_FOLDS               : {N_FOLDS}")
    print(f"  TARGET_COL            : {TARGET_COL}")
    print(f"  ID_COL                : {ID_COL}")
    print()
    print(f"  -- Device ------------------------------------------")
    print(f"  USE_GPU               : {USE_GPU}")
    print(f"  DEVICE                : {DEVICE}")
    print(f"  LGB_DEVICE            : {LGB_DEVICE}")
    print(f"  XGB_DEVICE            : {XGB_DEVICE}")
    print(f"  DEVICE_INFO           : {DEVICE_INFO}")
    print("=" * 60)
