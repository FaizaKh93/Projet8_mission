"""
data_loading.py — Chargement des fichiers CSV bruts.

Resout le chemin vers data/ a partir de la position de ce fichier,
independamment de la machine ou du repertoire de travail courant.
"""
from pathlib import Path
import pandas as pd
#from src.config import DATA_DIR

# Chemin absolu vers la racine du projet
# __file__ = chemin du fichier courant (data_loading.py)
# .resolve() = chemin absolu
# .parent.parent = remonte de deux niveaux (src/ → projet/)
BASE_DIR = Path(__file__).resolve().parent.parent 

# Chemin vers le dossier contenant les données brutes
DATA_DIR = BASE_DIR / "data"

def load_csv(filename, nrows=None):
    """
    Charge un fichier CSV depuis le dossier data/.

    Paramètres
    ----------
    filename : str
        Nom du fichier CSV.
    nrows : int, optionnel
        Nombre de lignes à lire.

    Retour
    ------
    pandas.DataFrame
    """
    return pd.read_csv(DATA_DIR / filename, nrows=nrows)