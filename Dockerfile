# ── Image de base ─────────────────────────────────────────────────────────────
# python:3.11-slim : Debian minimal, pas d'Alpine (compatibilité binaires scipy/numpy)
FROM python:3.11-slim

# ── Installation de uv ────────────────────────────────────────────────────────
# Copie du binaire uv depuis l'image officielle — plus rapide que pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Dépendances ───────────────────────────────────────────────────────────────
# Copie des fichiers de dépendances en premier : Docker met cette couche en cache
# tant que pyproject.toml et uv.lock ne changent pas → rebuild rapide
COPY pyproject.toml uv.lock ./

# --frozen        : utilise exactement les versions de uv.lock (reproductibilité)
# --no-dev        : exclut pytest, httpx (outils de développement)
# --no-group gpu  : exclut torch+CUDA (~2 Go) — inutile pour l'inférence LightGBM
# --no-cache      : pas de cache uv dans l'image (réduit la taille finale)
RUN uv sync --frozen --no-dev --no-group gpu --no-cache

# ── Code applicatif ───────────────────────────────────────────────────────────
COPY api/ ./api/

# ── Variables d'environnement ─────────────────────────────────────────────────
# PATH vers le virtualenv créé par uv dans /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# MLflow : le dossier mlruns/ est monté en volume au runtime (pas embarqué dans l'image)
# Exemple : docker run -v ./mlruns:/app/mlruns ...
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Logs en temps réel (pas de buffering Python)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Réseau ────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Utilise Python natif — pas besoin d'installer curl
# --start-period=60s : laisse le temps à load_artifacts() de charger le modèle MLflow
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Démarrage ─────────────────────────────────────────────────────────────────
# Volumes à monter au runtime :
#   -v ./mlruns:/app/mlruns                              (MLflow Registry)
#   -v ./data/processed:/app/data/processed              (dataset CSV pour /predict)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
