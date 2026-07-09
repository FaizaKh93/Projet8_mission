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

# MODEL_SOURCE=hf : charge le modèle et le dataset depuis HF Hub au démarrage
# Pour le dev local avec MLflow : docker run -e MODEL_SOURCE=mlflow -v ./mlruns:/app/mlruns
ENV MODEL_SOURCE=hf

# Logs en temps réel (pas de buffering Python)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Réseau ────────────────────────────────────────────────────────────────────
# Port 7860 : standard HF Spaces Docker
EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
# --start-period=120s : laisse le temps de télécharger le modèle et le dataset (~170MB) depuis HF Hub
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# ── Démarrage ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
