"""
Logging JSON structuré pour l'API de scoring.

Chaque appel aux endpoints /predict et /predict/new produit une ligne JSON
dans stdout — capturée automatiquement par Render, Docker, ou tout agrégateur
de logs (Fluentd, Logstash, CloudWatch...).

Format d'une ligne de log :
    {"timestamp": "2026-07-10T10:41:17+00:00", "level": "INFO",
     "endpoint": "/predict", "score": 0.83, "latency_ms": 45.2, ...}
"""
import json
import logging
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """
    Formatteur personnalisé : convertit chaque LogRecord en une ligne JSON.

    Si le message est déjà du JSON (cas normal dans cette API), ses champs
    sont fusionnés avec timestamp et level pour former un objet plat.
    Si le message est du texte libre, il est mis dans le champ "message".
    """

    def format(self, record: logging.LogRecord) -> str:
        # Champs de base présents dans tous les logs
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
        }
        msg = record.getMessage()
        try:
            # Le message est du JSON → on le fusionne dans l'objet de base
            base.update(json.loads(msg))
        except (ValueError, TypeError):
            # Le message est du texte libre (ex: erreur inattendue)
            base["message"] = msg
        return json.dumps(base, ensure_ascii=False)


def get_logger(name: str = "scoring_api") -> logging.Logger:
    """
    Retourne un logger JSON configuré, en évitant les handlers dupliqués.

    Le guard `if not logger.handlers` empêche d'ajouter plusieurs fois le même
    handler si get_logger() est appelé plusieurs fois (ex: rechargement uvicorn).

    propagate=False : évite que les logs remontent au logger root de Python
    et soient affichés deux fois dans la console.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()      # écrit dans stdout
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
