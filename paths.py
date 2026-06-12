"""
Résolution des chemins, valable que l'on lance le jeu en script Python ou en
application empaquetée (PyInstaller).

Deux besoins distincts :
  • ressources EN LECTURE embarquées dans l'app (modèles Whisper pré-téléchargés) :
    elles sont dans le bundle (lecture seule) -> `resource_base()`.
  • caches EN ÉCRITURE (images ARASAAC, modèles téléchargés à la volée) : on ne
    peut pas écrire dans un .app, donc on utilise un dossier utilisateur
    (~/Library/Caches sur macOS) -> `user_cache_dir()`.
"""

from __future__ import annotations

import os
import sys

APP_NAME = "jeu-oralisation"


def resource_base() -> str:
    """Dossier des ressources embarquées (lecture seule en app gelée)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_cache_dir() -> str:
    """Dossier de cache inscriptible, propre à l'utilisateur et persistant."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Caches", APP_NAME)
    elif os.name == "nt":
        base = os.path.join(os.environ.get("LOCALAPPDATA", home), APP_NAME, "Cache")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.join(home, ".cache"))
        base = os.path.join(xdg, APP_NAME)
    os.makedirs(base, exist_ok=True)
    return base


def pictos_cache_dir() -> str:
    """Cache des pictogrammes ARASAAC téléchargés."""
    d = os.path.join(user_cache_dir(), "pictos")
    os.makedirs(d, exist_ok=True)
    return d


def whisper_cache_dir() -> str:
    """Dossier où faster-whisper télécharge les modèles non embarqués."""
    d = os.path.join(user_cache_dir(), "whisper-models")
    os.makedirs(d, exist_ok=True)
    return d


def docs_dir() -> str:
    """Dossier de documents de l'utilisateur (listes de mots, récaps de séance) —
    visible et éditable par l'orthophoniste, contrairement au cache."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Documents", APP_NAME)
    elif os.name == "nt":
        base = os.path.join(home, "Documents", APP_NAME)
    else:
        base = os.path.join(home, APP_NAME)
    os.makedirs(base, exist_ok=True)
    return base


def lists_dir() -> str:
    """Dossier des listes de mots (.txt, un mot par ligne)."""
    d = os.path.join(docs_dir(), "listes")
    os.makedirs(d, exist_ok=True)
    return d


def exports_dir() -> str:
    """Dossier des récapitulatifs de séance exportés (CSV)."""
    d = os.path.join(docs_dir(), "seances")
    os.makedirs(d, exist_ok=True)
    return d


def bundled_model_dir(size: str) -> str | None:
    """Chemin d'un modèle Whisper embarqué dans l'app (s'il a été inclus au build),
    sinon None (le modèle sera téléchargé à la volée)."""
    d = os.path.join(resource_base(), "models", f"whisper-{size}")
    return d if os.path.isdir(d) else None
