"""
Synthèse vocale locale pour le mode « Écoute » — via la commande macOS `say`
(gratuite, hors-ligne, voix françaises de qualité). Aucune dépendance Python.

Sur un système sans `say` (Linux/Windows), `available()` renvoie False et le
mode Écoute affiche un message explicatif.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading

# Voix françaises préférées, par ordre de priorité (toutes fournies par macOS ;
# certaines doivent être téléchargées dans Réglages > Accessibilité > Contenu
# énoncé > Voix).
_PREFERRED = ["Thomas", "Audrey", "Amélie", "Amelie", "Aurélie", "Aurelie",
              "Marie", "Daniel"]

_voice: str | None = None
_checked = False
_lock = threading.Lock()


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("say") is not None


def _detect_voice() -> str | None:
    """Choisit une voix française installée (None = voix système par défaut)."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    french = []
    for line in out.splitlines():
        if "fr_" in line or "fr-" in line:
            french.append(line.split()[0])
    for pref in _PREFERRED:
        if pref in french:
            return pref
    return french[0] if french else None


def voice_name() -> str | None:
    """Voix utilisée (détectée une seule fois)."""
    global _voice, _checked
    with _lock:
        if not _checked:
            _voice = _detect_voice() if available() else None
            _checked = True
    return _voice


def speak(text: str, rate: int = 150) -> bool:
    """Prononce `text` (non bloquant). Renvoie False si la synthèse est
    indisponible."""
    if not available() or not text.strip():
        return False
    v = voice_name()
    cmd = ["say", "-r", str(rate)]
    if v:
        cmd += ["-v", v]
    cmd.append(text)
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
