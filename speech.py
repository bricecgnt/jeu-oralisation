"""
Reconnaissance de mots isolés via Whisper (open source), 100 % local.

On utilise `faster-whisper` (réimplémentation efficace de Whisper, tourne sur CPU,
sans réseau une fois le modèle téléchargé). Le modèle est chargé paresseusement au
premier usage ; le premier lancement télécharge le modèle (~75–150 Mo selon la
taille) depuis Hugging Face, puis tout fonctionne hors-ligne.

La correspondance mot cible / transcription est tolérante (accents, ponctuation,
petites erreurs) pour rester jouable avec des voix d'enfants.
"""

from __future__ import annotations

import difflib
import re
import threading
import unicodedata

import numpy as np

# Taille de modèle Whisper : "tiny" (rapide, moins précis) < "base" < "small"
# (plus précis, plus lent). "base" est un bon compromis sur Mac.
DEFAULT_MODEL = "base"


class WordRecognizer:
    def __init__(self, model_size: str = DEFAULT_MODEL):
        self.model_size = model_size
        self.model = None
        self.load_error: str | None = None
        self._lock = threading.Lock()

    # ---- chargement paresseux ------------------------------------------
    def ensure_model(self) -> bool:
        if self.model is not None:
            return True
        with self._lock:
            if self.model is not None:
                return True
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    self.model_size, device="cpu", compute_type="int8")
                return True
            except Exception as e:  # pragma: no cover - dépend de l'environnement
                self.load_error = str(e)
                print("Whisper indisponible :", e)
                return False

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    # ---- transcription --------------------------------------------------
    def transcribe(self, audio: np.ndarray, samplerate: int = 16000) -> str:
        """Transcrit une trame audio mono float32 (16 kHz) en texte français."""
        if not self.ensure_model():
            return ""
        a = np.asarray(audio, dtype=np.float32).flatten()
        if a.size < samplerate // 4:        # moins de 0,25 s : rien à transcrire
            return ""
        try:
            segments, _ = self.model.transcribe(
                a, language="fr", vad_filter=True, beam_size=5,
                condition_on_previous_text=False)
            return " ".join(s.text for s in segments).strip()
        except Exception as e:  # pragma: no cover
            print("Erreur de transcription :", e)
            return ""

    # ---- correspondance -------------------------------------------------
    @staticmethod
    def normalize(s: str) -> str:
        """Minuscule, sans accents ni ponctuation, espaces normalisés."""
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = s.lower()
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def match(cls, target: str, heard: str, ratio: float = 0.85) -> bool:
        """Vrai si le mot cible correspond (exact, présent, ou proche) au texte
        entendu."""
        t = cls.normalize(target)
        h = cls.normalize(heard)
        if not t or not h:
            return False
        words = h.split()
        if t == h or t in words:
            return True
        # tolérance aux petites erreurs (Whisper, prononciation enfantine)
        if any(difflib.SequenceMatcher(None, t, w).ratio() >= ratio for w in words):
            return True
        return difflib.SequenceMatcher(None, t, h).ratio() >= max(ratio, 0.85)
