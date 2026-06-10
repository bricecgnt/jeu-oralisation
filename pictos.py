"""
Récupération automatique d'un pictogramme depuis ARASAAC pour un mot donné.

ARASAAC (Gouvernement d'Aragon) propose une base de pictogrammes libres
(CC BY-NC-SA) très utilisée en orthophonie, avec une API publique.

  • recherche : https://api.arasaac.org/api/pictograms/fr/search/{mot}
  • image     : https://static.arasaac.org/pictograms/{id}/{id}_300.png

Les images sont mises en cache sur le disque (`pictos_cache/`) : le 1er accès à un
mot nécessite Internet, ensuite c'est hors-ligne. En cas d'absence de réseau ou de
résultat, la fonction renvoie None (le jeu fonctionne sans image).

Diagnostic en ligne de commande :
    python pictos.py chat

Attribution requise par la licence : « Pictogrammes : ARASAAC (arasaac.org) ».
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from paths import pictos_cache_dir

CACHE_DIR = pictos_cache_dir()
SEARCH_URL = "https://api.arasaac.org/api/pictograms/fr/search/{}"
IMAGE_URL = "https://static.arasaac.org/pictograms/{id}/{id}_300.png"
_UA = {"User-Agent": "Mozilla/5.0 (jeu-oralisation)"}


# --------------------------------------------------------------------------
# Réseau (gestion robuste du SSL, notamment sur macOS où les certificats Python
# ne sont pas toujours installés -> CERTIFICATE_VERIFY_FAILED).
# --------------------------------------------------------------------------
def _make_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_CTX = _make_context()


def _unverified_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _is_ssl_error(err: Exception) -> bool:
    """Vrai si l'erreur est (ou enveloppe) une erreur de vérification SSL.
    `urlopen` lève un URLError dont `.reason` est l'erreur SSL — d'où la
    nécessité de regarder à l'intérieur (sinon le repli ne se déclenche jamais)."""
    if isinstance(err, ssl.SSLError):
        return True
    reason = getattr(err, "reason", None)
    return isinstance(reason, ssl.SSLError)


def _open(url: str, timeout: float):
    req = urllib.request.Request(url, headers=_UA)
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=_CTX)
    except (urllib.error.URLError, ssl.SSLError) as e:
        # Repli macOS : certificats Python absents -> erreur de vérification SSL.
        # On réessaie sans vérification (acceptable pour des pictos publics).
        if _is_ssl_error(e):
            return urllib.request.urlopen(req, timeout=timeout,
                                          context=_unverified_ctx())
        raise


# --------------------------------------------------------------------------
# Normalisation / cache
# --------------------------------------------------------------------------
def _norm(word: str) -> str:
    """Minuscule, sans accents, espaces normalisés (pour la requête de recherche)."""
    s = unicodedata.normalize("NFD", word)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", s).strip()


def _slug(word: str) -> str:
    """Identifiant de fichier (cache) : lettres/chiffres, séparés par '_'."""
    return re.sub(r"[^a-z0-9]+", "_", _norm(word)).strip("_")


def cached_path(word: str) -> str | None:
    slug = _slug(word)
    if not slug:
        return None
    path = os.path.join(CACHE_DIR, slug + ".png")
    return path if os.path.exists(path) else None


# --------------------------------------------------------------------------
# API ARASAAC
# --------------------------------------------------------------------------
def search_id(word: str, timeout: float = 6.0):
    """Renvoie l'identifiant du 1er pictogramme correspondant, ou None."""
    q = urllib.parse.quote(_norm(word))
    with _open(SEARCH_URL.format(q), timeout) as r:
        data = json.load(r)
    if not data:
        return None
    return data[0].get("_id") or data[0].get("id")


def fetch_picto(word: str, timeout: float = 6.0) -> str | None:
    """Chemin local d'un pictogramme pour `word` (téléchargé puis mis en cache),
    ou None si introuvable / hors-ligne."""
    slug = _slug(word)
    if not slug:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, slug + ".png")
    if os.path.exists(path):
        return path
    try:
        pid = search_id(word, timeout)
        if pid is None:
            return None
        with _open(IMAGE_URL.format(id=pid), timeout) as r:
            content = r.read()
        with open(path, "wb") as f:
            f.write(content)
        return path
    except Exception as e:  # pragma: no cover - dépend du réseau
        print("ARASAAC indisponible :", repr(e))
        return None


# --------------------------------------------------------------------------
# Diagnostic : python pictos.py <mot>
# --------------------------------------------------------------------------
if __name__ == "__main__":
    word = " ".join(sys.argv[1:]) or "chat"
    print(f"Mot           : {word!r}")
    print(f"Requête       : {_norm(word)!r}")
    print(f"URL recherche : {SEARCH_URL.format(urllib.parse.quote(_norm(word)))}")
    try:
        import certifi
        print(f"certifi       : {certifi.where()}")
    except Exception:
        print("certifi       : absent (pip install certifi conseillé si erreur SSL)")
    try:
        pid = search_id(word)
        print(f"ID picto      : {pid}")
        if pid is not None:
            print(f"URL image     : {IMAGE_URL.format(id=pid)}")
    except Exception as e:
        print(f"ERREUR recherche : {e!r}")
    path = fetch_picto(word)
    print(f"Résultat      : {path}")
    if path:
        print(f"Taille fichier: {os.path.getsize(path)} octets")
