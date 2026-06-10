"""
Récupération automatique d'un pictogramme depuis ARASAAC pour un mot donné.

ARASAAC (Gouvernement d'Aragon) propose une base de pictogrammes libres
(CC BY-NC-SA) très utilisée en orthophonie, avec une API publique.

  • recherche : https://api.arasaac.org/api/pictograms/fr/search/{mot}
  • image     : https://static.arasaac.org/pictograms/{id}/{id}_300.png

Les images sont mises en cache sur le disque (`pictos_cache/`) : le 1er accès à un
mot nécessite Internet, ensuite c'est hors-ligne. En cas d'absence de réseau ou de
résultat, la fonction renvoie None (le jeu fonctionne sans image).

Attribution requise par la licence : « Pictogrammes : ARASAAC (arasaac.org) ».
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pictos_cache")
SEARCH_URL = "https://api.arasaac.org/api/pictograms/fr/search/{}"
IMAGE_URL = "https://static.arasaac.org/pictograms/{id}/{id}_300.png"
_UA = {"User-Agent": "jeu-oralisation/1.0"}


def _slug(word: str) -> str:
    s = unicodedata.normalize("NFD", word)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def cached_path(word: str) -> str | None:
    """Chemin du picto en cache pour ce mot, s'il existe déjà."""
    slug = _slug(word)
    if not slug:
        return None
    path = os.path.join(CACHE_DIR, slug + ".png")
    return path if os.path.exists(path) else None


def fetch_picto(word: str, timeout: float = 6.0) -> str | None:
    """Renvoie le chemin local d'un pictogramme pour `word` (téléchargé puis mis en
    cache), ou None si introuvable / hors-ligne."""
    slug = _slug(word)
    if not slug:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, slug + ".png")
    if os.path.exists(path):
        return path
    try:
        url = SEARCH_URL.format(urllib.parse.quote(slug))
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                    timeout=timeout) as r:
            data = json.load(r)
        if not data:
            return None
        pid = data[0].get("_id") or data[0].get("id")
        if pid is None:
            return None
        img = IMAGE_URL.format(id=pid)
        with urllib.request.urlopen(urllib.request.Request(img, headers=_UA),
                                    timeout=timeout) as r:
            content = r.read()
        with open(path, "wb") as f:
            f.write(content)
        return path
    except Exception as e:  # pragma: no cover - dépend du réseau
        print("ARASAAC indisponible :", e)
        return None
