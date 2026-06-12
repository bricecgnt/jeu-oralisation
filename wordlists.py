"""
Listes de mots pour le mode « Mot cible » et données de paires minimales.

Les listes sont de simples fichiers .txt (un mot par ligne, lignes « # »
ignorées) dans le dossier documents de l'utilisateur
(~/Documents/jeu-oralisation/listes). L'orthophoniste peut les créer/éditer avec
n'importe quel éditeur de texte ; des listes d'exemple sont créées au premier
lancement.
"""

from __future__ import annotations

import os

from paths import lists_dir

# Listes d'exemple écrites au premier lancement (si le dossier est vide).
DEFAULT_LISTS: dict[str, list[str]] = {
    "animaux": ["chat", "chien", "lapin", "vache", "poule", "mouton",
                "cheval", "canard", "souris", "poisson"],
    "son CH": ["chat", "chou", "chaussure", "château", "bouche", "vache",
               "cheveux", "chocolat", "chemise", "ruche"],
    "son S": ["sac", "soleil", "salade", "souris", "tasse", "poisson",
              "singe", "serpent", "sucre", "casserole"],
    "son R": ["rat", "roue", "robot", "carotte", "arbre", "fraise",
              "renard", "robe", "fourmi", "tortue"],
    "mots simples": ["papa", "maman", "ballon", "bateau", "pomme", "moto",
                     "lune", "main", "pain", "eau"],
}

# Paires minimales françaises classées par contraste travaillé.
MINIMAL_PAIRS: dict[str, list[tuple[str, str]]] = {
    "p / b": [("poule", "boule"), ("pain", "bain"), ("pelle", "belle"),
              ("pont", "bond")],
    "t / d": [("toux", "doux"), ("thé", "dé"), ("temps", "dent")],
    "k / g": [("car", "gare"), ("cou", "goût"), ("classe", "glace")],
    "f / v": [("faim", "vin"), ("fer", "vert"), ("fille", "ville")],
    "s / z": [("poisson", "poison"), ("coussin", "cousin"),
              ("dessert", "désert")],
    "ch / j": [("chou", "joue"), ("bouche", "bouge"), ("cache", "cage")],
    "s / ch": [("sou", "chou"), ("seau", "chaud"), ("casse", "cache"),
               ("mousse", "mouche")],
    "r / l": [("riz", "lit"), ("roue", "loup"), ("rampe", "lampe")],
}


def ensure_defaults() -> None:
    """Crée les listes d'exemple si le dossier des listes est vide."""
    d = lists_dir()
    if any(f.endswith(".txt") for f in os.listdir(d)):
        return
    for name, words in DEFAULT_LISTS.items():
        path = os.path.join(d, name.replace(" ", "_") + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Un mot par ligne. Les lignes commençant par # sont ignorées.\n")
            f.write("\n".join(words) + "\n")


def load_lists() -> list[tuple[str, list[str]]]:
    """Charge toutes les listes : [(nom, [mots…]), …], triées par nom."""
    ensure_defaults()
    out: list[tuple[str, list[str]]] = []
    d = lists_dir()
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(d, fname)
        try:
            with open(path, encoding="utf-8") as f:
                words = [ln.strip() for ln in f
                         if ln.strip() and not ln.strip().startswith("#")]
        except OSError:
            continue
        if words:
            out.append((os.path.splitext(fname)[0].replace("_", " "), words))
    return out
