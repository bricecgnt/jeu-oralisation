"""
Sons cibles et référents sonores (associations iconiques, type Borel-Maisonny) :
l'image évoque le SON, pas le mot (le serpent siffle « sssss », la moto fait
« vvvvv »…). Utilisé par le mode « Son cible » non lecteur et par la fusion
phonémique.

Type de détection acoustique :
  • "vowel"     : voyelle (formants)            -> fiable
  • "fric_unv"  : fricative sourde S/CH/F        -> fiable
  • "fric_v"    : fricative sonore V/Z           -> correct
  • "exp"       : R/L/AN (nasale, liquides)      -> expérimental (à valider à
                  l'oreille ; l'objet avance sur un son tenu voisé)
"""

from __future__ import annotations

# symbole -> (mot-référent, type, petite phrase d'aide)
REFERENTS: dict[str, tuple[str, str, str]] = {
    "S":  ("serpent",  "fric_unv", "le serpent siffle « sssss »"),
    "CH": ("douche",   "fric_unv", "la douche fait « chhhh »"),
    "F":  ("vent",     "fric_unv", "le vent souffle « fffff »"),
    "V":  ("moto",     "fric_v",   "la moto fait « vvvvv »"),
    "Z":  ("abeille",  "fric_v",   "l'abeille fait « zzzzz »"),
    "R":  ("lion",     "exp",      "le lion rugit « rrrrr »"),
    "L":  ("moulin",   "exp",      "« lllll »"),
    "AN": ("éléphant", "exp",      "on dit « annnn »"),
}

CONSONANTS = list(REFERENTS.keys())


def referent_word(symbol: str) -> str:
    r = REFERENTS.get(symbol)
    return r[0] if r else symbol


def referent_tip(symbol: str) -> str:
    r = REFERENTS.get(symbol)
    return r[2] if r else ""


def detect_type(symbol: str) -> str:
    r = REFERENTS.get(symbol)
    return r[1] if r else "exp"


# ---------------------------------------------------------------------------
# Fusion phonémique : (mot-image, son à fusionner, mot-résultat).
# L'enfant voit l'image du 1er mot + l'image-référent du son, et doit dire le mot
# fusionné. Le résultat est vérifié par Whisper.
#   loup + CH (douche) -> louche
# ---------------------------------------------------------------------------
FUSIONS: list[tuple[str, str, str]] = [
    ("loup", "CH", "louche"),
    ("chat", "S", "chasse"),
    ("nid", "CH", "niche"),
    ("roue", "L", "roule"),
    ("pou", "L", "poule"),
    ("fou", "L", "foule"),
    ("mou", "L", "moule"),
    ("tas", "S", "tasse"),
    ("lit", "R", "lire"),
    ("rat", "V", "rave"),
    ("rue", "Z", "ruse"),
]

# Sons présents dans les fusions (ordre d'apparition) — pour le filtre par cible.
FUSION_SOUNDS = list(dict.fromkeys(sound for _, sound, _ in FUSIONS))


def fusions_for(sound: str | None) -> list[tuple[str, str, str]]:
    """Fusions filtrées par son cible (None = toutes)."""
    if not sound:
        return FUSIONS
    return [f for f in FUSIONS if f[1] == sound] or FUSIONS
