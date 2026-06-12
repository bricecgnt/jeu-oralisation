"""
Reconnaissance de voyelles temps réel, locale et sans calibrage.

Principe (volontairement simple et « universel ») :
  1. extraction des formants par LPC (racines du polynôme prédicteur), bien plus
     fiable que le pic de FFT utilisé dans la version web ;
  2. conversion des formants en échelle Bark (perceptuelle) pour atténuer les
     différences homme / femme / enfant sans calibrage ;
  3. classement à la voyelle de référence la plus proche (k-plus-proche-voisin
     sur des points couvrant adulte masculin, adulte féminin et enfant).

Aucune dépendance audio ici : ce module ne fait que du calcul sur des trames
numpy, ce qui le rend testable sans micro.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Références de formants (F1, F2, F3) en Hz pour les voyelles françaises.
# Trois points par voyelle ≈ adulte masculin / adulte féminin / enfant, afin de
# couvrir la variabilité des conduits vocaux (c'est ce qui remplace le calibrage).
# Valeurs approximatives tirées de la littérature (Peterson-Barney / Hillenbrand
# adaptées au français ; « U » = /y/, voyelle antérieure arrondie).
# F3 sert surtout à séparer les arrondies (O, U : F3 bas) des non arrondies
# antérieures (I, E : F3 élevé) — clé pour distinguer I de U.
# --------------------------------------------------------------------------
VOWEL_REFS: dict[str, list[tuple[float, float, float]]] = {
    "A": [(700, 1350, 2500), (900, 1550, 2850), (1000, 1700, 3050)],
    "E": [(500, 1950, 2550), (600, 2300, 2850), (700, 2500, 3050)],
    "I": [(300, 2200, 3000), (350, 2750, 3300), (420, 3050, 3600)],
    "O": [(490, 840, 2550), (570, 1000, 2700), (645, 1140, 2850)],
    "U": [(330, 1600, 2150), (390, 1850, 2300), (450, 2050, 2500)],
    "OU": [(370, 770, 2250), (430, 930, 2650), (510, 1130, 2950)],
}

VOWELS = list(VOWEL_REFS.keys())

# Fricatives sourdes tenues reconnues (sigmatisme et confusions s/ch) :
# classées par la forme du spectre du bruit, pas par les formants.
FRICATIVES = ["S", "CH", "F"]

# Références (centroïde Hz, part d'énergie > 4,5 kHz, platitude spectrale).
# /s/ : énergie concentrée très aiguë ; /ʃ/ : concentrée médium (~3 kHz) ;
# /f/ : diffuse (plate) et faible. Ajustées sur bruits synthétiques.
FRICA_REFS: dict[str, tuple[float, float, float]] = {
    "S": (6300, 0.78, 0.30),
    "CH": (3100, 0.10, 0.25),
    "F": (4200, 0.45, 0.75),
}

# Poids de F3 dans la distance (plus faible : F3 est plus bruité que F1/F2).
F3_WEIGHT = 0.6


def hz_to_bark(f: np.ndarray | float) -> np.ndarray | float:
    """Échelle de Bark (Traunmüller). Comprime les hautes fréquences, dilate les
    basses → les écarts de F1 (clés pour A/E/I) pèsent davantage."""
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# Pré-calcul des références en espace de features (bark(F1), bark(F2), w·bark(F3)).
def _build_ref_points() -> tuple[np.ndarray, list[str]]:
    pts, labels = [], []
    for v, refs in VOWEL_REFS.items():
        for f1, f2, f3 in refs:
            pts.append([hz_to_bark(f1), hz_to_bark(f2), F3_WEIGHT * hz_to_bark(f3)])
            labels.append(v)
    return np.array(pts), labels


_REF_POINTS, _REF_LABELS = _build_ref_points()


class VowelRecognizer:
    """Analyse une trame audio et renvoie volume, voisement, formants et la
    voyelle la plus probable avec une confiance.

    Paramètres
    ----------
    samplerate : fréquence d'échantillonnage des trames fournies (Hz).
    lpc_order  : ordre du modèle LPC (≈ 2 + fs/1000).
    noise_floor: RMS en dessous duquel on considère qu'il n'y a pas de son.
    """

    def __init__(self, samplerate: int = 16000, lpc_order: int | None = None,
                 noise_floor: float = 0.01):
        self.fs = int(samplerate)
        self.order = lpc_order if lpc_order else int(2 + self.fs / 1000)
        self.noise_floor = float(noise_floor)
        self._softmax_temp = 1.2  # « température » du softmax sur les distances

    # ---- bas niveau -----------------------------------------------------
    def _lpc(self, x: np.ndarray) -> np.ndarray:
        """Coefficients LPC via Yule-Walker. Renvoie le polynôme A(z)
        = [1, -a1, ..., -ap]."""
        order = self.order
        n = len(x)
        # autocorrélation r[0..order]
        r = np.empty(order + 1)
        for k in range(order + 1):
            r[k] = np.dot(x[: n - k], x[k:])
        if r[0] == 0.0:
            return np.concatenate(([1.0], np.zeros(order)))
        # Toeplitz R a = r[1..order]
        R = np.empty((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = r[abs(i - j)]
        # petite régularisation pour la stabilité numérique
        R[np.diag_indices_from(R)] += 1e-6 * r[0]
        try:
            a = np.linalg.solve(R, r[1: order + 1])
        except np.linalg.LinAlgError:
            return np.concatenate(([1.0], np.zeros(order)))
        return np.concatenate(([1.0], -a))

    # bande passante max d'un « vrai » formant (les racines LPC parasites sont
    # larges ; les résonances du conduit vocal sont étroites)
    _BW_MAX = 350.0

    def _formant_candidates(self, x: np.ndarray) -> list[tuple[float, float]]:
        """Tous les candidats (freq, bande passante) issus des racines LPC, triés
        par fréquence."""
        a = self._lpc(x)
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0]  # une racine par paire conjuguée
        cands = []
        for z in roots:
            freq = np.arctan2(np.imag(z), np.real(z)) * (self.fs / (2 * np.pi))
            bw = -0.5 * (self.fs / np.pi) * np.log(np.abs(z) + 1e-12)
            if 90.0 < freq < (self.fs / 2 - 150):
                cands.append((float(freq), float(bw)))
        cands.sort()
        return cands

    def _pick_formants(self, cands: list[tuple[float, float]]) -> list[float]:
        """Choisit F1, F2, F3 parmi les candidats : les pics ÉTROITS (faible bande
        passante) les plus bas en fréquence. Évite les racines parasites larges
        qui s'intercalent entre les formants."""
        narrow = [(f, b) for f, b in cands if b < self._BW_MAX]
        if len(narrow) < 2:
            # repli : on garde les 5 racines les plus étroites
            narrow = sorted(cands, key=lambda t: t[1])[:5]
            narrow.sort()
        if len(narrow) < 2:
            return [f for f, _ in narrow]
        # F1 : pic étroit le plus bas dans la plage plausible de F1
        f1 = next((f for f, _ in narrow if 200 <= f <= 1100), narrow[0][0])
        # F2 : pic étroit le plus bas nettement au-dessus de F1
        f2 = next((f for f, _ in narrow if f > f1 + 150), narrow[-1][0])
        # F3 : pic étroit le plus bas nettement au-dessus de F2 (optionnel)
        f3 = next((f for f, _ in narrow if f > f2 + 150), None)
        return [f1, f2] if f3 is None else [f1, f2, f3]

    def _pitch(self, x: np.ndarray, rms: float) -> tuple[bool, float]:
        """Voisement + hauteur par autocorrélation : un pic net dans la plage
        70–600 Hz (voix d'adulte à voix d'enfant) indique un son voisé tenu.
        Renvoie (voisé, f0 en Hz)."""
        if rms < self.noise_floor:
            return False, 0.0
        lo = max(2, int(self.fs / 600))
        hi = int(self.fs / 70)
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        if ac[0] <= 0:
            return False, 0.0
        ac = ac / ac[0]
        seg = ac[lo:hi]
        if seg.size == 0:
            return False, 0.0
        i = int(np.argmax(seg))
        if float(seg[i]) <= 0.3:
            return False, 0.0
        return True, self.fs / (lo + i)

    def _fricative(self, x: np.ndarray) -> tuple[str | None, float]:
        """Classifie une fricative sourde tenue (S/CH/F) d'après la forme du
        spectre du bruit : centroïde (aigu pour S, médium pour CH), part
        d'énergie > 4,5 kHz, et platitude spectrale (élevée pour F, diffus)."""
        n = len(x)
        P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        sel = (freqs >= 500) & (freqs <= min(7800.0, self.fs / 2 - 100))
        Ps, Fs = P[sel], freqs[sel]
        tot = float(Ps.sum())
        if tot <= 0:
            return None, 0.0
        centroid = float((Ps * Fs).sum() / tot)
        hi = float(Ps[Fs >= 4500].sum() / tot)
        pos = Ps + 1e-12
        flat = float(np.exp(np.mean(np.log(pos))) / np.mean(pos))

        def vec(c, h, fl):
            return np.array([c / 1500.0, h * 3.0, fl * 3.0])

        feat = vec(centroid, hi, flat)
        ds = np.array([np.linalg.norm(feat - vec(*FRICA_REFS[k]))
                       for k in FRICATIVES])
        logits = -ds / 0.8
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        i = int(np.argmax(p))
        return FRICATIVES[i], float(p[i])

    # ---- haut niveau ----------------------------------------------------
    def process(self, frame: np.ndarray) -> dict:
        """Analyse une trame mono (float). Renvoie un dictionnaire de résultats."""
        x = np.asarray(frame, dtype=float)
        if x.size < self.order + 8:
            return self._empty()

        rms = float(np.sqrt(np.mean(x * x)))
        vol = float(np.clip((rms - self.noise_floor) / 0.15, 0.0, 1.0))
        voiced, f0 = self._pitch(x, rms)

        # fricative sourde : son non voisé mais énergique (S / CH / F)
        frica, frica_conf = (None, 0.0)
        if not voiced and vol > 0.04:
            frica, frica_conf = self._fricative(x)

        # pré-accentuation + fenêtre de Hamming avant LPC
        emph = np.append(x[0], x[1:] - 0.97 * x[:-1])
        win = emph * np.hamming(len(emph))
        formants = self._pick_formants(self._formant_candidates(win))

        result = {
            "vol": vol,
            "voiced": voiced,
            "f0": f0,
            "frica": frica,
            "frica_conf": frica_conf,
            "formants": formants,
            "vowel": None,
            "confidence": 0.0,
            "scores": {v: 0.0 for v in VOWELS},
        }
        if not voiced or len(formants) < 2:
            return result

        f1, f2 = formants[0], formants[1]
        if len(formants) >= 3:
            feat = np.array([hz_to_bark(f1), hz_to_bark(f2),
                             F3_WEIGHT * hz_to_bark(formants[2])])
            refs = _REF_POINTS                # comparaison en 3D (avec F3)
        else:
            feat = np.array([hz_to_bark(f1), hz_to_bark(f2)])
            refs = _REF_POINTS[:, :2]         # repli en 2D (sans F3)
        # distance à chaque voyelle = distance au plus proche de ses points de réf.
        dist = {}
        for v in VOWELS:
            idx = [i for i, lab in enumerate(_REF_LABELS) if lab == v]
            d = np.min(np.linalg.norm(refs[idx] - feat, axis=1))
            dist[v] = d
        # softmax sur -distance → pseudo-probabilités
        ds = np.array([dist[v] for v in VOWELS])
        logits = -ds / self._softmax_temp
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        scores = {v: float(pi) for v, pi in zip(VOWELS, p)}
        best = max(scores, key=scores.get)

        result["vowel"] = best
        result["confidence"] = scores[best]
        result["scores"] = scores
        return result

    @staticmethod
    def _empty() -> dict:
        return {
            "vol": 0.0, "voiced": False, "f0": 0.0,
            "frica": None, "frica_conf": 0.0, "formants": [],
            "vowel": None, "confidence": 0.0,
            "scores": {v: 0.0 for v in VOWELS},
        }
