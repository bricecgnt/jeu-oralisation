"""
Test de la reconnaissance sans micro : on synthétise des voyelles avec des
formants connus (cascade de résonateurs à 2 pôles excitée par un train
d'impulsions glottiques) pour 3 profils de voix, puis on vérifie que le
classifieur retrouve la bonne voyelle — en particulier que A n'est PAS confondu
avec E (le défaut de la version web).
"""

import numpy as np
from recognition import VowelRecognizer

FS = 16000


def resonator(x, f, bw, fs):
    """Filtre 2 pôles (un formant)."""
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * f / fs
    a1 = -2 * r * np.cos(theta)
    a2 = r * r
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = x[n] - a1 * (y[n - 1] if n >= 1 else 0) - a2 * (y[n - 2] if n >= 2 else 0)
    return y


def synth_vowel(f1, f2, f3, f0, dur=0.4, fs=FS):
    """Voyelle synthétique : train d'impulsions à f0 filtré par F1/F2/F3."""
    n = int(dur * fs)
    src = np.zeros(n)
    period = int(fs / f0)
    src[::period] = 1.0
    y = resonator(src, f1, 80, fs)
    y = resonator(y, f2, 100, fs)
    y = resonator(y, f3, 150, fs)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y * 0.6


# Formants « cibles » par voyelle et par profil de voix (F1, F2, F3 en Hz).
# Volontairement un peu différents des points de référence du classifieur.
VOICES = {
    "homme":  {"f0": 120, "A": (730, 1300, 2500), "E": (530, 1850, 2500),
               "I": (310, 2200, 3000), "O": (460, 820, 2500), "U": (330, 1700, 2300),
               "OU": (305, 760, 2300)},
    "femme":  {"f0": 210, "A": (900, 1500, 2900), "E": (620, 2250, 2900),
               "I": (370, 2700, 3300), "O": (540, 980, 2800), "U": (380, 1900, 2600),
               "OU": (375, 940, 2700)},
    "enfant": {"f0": 290, "A": (1000, 1650, 3200), "E": (720, 2500, 3200),
               "I": (430, 3000, 3700), "O": (610, 1100, 3100), "U": (440, 2050, 2900),
               "OU": (445, 1140, 3000)},
}

TEST_VOWELS = ["A", "E", "I", "O", "U", "OU"]


def synth_fricative(kind, fs=FS, dur=0.4, level=None):
    """Fricative synthétique : bruit blanc mis en forme spectralement.
    /s/ ≈ passe-haut 5 kHz ; /ʃ/ ≈ bande autour de 3 kHz ; /f/ ≈ large bande
    plate et faible."""
    n = int(dur * fs)
    rng = np.random.default_rng(42 + hash(kind) % 1000)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    if kind == "S":
        mask = 1.0 / (1.0 + np.exp(-(f - 5000) / 350))
    elif kind == "CH":
        mask = np.exp(-((f - 3000) / 950) ** 2)
    else:  # F
        mask = ((f > 700) & (f < 7600)).astype(float)
    y = np.fft.irfft(X * mask, n)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y * (level if level is not None else (0.18 if kind == "F" else 0.5))

rec = VowelRecognizer(samplerate=FS, noise_floor=0.005)

ok = 0
total = 0
print(f"{'voix':8} {'cible':5} {'-> reconnu':12} {'conf':>5}  formants(F1,F2)")
print("-" * 60)
for voice, params in VOICES.items():
    f0 = params["f0"]
    for vowel in TEST_VOWELS:
        f1, f2, f3 = params[vowel]
        sig = synth_vowel(f1, f2, f3, f0)
        # on analyse une trame centrale de 32 ms
        mid = len(sig) // 2
        frame = sig[mid: mid + 512]
        res = rec.process(frame)
        got = res["vowel"]
        total += 1
        ok += int(got == vowel)
        fmt = ",".join(f"{int(x)}" for x in res["formants"][:2])
        flag = "" if got == vowel else "  <-- ERREUR"
        print(f"{voice:8} {vowel:5} {str(got):12} {res['confidence']:.2f}  [{fmt}]{flag}")

print("-" * 60)
print(f"Score : {ok}/{total} correct ({100*ok/total:.0f}%)")

# Vérification ciblée : un A ne doit jamais être classé E
print("\nFocus A vs E :")
for voice, params in VOICES.items():
    f1, f2, f3 = params["A"]
    frame = synth_vowel(f1, f2, f3, params["f0"])[3000:3512]
    res = rec.process(frame)
    print(f"  {voice:8} A -> {res['vowel']}  (scores A={res['scores']['A']:.2f} "
          f"E={res['scores']['E']:.2f})")

# ---- fricatives S / CH / F (sons non voisés) ------------------------------
print("\nFricatives :")
fok = ftot = 0
for kind in ["S", "CH", "F"]:
    sig = synth_fricative(kind)
    hits = []
    for off in range(2000, 5000, 512):
        r = rec.process(sig[off: off + 512])
        hits.append((r["frica"], r["frica_conf"], r["voiced"]))
    votes = [h[0] for h in hits]
    best = max(set(votes), key=votes.count)
    conf = np.mean([c for f_, c, _ in hits if f_ == best])
    voiced_any = any(v for _, _, v in hits)
    ftot += 1
    fok += int(best == kind)
    flag = "" if best == kind else "  <-- ERREUR"
    print(f"  {kind:3} -> {best}  (conf moy {conf:.2f}, voisé={voiced_any}){flag}")

# une voyelle ne doit pas être prise pour une fricative (gate de voisement)
sig = synth_vowel(*VOICES["femme"]["A"], VOICES["femme"]["f0"])
r = rec.process(sig[3000:3512])
print(f"  voyelle A -> voisé={r['voiced']}, frica={r['frica']} (attendu None)")
print(f"Fricatives : {fok}/{ftot}")
