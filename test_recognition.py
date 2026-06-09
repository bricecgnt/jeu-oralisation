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
               "I": (310, 2200, 3000), "O": (460, 820, 2500), "U": (330, 1700, 2300)},
    "femme":  {"f0": 210, "A": (900, 1500, 2900), "E": (620, 2250, 2900),
               "I": (370, 2700, 3300), "O": (540, 980, 2800), "U": (380, 1900, 2600)},
    "enfant": {"f0": 290, "A": (1000, 1650, 3200), "E": (720, 2500, 3200),
               "I": (430, 3000, 3700), "O": (610, 1100, 3100), "U": (440, 2050, 2900)},
}

rec = VowelRecognizer(samplerate=FS, noise_floor=0.005)

ok = 0
total = 0
print(f"{'voix':8} {'cible':5} {'-> reconnu':12} {'conf':>5}  formants(F1,F2)")
print("-" * 60)
for voice, params in VOICES.items():
    f0 = params["f0"]
    for vowel in ["A", "E", "I", "O", "U"]:
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
