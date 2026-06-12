"""
Journal de séance : trace les essais (activité, cible, réussite) et exporte un
récapitulatif CSV dans ~/Documents/jeu-oralisation/seances/.

Le CSV utilise « ; » comme séparateur et un BOM UTF-8 pour s'ouvrir proprement
dans Excel/Numbers en français. Seul un pseudonyme (optionnel) figure dans le
nom de fichier — aucune donnée nominative n'est requise.
"""

from __future__ import annotations

import datetime as _dt
import os
import re

from paths import exports_dir


class SessionLog:
    def __init__(self):
        self.started = _dt.datetime.now()
        self.events: list[tuple[_dt.datetime, str, str, bool, str]] = []

    def log(self, activity: str, target: str, ok: bool, detail: str = ""):
        self.events.append((_dt.datetime.now(), activity, target, bool(ok), detail))

    # ---- statistiques ----------------------------------------------------
    def counts(self) -> tuple[int, int]:
        """(essais, réussites) sur toute la séance."""
        total = len(self.events)
        ok = sum(1 for e in self.events if e[3])
        return total, ok

    def by_activity(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for _, act, _, ok, _ in self.events:
            c = out.setdefault(act, [0, 0])
            c[0] += 1
            c[1] += int(ok)
        return {k: (v[0], v[1]) for k, v in out.items()}

    # ---- export -----------------------------------------------------------
    def export_csv(self, child: str = "") -> str:
        """Écrit le récapitulatif et renvoie le chemin du fichier créé."""
        stamp = self.started.strftime("%Y%m%d_%H%M")
        pseudo = re.sub(r"[^\w-]+", "", child.replace(" ", "_"))
        name = f"seance_{stamp}" + (f"_{pseudo}" if pseudo else "") + ".csv"
        path = os.path.join(exports_dir(), name)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("date;heure;activite;cible;resultat;detail\n")
            for t, act, target, ok, detail in self.events:
                detail = detail.replace(";", ",").replace("\n", " ")
                f.write(f"{t:%d/%m/%Y};{t:%H:%M:%S};{act};{target};"
                        f"{'réussi' if ok else 'raté'};{detail}\n")
            total, ok_n = self.counts()
            f.write(f";;TOTAL;;{ok_n}/{total} réussis;\n")
        return path
