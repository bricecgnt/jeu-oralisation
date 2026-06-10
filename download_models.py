"""
Pré-télécharge un ou plusieurs modèles Whisper dans `models/whisper-<taille>/`
afin de les EMBARQUER dans l'application (fonctionnement 100 % hors-ligne, même au
tout premier lancement).

Usage :
    python download_models.py             # télécharge "base"
    python download_models.py tiny base   # plusieurs tailles
    python download_models.py small

Les dossiers produits sont automatiquement inclus par le fichier .spec PyInstaller.
Sans cette étape, l'app téléchargera le modèle à la volée au 1er usage (puis cache).
"""

import os
import sys

from faster_whisper import download_model


def main():
    sizes = sys.argv[1:] or ["base"]
    for size in sizes:
        out = os.path.join("models", f"whisper-{size}")
        os.makedirs(out, exist_ok=True)
        print(f"Téléchargement du modèle « {size} » vers {out}/ …")
        download_model(size, output_dir=out)
        print(f"  OK : {out}")
    print("Terminé. Lance ensuite le build (build_mac.sh).")


if __name__ == "__main__":
    main()
