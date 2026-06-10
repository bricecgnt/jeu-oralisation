#!/usr/bin/env bash
# Construit l'application macOS cliquable « Jeu d'oralisation.app ».
#
#   chmod +x build_mac.sh
#   ./build_mac.sh           # embarque le modèle "base"
#   ./build_mac.sh small     # embarque "small" à la place
#
# Résultat : dist/Jeu d'oralisation.app  (glisse-le dans /Applications)

set -euo pipefail

MODEL="${1:-base}"

echo "==> Environnement virtuel + dépendances"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pyinstaller

echo "==> Pré-téléchargement du modèle Whisper « $MODEL » (pour l'embarquer hors-ligne)"
python download_models.py "$MODEL"

echo "==> Build PyInstaller"
pyinstaller --noconfirm jeu-oralisation.spec

echo ""
echo "Terminé : dist/Jeu d'oralisation.app"
echo "Astuce : au 1er lancement, clic droit > Ouvrir (app non signée)."
