# -*- mode: python ; coding: utf-8 -*-
"""
Spec PyInstaller : construit une application cliquable.
  • macOS  -> dist/Jeu d'oralisation.app
  • autres -> dist/jeu-oralisation/ (dossier exécutable)

Étapes :
    pip install -r requirements.txt pyinstaller
    python download_models.py base        # (optionnel) embarque un modèle
    pyinstaller --noconfirm jeu-oralisation.spec
"""

import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# Dépendances « lourdes » qui embarquent des binaires / données et que PyInstaller
# ne détecte pas toujours seul.
for pkg in ("faster_whisper", "ctranslate2", "onnxruntime", "tokenizers",
            "av", "sounddevice", "huggingface_hub", "tqdm"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # le paquet peut être absent (av, etc.)
        print(f"[spec] collect_all ignoré pour {pkg}: {exc}")

# Modèles Whisper pré-téléchargés (optionnels) -> embarqués dans l'app.
if os.path.isdir("models"):
    datas.append(("models", "models"))

a = Analysis(
    ["game.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="jeu-oralisation",
    console=False,            # pas de terminal (app fenêtrée)
    disable_windowed_traceback=False,
    argv_emulation=True,      # macOS : ouverture par double-clic
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="jeu-oralisation",
)

app = BUNDLE(
    coll,
    name="Jeu d'oralisation.app",
    icon=None,
    bundle_identifier="org.orthophonie.jeu-oralisation",
    info_plist={
        "NSMicrophoneUsageDescription":
            "Le jeu utilise le microphone pour écouter la voix de l'enfant.",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
