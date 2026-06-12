"""
Jeu d'oralisation — version Python / pygame.

Un enfant produit un son dans le micro pour faire avancer un objet (voiture,
ballon, fusée) jusqu'au drapeau d'arrivée.

  • Mode « son libre »  : n'importe quel son tenu fait avancer l'objet.
  • Mode « voyelle cible »: l'objet n'avance que si l'enfant prononce la bonne
    voyelle (A E I O U), reconnue par formants (voir recognition.py).

100 % local, sans réseau. Repli sans micro : maintenir la touche ESPACE.

Lancement :
    pip install -r requirements.txt
    python game.py
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading

# Réduit l'activité en arrière-plan de Whisper/HF (évite des sous-processus et des
# avertissements de parallélisme une fois empaqueté).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pygame

import random

from recognition import VowelRecognizer, VOWELS, FRICATIVES
from speech import WordRecognizer, DEFAULT_MODEL
from session import SessionLog
import pictos
import tts
import wordlists

# Tailles de modèle Whisper proposées à l'adulte (du plus rapide au plus précis).
WHISPER_MODELS = ["tiny", "base", "small"]

# Micro optionnel : si sounddevice est absent ou sans périphérique, on bascule
# automatiquement en mode clavier (ESPACE).
try:
    import sounddevice as sd
except Exception:  # pragma: no cover - dépend de l'environnement
    sd = None

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
FS = 16000           # fréquence d'échantillonnage du micro
FRAME = 512          # taille de trame analysée (~32 ms)
RING = 4096          # taille du tampon circulaire
REC_SECONDS = 2.2    # durée d'écoute pour le mode « mot cible »

W0, H0 = 1100, 720   # taille de fenêtre initiale
TOP = 0              # hauteur de la barre d'outils (calculée après layout)

COL = {
    "bg": (240, 248, 255),
    "panel": (255, 255, 255),
    "accent": (255, 138, 61),
    "accent2": (77, 171, 247),
    "ok": (81, 207, 102),
    "text": (43, 45, 66),
    "muted": (108, 117, 125),
    "btn": (233, 236, 239),
    "btnhover": (222, 226, 230),
    "danger": (224, 49, 49),
}


# --------------------------------------------------------------------------
# Entrée micro (tampon circulaire alimenté par le callback audio)
# --------------------------------------------------------------------------
class MicInput:
    def __init__(self, fs=FS):
        self.fs = fs
        self.buf = np.zeros(RING, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None
        self.available = sd is not None
        self.recording = False
        self.rec_frames = []

    def _callback(self, indata, frames, time_info, status):  # pragma: no cover
        x = indata[:, 0]
        with self.lock:
            n = len(x)
            if n >= RING:
                self.buf[:] = x[-RING:]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = x
            if self.recording:
                self.rec_frames.append(x.copy())

    def start_record(self):
        with self.lock:
            self.rec_frames = []
            self.recording = True

    def stop_record(self) -> np.ndarray:
        with self.lock:
            self.recording = False
            if not self.rec_frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.rec_frames)

    def start(self) -> bool:
        if not self.available:
            return False
        try:
            self.stream = sd.InputStream(
                samplerate=self.fs, channels=1, blocksize=256,
                dtype="float32", callback=self._callback)
            self.stream.start()
            return True
        except Exception as e:
            print("Micro indisponible :", e)
            self.stream = None
            return False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
        self.stream = None

    def frame(self) -> np.ndarray:
        with self.lock:
            return self.buf[-FRAME:].copy()


# --------------------------------------------------------------------------
# Curseur (slider) réglable à la souris
# --------------------------------------------------------------------------
class Slider:
    def __init__(self, label, value):
        self.label = label
        self.value = float(value)     # 0..1
        self.rect = pygame.Rect(0, 0, 10, 40)   # zone complète du widget
        self.dragging = False
        self.visible = True

    def _track(self):
        r = self.rect
        return pygame.Rect(r.x, r.y + 24, r.w, 8)

    def draw(self, surf, font):
        if not self.visible:
            return
        r = self.rect
        lbl = font.render(f"{self.label} : {int(round(self.value * 100))} %",
                          True, COL["text"])
        surf.blit(lbl, (r.x, r.y + 2))
        t = self._track()
        pygame.draw.rect(surf, (206, 212, 218), t, border_radius=4)
        pygame.draw.rect(surf, COL["accent2"],
                         (t.x, t.y, int(self.value * t.w), t.h), border_radius=4)
        hx = int(t.x + self.value * t.w)
        pygame.draw.circle(surf, COL["accent2"], (hx, t.centery), 9)
        pygame.draw.circle(surf, (255, 255, 255), (hx, t.centery), 5)

    def hit(self, pos):
        return self._track().inflate(0, 20).collidepoint(pos)

    def set_from_pos(self, pos):
        t = self._track()
        self.value = float(np.clip((pos[0] - t.x) / max(1, t.w), 0.0, 1.0))


# --------------------------------------------------------------------------
# Bouton cliquable
# --------------------------------------------------------------------------
class Button:
    def __init__(self, label, key=None, kind="normal"):
        self.label = label
        self.key = key            # raccourci clavier (str) optionnel
        self.kind = kind          # normal | primary
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.selected = False
        self.visible = True

    def draw(self, surf, font, mouse):
        if not self.visible:
            return
        hover = self.rect.collidepoint(mouse)
        if self.selected:
            bg = COL["accent2"]; fg = (255, 255, 255)
        elif self.kind == "primary":
            bg = COL["danger"] if self.label.startswith("⏹") else COL["accent"]
            fg = (255, 255, 255)
        else:
            bg = COL["btnhover"] if hover else COL["btn"]; fg = COL["text"]
        pygame.draw.rect(surf, bg, self.rect, border_radius=12)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


# --------------------------------------------------------------------------
# Jeu
# --------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        try:
            pygame.mixer.init()
            self.mixer_ok = True
        except Exception:
            self.mixer_ok = False
        self.screen = pygame.display.set_mode((W0, H0), pygame.RESIZABLE)
        pygame.display.set_caption("Jeu d'oralisation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_big = pygame.font.SysFont("Arial", 220, bold=True)
        self.font_hint = pygame.font.SysFont("Arial", 26)
        self.font_timer = pygame.font.SysFont("Arial", 54, bold=True)

        self.rec = VowelRecognizer(samplerate=FS, noise_floor=0.02)
        self.mic = MicInput()

        # état
        self.scene = "car"
        self.mode = "free"
        self.vowel = "A"
        self.energy = 0.0
        self.won = False
        self.win_time = 0
        self.running_mic = False
        self.space_down = False
        self.sensitivity = 0.35
        self.threshold = 0.45           # exigence (confiance mini en mode cible)
        self.hold = True                # le véhicule reste en place si le son cesse
        self.last = {"vol": 0.0, "voiced": False, "vowel": None, "confidence": 0.0}
        self.confetti = []
        # mode « mot cible »
        self.word = ""                  # mot tapé par l'adulte
        self.model_size = DEFAULT_MODEL
        self.word_rec = WordRecognizer(self.model_size)
        self.model_loading = False
        self.recording = False
        self.rec_start = 0
        self.transcribing = False
        self.heard = ""                 # dernière transcription
        self.word_ok = False            # le mot a-t-il été reconnu
        self.word_msg = ""              # message d'aide (mode mot)
        self._pending = None            # (texte, ok) renvoyé par le thread
        self.fireworks = []
        # pictogramme ARASAAC
        self.show_picto = True
        self.picto_word = None          # mot pour lequel picto_surface est valide
        self.picto_surface = None
        self.picto_found = False        # une image a-t-elle été trouvée pour picto_word
        self.picto_fetching = False
        self.word_change_t = 0
        self._picto_pending = None       # (mot, chemin|None) renvoyé par le thread
        # pack séance : listes de mots, jetons, dénomination, mains libres, journal
        self.session = SessionLog()
        self.lists = wordlists.load_lists()   # [(nom, [mots…]), …]
        self.list_idx = 0
        self.list_pos = 0
        self.tokens = 0
        self.tokens_goal = 5
        self.hide_word = False           # dénomination : picto seul, mot caché
        self.auto_listen = False         # mains libres : écoute déclenchée à la voix
        self._auto_cooldown = 0
        self.export_msg = ""
        # jeux de voix : intensité / hauteur / souffle
        self.dt = 1 / 60                 # durée de la dernière frame (s)
        self.zone = "moyen"              # intensité : doux / moyen / fort
        self.vol_ema = 0.0
        self.zone_prog = 0.0
        self.pzone = "medium"            # hauteur : grave / medium / aigu
        self.pitch_ema = 0.0
        self.pitch_prog = 0.0
        self.pitch_f0 = 0.0
        self.breath_goal = 5             # souffle : durée cible (s)
        self.breath_t = 0.0
        self.breath_gap = 0.0
        self.breath_best = 0.0
        # mode écoute (TTS + choix d'images)
        self.tts_ok = tts.available()
        self.listen_n = 3                # nombre d'images proposées
        self.listen_opts: list[str] = []
        self.listen_target = ""
        self.listen_wrong: set[str] = set()
        self.listen_done = 0
        # paires minimales
        self.pair_keys = list(wordlists.MINIMAL_PAIRS.keys())
        self.pair_ci = 0
        self.pair: tuple[str, str] | None = None
        self.pair_target = 0
        self.pair_ok = False
        self.pair_msg = ""
        self.pair_heard = ""
        self.pair_done = 0
        # cache de pictogrammes multi-mots (écoute / paires)
        self.pcache: dict[str, object] = {}
        self._pcache_results: list[tuple[str, str | None]] = []
        self._card_rects: list[tuple[pygame.Rect, str]] = []
        self._cheer = self._make_cheer() if self.mixer_ok else None
        self._apply_sensitivity()

        self._build_buttons()
        self._gradients = {}

    # ---- sons de récompense --------------------------------------------
    def _make_cheer(self):
        fs = 44100
        notes = [523.25, 659.25, 783.99, 1046.5]   # do mi sol do
        out = np.zeros(int(fs * 0.7), dtype=np.float32)
        for i, f in enumerate(notes):
            t0 = int(i * 0.12 * fs)
            dur = int(0.3 * fs)
            t = np.arange(dur) / fs
            env = np.exp(-t * 12)
            # onde triangulaire douce, enveloppe décroissante
            seg = 0.25 * env * (2 * np.abs(2 * (f * t - np.floor(f * t + 0.5))) - 1)
            end = min(t0 + dur, len(out))
            out[t0:end] += seg[: end - t0]
        out = np.clip(out, -1, 1)
        return pygame.sndarray.make_sound((out * 32767).astype(np.int16))

    def _play_cheer(self):
        if self._cheer is not None:
            self._cheer.play()

    # ---- boutons / mise en page ----------------------------------------
    def _build_buttons(self):
        self.b_scene = {
            "car": Button("🚗 Voiture"), "balloon": Button("🎈 Ballon"),
            "rocket": Button("🚀 Fusée"),
        }
        self.b_scene["car"].selected = True
        self.b_mode = {"free": Button("Son libre"), "target": Button("Son cible"),
                       "word": Button("Mot cible"), "loud": Button("Intensité"),
                       "pitch": Button("Hauteur"), "breath": Button("Souffle"),
                       "listen": Button("Écoute"), "pairs": Button("Paires")}
        self.b_mode["free"].selected = True
        self.b_vowel = {v: Button(v, key=v) for v in VOWELS}
        self.b_vowel["A"].selected = True
        self.b_frica = {k: Button(k) for k in FRICATIVES}   # fricatives S/CH/F
        self.b_hold = Button("Reste en place"); self.b_hold.selected = self.hold
        self.b_speak = Button("🎤 Parler", kind="primary")
        self.b_clear = Button("Effacer")
        self.b_model = Button(f"Modèle : {self.model_size}")
        self.b_picto = Button("Image"); self.b_picto.selected = self.show_picto
        # pack séance
        self.b_list = Button("Liste")
        self.b_next = Button("Mot suivant ▶")
        self.b_denom = Button("Cacher le mot")
        self.b_auto = Button("Mains libres")
        self.b_export = Button("Exporter séance")
        # jeux de voix
        self.b_zone = {"doux": Button("Doux"), "moyen": Button("Moyen"),
                       "fort": Button("Fort")}
        self.b_zone["moyen"].selected = True
        self.b_pzone = {"grave": Button("Grave"), "medium": Button("Médium"),
                        "aigu": Button("Aigu")}
        self.b_pzone["medium"].selected = True
        self.b_bgoal = {g: Button(f"{g} s") for g in (2, 3, 5, 8)}
        self.b_bgoal[5].selected = True
        # écoute
        self.b_replay = Button("🔊 Réécouter", kind="primary")
        self.b_choices = Button("Choix : 3")
        self.b_newround = Button("Nouveau mot")
        # paires minimales
        self.b_contrast = Button("Contraste")
        self.b_newpair = Button("Autre paire")
        self.s_sens = Slider("Sensibilité", self.sensitivity)
        self.s_str = Slider("Exigence", (self.threshold - 0.2) / 0.65)
        self.b_mic = Button("▶︎ Démarrer le micro", kind="primary")

    def _all_buttons(self):
        yield from self.b_scene.values()
        yield from self.b_mode.values()
        yield from self.b_vowel.values()
        yield from self.b_frica.values()
        yield self.b_hold
        yield self.b_speak
        yield self.b_clear
        yield self.b_list
        yield self.b_next
        yield self.b_denom
        yield self.b_auto
        yield self.b_model
        yield self.b_picto
        yield self.b_export
        yield from self.b_zone.values()
        yield from self.b_pzone.values()
        yield from self.b_bgoal.values()
        yield self.b_replay
        yield self.b_choices
        yield self.b_newround
        yield self.b_contrast
        yield self.b_newpair
        yield self.b_mic

    def _sliders(self):
        return (self.s_sens, self.s_str)

    def _layout(self, w):
        """Place les boutons en lignes avec retour à la ligne."""
        global TOP
        m = self.mode
        target = m == "target"
        word = m == "word"
        listen = m == "listen"
        pairs = m == "pairs"
        for v in self.b_vowel.values():
            v.visible = target
        for v in self.b_frica.values():
            v.visible = target
        for b in self.b_scene.values():
            b.visible = m in ("free", "target")
        self.s_str.visible = m in ("target", "loud", "pitch")
        self.s_sens.visible = m not in ("listen",)
        self.b_hold.visible = m in ("free", "target")
        self.b_speak.visible = word or pairs
        self.b_clear.visible = word
        self.b_model.visible = word or pairs
        self.b_picto.visible = word
        self.b_list.visible = word or listen
        self.b_next.visible = word
        self.b_denom.visible = word
        self.b_auto.visible = word or pairs
        self.b_export.visible = word or listen or pairs
        for b in self.b_zone.values():
            b.visible = m == "loud"
        for b in self.b_pzone.values():
            b.visible = m == "pitch"
        for b in self.b_bgoal.values():
            b.visible = m == "breath"
        self.b_replay.visible = listen
        self.b_choices.visible = listen
        self.b_newround.visible = listen
        self.b_contrast.visible = pairs
        self.b_newpair.visible = pairs
        self.b_mic.visible = not listen
        self.b_speak.label = ("● J'écoute…" if self.recording else
                              "⏳ …" if self.transcribing else "🎤 Parler")
        self.b_model.label = f"Modèle : {self.model_size}"
        self.b_list.label = (f"Liste : {self.lists[self.list_idx][0]}"
                             if self.lists else "Liste : —")
        self.b_choices.label = f"Choix : {self.listen_n}"
        self.b_contrast.label = f"Contraste : {self.pair_keys[self.pair_ci]}"
        self.b_denom.selected = self.hide_word
        self.b_auto.selected = self.auto_listen

        pad, gap, h = 12, 6, 40
        x, y = pad, pad

        def place(btns, widths):
            nonlocal x, y
            # séparateur de groupe
            for b, bw in zip(btns, widths):
                if not b.visible:
                    continue
                if x + bw > w - pad:
                    x = pad; y += h + gap
                b.rect = pygame.Rect(x, y, bw, h)
                x += bw + gap
            x += 14  # espace inter-groupe

        place(list(self.b_scene.values()), [108, 100, 92])
        place(list(self.b_mode.values()), [96, 96, 100, 100, 92, 88, 84, 78])
        place([self.b_vowel[v] for v in VOWELS], [46] * len(VOWELS))
        place([self.b_frica[k] for k in FRICATIVES], [52] * len(FRICATIVES))
        place([self.b_hold], [148])
        place([self.b_speak, self.b_clear], [150, 96])
        place([self.b_list, self.b_next], [188, 124])
        place([self.b_denom, self.b_auto], [126, 116])
        place([self.b_model, self.b_picto], [150, 96])
        place(list(self.b_zone.values()), [86, 92, 80])
        place(list(self.b_pzone.values()), [86, 96, 80])
        place(list(self.b_bgoal.values()), [62, 62, 62, 62])
        place([self.b_replay, self.b_choices, self.b_newround], [148, 104, 134])
        place([self.b_contrast, self.b_newpair], [180, 124])
        place([self.b_export], [148])
        place([self.s_sens], [156])
        place([self.s_str], [156])
        # micro aligné à droite si la place le permet, sinon à la suite
        if self.b_mic.visible:
            mic_w = 210
            if x + mic_w > w - pad:
                x = pad; y += h + gap
            self.b_mic.rect = pygame.Rect(max(x, w - pad - mic_w), y, mic_w, h)
        else:
            self.b_mic.rect = pygame.Rect(-10, -10, 0, 0)
        TOP = y + h + pad

    # ---- réglages -------------------------------------------------------
    def _apply_sensitivity(self):
        self.rec.noise_floor = float(np.clip(0.06 - self.sensitivity * 0.055,
                                             0.004, 0.06))

    # ---- audio / logique ------------------------------------------------
    def analyze(self):
        if self.space_down:
            # repli clavier : simule le son cible (voyelle ou fricative)
            frica = self.vowel if self.vowel in FRICATIVES else None
            return {"vol": 0.9, "voiced": frica is None, "f0": 200.0,
                    "vowel": self.vowel, "confidence": 1.0,
                    "frica": frica, "frica_conf": 1.0}
        if not self.running_mic:
            return {"vol": 0.0, "voiced": False, "f0": 0.0, "vowel": None,
                    "confidence": 0.0, "frica": None, "frica_conf": 0.0}
        return self.rec.process(self.mic.frame())

    def update(self):
        self._pump_pcache()
        now = pygame.time.get_ticks()
        if self.mode == "word":
            self._update_word()
        elif self.mode == "listen":
            self._update_listen(now)
        elif self.mode == "pairs":
            self._update_pairs(now)
        elif self.mode in ("loud", "pitch", "breath"):
            self._update_voice(now)
        else:
            self._update_vehicle(now)
        self._update_fireworks()

    def _update_vehicle(self, now):
        r = self.analyze()
        self.last = r
        if self.mode == "target":
            if self.vowel in FRICATIVES:
                conf = r.get("frica_conf", 0.0) if r.get("frica") == self.vowel else 0.0
                matched = r.get("vol", 0) > 0.04 and conf >= self.threshold
            else:
                conf = r["confidence"] if r["voiced"] and r["vowel"] == self.vowel else 0.0
                matched = conf >= self.threshold
            drive = r["vol"] * (0.4 + 0.6 * conf) if matched else 0.0
        else:
            drive = r["vol"] if r["voiced"] or self.space_down else 0.0

        if drive > 0.05:
            self.energy = min(1.0, self.energy + drive * 0.012)
        elif not self.hold:
            self.energy = max(0.0, self.energy - 0.006)
        # en mode « Reste en place », l'énergie ne redescend pas quand le son cesse

        if self.energy >= 1.0 and not self.won:
            self.won = True
            self.win_time = now
            self._play_cheer()
            if self.mode == "target":
                self.session.log("son cible", self.vowel, True)
            else:
                self.session.log("son libre", "libre", True)
        # après la célébration, on repart à zéro pour rejouer
        if self.won and now - self.win_time > 1600:
            self.reset_progress()

    # ---- jeux de voix : intensité / hauteur / souffle --------------------
    def _zone_params(self):
        """(centre, demi-largeur) de la zone cible — l'exigence resserre la zone."""
        hw = 0.32 - 0.20 * self.s_str.value
        if self.mode == "loud":
            c = {"doux": 0.22, "moyen": 0.5, "fort": 0.8}[self.zone]
        else:
            c = {"grave": 0.22, "medium": 0.52, "aigu": 0.8}[self.pzone]
        return c, hw

    def _voice_win(self, activity, target):
        self.won = True
        self.win_time = pygame.time.get_ticks()
        self._spawn_fireworks()
        self._play_cheer()
        self.session.log(activity, target, True)
        self._add_token()

    def _update_voice(self, now):
        r = self.analyze()
        self.last = r
        dt = self.dt
        if self.mode == "loud":
            self.vol_ema = 0.75 * self.vol_ema + 0.25 * r["vol"]
            c, hw = self._zone_params()
            if not self.won:
                if abs(self.vol_ema - c) <= hw and self.vol_ema > 0.03:
                    self.zone_prog = min(1.0, self.zone_prog + dt * 0.4)
                else:
                    self.zone_prog = max(0.0, self.zone_prog - dt * 0.15)
                if self.zone_prog >= 1.0:
                    self._voice_win("voix intensité", self.zone)
        elif self.mode == "pitch":
            if r["voiced"] and r.get("f0", 0) > 0:
                self.pitch_f0 = r["f0"]
                p = float(np.clip(np.log(max(r["f0"], 1) / 100.0) / np.log(5.0),
                                  0.0, 1.0))
                self.pitch_ema = 0.7 * self.pitch_ema + 0.3 * p
            c, hw = self._zone_params()
            if not self.won:
                if r["voiced"] and abs(self.pitch_ema - c) <= hw:
                    self.pitch_prog = min(1.0, self.pitch_prog + dt * 0.4)
                else:
                    self.pitch_prog = max(0.0, self.pitch_prog - dt * 0.15)
                if self.pitch_prog >= 1.0:
                    self._voice_win("voix hauteur", self.pzone)
        else:  # souffle
            active = (r["voiced"] and r["vol"] > 0.06) or self.space_down
            if not self.won:
                if active:
                    self.breath_t += dt
                    self.breath_gap = 0.0
                else:
                    self.breath_gap += dt
                    if self.breath_gap > 0.45:
                        self.breath_t = 0.0
                self.breath_best = max(self.breath_best, self.breath_t)
                if self.breath_t >= self.breath_goal:
                    self._voice_win("souffle", f"{self.breath_goal} s")
        # après la célébration : remise à zéro pour rejouer
        if self.won and now - self.win_time > 2000:
            self.won = False
            self.zone_prog = self.pitch_prog = 0.0
            self.breath_t = 0.0
            if self.tokens >= self.tokens_goal:
                self.tokens = 0

    def _add_token(self):
        self.tokens = min(self.tokens_goal, self.tokens + 1)
        if self.tokens >= self.tokens_goal:
            self._spawn_fireworks(12)    # objectif atteint : méga célébration

    # ---- mode « écoute » (TTS + choix d'images) --------------------------
    def _new_listen_round(self):
        pool = list(dict.fromkeys(
            self.lists[self.list_idx][1] if self.lists else []))
        if len(pool) < max(2, self.listen_n):
            pool = wordlists.DEFAULT_LISTS["animaux"]
        n = min(self.listen_n, len(pool))
        self.listen_opts = random.sample(pool, n)
        self.listen_target = random.choice(self.listen_opts)
        self.listen_wrong = set()
        self.listen_done = 0
        for wd in self.listen_opts:
            self._request_pcache(wd)
        tts.speak(self.listen_target)

    def _listen_click(self, wd):
        if self.listen_done:
            return
        now = pygame.time.get_ticks()
        if wd == self.listen_target:
            self.listen_done = now
            self.win_time = now
            self._spawn_fireworks()
            self._play_cheer()
            self.session.log("écoute", self.listen_target, True, wd)
            self._add_token()
        elif wd not in self.listen_wrong:
            self.listen_wrong.add(wd)
            self.session.log("écoute", self.listen_target, False, wd)
            tts.speak(self.listen_target)   # répète le mot pour réessayer

    def _update_listen(self, now):
        if self.listen_done and now - self.listen_done > 1800:
            if self.tokens >= self.tokens_goal:
                self.tokens = 0
            self._new_listen_round()

    # ---- mode « paires minimales » ---------------------------------------
    def _new_pair(self):
        key = self.pair_keys[self.pair_ci]
        self.pair = random.choice(wordlists.MINIMAL_PAIRS[key])
        self.pair_target = random.randint(0, 1)
        self.pair_ok = False
        self.pair_msg = ""
        self.pair_heard = ""
        self.pair_done = 0
        for wd in self.pair:
            self._request_pcache(wd)

    def _update_pairs(self, now):
        self._update_recording(now)
        if self._pending is not None:
            heard, ok_t = self._pending
            self._pending = None
            self.transcribing = False
            self.pair_heard = heard
            target = self.pair[self.pair_target]
            distr = self.pair[1 - self.pair_target]
            self.session.log("paires", target, ok_t, heard)
            if ok_t:
                self.pair_ok = True
                self.pair_done = now
                self.win_time = now
                self._spawn_fireworks()
                self._play_cheer()
                self._add_token()
            elif WordRecognizer.match(distr, heard):
                self.pair_msg = f"J'ai entendu « {distr} » — essaie encore !"
            elif heard:
                self.pair_msg = f"J'ai entendu « {heard} » — essaie encore"
            else:
                self.pair_msg = "Je n'ai rien entendu, réessaie."
            self._auto_cooldown = now + 1400
        if self.pair_done and now - self.pair_done > 2000:
            if self.tokens >= self.tokens_goal:
                self.tokens = 0
            self._new_pair()
        self._update_autolisten(now)

    # ---- cache de pictogrammes multi-mots --------------------------------
    def _request_pcache(self, word):
        key = word.strip().lower()
        if not key or key in self.pcache:
            return
        self.pcache[key] = "pending"

        def work():
            try:
                path = pictos.fetch_picto(key)
            except Exception:
                path = None
            self._pcache_results.append((key, path))

        threading.Thread(target=work, daemon=True).start()

    def _pump_pcache(self):
        while self._pcache_results:
            key, path = self._pcache_results.pop(0)
            surf = None
            if path:
                try:
                    surf = pygame.image.load(path).convert_alpha()
                except Exception:
                    surf = None
            self.pcache[key] = surf

    def _pcache_get(self, word):
        v = self.pcache.get(word.strip().lower())
        return v if isinstance(v, pygame.Surface) else None

    def reset_progress(self):
        self.energy = 0.0; self.won = False; self.confetti = []

    # ---- mode « mot cible » --------------------------------------------
    def _ensure_model_async(self):
        """Précharge le modèle Whisper en arrière-plan (téléchargement au 1er usage)."""
        if not self.word_rec.available or self.word_rec.model is not None \
                or self.model_loading:
            return
        self.model_loading = True

        def work():
            self.word_rec.ensure_model()
            self.model_loading = False

        threading.Thread(target=work, daemon=True).start()

    def _cycle_model(self):
        i = (WHISPER_MODELS.index(self.model_size) + 1) % len(WHISPER_MODELS) \
            if self.model_size in WHISPER_MODELS else 0
        self.model_size = WHISPER_MODELS[i]
        self.word_rec = WordRecognizer(self.model_size)
        self.model_loading = False
        self._ensure_model_async()

    def _word_edited(self):
        """Appelé quand le mot saisi change : réinitialise l'affichage du picto et
        relance le compte à rebours du débounce."""
        self.word_change_t = pygame.time.get_ticks()
        self.picto_surface = None
        self.word_ok = False
        self.heard = ""

    def _start_recording(self):
        if not self._speech_target().strip():
            self.word_msg = "Tape d'abord un mot à dire."; return
        if not self.running_mic:
            self.word_msg = "Démarre le micro pour écouter."; return
        if not self.word_rec.available:
            self.word_msg = "Installe faster-whisper (voir le README)."; return
        if self.model_loading:
            self.word_msg = f"Chargement du modèle « {self.model_size} »…"; return
        if self.recording or self.transcribing:
            return
        self.word_ok = False; self.heard = ""; self.word_msg = ""
        self.pair_msg = ""
        self.mic.start_record()
        self.recording = True
        self.rec_start = pygame.time.get_ticks()

    def _speech_target(self) -> str:
        """Mot que l'enfant doit prononcer (mode mot ou paire minimale)."""
        if self.mode == "pairs" and self.pair:
            return self.pair[self.pair_target]
        return self.word

    def _start_transcribe(self, audio):
        self.transcribing = True
        target = self._speech_target()
        rec = self.word_rec          # capturé au cas où le modèle change entre-temps

        def work():
            text = rec.transcribe(audio, FS)
            self._pending = (text, WordRecognizer.match(target, text))

        threading.Thread(target=work, daemon=True).start()

    def _update_recording(self, now):
        """Arrête l'enregistrement à la fin de la fenêtre d'écoute et lance la
        transcription (commun aux modes mot et paires)."""
        if self.recording and now - self.rec_start >= REC_SECONDS * 1000:
            audio = self.mic.stop_record()
            self.recording = False
            self._start_transcribe(audio)

    def _update_word(self):
        now = pygame.time.get_ticks()
        self._update_recording(now)
        if self._pending is not None:
            heard, ok = self._pending
            self._pending = None
            self.transcribing = False
            self.heard = heard
            self.word_ok = ok
            self.session.log("mot", self.word, ok, heard)
            if ok:
                self.win_time = now
                self._add_token()
                self._spawn_fireworks()
                self._play_cheer()
            elif self.word_rec.model is None and self.word_rec.load_error:
                self.word_msg = ("Modèle Whisper non chargé — connexion Internet "
                                 "requise au tout premier lancement.")
            elif not heard:
                self.word_msg = "Je n'ai rien entendu, réessaie."
            self._auto_cooldown = now + 1400

        # passage automatique au mot suivant après la célébration d'une réussite
        if (self.word_ok and self.lists and now - self.win_time > 2600):
            self.word_ok = False
            if self.tokens >= self.tokens_goal:
                self.tokens = 0          # objectif atteint : on repart à zéro
            self._next_word()

        self._update_autolisten(now)
        self._update_picto(now)

    def _update_autolisten(self, now):
        """Mains libres : déclenche l'écoute quand l'enfant se met à parler
        (énergie au-dessus d'un seuil), sans clic."""
        if not (self.auto_listen and self.mode in ("word", "pairs")):
            return
        if (self.recording or self.transcribing or self.model_loading
                or not self.running_mic or not self._speech_target().strip()
                or now < self._auto_cooldown):
            return
        frame = self.mic.frame()
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        if rms > self.rec.noise_floor + 0.02:
            self._start_recording()

    def _update_picto(self, now):
        """Récupère automatiquement le pictogramme ARASAAC du mot saisi
        (avec un délai pour ne pas interroger l'API à chaque frappe)."""
        if not self.show_picto:
            return
        word = self.word.strip().lower()
        stable = now - self.word_change_t > 700
        if (word and stable and word != self.picto_word
                and not self.picto_fetching):
            self.picto_fetching = True
            target = word

            def work():
                self._picto_pending = (target, pictos.fetch_picto(target))

            threading.Thread(target=work, daemon=True).start()
        if self._picto_pending is not None:
            pw, path = self._picto_pending
            self._picto_pending = None
            self.picto_fetching = False
            self.picto_word = pw
            self.picto_surface = None
            self.picto_found = bool(path)
            if path:
                try:
                    self.picto_surface = pygame.image.load(path).convert_alpha()
                except Exception as e:
                    print("Chargement image échoué :", e)
                    self.picto_surface = None
                    self.picto_found = False

    # ---- feu d'artifice -------------------------------------------------
    def _spawn_fireworks(self, bursts=6):
        w, h = self.screen.get_size()
        cols = [(255, 107, 107), (255, 212, 59), (105, 219, 124),
                (77, 171, 247), (218, 119, 242), (255, 138, 61)]
        self.fireworks = []
        for _ in range(bursts):
            cx = np.random.uniform(w * 0.15, w * 0.85)
            cy = np.random.uniform(TOP + 40, max(TOP + 60, h * 0.6))
            col = cols[np.random.randint(len(cols))]
            for k in range(28):
                ang = 2 * np.pi * k / 28
                spd = np.random.uniform(2, 5)
                self.fireworks.append([cx, cy, np.cos(ang) * spd,
                                       np.sin(ang) * spd, 60.0, 60.0, col])

    def _update_fireworks(self):
        if not self.fireworks:
            return
        alive = []
        for p in self.fireworks:
            p[0] += p[2]; p[1] += p[3]; p[3] += 0.12; p[4] -= 1
            if p[4] > 0:
                alive.append(p)
        self.fireworks = alive

    def _draw_fireworks(self):
        for p in self.fireworks:
            frac = p[4] / p[5]
            a = max(0, min(255, int(255 * frac)))
            rad = max(1, int(3 * frac) + 1)
            s = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p[6], a), (rad, rad), rad)
            self.screen.blit(s, (p[0] - rad, p[1] - rad))

    # ---- événements -----------------------------------------------------
    def on_click(self, pos):
        for key, b in self.b_scene.items():
            if b.visible and b.rect.collidepoint(pos):
                for x in self.b_scene.values():
                    x.selected = False
                b.selected = True; self.scene = key; self.reset_progress(); return
        for key, b in self.b_mode.items():
            if b.rect.collidepoint(pos):
                for x in self.b_mode.values():
                    x.selected = False
                b.selected = True
                self._switch_mode(key)
                return
        for key, b in self.b_vowel.items():
            if b.visible and b.rect.collidepoint(pos):
                self._select_target(key); return
        for key, b in self.b_frica.items():
            if b.visible and b.rect.collidepoint(pos):
                self._select_target(key); return
        if self.b_hold.visible and self.b_hold.rect.collidepoint(pos):
            self.hold = not self.hold
            self.b_hold.selected = self.hold
            return
        if self.b_speak.visible and self.b_speak.rect.collidepoint(pos):
            self._start_recording(); return
        if self.b_clear.visible and self.b_clear.rect.collidepoint(pos):
            self.word = ""; self.heard = ""; self.word_ok = False
            self.word_msg = ""; self.picto_word = None; self.picto_surface = None
            return
        if self.b_model.visible and self.b_model.rect.collidepoint(pos):
            self._cycle_model(); return
        if self.b_picto.visible and self.b_picto.rect.collidepoint(pos):
            self.show_picto = not self.show_picto
            self.b_picto.selected = self.show_picto
            self.picto_surface = None
            self.picto_word = None       # forcera une nouvelle récupération si réactivé
            return
        if self.b_list.visible and self.b_list.rect.collidepoint(pos):
            self._cycle_list(); return
        if self.b_next.visible and self.b_next.rect.collidepoint(pos):
            self._next_word(); return
        if self.b_denom.visible and self.b_denom.rect.collidepoint(pos):
            self.hide_word = not self.hide_word; return
        if self.b_auto.visible and self.b_auto.rect.collidepoint(pos):
            self.auto_listen = not self.auto_listen; return
        if self.b_export.visible and self.b_export.rect.collidepoint(pos):
            self._export_session(); return
        for key, b in self.b_zone.items():
            if b.visible and b.rect.collidepoint(pos):
                for x in self.b_zone.values():
                    x.selected = False
                b.selected = True; self.zone = key; self.zone_prog = 0.0; return
        for key, b in self.b_pzone.items():
            if b.visible and b.rect.collidepoint(pos):
                for x in self.b_pzone.values():
                    x.selected = False
                b.selected = True; self.pzone = key; self.pitch_prog = 0.0; return
        for key, b in self.b_bgoal.items():
            if b.visible and b.rect.collidepoint(pos):
                for x in self.b_bgoal.values():
                    x.selected = False
                b.selected = True; self.breath_goal = key; self.breath_t = 0.0; return
        if self.b_replay.visible and self.b_replay.rect.collidepoint(pos):
            tts.speak(self.listen_target); return
        if self.b_choices.visible and self.b_choices.rect.collidepoint(pos):
            self.listen_n = {2: 3, 3: 4, 4: 2}[self.listen_n]
            self._new_listen_round(); return
        if self.b_newround.visible and self.b_newround.rect.collidepoint(pos):
            self._new_listen_round(); return
        if self.b_contrast.visible and self.b_contrast.rect.collidepoint(pos):
            self.pair_ci = (self.pair_ci + 1) % len(self.pair_keys)
            self._new_pair(); return
        if self.b_newpair.visible and self.b_newpair.rect.collidepoint(pos):
            self._new_pair(); return
        if self.mode == "listen":
            for rect, wd in self._card_rects:
                if rect.collidepoint(pos):
                    self._listen_click(wd); return
        if self.b_mic.visible and self.b_mic.rect.collidepoint(pos):
            self.toggle_mic(); return

    def _switch_mode(self, key):
        """Change de mode et initialise l'état du nouveau mode."""
        self.mode = key
        self.reset_progress()
        # interrompt proprement une écoute en cours
        if self.recording:
            self.mic.stop_record()
            self.recording = False
        self._pending = None
        self.transcribing = False
        if key == "word":
            self._ensure_model_async()
            if not self.word.strip() and self.lists:
                self._set_word(self.lists[self.list_idx][1][self.list_pos])
        elif key == "listen":
            self._new_listen_round()
        elif key == "pairs":
            self._ensure_model_async()
            self._new_pair()
        elif key in ("loud", "pitch", "breath"):
            self.zone_prog = self.pitch_prog = 0.0
            self.vol_ema = self.pitch_ema = 0.0
            self.breath_t = self.breath_gap = 0.0

    # ---- pack séance : listes, jetons, export ---------------------------
    def _set_word(self, w):
        self.word = w
        self.word_ok = False; self.heard = ""; self.word_msg = ""
        self.word_change_t = pygame.time.get_ticks()
        self.picto_surface = None; self.picto_word = None

    def _cycle_list(self):
        if not self.lists:
            return
        self.list_idx = (self.list_idx + 1) % len(self.lists)
        self.list_pos = 0
        self._set_word(self.lists[self.list_idx][1][0])

    def _next_word(self, step=1):
        if not self.lists:
            return
        words = self.lists[self.list_idx][1]
        self.list_pos = (self.list_pos + step) % len(words)
        self._set_word(words[self.list_pos])

    def _export_session(self):
        try:
            path = self.session.export_csv()
            self.export_msg = "Séance exportée : " + os.path.basename(path)
        except Exception as e:
            self.export_msg = f"Échec export : {e}"

    def _apply_sliders(self):
        self.sensitivity = self.s_sens.value
        self._apply_sensitivity()
        self.threshold = 0.2 + self.s_str.value * 0.65

    def _slider_mousedown(self, pos):
        for s in self._sliders():
            if s.visible and s.hit(pos):
                s.dragging = True
                s.set_from_pos(pos)
                self._apply_sliders()
                return True
        return False

    def _slider_drag(self, pos):
        moved = False
        for s in self._sliders():
            if s.dragging:
                s.set_from_pos(pos); moved = True
        if moved:
            self._apply_sliders()

    def toggle_mic(self):
        if self.running_mic:
            self.mic.stop(); self.running_mic = False
            self.b_mic.label = "▶︎ Démarrer le micro"
        else:
            if self.mic.start():
                self.running_mic = True
                self.b_mic.label = "⏹ Arrêter le micro"
            else:
                self.running_mic = False

    def select_vowel(self, v):
        self._select_target(v)

    def _select_target(self, sound):
        """Sélectionne une cible (voyelle ou fricative) et désélectionne l'autre
        groupe."""
        for x in self.b_vowel.values():
            x.selected = False
        for x in self.b_frica.values():
            x.selected = False
        if sound in self.b_vowel:
            self.b_vowel[sound].selected = True
        elif sound in self.b_frica:
            self.b_frica[sound].selected = True
        self.vowel = sound

    # ---- rendu ----------------------------------------------------------
    def gradient(self, key, w, h, top, bottom):
        ck = (key, w, h)
        if ck not in self._gradients:
            surf = pygame.Surface((w, h))
            for yy in range(h):
                t = yy / max(1, h - 1)
                c = [int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)]
                pygame.draw.line(surf, c, (0, yy), (w, yy))
            self._gradients[ck] = surf
        return self._gradients[ck]

    def draw(self):
        w, h = self.screen.get_size()
        self._layout(w)
        self.screen.fill(COL["bg"])
        scene_rect = pygame.Rect(0, TOP, w, h - TOP)
        self.draw_scene(scene_rect)
        if self.mode not in ("free", "target"):
            self._draw_fireworks()
            self._draw_tokens(scene_rect)
        self.draw_toolbar(w)
        self.draw_overlay(scene_rect)
        self.draw_meters()
        self.draw_status(scene_rect)
        pygame.display.flip()

    def _draw_tokens(self, r):
        """Jetons (étoiles) vers l'objectif + message d'export éventuel."""
        x0, y0, w, h = r
        for i in range(self.tokens_goal):
            col = COL["accent"] if i < self.tokens else (222, 226, 230)
            s = self.font_hint.render("★", True, col)
            self.screen.blit(s, (x0 + w - 40 * (self.tokens_goal - i), y0 + 10))
        if self.export_msg:
            e = self.font_small.render(self.export_msg, True, COL["muted"])
            self.screen.blit(e, (x0 + 12, y0 + h - 24))

    def draw_toolbar(self, w):
        pygame.draw.rect(self.screen, COL["panel"], (0, 0, w, TOP))
        mouse = pygame.mouse.get_pos()
        for b in self._all_buttons():
            b.draw(self.screen, self.font, mouse)
        for s in self._sliders():
            s.draw(self.screen, self.font_small)

    def draw_scene(self, r):
        if self.mode == "word":
            self._draw_word_scene(r)
            return
        if self.mode == "loud":
            label = {"doux": "Voix douce", "moyen": "Voix moyenne",
                     "fort": "Voix forte"}[self.zone]
            self._draw_voice_scene(r, self.vol_ema, self.zone_prog, label, "")
            return
        if self.mode == "pitch":
            extra = f"{self.pitch_f0:.0f} Hz" if self.pitch_f0 > 0 else ""
            label = {"grave": "Voix grave", "medium": "Voix médium",
                     "aigu": "Voix aiguë"}[self.pzone]
            self._draw_voice_scene(r, self.pitch_ema, self.pitch_prog,
                                   label, extra)
            return
        if self.mode == "breath":
            self._draw_breath(r)
            return
        if self.mode == "listen":
            self._draw_listen(r)
            return
        if self.mode == "pairs":
            self._draw_pairs(r)
            return
        if self.scene == "car":
            self._draw_car(r)
        elif self.scene == "balloon":
            self._draw_balloon(r)
        else:
            self._draw_rocket(r)
        if self.won and pygame.time.get_ticks() - self.win_time < 1600:
            self._draw_confetti(r)

    def _draw_word_scene(self, r):
        x0, y0, w, h = r
        cx = x0 + w // 2
        self.screen.blit(self.gradient("word", w, h, (224, 242, 255),
                                       (255, 241, 230)), (x0, y0))
        success = self.word_ok and pygame.time.get_ticks() - self.win_time < 2600

        # pictogramme ARASAAC (si disponible et correspondant au mot courant)
        pic = self.picto_surface if (
            self.show_picto and self.picto_surface is not None
            and self.picto_word == self.word.strip().lower()) else None
        if pic is not None:
            ph = int(min(h * 0.42, 300))
            scale = ph / pic.get_height()
            pic2 = pygame.transform.smoothscale(
                pic, (int(pic.get_width() * scale), ph))
            self.screen.blit(pic2, pic2.get_rect(center=(cx, y0 + int(h * 0.40))))

        # mot (sous le picto s'il y en a un, sinon centré).
        # En mode dénomination, le mot est caché tant que l'enfant n'a pas réussi.
        if self.hide_word and not success and self.word.strip():
            word = "•" * len(self.word.strip())
        else:
            word = self.word if self.word.strip() else "…"
        color = COL["ok"] if success else COL["text"]
        glyph = self.font_big.render(word, True, color)
        maxw = w * 0.9
        maxh = h * 0.26 if pic is not None else h * 0.55
        sc = min(maxw / glyph.get_width(), maxh / glyph.get_height(), 1.0)
        if sc < 1.0:
            glyph = pygame.transform.smoothscale(
                glyph, (int(glyph.get_width() * sc), int(glyph.get_height() * sc)))
        wy = y0 + int(h * 0.76) if pic is not None else y0 + h // 2 - 10
        self.screen.blit(glyph, glyph.get_rect(center=(cx, wy)))

        # mention de la source / indicateur de recherche (coin haut-gauche)
        current = self.word.strip().lower()
        if pic is not None:
            note = "Pictogrammes : ARASAAC (arasaac.org)"
        elif self.show_picto and self.picto_fetching:
            note = "Recherche d'image…"
        elif (self.show_picto and current and self.picto_word == current
              and not self.picto_found):
            note = "Aucune image trouvée pour ce mot (voir console)."
        else:
            note = ""
        if note:
            self.screen.blit(self.font_small.render(note, True, COL["muted"]),
                             (x0 + 12, y0 + 10))

    # ---- scènes des jeux de voix -----------------------------------------
    def _balloon_at(self, cx, cy):
        pygame.draw.line(self.screen, (134, 142, 150), (cx, cy + 56),
                         (cx, cy + 92), 2)
        pygame.draw.ellipse(self.screen, (230, 73, 128),
                            (cx - 44, cy - 54, 88, 108))
        pygame.draw.ellipse(self.screen, (255, 255, 255),
                            (cx - 26, cy - 34, 22, 36))

    def _draw_voice_scene(self, r, value, prog, label, extra):
        """Scène commune intensité / hauteur : le ballon monte avec la valeur,
        l'enfant doit le maintenir dans la zone verte pour remplir la jauge."""
        x0, y0, w, h = r
        self.screen.blit(self.gradient("voice", w, h, (208, 236, 255),
                                       (255, 243, 224)), (x0, y0))
        ytop, ybot = y0 + 90, y0 + h - 80
        span = max(1, ybot - ytop)
        c, hw = self._zone_params()
        # zone cible (bande verte translucide)
        bh = max(8, int(2 * hw * span))
        by = ybot - int((c + hw) * span)
        band = pygame.Surface((w - 120, bh), pygame.SRCALPHA)
        band.fill((81, 207, 102, 70))
        self.screen.blit(band, (x0 + 60, by))
        pygame.draw.rect(self.screen, COL["ok"], (x0 + 60, by, w - 120, bh),
                         2, border_radius=6)
        # ballon à la hauteur courante
        cy = ybot - int(float(np.clip(value, 0, 1)) * span)
        self._balloon_at(x0 + w // 2, cy)
        # jauge de progression
        self._bar("Progression", prog, COL["ok"] if prog > 0.99 else COL["accent2"],
                  x0 + w // 2 - 130, y0 + 16, 260)
        # libellés
        self.screen.blit(self.font_hint.render(label, True, COL["muted"]),
                         (x0 + 16, y0 + 12))
        if extra:
            self.screen.blit(self.font_hint.render(extra, True, COL["accent2"]),
                             (x0 + 16, y0 + 46))

    def _draw_breath(self, r):
        """Souffle : le ballon gonfle tant que le son est tenu."""
        x0, y0, w, h = r
        self.screen.blit(self.gradient("breath", w, h, (224, 242, 255),
                                       (236, 253, 245)), (x0, y0))
        frac = min(1.0, self.breath_t / max(0.5, self.breath_goal))
        rad = 44 + int(frac * min(w, h) * 0.22)
        cx, cy = x0 + w // 2, y0 + h // 2 + 10
        col = (230, 73, 128) if not self.won else (81, 207, 102)
        pygame.draw.ellipse(self.screen, col,
                            (cx - rad, cy - int(rad * 1.15), rad * 2,
                             int(rad * 2.3)))
        pygame.draw.ellipse(self.screen, (255, 255, 255),
                            (cx - int(rad * 0.55), cy - int(rad * 0.75),
                             int(rad * 0.45), int(rad * 0.7)))
        timer = self.font_timer.render(
            f"{self.breath_t:.1f} s / {self.breath_goal} s", True, COL["text"])
        self.screen.blit(timer, timer.get_rect(center=(cx, y0 + 46)))
        best = self.font_small.render(
            f"Record de la séance : {self.breath_best:.1f} s", True, COL["muted"])
        self.screen.blit(best, (x0 + 16, y0 + 12))

    # ---- scènes écoute / paires -------------------------------------------
    def _draw_card(self, rect, word, border, show_label=True):
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=18)
        pygame.draw.rect(self.screen, border, rect, 4, border_radius=18)
        pic = self._pcache_get(word)
        if pic is not None:
            area = rect.inflate(-28, -70)
            sc = min(area.w / pic.get_width(), area.h / pic.get_height(), 1.0)
            img = pygame.transform.smoothscale(
                pic, (int(pic.get_width() * sc), int(pic.get_height() * sc)))
            self.screen.blit(img, img.get_rect(
                center=(rect.centerx, rect.centery - 16)))
            if show_label:
                lab = self.font.render(word, True, COL["muted"])
                self.screen.blit(lab, lab.get_rect(
                    center=(rect.centerx, rect.bottom - 24)))
        else:
            pending = self.pcache.get(word.strip().lower()) == "pending"
            txt = "…" if pending else word
            lab = self.font_hint.render(txt, True, COL["text"])
            self.screen.blit(lab, lab.get_rect(center=rect.center))

    def _draw_listen(self, r):
        x0, y0, w, h = r
        self.screen.blit(self.gradient("listen", w, h, (255, 244, 224),
                                       (224, 242, 255)), (x0, y0))
        self._card_rects = []
        n = max(1, len(self.listen_opts))
        cw = min(250, (w - 80 - (n - 1) * 24) // n)
        ch = min(280, h - 150)
        total = n * cw + (n - 1) * 24
        xs = x0 + (w - total) // 2
        yc = y0 + (h - ch) // 2
        for i, wd in enumerate(self.listen_opts):
            rect = pygame.Rect(xs + i * (cw + 24), yc, cw, ch)
            if self.listen_done and wd == self.listen_target:
                border = COL["ok"]
            elif wd in self.listen_wrong:
                border = COL["danger"]
            else:
                border = (222, 226, 230)
            self._draw_card(rect, wd, border, show_label=False)
            self._card_rects.append((rect, wd))
        note = self.font_small.render("Pictogrammes : ARASAAC (arasaac.org)",
                                      True, COL["muted"])
        self.screen.blit(note, (x0 + 12, y0 + h - 24))

    def _draw_pairs(self, r):
        x0, y0, w, h = r
        self.screen.blit(self.gradient("pairs", w, h, (236, 229, 255),
                                       (255, 241, 230)), (x0, y0))
        if not self.pair:
            return
        target = self.pair[self.pair_target]
        hint = self.font_hint.render(f"Dis : « {target} »", True, COL["text"])
        self.screen.blit(hint, hint.get_rect(center=(x0 + w // 2, y0 + 34)))
        cw = min(290, (w - 120) // 2)
        ch = min(300, h - 170)
        yc = y0 + 70 + (h - 70 - ch) // 2 - 10
        for i, wd in enumerate(self.pair):
            rect = pygame.Rect(x0 + w // 2 + (-cw - 18 if i == 0 else 18),
                               yc, cw, ch)
            if self.pair_ok and i == self.pair_target:
                border = COL["ok"]
            elif i == self.pair_target:
                border = COL["accent"]
            else:
                border = (222, 226, 230)
            self._draw_card(rect, wd, border)
        note = self.font_small.render("Pictogrammes : ARASAAC (arasaac.org)",
                                      True, COL["muted"])
        self.screen.blit(note, (x0 + 12, y0 + h - 24))

    def _flag(self, x, y):
        pygame.draw.line(self.screen, (52, 58, 64), (x, y), (x, y - 46), 3)
        pygame.draw.polygon(self.screen, COL["ok"],
                            [(x, y - 46), (x + 30, y - 38), (x, y - 30)])

    def _draw_car(self, r):
        x0, y0, w, h = r
        self.screen.blit(self.gradient("car", w, h, (189, 224, 254), (232, 245, 233)), (x0, y0))
        road_y = y0 + int(h * 0.72)
        pygame.draw.rect(self.screen, (73, 80, 87), (x0, road_y, w, y0 + h - road_y))
        midy = road_y + (y0 + h - road_y) // 2
        for dx in range(0, w, 55):
            pygame.draw.line(self.screen, (255, 224, 102), (x0 + dx, midy), (x0 + dx + 30, midy), 5)
        self._flag(x0 + w - 50, road_y)
        e = self.energy
        x = x0 + 40 + e * (w - 140)
        y = road_y - 30
        pygame.draw.rect(self.screen, COL["danger"], (x, y - 26, 96, 30), border_radius=8)
        pygame.draw.rect(self.screen, COL["danger"], (x + 16, y - 48, 56, 26), border_radius=8)
        pygame.draw.rect(self.screen, (165, 216, 255), (x + 22, y - 44, 44, 18), border_radius=4)
        for cx in (x + 24, x + 72):
            pygame.draw.circle(self.screen, (33, 37, 41), (int(cx), int(y + 6)), 13)
            pygame.draw.circle(self.screen, (173, 181, 189), (int(cx), int(y + 6)), 5)

    def _draw_balloon(self, r):
        x0, y0, w, h = r
        self.screen.blit(self.gradient("balloon", w, h, (165, 216, 255), (231, 245, 255)), (x0, y0))
        cx = x0 + w // 2
        self._flag(cx - 18, y0 + 70)
        cy = y0 + h - 60 - self.energy * (h - 170)
        pygame.draw.line(self.screen, (134, 142, 150), (cx, cy + 60), (cx, cy + 110), 2)
        pygame.draw.rect(self.screen, (141, 85, 36), (cx - 18, cy + 110, 36, 24), border_radius=5)
        pygame.draw.ellipse(self.screen, (230, 73, 128), (cx - 48, cy - 58, 96, 116))
        pygame.draw.ellipse(self.screen, (255, 255, 255), (cx - 28, cy - 36, 24, 40))

    def _draw_rocket(self, r):
        x0, y0, w, h = r
        self.screen.blit(self.gradient("rocket", w, h, (28, 37, 65), (58, 80, 107)), (x0, y0))
        for i in range(40):
            sxp = x0 + (i * 97 % w); syp = y0 + (i * 53 % int(h * 0.8))
            self.screen.fill((255, 255, 255), (sxp, syp, 2, 2))
        cx = x0 + w // 2
        self._flag(cx - 18, y0 + 60)
        cy = y0 + h - 70 - self.energy * (h - 180)
        if self.energy > 0.02:
            fl = 20 + np.random.random() * 30 * self.energy
            pygame.draw.polygon(self.screen, (255, 169, 77),
                                [(cx - 12, cy + 40), (cx, cy + 40 + fl), (cx + 12, cy + 40)])
            pygame.draw.polygon(self.screen, (255, 224, 102),
                                [(cx - 7, cy + 40), (cx, cy + 40 + fl * 0.6), (cx + 7, cy + 40)])
        pygame.draw.polygon(self.screen, (241, 243, 245),
                            [(cx, cy - 50), (cx + 20, cy + 40), (cx - 20, cy + 40)])
        pygame.draw.polygon(self.screen, COL["danger"],
                            [(cx - 18, cy + 20), (cx - 34, cy + 46), (cx - 18, cy + 40)])
        pygame.draw.polygon(self.screen, COL["danger"],
                            [(cx + 18, cy + 20), (cx + 34, cy + 46), (cx + 18, cy + 40)])
        pygame.draw.circle(self.screen, COL["accent2"], (cx, int(cy - 5)), 9)

    def _draw_confetti(self, r):
        x0, y0, w, h = r
        if not self.confetti:
            cols = [(255, 107, 107), (255, 212, 59), (105, 219, 124),
                    (77, 171, 247), (218, 119, 242)]
            for i in range(90):
                self.confetti.append([x0 + np.random.random() * w,
                                      y0 - np.random.random() * h,
                                      2 + np.random.random() * 3,
                                      5 + np.random.random() * 6,
                                      cols[i % len(cols)]])
        for p in self.confetti:
            p[1] += p[2]
            pygame.draw.rect(self.screen, p[4], (p[0], p[1], p[3], p[3]))

    def draw_overlay(self, r):
        if self.mode != "target":
            return
        x0, y0, w, h = r
        hit = self._target_conf() >= self.threshold
        color = (81, 207, 102, 130) if hit else (255, 138, 61, 60)
        glyph = self.font_big.render(self.vowel, True, color[:3])
        glyph.set_alpha(color[3])
        self.screen.blit(glyph, glyph.get_rect(center=(x0 + w // 2, y0 + h // 2 - 20)))
        hint = self.font_hint.render(f"Fais le son « {self.vowel} » 👄", True, COL["muted"])
        self.screen.blit(hint, hint.get_rect(center=(x0 + w // 2, y0 + h // 2 + 130)))

    def draw_meters(self):
        if self.mode not in ("free", "target"):
            return
        x, y = 14, TOP + 14
        panel = pygame.Surface((196, 96 if self.mode == "target" else 56), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 220))
        self.screen.blit(panel, (x, y))
        self._bar("Volume", self.last.get("vol", 0), COL["accent"], x + 12, y + 12, 170)
        if self.mode == "target":
            conf = self._target_conf()
            heard = (self.last.get("frica") if self.vowel in FRICATIVES
                     else self.last.get("vowel")) or "—"
            self._bar(f"Ressemblance ({heard})", conf,
                      COL["ok"] if conf >= self.threshold else (206, 212, 218),
                      x + 12, y + 52, 170)
            mx = x + 12 + int(self.threshold * 170)
            pygame.draw.line(self.screen, COL["danger"], (mx, y + 66), (mx, y + 82), 2)

    def _target_conf(self) -> float:
        """Confiance courante pour la cible sélectionnée (voyelle ou fricative)."""
        if self.vowel in FRICATIVES:
            return (self.last.get("frica_conf", 0.0)
                    if self.last.get("frica") == self.vowel else 0.0)
        if self.last.get("voiced") and self.last.get("vowel") == self.vowel:
            return self.last.get("confidence", 0.0)
        return 0.0

    def _bar(self, label, val, color, x, y, w):
        self.screen.blit(self.font_small.render(label, True, COL["text"]), (x, y))
        pygame.draw.rect(self.screen, (233, 236, 239), (x, y + 18, w, 12), border_radius=6)
        pygame.draw.rect(self.screen, color, (x, y + 18, int(w * float(np.clip(val, 0, 1))), 12), border_radius=6)

    def draw_status(self, r):
        x0, y0, w, h = r
        now = pygame.time.get_ticks()
        if self.mode == "word":
            if self.model_loading:
                msg = f"⏳ Chargement du modèle « {self.model_size} »… (1re fois : téléchargement)"
            elif self.recording:
                msg = "🎤 J'écoute… dis le mot !"
            elif self.transcribing:
                msg = "⏳ Je réfléchis…"
            elif self.word_ok and now - self.win_time < 2600:
                msg = f"Bravo ! 🎉  « {self.word} »"
            elif self.heard:
                msg = f"J'ai entendu : « {self.heard} » — essaie encore"
            elif self.word_msg:
                msg = self.word_msg
            elif not self.word.strip():
                msg = "Tape un mot, puis clique « 🎤 Parler » (ou Entrée)."
            else:
                msg = "Clique « 🎤 Parler » et dis le mot."
            self._status_box(r, msg); return
        if self.mode == "pairs":
            t = self.pair[self.pair_target] if self.pair else "…"
            if self.model_loading:
                msg = f"⏳ Chargement du modèle « {self.model_size} »…"
            elif self.recording:
                msg = "🎤 J'écoute… dis le mot !"
            elif self.transcribing:
                msg = "⏳ Je réfléchis…"
            elif self.pair_ok and now - self.win_time < 2000:
                msg = f"Bravo ! 🎉  « {t} »"
            elif self.pair_msg:
                msg = self.pair_msg
            elif not self.running_mic:
                msg = "Démarre le micro, puis l'enfant dit le mot encadré."
            else:
                msg = f"Dis : « {t} » (clique « 🎤 Parler » ou Entrée)"
            self._status_box(r, msg); return
        if self.mode == "listen":
            if self.listen_done:
                msg = "Bravo ! 🎉"
            elif not self.tts_ok:
                msg = ("Pas de voix système : dis le mot « "
                       + self.listen_target + " » toi-même, l'enfant clique l'image.")
            else:
                msg = "Écoute bien… et clique sur la bonne image. (🔊 pour réécouter)"
            self._status_box(r, msg); return
        if self.mode == "loud":
            if self.won and now - self.win_time < 2000:
                msg = "Bravo ! 🎉"
            else:
                lib = {"doux": "douce 🤫", "moyen": "moyenne 🙂", "fort": "forte 📢"}[self.zone]
                msg = f"Garde ta voix {lib} dans la zone verte !"
            self._status_box(r, msg); return
        if self.mode == "pitch":
            if self.won and now - self.win_time < 2000:
                msg = "Bravo ! 🎉"
            else:
                lib = {"grave": "grave 🐻", "medium": "médium 🙂", "aigu": "aiguë 🐭"}[self.pzone]
                msg = f"Fais une voix {lib} et reste dans la zone verte !"
            self._status_box(r, msg); return
        if self.mode == "breath":
            if self.won and now - self.win_time < 2000:
                msg = "Bravo ! 🎉"
            else:
                msg = f"Tiens le son « aaaa » pendant {self.breath_goal} s pour gonfler le ballon !"
            self._status_box(r, msg); return
        if self.won and now - self.win_time < 1600:
            msg = "Bravo ! 🎉"
        elif not self.running_mic and not self.space_down:
            msg = "Clique sur « Démarrer le micro ». (Astuce : maintiens ESPACE pour tester sans micro.)"
        elif self.mode == "target":
            msg = f"Fais le son « {self.vowel} » pour avancer."
        else:
            msg = "Fais un son ! 🎤"
        self._status_box(r, msg)

    def _status_box(self, r, msg):
        x0, y0, w, h = r
        surf = self.font.render(msg, True, (255, 255, 255))
        bg = surf.get_rect(center=(x0 + w // 2, y0 + h - 26)).inflate(24, 14)
        box = pygame.Surface(bg.size, pygame.SRCALPHA); box.fill((43, 45, 66, 220))
        self.screen.blit(box, bg.topleft)
        self.screen.blit(surf, surf.get_rect(center=bg.center))

    # ---- boucle principale ---------------------------------------------
    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if not self._slider_mousedown(ev.pos):
                        self.on_click(ev.pos)
                elif ev.type == pygame.MOUSEMOTION:
                    self._slider_drag(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    for s in self._sliders():
                        s.dragging = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif self.mode == "word":
                        # navigation dans la liste + saisie du mot par l'adulte
                        if ev.key == pygame.K_RIGHT:
                            self._next_word(1)
                        elif ev.key == pygame.K_LEFT:
                            self._next_word(-1)
                        elif ev.key == pygame.K_BACKSPACE:
                            self.word = self.word[:-1]; self._word_edited()
                        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self._start_recording()
                        elif ev.unicode and (ev.unicode.isalpha() or ev.unicode in " -'") \
                                and len(self.word) < 22:
                            self.word += ev.unicode; self._word_edited()
                    elif ev.key == pygame.K_SPACE:
                        self.space_down = True
                    elif ev.unicode and ev.unicode.upper() in VOWELS and self.mode == "target":
                        self.select_vowel(ev.unicode.upper())
                elif ev.type == pygame.KEYUP and ev.key == pygame.K_SPACE:
                    self.space_down = False
            self.update()
            self.draw()
            self.dt = min(self.clock.tick(60) / 1000.0, 0.05)
        self.mic.stop()
        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    # IMPORTANT (app empaquetée) : empêche le ré-lancement de l'application — donc
    # l'ouverture d'une 2e fenêtre — quand une dépendance crée un sous-processus
    # (multiprocessing « spawn » sur macOS). Doit être la toute première instruction.
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit(); sys.exit(0)
