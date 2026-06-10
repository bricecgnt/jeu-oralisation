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

import sys
import threading

import numpy as np
import pygame

from recognition import VowelRecognizer, VOWELS
from speech import WordRecognizer

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
        self.word_rec = WordRecognizer()
        self.recording = False
        self.rec_start = 0
        self.transcribing = False
        self.heard = ""                 # dernière transcription
        self.word_ok = False            # le mot a-t-il été reconnu
        self.word_msg = ""              # message d'aide (mode mot)
        self._pending = None            # (texte, ok) renvoyé par le thread
        self.fireworks = []
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
                       "word": Button("Mot cible")}
        self.b_mode["free"].selected = True
        self.b_vowel = {v: Button(v, key=v) for v in VOWELS}
        self.b_vowel["A"].selected = True
        self.b_hold = Button("Reste en place"); self.b_hold.selected = self.hold
        self.b_speak = Button("🎤 Parler", kind="primary")
        self.b_clear = Button("Effacer")
        self.s_sens = Slider("Sensibilité", self.sensitivity)
        self.s_str = Slider("Exigence", (self.threshold - 0.2) / 0.65)
        self.b_mic = Button("▶︎ Démarrer le micro", kind="primary")

    def _all_buttons(self):
        yield from self.b_scene.values()
        yield from self.b_mode.values()
        yield from self.b_vowel.values()
        yield self.b_hold
        yield self.b_speak
        yield self.b_clear
        yield self.b_mic

    def _sliders(self):
        return (self.s_sens, self.s_str)

    def _layout(self, w):
        """Place les boutons en lignes avec retour à la ligne."""
        global TOP
        target = self.mode == "target"
        word = self.mode == "word"
        for v in self.b_vowel.values():
            v.visible = target
        self.s_str.visible = target
        self.s_sens.visible = not word            # Whisper n'utilise pas la sensibilité
        self.b_hold.visible = not word
        self.b_speak.visible = word
        self.b_clear.visible = word
        self.b_speak.label = ("● J'écoute…" if self.recording else
                              "⏳ …" if self.transcribing else "🎤 Parler")

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
        place(list(self.b_mode.values()), [96, 96, 100])
        place([self.b_vowel[v] for v in VOWELS], [40] * 5)
        place([self.b_hold], [148])
        place([self.b_speak, self.b_clear], [150, 96])
        place([self.s_sens], [156])
        place([self.s_str], [156])
        # micro aligné à droite si la place le permet, sinon à la suite
        mic_w = 210
        if x + mic_w > w - pad:
            x = pad; y += h + gap
        self.b_mic.rect = pygame.Rect(max(x, w - pad - mic_w), y, mic_w, h)
        TOP = y + h + pad

    # ---- réglages -------------------------------------------------------
    def _apply_sensitivity(self):
        self.rec.noise_floor = float(np.clip(0.06 - self.sensitivity * 0.055,
                                             0.004, 0.06))

    # ---- audio / logique ------------------------------------------------
    def analyze(self):
        if self.space_down:
            return {"vol": 0.9, "voiced": True, "vowel": self.vowel,
                    "confidence": 1.0}
        if not self.running_mic:
            return {"vol": 0.0, "voiced": False, "vowel": None, "confidence": 0.0}
        return self.rec.process(self.mic.frame())

    def update(self):
        if self.mode == "word":
            self._update_word()
            self._update_fireworks()
            return
        r = self.analyze()
        self.last = r
        if self.mode == "target":
            matched = (r["voiced"] and r["vowel"] == self.vowel
                       and r["confidence"] >= self.threshold)
            drive = r["vol"] * (0.4 + 0.6 * r["confidence"]) if matched else 0.0
        else:
            drive = r["vol"] if r["voiced"] or self.space_down else 0.0

        if drive > 0.05:
            self.energy = min(1.0, self.energy + drive * 0.012)
        elif not self.hold:
            self.energy = max(0.0, self.energy - 0.006)
        # en mode « Reste en place », l'énergie ne redescend pas quand le son cesse

        now = pygame.time.get_ticks()
        if self.energy >= 1.0 and not self.won:
            self.won = True
            self.win_time = now
            self._play_cheer()
        # après la célébration, on repart à zéro pour rejouer
        if self.won and now - self.win_time > 1600:
            self.reset_progress()
        self._update_fireworks()

    def reset_progress(self):
        self.energy = 0.0; self.won = False; self.confetti = []

    # ---- mode « mot cible » --------------------------------------------
    def _start_recording(self):
        if not self.word.strip():
            self.word_msg = "Tape d'abord un mot à dire."; return
        if not self.running_mic:
            self.word_msg = "Démarre le micro pour écouter."; return
        if not self.word_rec.available:
            self.word_msg = "Installe faster-whisper (voir le README)."; return
        if self.recording or self.transcribing:
            return
        self.word_ok = False; self.heard = ""; self.word_msg = ""
        self.mic.start_record()
        self.recording = True
        self.rec_start = pygame.time.get_ticks()

    def _start_transcribe(self, audio):
        self.transcribing = True
        target = self.word

        def work():
            text = self.word_rec.transcribe(audio, FS)
            self._pending = (text, WordRecognizer.match(target, text))

        threading.Thread(target=work, daemon=True).start()

    def _update_word(self):
        now = pygame.time.get_ticks()
        if self.recording and now - self.rec_start >= REC_SECONDS * 1000:
            audio = self.mic.stop_record()
            self.recording = False
            self._start_transcribe(audio)
        if self._pending is not None:
            heard, ok = self._pending
            self._pending = None
            self.transcribing = False
            self.heard = heard
            self.word_ok = ok
            if ok:
                self.win_time = now
                self._spawn_fireworks()
                self._play_cheer()
            elif self.word_rec.model is None and self.word_rec.load_error:
                self.word_msg = ("Modèle Whisper non chargé — connexion Internet "
                                 "requise au tout premier lancement.")
            elif not heard:
                self.word_msg = "Je n'ai rien entendu, réessaie."

    # ---- feu d'artifice -------------------------------------------------
    def _spawn_fireworks(self):
        w, h = self.screen.get_size()
        cols = [(255, 107, 107), (255, 212, 59), (105, 219, 124),
                (77, 171, 247), (218, 119, 242), (255, 138, 61)]
        self.fireworks = []
        for _ in range(6):
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
            if b.rect.collidepoint(pos):
                for x in self.b_scene.values():
                    x.selected = False
                b.selected = True; self.scene = key; self.reset_progress(); return
        for key, b in self.b_mode.items():
            if b.rect.collidepoint(pos):
                for x in self.b_mode.values():
                    x.selected = False
                b.selected = True; self.mode = key; self.reset_progress(); return
        for key, b in self.b_vowel.items():
            if b.visible and b.rect.collidepoint(pos):
                for x in self.b_vowel.values():
                    x.selected = False
                b.selected = True; self.vowel = key; return
        if self.b_hold.visible and self.b_hold.rect.collidepoint(pos):
            self.hold = not self.hold
            self.b_hold.selected = self.hold
            return
        if self.b_speak.visible and self.b_speak.rect.collidepoint(pos):
            self._start_recording(); return
        if self.b_clear.visible and self.b_clear.rect.collidepoint(pos):
            self.word = ""; self.heard = ""; self.word_ok = False
            self.word_msg = ""; return
        if self.b_mic.rect.collidepoint(pos):
            self.toggle_mic(); return

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
        for x in self.b_vowel.values():
            x.selected = False
        self.b_vowel[v].selected = True
        self.vowel = v

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
        self.draw_toolbar(w)
        self.draw_overlay(scene_rect)
        self.draw_meters()
        self.draw_status(scene_rect)
        pygame.display.flip()

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
        self.screen.blit(self.gradient("word", w, h, (224, 242, 255),
                                       (255, 241, 230)), (x0, y0))
        success = self.word_ok and pygame.time.get_ticks() - self.win_time < 2600
        word = self.word if self.word.strip() else "…"
        color = COL["ok"] if success else COL["text"]
        glyph = self.font_big.render(word, True, color)
        maxw, maxh = w * 0.9, (h) * 0.55
        sc = min(maxw / glyph.get_width(), maxh / glyph.get_height(), 1.0)
        if sc < 1.0:
            glyph = pygame.transform.smoothscale(
                glyph, (int(glyph.get_width() * sc), int(glyph.get_height() * sc)))
        self.screen.blit(glyph, glyph.get_rect(center=(x0 + w // 2, y0 + h // 2 - 10)))
        self._draw_fireworks()

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
        hit = (self.last.get("voiced") and self.last.get("vowel") == self.vowel
               and self.last.get("confidence", 0) >= self.threshold)
        color = (81, 207, 102, 130) if hit else (255, 138, 61, 60)
        glyph = self.font_big.render(self.vowel, True, color[:3])
        glyph.set_alpha(color[3])
        self.screen.blit(glyph, glyph.get_rect(center=(x0 + w // 2, y0 + h // 2 - 20)))
        hint = self.font_hint.render(f"Fais le son « {self.vowel} » 👄", True, COL["muted"])
        self.screen.blit(hint, hint.get_rect(center=(x0 + w // 2, y0 + h // 2 + 130)))

    def draw_meters(self):
        if self.mode == "word":
            return
        x, y = 14, TOP + 14
        panel = pygame.Surface((196, 96 if self.mode == "target" else 56), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 220))
        self.screen.blit(panel, (x, y))
        self._bar("Volume", self.last.get("vol", 0), COL["accent"], x + 12, y + 12, 170)
        if self.mode == "target":
            conf = self.last.get("confidence", 0) if self.last.get("vowel") == self.vowel else 0
            ok = self.last.get("voiced") and self.last.get("vowel") == self.vowel and conf >= self.threshold
            self._bar(f"Ressemblance ({self.last.get('vowel') or '—'})", conf,
                      COL["ok"] if ok else (206, 212, 218), x + 12, y + 52, 170)
            mx = x + 12 + int(self.threshold * 170)
            pygame.draw.line(self.screen, COL["danger"], (mx, y + 66), (mx, y + 82), 2)

    def _bar(self, label, val, color, x, y, w):
        self.screen.blit(self.font_small.render(label, True, COL["text"]), (x, y))
        pygame.draw.rect(self.screen, (233, 236, 239), (x, y + 18, w, 12), border_radius=6)
        pygame.draw.rect(self.screen, color, (x, y + 18, int(w * float(np.clip(val, 0, 1))), 12), border_radius=6)

    def draw_status(self, r):
        x0, y0, w, h = r
        now = pygame.time.get_ticks()
        if self.mode == "word":
            if self.recording:
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
                        # saisie du mot par l'adulte
                        if ev.key == pygame.K_BACKSPACE:
                            self.word = self.word[:-1]
                        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self._start_recording()
                        elif ev.unicode and (ev.unicode.isalpha() or ev.unicode in " -'") \
                                and len(self.word) < 22:
                            self.word += ev.unicode
                    elif ev.key == pygame.K_SPACE:
                        self.space_down = True
                    elif ev.unicode and ev.unicode.upper() in VOWELS and self.mode == "target":
                        self.select_vowel(ev.unicode.upper())
                elif ev.type == pygame.KEYUP and ev.key == pygame.K_SPACE:
                    self.space_down = False
            self.update()
            self.draw()
            self.clock.tick(60)
        self.mic.stop()
        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit(); sys.exit(0)
