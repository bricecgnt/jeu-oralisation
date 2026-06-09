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

    def _callback(self, indata, frames, time_info, status):  # pragma: no cover
        x = indata[:, 0]
        with self.lock:
            n = len(x)
            if n >= RING:
                self.buf[:] = x[-RING:]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = x

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
        self.last = {"vol": 0.0, "voiced": False, "vowel": None, "confidence": 0.0}
        self.confetti = []
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
        self.b_mode = {"free": Button("Son libre"), "target": Button("Son cible")}
        self.b_mode["free"].selected = True
        self.b_vowel = {v: Button(v, key=v) for v in VOWELS}
        self.b_vowel["A"].selected = True
        self.b_sens_m = Button("Sens −"); self.b_sens_p = Button("Sens +")
        self.b_str_m = Button("Exig. −"); self.b_str_p = Button("Exig. +")
        self.b_mic = Button("▶︎ Démarrer le micro", kind="primary")

    def _all_buttons(self):
        yield from self.b_scene.values()
        yield from self.b_mode.values()
        yield from self.b_vowel.values()
        yield from (self.b_sens_m, self.b_sens_p, self.b_str_m, self.b_str_p,
                    self.b_mic)

    def _layout(self, w):
        """Place les boutons en lignes avec retour à la ligne."""
        global TOP
        target = self.mode == "target"
        for v in self.b_vowel.values():
            v.visible = target
        self.b_str_m.visible = self.b_str_p.visible = target

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
        place(list(self.b_mode.values()), [96, 96])
        place([self.b_vowel[v] for v in VOWELS], [40] * 5)
        place([self.b_sens_m, self.b_sens_p], [78, 78])
        place([self.b_str_m, self.b_str_p], [78, 78])
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
        else:
            self.energy = max(0.0, self.energy - 0.006)

        if self.energy >= 1.0 and not self.won:
            self.won = True
            self.win_time = pygame.time.get_ticks()
            self._play_cheer()
        if self.won and self.energy < 0.6:
            self.won = False

    def reset_progress(self):
        self.energy = 0.0; self.won = False; self.confetti = []

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
        if self.b_sens_m.rect.collidepoint(pos):
            self.sensitivity = max(0.0, self.sensitivity - 0.1); self._apply_sensitivity(); return
        if self.b_sens_p.rect.collidepoint(pos):
            self.sensitivity = min(1.0, self.sensitivity + 0.1); self._apply_sensitivity(); return
        if self.b_str_m.visible and self.b_str_m.rect.collidepoint(pos):
            self.threshold = max(0.2, self.threshold - 0.05); return
        if self.b_str_p.visible and self.b_str_p.rect.collidepoint(pos):
            self.threshold = min(0.85, self.threshold + 0.05); return
        if self.b_mic.rect.collidepoint(pos):
            self.toggle_mic(); return

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

    def draw_scene(self, r):
        sx, sy, sw, sh = r
        if self.scene == "car":
            self._draw_car(r)
        elif self.scene == "balloon":
            self._draw_balloon(r)
        else:
            self._draw_rocket(r)
        if self.won and pygame.time.get_ticks() - self.win_time < 1600:
            self._draw_confetti(r)

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
        if self.won and pygame.time.get_ticks() - self.win_time < 1600:
            msg = "Bravo ! 🎉"
        elif not self.running_mic and not self.space_down:
            msg = "Clique sur « Démarrer le micro ». (Astuce : maintiens ESPACE pour tester sans micro.)"
        elif self.mode == "target":
            msg = f"Fais le son « {self.vowel} » pour avancer."
        else:
            msg = "Fais un son ! 🎤"
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
                    self.on_click(ev.pos)
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE:
                        self.space_down = True
                    elif ev.key == pygame.K_ESCAPE:
                        running = False
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
