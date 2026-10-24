"""Cute crying eyes on Pupper's screen: big glossy eyes welling up, blinking, sobbing, tears falling into puddles.

Run from the laptop with:  ./pup gui examples/crying_eyes.py [seconds]   (default 10, Esc quits)
"""
import math
import random
import sys
import time

import pygame
from pygame import gfxdraw

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
W, H = screen.get_size()
S = min(W, H) / 720  # everything is designed at 720x720 and scaled
CX, CY = W // 2, H // 2


def px(v):
    return int(round(v * S))


# Soft pastel vertical gradient, drawn once.
background = pygame.Surface((W, H))
top, bottom = (255, 214, 228), (214, 200, 255)
for y in range(H):
    t = y / max(H - 1, 1)
    background.fill([int(a + (b - a) * t) for a, b in zip(top, bottom)], (0, y, W, 1))

EYE_RX, EYE_RY = px(92), px(112)
EYE_DX, EYE_Y = px(150), px(300)
EYE_COLOR = (38, 30, 64)
WATER_COLOR = (96, 150, 235)
TEAR_COLOR = (140, 200, 255)
BLUSH_COLOR = (255, 130, 160, 90)
PUDDLE_Y = H - px(40)


def aa_ellipse(surf, color, cx, cy, rx, ry):
    if rx < 1 or ry < 1:
        return
    gfxdraw.filled_ellipse(surf, int(cx), int(cy), int(rx), int(ry), color)
    gfxdraw.aaellipse(surf, int(cx), int(cy), int(rx), int(ry), color)


def aa_circle(surf, color, cx, cy, r):
    if r < 1:
        return
    gfxdraw.filled_circle(surf, int(cx), int(cy), int(r), color)
    gfxdraw.aacircle(surf, int(cx), int(cy), int(r), color)


def thick_curve(surf, color, points, width):
    pygame.draw.lines(surf, color, False, points, width)
    for p in (points[0], points[-1]):
        aa_circle(surf, color, p[0], p[1], width // 2)


def draw_eye(surf, cx, cy, openness, t, side):
    """One glossy eye. openness 1 = wide open, 0 = shut; side -1 left, +1 right."""
    ry = EYE_RY * openness
    if openness < 0.18:
        # Shut tight: a squeezed, downturned curve.
        pts = [(cx + EYE_RX * (i / 10 * 2 - 1), cy + px(10) - px(22) * math.sin(math.pi * i / 10)) for i in range(11)]
        thick_curve(surf, EYE_COLOR, pts, px(12))
        return

    # Eye body clipped to an ellipse, with a wobbling pool of tears in the lower part.
    w, h = EYE_RX * 2 + 4, int(ry * 2) + 4
    body = pygame.Surface((w, h), pygame.SRCALPHA)
    body.fill(EYE_COLOR)
    level = h * (0.62 - 0.04 * math.sin(t * 1.3 + side))
    wave = [(x, level + px(5) * math.sin(x / px(18) + t * 4 + side)) for x in range(0, w + 8, 8)]
    pygame.draw.polygon(body, WATER_COLOR, wave + [(w, h), (0, h)])
    pygame.draw.lines(body, (170, 210, 255), False, wave, max(2, px(3)))
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    aa_ellipse(mask, (255, 255, 255, 255), w // 2, h // 2, EYE_RX, int(ry))
    body.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(body, (cx - w // 2, cy - h // 2))

    # Sparkly highlights that shimmer, so the eyes look wet.
    if openness > 0.5:
        shimmer = 1 + 0.08 * math.sin(t * 6 + side)
        aa_circle(surf, (255, 255, 255), cx - px(30), cy - ry * 0.42, px(30) * shimmer * openness)
        aa_circle(surf, (255, 255, 255), cx + px(28), cy - ry * 0.05, px(13) * shimmer * openness)
        aa_circle(surf, (230, 240, 255), cx - px(8), cy + ry * 0.45, px(8) * openness)


def draw_brow(surf, cx, cy, side, t):
    # Inner end raised: the classic worried, sad brow. It quivers a little.
    lift = px(8) * math.sin(t * 9) * 0.3
    inner = (cx - side * px(70), cy - EYE_RY - px(36) + lift)
    outer = (cx + side * px(80), cy - EYE_RY - px(4))
    mid = ((inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2 - px(10))
    pts = [inner, mid, outer]
    thick_curve(surf, EYE_COLOR, pts, px(14))


def draw_tear(surf, x, y, r, alpha=230):
    r = max(r, 1)
    tip = (x, y - r * 2.4)
    gfxdraw.filled_trigon(surf, int(x - r * 0.92), int(y - r * 0.38), int(x + r * 0.92), int(y - r * 0.38),
                          int(tip[0]), int(tip[1]), (*TEAR_COLOR, alpha))
    gfxdraw.aatrigon(surf, int(x - r * 0.92), int(y - r * 0.38), int(x + r * 0.92), int(y - r * 0.38),
                     int(tip[0]), int(tip[1]), (*TEAR_COLOR, alpha))
    aa_circle(surf, (*TEAR_COLOR, alpha), x, y, r)
    aa_circle(surf, (255, 255, 255, alpha), x - r * 0.35, y - r * 0.3, r * 0.3)


class Tear:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.r = 1.0
        self.target_r = px(random.uniform(11, 17))
        self.vy = 0.0
        self.vx = random.uniform(-0.3, 0.3) * S
        self.growing = True

    def update(self, dt):
        if self.growing:  # well up at the corner of the eye first, then let go
            self.r += (self.target_r - self.r) * min(1, dt * 5)
            if self.r > self.target_r * 0.95:
                self.growing = False
            return
        self.vy += 1400 * S * dt
        self.y += self.vy * dt
        self.x += self.vx


class Puddle:
    def __init__(self, x):
        self.x, self.size, self.ripples = x, 0.0, []

    def splash(self, r):
        self.size = min(self.size + r * 0.5, px(120))
        self.ripples.append([0.0, 1.0])

    def update(self, dt):
        for rip in self.ripples:
            rip[0] += px(110) * dt
            rip[1] -= dt * 1.4
        self.ripples = [r for r in self.ripples if r[1] > 0]

    def draw(self, surf):
        if self.size > 1:
            aa_ellipse(surf, (*TEAR_COLOR, 170), self.x, PUDDLE_Y, self.size, self.size * 0.22)
            aa_ellipse(surf, (220, 240, 255, 150), self.x - self.size * 0.3, PUDDLE_Y - self.size * 0.05,
                       self.size * 0.25, self.size * 0.05)
        for radius, life in self.ripples:
            rx = self.size + radius
            if rx >= 2:
                gfxdraw.aaellipse(surf, int(self.x), PUDDLE_Y, int(rx), max(1, int(rx * 0.22)),
                                  (255, 255, 255, int(200 * life)))


tears = []
puddles = {-1: Puddle(CX - EYE_DX - px(60)), 1: Puddle(CX + EYE_DX + px(60))}
next_tear = {-1: 0.2, 1: 0.6}
next_blink = 2.0
blink_start = -10.0
clock = pygame.time.Clock()
start = time.monotonic()
last = start

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    now = time.monotonic()
    t, dt = now - start, min(now - last, 0.05)
    last = now
    if t >= DURATION_S:
        running = False

    # Sobbing: a hiccupy bob of the whole face every ~1.7 s.
    sob_phase = (t % 1.7) / 1.7
    sob = px(10) * math.sin(sob_phase * math.pi * 2) * math.exp(-sob_phase * 3)
    face_y = EYE_Y + sob

    # Blink: quick close and reopen; during a big sob the eyes squeeze shut.
    if t >= next_blink:
        blink_start, next_blink = t, t + random.uniform(2.2, 3.5)
    bt = (t - blink_start) / 0.22
    openness = abs(bt * 2 - 1) if bt < 1 else 1.0
    if sob_phase < 0.12:
        openness = min(openness, 0.1)
    openness = 0.15 + 0.85 * openness if openness >= 0.18 else openness

    # New tears well up at the outer-bottom corner of each eye.
    for side in (-1, 1):
        if t >= next_tear[side]:
            next_tear[side] = t + random.uniform(0.35, 0.8)
            tears.append(Tear(CX + side * (EYE_DX + px(58)), face_y + EYE_RY * 0.7))
    for tear in tears:
        tear.update(dt)
    for tear in tears:
        if tear.y >= PUDDLE_Y:
            side = -1 if tear.x < CX else 1
            puddles[side].splash(tear.r)
    tears = [tear for tear in tears if tear.y < PUDDLE_Y]
    for p in puddles.values():
        p.update(dt)

    # Draw.
    screen.blit(background, (0, 0))
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)

    # Tear streams running down the cheeks.
    for side in (-1, 1):
        x0 = CX + side * (EYE_DX + px(58))
        pts = [(x0 + px(4) * math.sin(t * 5 + i * 0.6 + side), face_y + EYE_RY * 0.7 + i * px(22)) for i in range(12)]
        pygame.draw.lines(overlay, (*TEAR_COLOR, 110), False, pts, px(16))

    # Rosy cheeks.
    for side in (-1, 1):
        aa_ellipse(overlay, BLUSH_COLOR, CX + side * (EYE_DX + px(20)), face_y + EYE_RY + px(48), px(52), px(26))

    for p in puddles.values():
        p.draw(overlay)
    for tear in tears:
        draw_tear(overlay, tear.x, tear.y, tear.r)
    screen.blit(overlay, (0, 0))

    for side in (-1, 1):
        cx = CX + side * EYE_DX
        draw_eye(screen, cx, face_y, openness, t, side)
        draw_brow(screen, cx, face_y, side, t)

    # Little wobbly frown that quivers.
    mouth_y = face_y + EYE_RY + px(110)
    mouth = [(CX + px(70) * (i / 14 * 2 - 1),
              mouth_y - px(26) * math.sin(math.pi * i / 14) + px(3) * math.sin(t * 22 + i))
             for i in range(15)]
    thick_curve(screen, EYE_COLOR, mouth, px(10))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
print(f"done after {time.monotonic() - start:.1f}s", flush=True)
