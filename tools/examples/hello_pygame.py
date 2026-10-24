"""Hello world on Pupper's screen: a full-screen pygame window that closes after 10 seconds.

Run from the laptop with:  ./pup gui examples/hello_pygame.py
"""
import sys
import time

import pygame

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
width, height = screen.get_size()
print(f"display {width}x{height}, driver {pygame.display.get_driver()}", flush=True)

title_font = pygame.font.Font(None, height // 5)
small_font = pygame.font.Font(None, height // 14)
clock = pygame.time.Clock()
start = time.monotonic()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    remaining = DURATION_S - (time.monotonic() - start)
    if remaining <= 0:
        running = False

    screen.fill((20, 24, 40))
    title = title_font.render("Hello, world!", True, (255, 255, 255))
    screen.blit(title, title.get_rect(center=(width // 2, height // 2 - height // 12)))
    countdown = small_font.render(f"closing in {max(remaining, 0):.0f}s", True, (150, 170, 255))
    screen.blit(countdown, countdown.get_rect(center=(width // 2, height // 2 + height // 8)))
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
print("closed", flush=True)
