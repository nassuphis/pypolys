import math
import random
import sys
import pygame

# -------------- Config --------------
WIDTH, HEIGHT = 900, 600
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
FOV = math.radians(70)          # field of view
HALF_FOV = FOV / 2
MAX_ENEMIES = 10
SPAWN_RADIUS = 18.0
FAR_CLIP = 40.0
NEAR_CLIP = 0.25
RELOAD_T = 0.18                 # seconds between pistol shots
MUZZLE_FLASH_T = 0.06
PLAYER_SPEED = 5.2
TURN_SPEED_KEYS = math.radians(120)  # deg/s via arrows
MOUSE_SENS = 0.0022
START_HEALTH = 100
START_AMMO = 60
START_ARMOR = 0

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tiny DOOM-ish")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("Consolas", 28, bold=True)
font_small = pygame.font.SysFont("Consolas", 18, bold=True)

# -------------- Helpers --------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def lerp(a, b, t):
    return a + (b - a) * t

# -------------- World / Entities --------------
class Player:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.ang = 0.0
        self.health = START_HEALTH
        self.ammo = START_AMMO
        self.armor = START_ARMOR
        self.last_shot = -999.0
        self.muzzle_until = 0.0

class Enemy:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.hp = 3
        self.speed = 1.9
        self.alive = True
        self.attack_cooldown = 0.0

    def dir_to_player(self, p):
        dx, dy = p.x - self.x, p.y - self.y
        dist = math.hypot(dx, dy) + 1e-9
        return dx / dist, dy / dist, dist

# simple rectangular “arena” walls (axis-aligned)
WALLS = [
    (-20, -12, 20, -11),  # top
    (-20,  11, 20,  12),  # bottom
    (-20, -12, -19, 12),  # left
    ( 19, -12,  20, 12),  # right
]

def spawn_enemy(p):
    # spawn around a ring at random angle
    a = random.random() * math.tau
    r = SPAWN_RADIUS
    x = p.x + math.cos(a) * r
    y = p.y + math.sin(a) * r
    return Enemy(x, y)

# -------------- Rendering (simple billboard pseudo-3D) --------------
def world_to_screen(px, py, ang, ex, ey):
    """Project enemy position (ex,ey) into screen space."""
    # translate to player
    dx, dy = ex - px, ey - py
    # rotate opposite of player angle
    ca, sa = math.cos(-ang), math.sin(-ang)
    rx = dx * ca - dy * sa
    ry = dx * sa + dy * ca

    if ry <= NEAR_CLIP:  # behind or too close
        return None

    # perspective divide
    f = (WIDTH / 2) / math.tan(HALF_FOV)
    sx = CENTER_X + (rx * f) / ry
    # size inversely proportional to depth
    scale = clamp(1.8 / ry, 0.001, 3.0)
    return sx, scale, ry

def draw_floor_and_ceiling():
    # simple gradient bands
    screen.fill((35, 35, 45))
    horizon = int(HEIGHT * 0.55)
    pygame.draw.rect(screen, (160, 154, 140), (0, horizon, WIDTH, HEIGHT - horizon))  # floor
    pygame.draw.rect(screen, (70, 70, 80), (0, 0, WIDTH, horizon))                    # ceiling
    # fake lights
    for i in range(6):
        x = i * (WIDTH // 6) + (WIDTH // 12)
        pygame.draw.rect(screen, (210, 210, 210), (x - 40, 30, 80, 14), border_radius=6)

def draw_enemy_billboard(sx, scale, depth, alive=True):
    # enemy silhouette: pinky-like blob
    base_h = 250
    h = int(base_h * scale)
    w = int(h * 0.6)
    x = int(sx - w // 2)
    y = int(HEIGHT * 0.55 - h)

    body_col = (205, 170, 150) if alive else (90, 90, 90)
    outline = (60, 50, 45)
    eye = (230, 20, 20) if alive else (40, 40, 40)

    pygame.draw.ellipse(screen, body_col, (x, y, w, h))
    pygame.draw.ellipse(screen, outline, (x, y, w, h), 3)

    # arms
    aw, ah = int(w * 0.35), int(h * 0.35)
    pygame.draw.ellipse(screen, body_col, (x - aw//2, y + ah//2, aw, ah))
    pygame.draw.ellipse(screen, body_col, (x + w - aw//2, y + ah//2, aw, ah))

    # eyes
    ex = x + w * 0.35
    ey = y + h * 0.35
    ew = int(w * 0.08)
    eh = int(h * 0.06)
    pygame.draw.ellipse(screen, eye, (int(ex), int(ey), ew, eh))
    pygame.draw.ellipse(screen, eye, (int(x + w - w*0.35 - ew), int(ey), ew, eh))

    # depth-based shadow
    alpha = clamp(int(220 * (1.0 - clamp((depth - 2.0) / 12.0, 0, 1))), 0, 120)
    if alpha > 0:
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((0, 0, 0, alpha))
        screen.blit(s, (x, y))

def draw_pistol(muzzle_on):
    # pistol sprite in hands
    gun_w, gun_h = 120, 150
    gx = CENTER_X - gun_w // 2
    gy = HEIGHT - gun_h - 20

    # hands
    pygame.draw.rect(screen, (210, 180, 150), (gx - 12, gy + gun_h - 40, 70, 40), border_radius=8)
    pygame.draw.rect(screen, (210, 180, 150), (gx + gun_w - 58, gy + gun_h - 40, 70, 40), border_radius=8)

    # pistol body
    pygame.draw.rect(screen, (60, 60, 65), (gx + 10, gy + 40, gun_w - 20, gun_h - 40), border_radius=8)
    pygame.draw.rect(screen, (35, 35, 38), (gx + 22, gy + 52, gun_w - 44, gun_h - 64), border_radius=8)
    pygame.draw.rect(screen, (220, 210, 200), (gx + gun_w//2 - 8, gy + 10, 16, 32), border_radius=3)  # barrel

    if muzzle_on:
        pygame.draw.circle(screen, (255, 240, 180), (gx + gun_w//2, gy + 12), 30)
        pygame.draw.circle(screen, (255, 120, 0), (gx + gun_w//2, gy + 12), 18)
        for i in range(6):
            ang = random.random() * math.tau
            r = random.randint(26, 40)
            pygame.draw.circle(screen, (255, 60, 0),
                               (gx + gun_w//2 + int(math.cos(ang) * r),
                                gy + 12 + int(math.sin(ang) * r)), 3)

def draw_hud(p: Player, time_s):
    hud_h = 82
    y0 = HEIGHT - hud_h
    pygame.draw.rect(screen, (40, 40, 40), (0, y0, WIDTH, hud_h))
    pygame.draw.rect(screen, (20, 20, 20), (0, y0, WIDTH, hud_h), 4)

    # AMMO
    ammo_text = font_big.render(f"{p.ammo:02d}", True, (240, 80, 40))
    screen.blit(ammo_text, (24, y0 + 18))
    screen.blit(font_small.render("AMMO", True, (230, 230, 230)), (24, y0 + 50))

    # HEALTH
    col = (60, 230, 80) if p.health >= 60 else (230, 200, 50) if p.health >= 30 else (230, 60, 60)
    hp_text = font_big.render(f"{p.health:3d}%", True, col)
    screen.blit(hp_text, (140, y0 + 18))
    screen.blit(font_small.render("HEALTH", True, (230, 230, 230)), (140, y0 + 50))

    # FACE
    face = pygame.Surface((78, 58))
    face.fill((20, 20, 20))
    # base face color varies with health
    base = int(lerp(80, 220, clamp(p.health / 100.0, 0, 1)))
    pygame.draw.rect(face, (base, base - 20, base - 30), (4, 4, 70, 50), border_radius=6)
    # eyes
    pygame.draw.rect(face, (200, 0, 0), (18, 22, 12, 6))
    pygame.draw.rect(face, (200, 0, 0), (48, 22, 12, 6))
    # mouth (grimace if hurt recently)
    hurt = int((math.sin(time_s * 10) + 1) * 0.5) if p.health < 35 else 0
    pygame.draw.rect(face, (50, 30, 30), (24, 38 + 4*hurt, 30, 6), border_radius=3)
    screen.blit(face, (WIDTH // 2 - 39, y0 + 12))
    pygame.draw.rect(screen, (200, 200, 200), (WIDTH // 2 - 41, y0 + 10, 82, 62), 3, border_radius=8)

    # ARMOR
    ar_text = font_big.render(f"{p.armor:3d}%", True, (220, 220, 220))
    screen.blit(ar_text, (WIDTH - 150, y0 + 18))
    screen.blit(font_small.render("ARMOR", True, (230, 230, 230)), (WIDTH - 150, y0 + 50))

def hitscan(player, enemies, time_s):
    """Simple hitscan: pick closest enemy along the crosshair direction within small angle."""
    if player.ammo <= 0:
        return None

    # fire
    player.ammo -= 1
    player.last_shot = time_s
    player.muzzle_until = time_s + MUZZLE_FLASH_T

    best, best_d = None, 1e9
    for e in enemies:
        if not e.alive:
            continue
        _, _, dist = e.dir_to_player(player)
        if dist < NEAR_CLIP or dist > FAR_CLIP:
            continue
        # angle between aim vector and target vector
        aimx, aimy = math.cos(player.ang), math.sin(player.ang)
        tx, ty = e.x - player.x, e.y - player.y
        tlen = math.hypot(tx, ty) + 1e-9
        dot = (aimx * tx + aimy * ty) / tlen
        # cosine of small cone ~ 4 degrees
        if dot / tlen > 1e6:  # numeric guard (never true)
            pass
        ang_diff = math.acos(clamp((aimx * tx + aimy * ty) / (tlen), -1, 1))
        if ang_diff < math.radians(4.2):
            if dist < best_d:
                best, best_d = e, dist

    if best:
        best.hp -= 1
        if best.hp <= 0:
            best.alive = False
    return best

def move_player(p, dt, keys):
    vx = vy = 0.0
    if keys[pygame.K_w]: vy += 1
    if keys[pygame.K_s]: vy -= 1
    if keys[pygame.K_a]: vx -= 1
    if keys[pygame.K_d]: vx += 1
    mag = math.hypot(vx, vy)
    if mag > 0:
        vx, vy = vx / mag, vy / mag
        # rotate move into world
        ca, sa = math.cos(p.ang), math.sin(p.ang)
        wx = vx * ca - vy * sa
        wy = vx * sa + vy * ca
        p.x += wx * PLAYER_SPEED * dt
        p.y += wy * PLAYER_SPEED * dt

    # rotate with arrows as fallback
    if keys[pygame.K_LEFT]:
        p.ang -= TURN_SPEED_KEYS * dt
    if keys[pygame.K_RIGHT]:
        p.ang += TURN_SPEED_KEYS * dt

    # collide with simple rectangular walls
    for (x0, y0, x1, y1) in WALLS:
        if x0 < p.x < x1 and y0 < p.y < y1:
            # push out along smallest overlap
            dx = min(abs(p.x - x0), abs(x1 - p.x))
            dy = min(abs(p.y - y0), abs(y1 - p.y))
            if dx < dy:
                p.x = x0 + 0.001 if abs(p.x - x0) < abs(x1 - p.x) else x1 - 0.001
            else:
                p.y = y0 + 0.001 if abs(p.y - y0) < abs(y1 - p.y) else y1 - 0.001

def update_enemies(enemies, p, dt, time_s):
    for e in enemies:
        if not e.alive:
            continue
        ux, uy, dist = e.dir_to_player(p)
        # advance toward player
        e.x += ux * e.speed * dt
        e.y += uy * e.speed * dt

        # attack if close
        e.attack_cooldown = max(0.0, e.attack_cooldown - dt)
        if dist < 1.6 and e.attack_cooldown <= 0.0:
            dmg = random.randint(4, 14)
            if p.armor > 0:
                soak = min(p.armor, int(dmg * 0.6))
                p.armor -= soak
                dmg -= soak
            p.health = max(0, p.health - dmg)
            e.attack_cooldown = 0.8

    # remove very far dead ones
    enemies[:] = [e for e in enemies if (e.alive or random.random() > 0.002)]

# -------------- Main loop --------------
def main():
    player = Player()
    enemies = [spawn_enemy(player) for _ in range(5)]

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    running = True
    time_s = 0.0
    while running:
        dt = clock.tick(60) / 1000.0
        time_s += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.rel
                player.ang += mx * MOUSE_SENS
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if time_s - player.last_shot >= RELOAD_T and player.ammo > 0 and player.health > 0:
                    hitscan(player, enemies, time_s)

        keys = pygame.key.get_pressed()
        # keyboard turning if no mouse
        if any(keys[k] for k in (pygame.K_LEFT, pygame.K_RIGHT)):
            pygame.mouse.get_rel()  # flush mouse delta to avoid drift

        # shooting via Space
        if keys[pygame.K_SPACE] and (time_s - player.last_shot >= RELOAD_T) and player.ammo > 0 and player.health > 0:
            hitscan(player, enemies, time_s)

        move_player(player, dt, keys)
        update_enemies(enemies, player, dt, time_s)

        # keep enemy count
        if len([e for e in enemies if e.alive]) < MAX_ENEMIES:
            if random.random() < 0.02:
                enemies.append(spawn_enemy(player))

        # -------- Render --------
        draw_floor_and_ceiling()

        # project enemies and draw back-to-front
        proj = []
        for e in enemies:
            pr = world_to_screen(player.x, player.y, player.ang, e.x, e.y)
            if pr:
                sx, scale, depth = pr
                proj.append((depth, sx, scale, e.alive))
        for depth, sx, scale, alive in sorted(proj, key=lambda t: -t[0]):
            if NEAR_CLIP < depth < FAR_CLIP:
                draw_enemy_billboard(sx, scale, depth, alive)

        # center reticle
        pygame.draw.circle(screen, (255, 255, 255), (CENTER_X, int(HEIGHT * 0.55) - 8), 2)

        # pistol & muzzle
        draw_pistol(time_s < player.muzzle_until)

        # HUD
        draw_hud(player, time_s)

        # “game over” overlay
        if player.health <= 0:
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            screen.blit(s, (0, 0))
            txt = font_big.render("YOU DIED  (Esc to quit)", True, (240, 60, 60))
            screen.blit(txt, (CENTER_X - txt.get_width() // 2, CENTER_Y - 14))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

    