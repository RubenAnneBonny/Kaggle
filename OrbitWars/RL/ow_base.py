# ===== notebook cell 1 =====


import math
from collections import namedtuple
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

CENTER = 50.0
SUN_RADIUS = 10.0
SHIP_MAX_SPEED = 6.0


def get_field(obs, name, default):
    """obs may be a dict (local) or an attribute object (sandbox)."""
    return obs.get(name, default) if isinstance(obs, dict) else getattr(obs, name)


def parse_obs(obs):
    """Pull the common fields out of an observation once.

    Returns: player, list[Planet] mine, list[Planet] targets, set comet_ids, ang_vel
    """
    player = get_field(obs, "player", 0)
    raw_planets = get_field(obs, "planets", [])
    comet_ids = set(get_field(obs, "comet_planet_ids", []))
    ang_vel = get_field(obs, "angular_velocity", 0)

    planets = [Planet(*p) for p in raw_planets]
    mine = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    return player, mine, targets, comet_ids, ang_vel


def is_orbiting(p):
    return math.hypot(p.x - CENTER, p.y - CENTER) + p.radius < CENTER


def fleet_speed(ships):
    s = 1.0 + (SHIP_MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, SHIP_MAX_SPEED)


def sq_dist(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def bad_target_ids(obs, comet_ids):
    """Planets we should not aim at this turn: comets + planets already
    being intercepted by one of our in-flight fleets."""
    incoming = {pid for (_fid, pid, _t) in fleet_planet_collision(obs)}
    return comet_ids | incoming

# ===== notebook cell 2 =====

def how_many_send_no_cap(my_planet, enemy_planet, spare, obs):
    """Smallest fleet that intercepts AND captures enemy_planet, via a clear path.

    Returns (ships, angle, timesteps).
      - ships == 1e9 and timesteps is None  -> no viable shot. This happens when no
        fleet can capture, OR when every capturing shot's straight path is blocked
        by another planet (or the sun) before it reaches the target.
      - timesteps is the (fractional) number of turns until the fleet arrives.
    """
    ang_vel = get_field(obs, "angular_velocity", 0)
    planets = [Planet(*p) for p in get_field(obs, "planets", [])]
    orbits = is_orbiting(enemy_planet)

    def planet_pos(t):
        if not orbits:
            return enemy_planet.x, enemy_planet.y
        a = t * ang_vel
        dx, dy = enemy_planet.x - CENTER, enemy_planet.y - CENTER
        return (CENTER + math.cos(a) * dx - math.sin(a) * dy,
                CENTER + math.sin(a) * dx + math.cos(a) * dy)

    def f(t, speed):
        px, py = planet_pos(t)
        return math.hypot(px - my_planet.x, py - my_planet.y) - (my_planet.radius + 0.1) - t * speed

    def path_hits_sun(px, py):
        sx, sy = px - my_planet.x, py - my_planet.y
        seg2 = sx * sx + sy * sy
        if seg2 == 0:
            return False
        u = ((CENTER - my_planet.x) * sx + (CENTER - my_planet.y) * sy) / seg2
        u = max(0.0, min(1.0, u))
        cx, cy = my_planet.x + u * sx, my_planet.y + u * sy
        return math.hypot(cx - CENTER, cy - CENTER) <= SUN_RADIUS

    def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
        d0x, d0y = ax - p0x, ay - p0y
        dvx = (bx - ax) - (p1x - p0x)
        dvy = (by - ay) - (p1y - p0y)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0

    # Precompute OTHER planets' positions per tick ONCE (independent of fleet size),
    # so the obstacle check is cheap to reuse across the binary search.
    MAX_T = 101
    others = [p for p in planets if p.id != enemy_planet.id]
    o_static = [(p.x, p.y, p.radius) for p in others if not is_orbiting(p)]
    o_orbit = [p for p in others if is_orbiting(p)]
    o_pos = []
    for t in range(MAX_T):
        a = t * ang_vel
        ca, sa = math.cos(a), math.sin(a)
        o_pos.append([(CENTER + ca * (p.x - CENTER) - sa * (p.y - CENTER),
                       CENTER + sa * (p.x - CENTER) + ca * (p.y - CENTER),
                       p.radius) for p in o_orbit])

    def path_blocked(angle, speed, t_arrive):
        """True if the fleet sweeps any non-target planet before reaching the target."""
        dx, dy = math.cos(angle) * speed, math.sin(angle) * speed
        fx = my_planet.x + math.cos(angle) * (my_planet.radius + 0.1)
        fy = my_planet.y + math.sin(angle) * (my_planet.radius + 0.1)
        last = min(int(t_arrive), MAX_T - 1)
        for t in range(1, last + 1):
            nfx, nfy = fx + dx, fy + dy
            mx, my_ = (fx + nfx) * 0.5, (fy + nfy) * 0.5
            for px, py, pr in o_static:
                if (px - mx) ** 2 + (py - my_) ** 2 > (speed + 0.5 + pr) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, px, py, px, py, pr):
                    return True
            for (p0x, p0y, pr), (p1x, p1y, _) in zip(o_pos[t - 1], o_pos[t]):
                if (p1x - mx) ** 2 + (p1y - my_) ** 2 > (speed + 0.5 + pr + 3) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, p0x, p0y, p1x, p1y, pr):
                    return True
            fx, fy = nfx, nfy
        return False

    def solve_intercept(speed):
        prev = f(0, speed)
        for t in range(1, 101):
            cur = f(t, speed)
            if cur <= 0 and prev > 0:
                lo, hi = t - 1, t
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if f(mid, speed) > 0:
                        lo = mid
                    else:
                        hi = mid
                t_star = (lo + hi) / 2
                px, py = planet_pos(t_star)
                if path_hits_sun(px, py):
                    prev = cur
                    continue
                angle = math.atan2(py - my_planet.y, px - my_planet.x)
                return t_star, angle
            prev = cur
        return None

    lo_size, hi_size = enemy_planet.ships + spare, my_planet.ships
    best, best_angle, best_t = 1e9, 0, None

    while lo_size <= hi_size:
        ships = (lo_size + hi_size) // 2
        speed = fleet_speed(ships)

        res = solve_intercept(speed)
        if res is None:
            lo_size = ships + 1
            continue

        t_star, angle = res
        prod = enemy_planet.production * t_star if enemy_planet.owner != -1 else 0
        if ships >= enemy_planet.ships + prod + spare:
            if path_blocked(angle, speed, t_star):
                lo_size = ships + 1          # blocked -> try a faster (bigger) fleet
            else:
                best, best_angle, best_t = ships, angle, t_star
                hi_size = ships - 1
        else:
            lo_size = ships + 1

    return best, best_angle, best_t

# ===== notebook cell 3 =====

def how_many_send_improved(my_planet, enemy_planet, spare, obs):
    """Smallest fleet that intercepts AND captures enemy_planet, via a clear path.

    Returns (ships, angle, timesteps).
      - ships == 1e9 and timesteps is None  -> no viable shot. This happens when no
        fleet can capture, OR when every capturing shot's straight path is blocked
        by another planet (or the sun) before it reaches the target.
      - timesteps is the (fractional) number of turns until the fleet arrives.
    """
    ang_vel = get_field(obs, "angular_velocity", 0)
    planets = [Planet(*p) for p in get_field(obs, "planets", [])]
    orbits = is_orbiting(enemy_planet)

    def planet_pos(t):
        if not orbits:
            return enemy_planet.x, enemy_planet.y
        a = t * ang_vel
        dx, dy = enemy_planet.x - CENTER, enemy_planet.y - CENTER
        return (CENTER + math.cos(a) * dx - math.sin(a) * dy,
                CENTER + math.sin(a) * dx + math.cos(a) * dy)

    def f(t, speed):
        px, py = planet_pos(t)
        return math.hypot(px - my_planet.x, py - my_planet.y) - (my_planet.radius + 0.1) - t * speed

    def path_hits_sun(px, py):
        sx, sy = px - my_planet.x, py - my_planet.y
        seg2 = sx * sx + sy * sy
        if seg2 == 0:
            return False
        u = ((CENTER - my_planet.x) * sx + (CENTER - my_planet.y) * sy) / seg2
        u = max(0.0, min(1.0, u))
        cx, cy = my_planet.x + u * sx, my_planet.y + u * sy
        return math.hypot(cx - CENTER, cy - CENTER) <= SUN_RADIUS

    def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
        d0x, d0y = ax - p0x, ay - p0y
        dvx = (bx - ax) - (p1x - p0x)
        dvy = (by - ay) - (p1y - p0y)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0

    # Precompute OTHER planets' positions per tick ONCE (independent of fleet size),
    # so the obstacle check is cheap to reuse across the binary search.
    MAX_T = 101
    others = [p for p in planets if p.id != enemy_planet.id]
    o_static = [(p.x, p.y, p.radius) for p in others if not is_orbiting(p)]
    o_orbit = [p for p in others if is_orbiting(p)]
    o_pos = []
    for t in range(MAX_T):
        a = t * ang_vel
        ca, sa = math.cos(a), math.sin(a)
        o_pos.append([(CENTER + ca * (p.x - CENTER) - sa * (p.y - CENTER),
                       CENTER + sa * (p.x - CENTER) + ca * (p.y - CENTER),
                       p.radius) for p in o_orbit])

    def path_blocked(angle, speed, t_arrive):
        """True if the fleet sweeps any non-target planet before reaching the target."""
        dx, dy = math.cos(angle) * speed, math.sin(angle) * speed
        fx = my_planet.x + math.cos(angle) * (my_planet.radius + 0.1)
        fy = my_planet.y + math.sin(angle) * (my_planet.radius + 0.1)
        last = min(int(t_arrive), MAX_T - 1)
        for t in range(1, last + 1):
            nfx, nfy = fx + dx, fy + dy
            mx, my_ = (fx + nfx) * 0.5, (fy + nfy) * 0.5
            for px, py, pr in o_static:
                if (px - mx) ** 2 + (py - my_) ** 2 > (speed + 0.5 + pr) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, px, py, px, py, pr):
                    return True
            for (p0x, p0y, pr), (p1x, p1y, _) in zip(o_pos[t - 1], o_pos[t]):
                if (p1x - mx) ** 2 + (p1y - my_) ** 2 > (speed + 0.5 + pr + 3) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, p0x, p0y, p1x, p1y, pr):
                    return True
            fx, fy = nfx, nfy
        return False

    def solve_intercept(speed):
        prev = f(0, speed)
        for t in range(1, 101):
            cur = f(t, speed)
            if cur <= 0 and prev > 0:
                lo, hi = t - 1, t
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if f(mid, speed) > 0:
                        lo = mid
                    else:
                        hi = mid
                t_star = (lo + hi) / 2
                px, py = planet_pos(t_star)
                if path_hits_sun(px, py):
                    prev = cur
                    continue
                angle = math.atan2(py - my_planet.y, px - my_planet.x)
                return t_star, angle
            prev = cur
        return None

    lo_size, hi_size = enemy_planet.ships + spare, my_planet.ships
    best, best_angle, best_t = 1e9, 0, None

    while lo_size <= hi_size:
        ships = (lo_size + hi_size) // 2
        speed = fleet_speed(ships)

        res = solve_intercept(speed)
        if res is None:
            lo_size = ships + 1
            continue

        t_star, angle = res
        prod = enemy_planet.production * math.ceil(t_star) if enemy_planet.owner != -1 else 0
        if ships >= enemy_planet.ships + prod + spare:
            if path_blocked(angle, speed, t_star):
                lo_size = ships + 1          # blocked -> try a faster (bigger) fleet
            else:
                best, best_angle, best_t = ships, angle, t_star
                hi_size = ships - 1
        else:
            lo_size = ships + 1

    return best, best_angle, best_t

# ===== notebook cell 4 =====

def how_many_send(my_planet, enemy_planet, spare, obs):
    """Smallest fleet that intercepts AND captures enemy_planet, via a clear path.

    Returns (ships, angle, timesteps).
      - ships == 1e9 and timesteps is None  -> no viable shot. This happens when no
        fleet can capture, OR when every capturing shot's straight path is blocked
        by another planet (or the sun) before it reaches the target.
      - timesteps is the (fractional) number of turns until the fleet arrives.
    """
    ang_vel = get_field(obs, "angular_velocity", 0)
    planets = [Planet(*p) for p in get_field(obs, "planets", [])]
    orbits = is_orbiting(enemy_planet)

    def planet_pos(t):
        if not orbits:
            return enemy_planet.x, enemy_planet.y
        a = t * ang_vel
        dx, dy = enemy_planet.x - CENTER, enemy_planet.y - CENTER
        return (CENTER + math.cos(a) * dx - math.sin(a) * dy,
                CENTER + math.sin(a) * dx + math.cos(a) * dy)

    def f(t, speed):
        px, py = planet_pos(t)
        return math.hypot(px - my_planet.x, py - my_planet.y) - (my_planet.radius + 0.1) - t * speed

    def path_hits_sun(px, py):
        sx, sy = px - my_planet.x, py - my_planet.y
        seg2 = sx * sx + sy * sy
        if seg2 == 0:
            return False
        u = ((CENTER - my_planet.x) * sx + (CENTER - my_planet.y) * sy) / seg2
        u = max(0.0, min(1.0, u))
        cx, cy = my_planet.x + u * sx, my_planet.y + u * sy
        return math.hypot(cx - CENTER, cy - CENTER) <= SUN_RADIUS

    def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
        d0x, d0y = ax - p0x, ay - p0y
        dvx = (bx - ax) - (p1x - p0x)
        dvy = (by - ay) - (p1y - p0y)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0

    # Precompute OTHER planets' positions per tick ONCE (independent of fleet size),
    # so the obstacle check is cheap to reuse across the binary search.
    MAX_T = 101
    others = [p for p in planets if p.id != enemy_planet.id]
    o_static = [(p.x, p.y, p.radius) for p in others if not is_orbiting(p)]
    o_orbit = [p for p in others if is_orbiting(p)]
    o_pos = []
    for t in range(MAX_T):
        a = t * ang_vel
        ca, sa = math.cos(a), math.sin(a)
        o_pos.append([(CENTER + ca * (p.x - CENTER) - sa * (p.y - CENTER),
                       CENTER + sa * (p.x - CENTER) + ca * (p.y - CENTER),
                       p.radius) for p in o_orbit])

    def path_blocked(angle, speed, t_arrive):
        """True if the fleet sweeps any non-target planet before reaching the target."""
        dx, dy = math.cos(angle) * speed, math.sin(angle) * speed
        fx = my_planet.x + math.cos(angle) * (my_planet.radius + 0.1)
        fy = my_planet.y + math.sin(angle) * (my_planet.radius + 0.1)
        last = min(int(t_arrive), MAX_T - 1)
        for t in range(1, last + 1):
            nfx, nfy = fx + dx, fy + dy
            mx, my_ = (fx + nfx) * 0.5, (fy + nfy) * 0.5
            for px, py, pr in o_static:
                if (px - mx) ** 2 + (py - my_) ** 2 > (speed + 0.5 + pr) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, px, py, px, py, pr):
                    return True
            for (p0x, p0y, pr), (p1x, p1y, _) in zip(o_pos[t - 1], o_pos[t]):
                if (p1x - mx) ** 2 + (p1y - my_) ** 2 > (speed + 0.5 + pr + 3) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, p0x, p0y, p1x, p1y, pr):
                    return True
            fx, fy = nfx, nfy
        return False

    def solve_intercept(speed):
        prev = f(0, speed)
        for t in range(1, 101):
            cur = f(t, speed)
            if cur <= 0 and prev > 0:
                lo, hi = t - 1, t
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if f(mid, speed) > 0:
                        lo = mid
                    else:
                        hi = mid
                t_star = (lo + hi) / 2
                px, py = planet_pos(t_star)
                if path_hits_sun(px, py):
                    prev = cur
                    continue
                angle = math.atan2(py - my_planet.y, px - my_planet.x)
                return t_star, angle
            prev = cur
        return None

    lo_size, hi_size = enemy_planet.ships + spare, my_planet.ships
    best, best_angle, best_t = 1e9, 0, None

    while lo_size <= hi_size:
        ships = (lo_size + hi_size) // 2
        speed = fleet_speed(ships)

        res = solve_intercept(speed)
        if res is None:
            lo_size = ships + 1
            continue

        t_star, angle = res
        prod = enemy_planet.production * min(t_star, 7) if enemy_planet.owner != -1 else 0
        if ships >= enemy_planet.ships + prod + spare:
            if path_blocked(angle, speed, t_star):
                lo_size = ships + 1          # blocked -> try a faster (bigger) fleet
            else:
                best, best_angle, best_t = ships, angle, t_star
                hi_size = ships - 1
        else:
            lo_size = ships + 1

    return best, best_angle, best_t

# ===== notebook cell 5 =====

def fleet_planet_collision(obs):
    """For each of my in-flight fleets, find the first planet its path hits.

    Uses the engine's exact continuous (swept-segment) test, with planet
    positions precomputed once per tick and a cheap bounding-radius reject.

    Returns list of (fleet_id, planet_id, tick).
    """
    player = get_field(obs, "player", 0)
    raw_fleets = get_field(obs, "fleets", [])
    raw_planets = get_field(obs, "planets", [])
    ang_vel = get_field(obs, "angular_velocity", 0)

    my_fleets = [Fleet(*f) for f in raw_fleets if Fleet(*f).owner == player]
    if not my_fleets:
        return []

    planets = [Planet(*p) for p in raw_planets]

    # Split once: static planets never move; orbiting ones get cached per tick.
    static = []
    orbiting = []
    for p in planets:
        if is_orbiting(p):
            orbiting.append(p)
        else:
            static.append((p.id, p.x, p.y, p.radius))

    MAX_T = 101

    # Precompute orbiting positions for every tick once (shared across fleets).
    orbit_pos = []
    for t in range(MAX_T):
        a = t * ang_vel
        ca, sa = math.cos(a), math.sin(a)
        row = []
        for p in orbiting:
            dx, dy = p.x - CENTER, p.y - CENTER
            row.append((p.id, CENTER + ca * dx - sa * dy, CENTER + sa * dx + ca * dy, p.radius))
        orbit_pos.append(row)

    def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
        d0x, d0y = ax - p0x, ay - p0y
        dvx = (bx - ax) - (p1x - p0x)
        dvy = (by - ay) - (p1y - p0y)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0

    hits = []
    for fl in my_fleets:
        speed = fleet_speed(fl.ships)
        dx, dy = math.cos(fl.angle) * speed, math.sin(fl.angle) * speed
        seg_r = speed + 0.5

        fx, fy = fl.x, fl.y
        hit = False
        for t in range(1, MAX_T):
            nfx, nfy = fx + dx, fy + dy
            mx, my = (fx + nfx) * 0.5, (fy + nfy) * 0.5

            for pid, px, py, pr in static:
                if (px - mx) ** 2 + (py - my) ** 2 > (seg_r + pr) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, px, py, px, py, pr):
                    hits.append((fl.id, pid, t))
                    hit = True
                    break
            if hit:
                break

            for (pid, p0x, p0y, pr), (_, p1x, p1y, _) in zip(orbit_pos[t - 1], orbit_pos[t]):
                if (p1x - mx) ** 2 + (p1y - my) ** 2 > (seg_r + pr + 3) ** 2:
                    continue
                if swept_pair_hit(fx, fy, nfx, nfy, p0x, p0y, p1x, p1y, pr):
                    hits.append((fl.id, pid, t))
                    hit = True
                    break
            if hit:
                break

            fx, fy = nfx, nfy

    return hits

# ===== notebook cell 6 =====

def predict_all_fleet_hits(obs):
    """Predict the first planet every in-flight fleet (any owner) will hit.
    Returns list of (owner, ships, planet_id, tick)."""
    raw_fleets = get_field(obs, "fleets", [])
    if not raw_fleets:
        return []
    ang_vel = get_field(obs, "angular_velocity", 0)
    fleets = [Fleet(*f) for f in raw_fleets]
    planets = [Planet(*p) for p in get_field(obs, "planets", [])]

    static, orbiting = [], []
    for p in planets:
        (orbiting if is_orbiting(p) else static).append(p)
    static = [(p.id, p.x, p.y, p.radius) for p in static]

    MAX_T = 101
    orbit_pos = []
    for t in range(MAX_T):
        a = t * ang_vel
        ca, sa = math.cos(a), math.sin(a)
        orbit_pos.append([(p.id, CENTER + ca*(p.x-CENTER) - sa*(p.y-CENTER),
                           CENTER + sa*(p.x-CENTER) + ca*(p.y-CENTER), p.radius) for p in orbiting])

    def swept(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
        d0x, d0y = ax-p0x, ay-p0y
        dvx = (bx-ax)-(p1x-p0x); dvy = (by-ay)-(p1y-p0y)
        a = dvx*dvx+dvy*dvy; b = 2.0*(d0x*dvx+d0y*dvy); c = d0x*d0x+d0y*d0y-r*r
        if a < 1e-12: return c <= 0.0
        disc = b*b-4.0*a*c
        if disc < 0.0: return False
        sq = math.sqrt(disc)
        return (-b+sq)/(2.0*a) >= 0.0 and (-b-sq)/(2.0*a) <= 1.0

    hits = []
    for fl in fleets:
        speed = fleet_speed(fl.ships)
        dx, dy = math.cos(fl.angle)*speed, math.sin(fl.angle)*speed
        seg_r = speed + 0.5
        fx, fy = fl.x, fl.y
        done = False
        for t in range(1, MAX_T):
            nfx, nfy = fx+dx, fy+dy
            mx, my = (fx+nfx)*0.5, (fy+nfy)*0.5
            for pid, px, py, pr in static:
                if (px-mx)**2 + (py-my)**2 > (seg_r+pr)**2: continue
                if swept(fx, fy, nfx, nfy, px, py, px, py, pr):
                    hits.append((fl.owner, fl.ships, pid, t)); done = True; break
            if done: break
            for (pid, p0x, p0y, pr), (_, p1x, p1y, _) in zip(orbit_pos[t-1], orbit_pos[t]):
                if (p1x-mx)**2 + (p1y-my)**2 > (seg_r+pr+3)**2: continue
                if swept(fx, fy, nfx, nfy, p0x, p0y, p1x, p1y, pr):
                    hits.append((fl.owner, fl.ships, pid, t)); done = True; break
            if done: break
            fx, fy = nfx, nfy
    return hits


def planet_balance(obs):
    """Per-planet incoming balance: my fleets +ships, enemy fleets -ships.
    Returns dict planet_id -> (net_incoming, earliest_enemy_tick, enemy_ships)."""
    player = get_field(obs, "player", 0)
    info = {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        net, etick, esh = info.get(pid, (0, 1e9, 0))
        if owner == player:
            net += ships
        else:
            net -= ships
            esh += ships
            etick = min(etick, tick)
        info[pid] = (net, etick, esh)
    return info


def reach(src, dst, ships, obs):
    """Earliest (t_star, angle) for a fleet of `ships` from src to reach dst
    (dst may be orbiting), avoiding the sun. None if unreachable in 100 turns."""
    ang_vel = get_field(obs, "angular_velocity", 0)
    orbits = is_orbiting(dst)
    speed = fleet_speed(ships)

    def pos(t):
        if not orbits: return dst.x, dst.y
        a = t * ang_vel
        dx, dy = dst.x-CENTER, dst.y-CENTER
        return (CENTER + math.cos(a)*dx - math.sin(a)*dy,
                CENTER + math.sin(a)*dx + math.cos(a)*dy)

    def f(t):
        px, py = pos(t)
        return math.hypot(px-src.x, py-src.y) - (src.radius+0.1) - t*speed

    def hits_sun(px, py):
        sx, sy = px-src.x, py-src.y
        seg2 = sx*sx+sy*sy
        if seg2 == 0: return False
        u = max(0.0, min(1.0, ((CENTER-src.x)*sx + (CENTER-src.y)*sy)/seg2))
        cx, cy = src.x+u*sx, src.y+u*sy
        return math.hypot(cx-CENTER, cy-CENTER) <= SUN_RADIUS

    prev = f(0)
    for t in range(1, 101):
        cur = f(t)
        if cur <= 0 and prev > 0:
            lo, hi = t-1, t
            for _ in range(40):
                mid = (lo+hi)/2
                if f(mid) > 0: lo = mid
                else: hi = mid
            ts = (lo+hi)/2
            px, py = pos(ts)
            if hits_sun(px, py):
                prev = cur; continue
            return ts, math.atan2(py-src.y, px-src.x)
        prev = cur
    return None

# ===== notebook cell 7 =====

def nearest_planet(obs):
    """Each owned planet attacks its closest weaker, non-comet target."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not targets:
        return moves

    for planet in mine:
        best = None
        best_d = 1e9
        for p in targets:
            if p.id in comet_ids or p.ships >= planet.ships:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d

        if best is not None:
            ships, angle, _ = how_many_send(planet, best, 1, obs)
            if ships <= planet.ships:
                moves.append([planet.id, angle, ships])

    return moves

# ===== notebook cell 8 =====

def nearest_planet_smart(obs):
    """Like nearest_planet, but skips comets AND planets already targeted by
    one of my in-flight fleets. Avoids double-committing within the turn too."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not targets:
        return moves

    bad = bad_target_ids(obs, comet_ids)   # computed once per turn

    for planet in mine:
        best = None
        best_d = 1e9
        for p in targets:
            if p.id in bad or p.ships >= planet.ships:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d

        if best is not None:
            ships, angle, _ = how_many_send(planet, best, 1, obs)
            if ships <= planet.ships:
                moves.append([planet.id, angle, ships])
                bad.add(best.id)   # claim it so another planet won't also pick it

    return moves

# ===== notebook cell 9 =====

def most_production(obs):
    """Prefer the highest-production targets. Skips comets and planets already
    being intercepted, and won't double-commit two planets to one target."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not targets:
        return moves

    bad = bad_target_ids(obs, comet_ids)
    # Highest production first.
    ranked = sorted(targets, key=lambda p: p.production, reverse=True)

    for planet in mine:
        for p in ranked:
            if p.id in bad:
                continue
            ships, angle, _ = how_many_send(planet, p, 1, obs)
            if ships < planet.ships:
                moves.append([planet.id, angle, max(ships, planet.ships // 2)])
                bad.add(p.id)
                break

    return moves

# ===== notebook cell 10 =====

def fastest_planet(obs):
    """Like nearest_planet_smart, but ranks targets by INTERCEPTION TIME
    (timesteps to arrival) rather than euclidean distance. Skips comets,
    planets already targeted by in-flight fleets, and blocked paths."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not targets:
        return moves

    bad = bad_target_ids(obs, comet_ids)

    for planet in mine:
        best_t = 1e9
        choice = None
        for p in targets:
            if p.id in bad:
                continue
            ships, angle, t = how_many_send(planet, p, 1, obs)
            if ships <= planet.ships and t is not None and t < best_t:
                best_t = t
                choice = (p, ships, angle)

        if choice:
            p, ships, angle = choice
            moves.append([planet.id, angle, ships])
            bad.add(p.id)

    return moves

# ===== notebook cell 11 =====

def smart_check_fleets(obs):
    """Two-pass policy:
      1) Expansion — every planet tries to capture the nearest new target it
         can afford (this is what actually grows your empire).
      2) Balancing — leftover ships are nudged so garrisons trend toward a
         share proportional to each planet's production (over-full planets
         send their surplus to the most under-full one).
    Skips comets and planets already targeted by an in-flight fleet."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    bad = bad_target_ids(obs, comet_ids)

    # track ships spent this turn so pass 2 sees realistic remaining garrisons
    spent = {p.id: 0 for p in mine}

    # ---- Pass 1: expansion (priority) ----
    if targets:
        for planet in mine:
            avail = planet.ships - spent[planet.id]
            best = None
            best_d = 1e9
            for p in targets:
                if p.id in bad or p.ships >= avail:
                    continue
                d = sq_dist(p, planet)
                if d < best_d:
                    best, best_d = p, d

            if best is not None:
                ships, angle, _ = how_many_send(planet, best, 1, obs)
                if ships <= avail:
                    moves.append([planet.id, angle, ships])
                    spent[planet.id] += ships
                    bad.add(best.id)

    # ---- Pass 2: balance remaining ships by production share ----
    if len(mine) >= 2:
        remaining = {p.id: p.ships - spent[p.id] for p in mine}
        total_ships = sum(remaining.values())
        total_prod = sum(p.production for p in mine)
        if total_prod > 0:
            by_id = {p.id: p for p in mine}
            target_reserve = {p.id: total_ships * p.production / total_prod for p in mine}

            # most under-supplied planet (largest deficit) is the sink
            sink = max(mine, key=lambda p: target_reserve[p.id] - remaining[p.id])
            deficit = target_reserve[sink.id] - remaining[sink.id]
            if deficit > 1:
                for planet in mine:
                    if planet.id == sink.id:
                        continue
                    surplus = remaining[planet.id] - target_reserve[planet.id]
                    send = int(min(surplus, deficit))
                    if send < 20:
                        continue
                    angle = math.atan2(sink.y - planet.y, sink.x - planet.x)
                    moves.append([planet.id, angle, send])
                    remaining[planet.id] -= send
                    deficit -= send
                    if deficit <= 1:
                        break

    return moves

# ===== notebook cell 12 =====

def defender(obs):
    """Defend first, then expand.
      1) For each owned planet with negative incoming balance (will be lost),
         pull reinforcements from the nearest owned planets that can ARRIVE
         BEFORE the enemy fleet, until the deficit is covered.
      2) With whatever ships remain, capture the nearest new targets."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    balance = planet_balance(obs)
    spent = {p.id: 0 for p in mine}

    # ---- Pass 1: defense ----
    threatened = []
    for p in mine:
        net, etick, esh = balance.get(p.id, (0, 1e9, 0))
        garrison_net = p.ships + net          # here + incoming friendly - incoming enemy
        if garrison_net < 0:
            threatened.append((p, etick, -garrison_net))
    threatened.sort(key=lambda x: x[1])       # soonest to fall first

    for tp, etick, deficit in threatened:
        need = deficit + 1
        helpers = sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp))
        for h in helpers:
            if need <= 0:
                break
            avail = h.ships - spent[h.id]
            if avail <= 0:
                continue
            send = int(min(avail, need))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star <= etick:               # arrives before the enemy hits
                moves.append([h.id, angle, send])
                spent[h.id] += send
                need -= send

    # ---- Pass 2: expansion ----
    if targets:
        bad = bad_target_ids(obs, comet_ids)
        for planet in mine:
            avail = planet.ships - spent[planet.id]
            if avail <= 0:
                continue
            best, best_d = None, 1e9
            for p in targets:
                if p.id in bad or p.ships >= avail:
                    continue
                d = sq_dist(p, planet)
                if d < best_d:
                    best, best_d = p, d
            if best is not None:
                ships, angle, _ = how_many_send(planet, best, 1, obs)
                if ships <= avail:
                    moves.append([planet.id, angle, ships])
                    spent[planet.id] += ships
                    bad.add(best.id)

    return moves

# ===== notebook cell 13 =====

def comet_user(obs):
    """nearest_planet_smart, plus one comet rule: any comet we own that is about
    to leave the board dumps its whole garrison outward, so those ships aren't
    lost when the comet expires. (Capturing comets was tested and consistently
    lost ships for no gain, so this deliberately does NOT chase comets.)"""
    EVAC_MARGIN = 3

    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    # comet id -> (path, current index along it)
    comet_path = {}
    for g in get_field(obs, "comets", []):
        for i, pid in enumerate(g["planet_ids"]):
            comet_path[pid] = (g["paths"][i], g["path_index"])

    spent = {p.id: 0 for p in mine}

    # Evacuate any owned comet within EVAC_MARGIN steps of the end of its path.
    for planet in mine:
        if planet.id in comet_path:
            path, idx = comet_path[planet.id]
            if 0 <= idx and idx >= len(path) - EVAC_MARGIN and planet.ships > 0:
                angle = math.atan2(planet.y - CENTER, planet.x - CENTER)  # fire outward
                moves.append([planet.id, angle, planet.ships])
                spent[planet.id] = planet.ships

    # Identical to nearest_planet_smart from here (comets stay excluded as targets).
    bad = bad_target_ids(obs, comet_ids)
    for planet in mine:
        avail = planet.ships - spent[planet.id]
        if avail <= 0:
            continue
        best, best_d = None, 1e9
        for p in targets:
            if p.id in bad or p.ships >= avail:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d
        if best is not None:
            ships, angle, _ = how_many_send(planet, best, 1, obs)
            if ships <= avail:
                moves.append([planet.id, angle, ships])
                spent[planet.id] += ships
                bad.add(best.id)

    return moves

# ===== notebook cell 14 =====

def net_attacker(obs):
    """nearest-target attacker that accounts for fleets already in transit.

    For each candidate target the effective garrison is
        eff = garrison + (enemy ships inbound) - (my ships inbound) - (ships I commit this turn)
    and the fleet is sized to beat `eff` instead of the raw garrison. So if I
    already have 30 inbound and the enemy has 40 inbound to the same planet,
    eff rises by 10 and I top up by ~10 rather than ignoring it or over-sending.

    Requires predict_all_fleet_hits(obs) to be defined in an earlier cell."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    # tally inbound ships per planet, split by side
    inc_mine, inc_enemy = {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships

    committed = {}  # ships I queue THIS turn, so planets don't double up

    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - committed.get(p.id, 0))

    for planet in mine:
        best, best_d = None, 1e9
        for p in targets:
            if p.id in comet_ids:
                continue
            eff = eff_garrison(p)
            if eff <= 0:                 # our inbound fleets already cover it
                continue
            if eff >= planet.ships:      # can't afford it from this planet
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d

        if best is None:
            continue

        eff = eff_garrison(best)
        if eff <= 0:
            continue
        adj = best._replace(ships=int(eff))          # size against effective garrison
        ships, angle, _ = how_many_send(planet, adj, 1, obs)
        if ships <= planet.ships:
            moves.append([planet.id, angle, ships])
            committed[best.id] = committed.get(best.id, 0) + ships

    return moves

# ===== notebook cell 15 =====

def net_attacker_v2(obs):
    """net_attacker, but invests in high-value captures: if the target produces
    MORE than the source planet, send all-but-one ship (and recompute the aim
    angle for that larger, faster fleet). Otherwise send just enough.

    Requires predict_all_fleet_hits(obs) and reach(src, dst, ships, obs)."""
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    inc_mine, inc_enemy = {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships

    committed = {}

    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - committed.get(p.id, 0))

    for planet in mine:
        best, best_d = None, 1e9
        for p in targets:
            if p.id in comet_ids:
                continue
            eff = eff_garrison(p)
            if eff <= 0 or eff >= planet.ships:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d

        if best is None:
            continue

        eff = eff_garrison(best)
        if eff <= 0:
            continue
        adj = best._replace(ships=int(eff))
        need, angle, _ = how_many_send(planet, adj, 1, obs)   # minimum + correct angle
        if need > planet.ships:
            continue

        if best.production > planet.production:
            send = planet.ships - 1                 # invest everything but one
            if send < need:                         # not enough to actually capture
                send = need
            if send != need:                        # recompute aim for the bigger fleet
                sol = reach(planet, best, send, obs)
                if sol is not None:
                    _, angle = sol
        else:
            send = need

        if 0 < send <= planet.ships:
            moves.append([planet.id, angle, send])
            committed[best.id] = committed.get(best.id, 0) + send

    return moves

# ===== notebook cell 16 =====

def net_attacker_defend(obs):
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    # One prediction pass: inbound ships per planet + earliest enemy arrival
    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing = {}      # ships LEAVING a planet this turn (reduces what it can spend)
    reinforce_in = {}  # friendly ships ARRIVING at a planet this turn (helps it survive)
    attack_committed = {}  # ships committed to an enemy target this turn

    # Enemy-target garrison: subtract my inbound (they help capture it)
    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))

    # Friendly net: my inbound HELPS, enemy inbound HURTS (correct signs)
    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))

    # Ships a planet must keep so it stays net-positive after sending
    def reserve(src):
        shortfall = max(0, inc_enemy.get(src.id, 0)
                        - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0))
        return shortfall + 1

    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)

    # ---- Pass 1: defense — only fires when ENEMY ships actually threaten a planet ----
    threatened = sorted(
        (p for p in mine if friendly_net(p) < 0),
        key=lambda p: etick.get(p.id, 1e9)        # soonest to fall first
    )
    for tp in threatened:
        deficit = -friendly_net(tp) + 1
        helpers = sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp))
        for h in helpers:
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):    # too late to matter
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: attack — nearest enemy target we can afford ----
    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        best, best_d = None, 1e9
        for p in targets:
            if p.id in comet_ids:
                continue
            eff = eff_garrison(p)
            if eff <= 0 or eff >= a:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d

        if best is None:
            continue

        eff = eff_garrison(best)
        if eff <= 0:
            continue
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send(planet, adj, 1, obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    return moves

# ===== notebook cell 17 =====

def net_attacker_defend_no_cap(obs):
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    # One prediction pass: inbound ships per planet + earliest enemy arrival
    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing = {}      # ships LEAVING a planet this turn (reduces what it can spend)
    reinforce_in = {}  # friendly ships ARRIVING at a planet this turn (helps it survive)
    attack_committed = {}  # ships committed to an enemy target this turn

    # Enemy-target garrison: subtract my inbound (they help capture it)
    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))

    # Friendly net: my inbound HELPS, enemy inbound HURTS (correct signs)
    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))

    # Ships a planet must keep so it stays net-positive after sending
    def reserve(src):
        shortfall = max(0, inc_enemy.get(src.id, 0)
                        - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0))
        return shortfall + 1

    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)

    # ---- Pass 1: defense — only fires when ENEMY ships actually threaten a planet ----
    threatened = sorted(
        (p for p in mine if friendly_net(p) < 0),
        key=lambda p: etick.get(p.id, 1e9)        # soonest to fall first
    )
    for tp in threatened:
        deficit = -friendly_net(tp) + 1
        helpers = sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp))
        for h in helpers:
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):    # too late to matter
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: attack — nearest enemy target we can afford ----
    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        best, best_d = None, 1e9
        for p in targets:
            if p.id in comet_ids:
                continue
            eff = eff_garrison(p)
            if eff <= 0 or eff >= a:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d

        if best is None:
            continue

        eff = eff_garrison(best)
        if eff <= 0:
            continue
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send_no_cap(planet, adj, 1, obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    return moves

# ===== notebook cell 18 =====

def net_attacker_defend_coord(obs):
    moves = []
    player, mine, targets, comet_ids, ang_vel = parse_obs(obs)
    if not mine:
        return moves

    planets = [Planet(*p) for p in get_field(obs, "planets", [])]

    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing, reinforce_in, attack_committed = {}, {}, {}

    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))

    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))

    def reserve(src):
        shortfall = max(0, inc_enemy.get(src.id, 0)
                        - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0))
        return shortfall + 1

    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)

    # ---- Pass 1: defense (identical to no_cap) ----
    threatened = sorted((p for p in mine if friendly_net(p) < 0),
                        key=lambda p: etick.get(p.id, 1e9))
    for tp in threatened:
        deficit = -friendly_net(tp) + 1
        helpers = sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp))
        for h in helpers:
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: solo attack on nearest affordable target (identical to no_cap) ----
    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        best, best_d = None, 1e9
        for p in targets:
            if p.id in comet_ids:
                continue
            eff = eff_garrison(p)
            if eff <= 0 or eff >= a:
                continue
            d = sq_dist(p, planet)
            if d < best_d:
                best, best_d = p, d
        if best is None:
            continue
        eff = eff_garrison(best)
        if eff <= 0:
            continue
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send_no_cap(planet, adj, 1, obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    # ---------- obstacle-aware path check (the key fix vs naive reach()) ----------
    def swept(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
        d0x, d0y = ax - p0x, ay - p0y
        dvx = (bx - ax) - (p1x - p0x); dvy = (by - ay) - (p1y - p0y)
        a = dvx*dvx + dvy*dvy; b = 2.0*(d0x*dvx + d0y*dvy); c = d0x*d0x + d0y*d0y - r*r
        if a < 1e-12:
            return c <= 0.0
        disc = b*b - 4.0*a*c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        return (-b + sq)/(2.0*a) >= 0.0 and (-b - sq)/(2.0*a) <= 1.0

    def clear_reach(src, dst, ships):
        """(t_star, angle) if a fleet of `ships` reaches dst on a path clear of the
        sun AND every other planet; else None."""
        sol = reach(src, dst, ships, obs)
        if sol is None:
            return None
        t_star, angle = sol
        speed = fleet_speed(ships)
        dx, dy = math.cos(angle)*speed, math.sin(angle)*speed
        fx = src.x + math.cos(angle)*(src.radius + 0.1)
        fy = src.y + math.sin(angle)*(src.radius + 0.1)
        last = min(int(t_star), 100)
        others = [p for p in planets if p.id != dst.id]
        for t in range(1, last + 1):
            nfx, nfy = fx + dx, fy + dy
            for p in others:
                if is_orbiting(p):
                    a0, a1 = (t-1)*ang_vel, t*ang_vel
                    qx0 = CENTER + math.cos(a0)*(p.x-CENTER) - math.sin(a0)*(p.y-CENTER)
                    qy0 = CENTER + math.sin(a0)*(p.x-CENTER) + math.cos(a0)*(p.y-CENTER)
                    qx1 = CENTER + math.cos(a1)*(p.x-CENTER) - math.sin(a1)*(p.y-CENTER)
                    qy1 = CENTER + math.sin(a1)*(p.x-CENTER) + math.cos(a1)*(p.y-CENTER)
                else:
                    qx0 = qx1 = p.x; qy0 = qy1 = p.y
                if swept(fx, fy, nfx, nfy, qx0, qy0, qx1, qy1, p.radius):
                    return None
            fx, fy = nfx, nfy
        return t_star, angle

    # ---- Pass 3: coordinated capture of leftover targets ----
    # "extra each planet has to give" = avail() (net-safe surplus). No planet
    # contributes more than that, so it never leaves itself under-defended.
    pool = {p.id: max(0, avail(p)) for p in mine}
    by_id = {p.id: p for p in mine}

    coord_targets = sorted(
        (p for p in targets
         if p.id not in comet_ids
         and attack_committed.get(p.id, 0) == 0
         and inc_mine.get(p.id, 0) == 0),
        key=lambda p: p.production, reverse=True)

    for tgt in coord_targets:
        need_base = eff_garrison(tgt)
        if need_base <= 0:
            continue
        contribs = []        # [pid, ships, angle, t]
        landed = 0; max_t = 0.0
        for h in sorted(mine, key=lambda h: sq_dist(h, tgt)):
            a = pool[h.id]
            if a <= 0:
                continue
            sol = clear_reach(h, tgt, a)      # obstacle-aware
            if sol is None:
                continue
            t_star, angle = sol
            contribs.append([h.id, a, angle, t_star])
            landed += a; max_t = max(max_t, t_star)
            # enemy planets regrow production/tick; neutral never do -> buffer only for enemy
            prod_buf = tgt.production * max_t if tgt.owner != -1 else 0.0
            if landed >= need_base + prod_buf + 1:
                break
        prod_buf = tgt.production * max_t if tgt.owner != -1 else 0.0
        required = need_base + prod_buf + 1
        if landed < required:
            continue
        overshoot = landed - required
        if overshoot > 0:
            last = contribs[-1]
            new_ships = int(last[1] - overshoot)
            if new_ships <= 0:
                contribs.pop()
            else:
                resol = clear_reach(by_id[last[0]], tgt, new_ships)
                if resol is not None:
                    last[1] = new_ships; last[2] = resol[1]
        for pid, ships, angle, _t in contribs:
            ships = int(ships)
            if ships <= 0:
                continue
            moves.append([pid, angle, ships])
            outgoing[pid] = outgoing.get(pid, 0) + ships
            pool[pid] -= ships
            attack_committed[tgt.id] = attack_committed.get(tgt.id, 0) + ships

    return moves

# ===== notebook cell 19 =====

def net_attacker_defend_geo(obs):
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing, reinforce_in, attack_committed = {}, {}, {}

    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))

    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))

    def reserve(src):
        shortfall = max(0, inc_enemy.get(src.id, 0)
                        - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0))
        return shortfall + 1

    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)

    # ---- Pass 1: defense (identical to no_cap) ----
    threatened = sorted((p for p in mine if friendly_net(p) < 0),
                        key=lambda p: etick.get(p.id, 1e9))
    for tp in threatened:
        deficit = -friendly_net(tp) + 1
        helpers = sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp))
        for h in helpers:
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: attack, GEOMETRY-RANKED instead of nearest-cheapest ----
    # tier 0 = orbiting (central) planets, nearest first
    # tier 1 = edge planets, largest production first, then nearest
    def target_key(planet, p, lean_orbit):
            d = math.sqrt(sq_dist(planet, p))
            bonus = 0
            if is_orbiting(p) and lean_orbit:
                bonus = 8.0
            elif not is_orbiting(p) and not lean_orbit:
                bonus = 8.0
            return -(p.production + bonus - 0.5*d)    # higher = better, so negate
    
    orbits_prod = 0
    other_prod = 0
    for t in [mine, targets]:
        for p in t:
            if is_orbiting(p):
                orbits_prod += p.production
            else:
                other_prod += p.production

    orbit_better = False
    if orbits_prod > other_prod:
        orbit_better = True

    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        affordable = [p for p in targets
                      if p.id not in comet_ids and 0 < eff_garrison(p) < a]
        if not affordable:
            continue
        affordable.sort(key=lambda p: target_key(planet, p, orbit_better))
        best = affordable[0]

        eff = eff_garrison(best)
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send_no_cap(planet, adj, 1, obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    return moves

# ===== notebook cell 20 =====

def net_attacker_defend_roi(obs):
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing, reinforce_in, attack_committed = {}, {}, {}
    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))
    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))
    def reserve(src):
        return max(0, inc_enemy.get(src.id, 0)
                   - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0)) + 1
    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)

    # ---- Pass 1: defense (identical to no_cap) ----
    for tp in sorted((p for p in mine if friendly_net(p) < 0), key=lambda p: etick.get(p.id, 1e9)):
        deficit = -friendly_net(tp) + 1
        for h in sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp)):
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: attack, ranked by ROI (production per ship), not distance ----
    def value(planet, p):
        d = math.sqrt(sq_dist(planet, p))
        return p.production / (eff_garrison(p) + 1) - 0.02 * d   # higher = better

    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        affordable = [p for p in targets
                      if p.id not in comet_ids and 0 < eff_garrison(p) < a]
        if not affordable:
            continue
        best = max(affordable, key=lambda p: value(planet, p))
        eff = eff_garrison(best)
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send_no_cap(planet, adj, 1, obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    return moves

# ===== notebook cell 21 =====

def net_roi_support(obs):
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing, reinforce_in, attack_committed = {}, {}, {}
    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))
    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))
    def reserve(src):
        return max(0, inc_enemy.get(src.id, 0)
                   - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0)) + 1
    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)
    def hold_buffer(p):
        # how hard will this be to hold? scale with value and enemy proximity
        enemies = [e for e in targets if e.owner not in (-1, player)]
        if not enemies:
            return 1
        nearest = math.sqrt(min(sq_dist(p, e) for e in enemies))
        threat = sum(e.production for e in enemies if sq_dist(p, e) < 30**2)
        return 1 + int(threat * (1.0 if nearest < 25 else 0.3))

    # ---- Pass 1: defense (identical to no_cap) ----
    for tp in sorted((p for p in mine if friendly_net(p) < 0), key=lambda p: etick.get(p.id, 1e9)):
        deficit = -friendly_net(tp) + 1
        for h in sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp)):
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: attack, ranked by ROI (production per ship), not distance ----
    def value(planet, p):
        d = math.sqrt(sq_dist(planet, p))
        return p.production / (eff_garrison(p) + 1) - 0.02 * d   # higher = better

    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        affordable = [p for p in targets
                      if p.id not in comet_ids and 0 < eff_garrison(p) < a]
        if not affordable:
            continue
        best = max(affordable, key=lambda p: value(planet, p))
        eff = eff_garrison(best)
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send_improved(planet, adj, hold_buffer(planet), obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    # ---- Pass 3: rear-to-front reinforcement (consolidate idle hoards) ----
    FRONT_RANGE   = 35.0   # a planet within this of any enemy is "front"
    MIN_HOARD     = 40     # only donate if a rear planet is sitting on this many spare
    LOCAL_RESERVE = 20     # ships a donor keeps behind regardless

    enemies = [p for p in targets if p.owner not in (-1, player)]
    if enemies:
        def nearest_enemy_dist(m):
            return math.sqrt(min(sq_dist(m, e) for e in enemies))

        front  = [m for m in mine if nearest_enemy_dist(m) <= FRONT_RANGE]
        donors = [m for m in mine if nearest_enemy_dist(m) > FRONT_RANGE
                  and inc_enemy.get(m.id, 0) == 0]              # safe: no enemy near, none inbound

        if front:
            for d in donors:
                a = avail(d)
                if a < MIN_HOARD:                # not a real hoard -> leave it for local use
                    continue
                f = min(front, key=lambda f: sq_dist(d, f))     # ferry toward nearest front planet
                D = math.sqrt(sq_dist(d, f))
                # speed threshold: a batch this big travels >=~2x a single ship (see math)
                min_batch = max(11, int(0.35 * D))
                send = int(a - LOCAL_RESERVE)
                if send < min_batch:             # below threshold -> hold & accumulate
                    continue
                sol = reach(d, f, send, obs)
                if sol is None:                  # path blocked by sun
                    continue
                _t, angle = sol
                moves.append([d.id, angle, send])
                outgoing[d.id] = outgoing.get(d.id, 0) + send

    return moves

# ===== notebook cell 22 =====

def net_roi_opt(obs):
    moves = []
    player, mine, targets, comet_ids, _ = parse_obs(obs)
    if not mine:
        return moves

    inc_mine, inc_enemy, etick = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine[pid] = inc_mine.get(pid, 0) + ships
        else:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships
            etick[pid] = min(etick.get(pid, 1e9), tick)

    outgoing, reinforce_in, attack_committed = {}, {}, {}
    def eff_garrison(p):
        return (p.ships + inc_enemy.get(p.id, 0)
                - inc_mine.get(p.id, 0) - attack_committed.get(p.id, 0))
    def friendly_net(p):
        return (p.ships + inc_mine.get(p.id, 0) + reinforce_in.get(p.id, 0)
                - inc_enemy.get(p.id, 0))
    def reserve(src):
        return max(0, inc_enemy.get(src.id, 0)
                   - inc_mine.get(src.id, 0) - reinforce_in.get(src.id, 0)) + 1
    def avail(src):
        return src.ships - outgoing.get(src.id, 0) - reserve(src)
    def hold_buffer(p):
        # how hard will this be to hold? scale with value and enemy proximity
        enemies = [e for e in targets if e.owner not in (-1, player)]
        if not enemies:
            return 1
        nearest = math.sqrt(min(sq_dist(p, e) for e in enemies))
        threat = sum(e.production for e in enemies if sq_dist(p, e) < 30**2)
        return 1 + int(threat * (1.0 if nearest < 25 else 0.3))

    # ---- Pass 1: defense (identical to no_cap) ----
    for tp in sorted((p for p in mine if friendly_net(p) < 0), key=lambda p: etick.get(p.id, 1e9)):
        deficit = -friendly_net(tp) + 1
        for h in sorted((h for h in mine if h.id != tp.id), key=lambda h: sq_dist(h, tp)):
            if deficit <= 0:
                break
            a = avail(h)
            if a <= 0:
                continue
            send = int(min(a, deficit))
            sol = reach(h, tp, send, obs)
            if sol is None:
                continue
            t_star, angle = sol
            if t_star > etick.get(tp.id, 1e9):
                continue
            moves.append([h.id, angle, send])
            outgoing[h.id] = outgoing.get(h.id, 0) + send
            reinforce_in[tp.id] = reinforce_in.get(tp.id, 0) + send
            deficit -= send

    # ---- Pass 2: attack, ranked by ROI (production per ship), not distance ----
    def value(planet, p):
        d = math.sqrt(sq_dist(planet, p))
        return p.production / (eff_garrison(p) + 1) - 0.02 * d   # higher = better

    for planet in mine:
        a = avail(planet)
        if a <= 0:
            continue
        affordable = [p for p in targets
                      if p.id not in comet_ids and 0 < eff_garrison(p) < a]
        if not affordable:
            continue
        best = max(affordable, key=lambda p: value(planet, p))
        eff = eff_garrison(best)
        adj = best._replace(ships=int(eff))
        ships, angle, _ = how_many_send_improved(planet, adj, hold_buffer(planet), obs)
        if ships <= a:
            moves.append([planet.id, angle, ships])
            attack_committed[best.id] = attack_committed.get(best.id, 0) + ships
            outgoing[planet.id] = outgoing.get(planet.id, 0) + ships

    return moves