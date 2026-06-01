import math
from collections import namedtuple
Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
 
 
def how_many_send(my_planet, enemy_planet, spare, obs):
    def planet_pos(t):
        if not orbits:
            return enemy_planet.x, enemy_planet.y
        a = t * ang_vel
        dx, dy = enemy_planet.x - 50, enemy_planet.y - 50
        px = 50 + math.cos(a) * dx - math.sin(a) * dy
        py = 50 + math.sin(a) * dx + math.cos(a) * dy
        return px, py
 
    def f(t):
        px, py = planet_pos(t)
        return math.hypot(px - my_planet.x, py - my_planet.y) - (my_planet.radius + 0.1) - t * speed
 
    l_fleet_size, r_fleet_size = enemy_planet.ships + spare, my_planet.ships
    sun_dist = math.hypot(enemy_planet.x - 50, enemy_planet.y - 50)
 
    ang_vel = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity
 
    orbits = False
    if sun_dist + enemy_planet.radius < 50:
        orbits = True
 
    ship_max_speed = 6
    best = 1e9
    angle = 0
 
    while l_fleet_size <= r_fleet_size:
        ships = (l_fleet_size + r_fleet_size) // 2
 
        speed = 1.0 + (ship_max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, ship_max_speed)
 
        found_angle = 0.0
        moves = -1
        feasible = False
 
        prev = f(0)
        for t in range(1, 101):
            cur = f(t)
            if cur <= 0 and prev > 0:
                lo, hi = t - 1, t
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if f(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                t_star = (lo + hi) / 2
 
                px, py = planet_pos(t_star)
 
                # closest point on the launch segment to the sun (not infinite line)
                sx, sy = px - my_planet.x, py - my_planet.y
                seg2 = sx * sx + sy * sy
                u = ((50 - my_planet.x) * sx + (50 - my_planet.y) * sy) / seg2
                u = max(0.0, min(1.0, u))
                cxs, cys = my_planet.x + u * sx, my_planet.y + u * sy
                if math.hypot(cxs - 50, cys - 50) <= 10.0:
                    prev = cur
                    continue
 
                found_angle = math.atan2(py - my_planet.y, px - my_planet.x)
                moves = t_star
                feasible = True
                break
            prev = cur
 
        if feasible:
            prod = enemy_planet.production * moves if enemy_planet.owner != -1 else 0
            if ships >= enemy_planet.ships + prod + spare:
                best = ships
                angle = found_angle
                r_fleet_size = ships - 1
            else:
                l_fleet_size = ships + 1
        else:
            l_fleet_size = ships + 1
 
    return best, angle
 
 
def nearest_planet(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planet = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    comets = obs.get("comet_planet_ids", []) if isinstance(obs, dict) else obs.comet_planet_ids
 
    planets = [Planet(*p) for p in raw_planet]
 
    mine = [p for p in planets if p.owner == player]
    target = [p for p in planets if p.owner != player]
 
    if len(target) == 0:
        return moves
 
    for planet in mine:
        nearest = -1
        dist = 1e9
 
        for p in target:
            is_comet = False
            for c in comets:
                if p.id == c:
                    is_comet = True
 
            if (p.x - planet.x) ** 2 + (p.y - planet.y) ** 2 < dist and p.ships < planet.ships and not is_comet:
                nearest = p.id
                dist = (p.x - planet.x) ** 2 + (p.y - planet.y) ** 2
 
        for p in target:
            if p.id == nearest:
                ships, angle = how_many_send(planet, p, 1, obs)
 
                if ships <= planet.ships:
                    moves.append([planet.id, angle, ships])
 
    return moves