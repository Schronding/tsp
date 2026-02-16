"""
TSP Route Animation on a Map of Mexico
=======================================
Visualizes the brute-force TSP solution for 6 Mexican cities.

The animation has three phases:
  1. Rapidly cycles through all 120 possible routes (search phase)
  2. Highlights the best route found
  3. Animates a truck marker traveling along the optimal path

Dependencies: pip install matplotlib numpy pillow
Run:          python tsp_animation.py
Output:       tsp_mexico.gif  (saved in same folder)
"""

import sys
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install matplotlib numpy pillow")
    sys.exit(1)

from itertools import permutations

# ──────────────────────────────────────────────
#  Data (same as your notebook)
# ──────────────────────────────────────────────

city_names = ["GDL", "MLA", "MID", "PUE", "MTY", "SLP"]
city_full  = {
    "GDL": "Guadalajara",
    "MLA": "Morelia",
    "MID": "Mérida",
    "PUE": "Puebla",
    "MTY": "Monterrey",
    "SLP": "San Luis Potosí",
}

# Geographic coordinates (longitude, latitude) for plotting
coords = {
    "GDL": (-103.35, 20.66),
    "MLA": (-101.19, 19.71),
    "MID": ( -89.59, 20.97),
    "PUE": ( -98.21, 19.04),
    "MTY": (-100.32, 25.69),
    "SLP": (-100.99, 22.16),
}

# Adjacency matrix – hours of transit for a cargo truck (from your report)
#             GDL   MLA    MID   PUE   MTY   SLP
adj_matrix = [
    [ 0.0,  4.5, 34.5, 11.5, 14.0,  6.0],  # GDL
    [ 4.5,  0.0, 30.0,  7.0, 14.0,  6.0],  # MLA
    [34.5, 30.0,  0.0, 23.0, 40.0, 29.0],  # MID
    [11.5,  7.0, 23.0,  0.0, 17.5,  8.5],  # PUE
    [14.0, 14.0, 40.0, 17.5,  0.0,  8.0],  # MTY
    [ 6.0,  6.0, 29.0,  8.5,  8.0,  0.0],  # SLP
]

idx = {name: i for i, name in enumerate(city_names)}

def route_weight(route):
    """Total weight (hours) of a round-trip route starting and ending at route[0]."""
    total = 0.0
    for k in range(len(route) - 1):
        total += adj_matrix[idx[route[k]]][idx[route[k + 1]]]
    # Return to start
    total += adj_matrix[idx[route[-1]]][idx[route[0]]]
    return total

# ──────────────────────────────────────────────
#  Generate all 120 routes from SLP
# ──────────────────────────────────────────────

start = "SLP"
others = [c for c in city_names if c != start]
all_routes = []
all_weights = []

for perm in permutations(others):
    route = [start] + list(perm)
    w = route_weight(route)
    all_routes.append(route)
    all_weights.append(w)

best_idx = int(np.argmin(all_weights))
worst_idx = int(np.argmax(all_weights))
best_route = all_routes[best_idx]
best_weight = all_weights[best_idx]
worst_route = all_routes[worst_idx]
worst_weight = all_weights[worst_idx]

print(f"Total routes: {len(all_routes)}")
print(f"Best:  {' → '.join(best_route)} → {start}  ({best_weight}h)")
print(f"Worst: {' → '.join(worst_route)} → {start}  ({worst_weight}h)")

# ──────────────────────────────────────────────
#  Simplified Mexico border outline (lon, lat)
# ──────────────────────────────────────────────

mexico_border = np.array([
    # Pacific NW coast → Northern border
    (-117.1, 32.5), (-115.0, 32.1), (-112.0, 31.3), (-111.0, 31.3),
    (-109.0, 31.3), (-108.0, 31.3), (-106.5, 31.8), (-105.0, 30.7),
    (-104.0, 29.5), (-103.3, 29.0), (-101.4, 29.8), (-100.5, 28.7),
    (-99.7,  27.5), (-99.1,  26.4), (-97.8,  25.9),
    # Gulf coast going south
    (-97.5,  24.5), (-97.8,  22.5), (-97.2,  21.5), (-96.5,  19.8),
    (-96.0,  19.0), (-95.0,  18.5), (-94.5,  18.2), (-93.5,  18.4),
    (-92.5,  18.6), (-91.5,  18.5),
    # Yucatán peninsula
    (-90.5,  19.5), (-90.3,  20.0), (-90.0,  21.0), (-89.5,  21.4),
    (-88.0,  21.5), (-87.4,  21.2), (-87.1,  20.3), (-87.5,  19.5),
    (-87.8,  18.5), (-88.3,  18.3), (-89.0,  18.0), (-89.5,  17.8),
    (-90.5,  17.8), (-91.0,  17.3), (-91.5,  17.0),
    # Southern border → Pacific coast
    (-92.0,  15.5), (-92.5,  15.0), (-93.5,  15.7), (-94.5,  16.2),
    (-96.0,  15.8), (-97.5,  16.0), (-99.0,  16.4), (-100.0, 17.0),
    (-101.5, 17.8), (-102.5, 18.0), (-103.5, 18.3), (-104.5, 19.0),
    (-105.0, 19.5), (-105.5, 20.5), (-105.3, 21.5), (-105.8, 22.0),
    (-106.5, 23.0), (-108.0, 24.0), (-109.5, 23.0), (-110.0, 23.5),
    (-110.5, 24.5), (-112.0, 29.0), (-114.5, 30.5), (-116.0, 31.5),
    (-117.1, 32.5),
])

# ──────────────────────────────────────────────
#  Build the animation
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# Draw Mexico outline
ax.fill(mexico_border[:, 0], mexico_border[:, 1],
        color="#0f3460", edgecolor="#4a6fa5", linewidth=1.2, alpha=0.7)

# Draw city dots and labels
for name, (lon, lat) in coords.items():
    ax.plot(lon, lat, "o", color="#e94560", markersize=10, zorder=5,
            markeredgecolor="white", markeredgewidth=1.2)
    offset_x, offset_y = 0.4, 0.5
    if name == "MTY":
        offset_y = -0.8
    if name == "MID":
        offset_x = -1.5
    if name == "PUE":
        offset_y = -0.8
    ax.text(lon + offset_x, lat + offset_y,
            f"{name}\n({city_full[name]})",
            color="white", fontsize=7.5, fontweight="bold",
            ha="left", va="center", zorder=6)

# Light connection lines (all edges of the complete graph)
for i in range(len(city_names)):
    for j in range(i + 1, len(city_names)):
        c1 = coords[city_names[i]]
        c2 = coords[city_names[j]]
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                color="#4a6fa5", linewidth=0.3, alpha=0.25, zorder=1)

ax.set_xlim(-118, -86)
ax.set_ylim(14, 34)
ax.set_aspect("equal")
ax.axis("off")

# ── Animated elements ──
route_line, = ax.plot([], [], color="#ff6b6b", linewidth=2.5, alpha=0.8, zorder=3)
best_line,  = ax.plot([], [], color="#00ff88", linewidth=3.5, alpha=0.0, zorder=4)
truck = ax.plot([], [], "s", color="#ffd700", markersize=14,
                markeredgecolor="white", markeredgewidth=2, zorder=10)[0]

# Text overlays
title_text = ax.text(0.5, 0.97, "", transform=ax.transAxes,
                     color="white", fontsize=14, fontweight="bold",
                     ha="center", va="top", zorder=7)
info_text  = ax.text(0.02, 0.03, "", transform=ax.transAxes,
                     color="#aaaaaa", fontsize=9, ha="left", va="bottom",
                     zorder=7, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                               edgecolor="#4a6fa5", alpha=0.9))
phase_text = ax.text(0.98, 0.03, "", transform=ax.transAxes,
                     color="#ffd700", fontsize=10, fontweight="bold",
                     ha="right", va="bottom", zorder=7)

# ── Animation parameters ──
N_ROUTES       = len(all_routes)          # 120
SEARCH_FRAMES  = N_ROUTES                 # 1 frame per route
PAUSE_FRAMES   = 30                       # pause showing best
TRAVEL_STEPS   = 60                       # frames for truck to travel
TOTAL_FRAMES   = SEARCH_FRAMES + PAUSE_FRAMES + TRAVEL_STEPS + 30  # +30 hold

# Precompute the best route coordinates for the truck phase
best_coords = [coords[c] for c in best_route] + [coords[start]]  # close the loop
# Interpolate for smooth movement
interp_pts = []
pts_per_leg = TRAVEL_STEPS // len(best_coords)
if pts_per_leg < 2:
    pts_per_leg = 2
for k in range(len(best_coords) - 1):
    x0, y0 = best_coords[k]
    x1, y1 = best_coords[k + 1]
    for t in np.linspace(0, 1, pts_per_leg, endpoint=False):
        interp_pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
interp_pts.append(best_coords[-1])  # final point

# Track best-so-far during the search phase
running_best = float("inf")
running_best_route = None


def get_route_xy(route, close=True):
    """Return x, y arrays for a route (optionally closing the loop)."""
    xs = [coords[c][0] for c in route]
    ys = [coords[c][1] for c in route]
    if close:
        xs.append(coords[route[0]][0])
        ys.append(coords[route[0]][1])
    return xs, ys


def animate(frame):
    global running_best, running_best_route

    # ── PHASE 1: Search ──
    if frame < SEARCH_FRAMES:
        r = all_routes[frame]
        w = all_weights[frame]
        xs, ys = get_route_xy(r)
        route_line.set_data(xs, ys)
        route_line.set_alpha(0.6)

        if w < running_best:
            running_best = w
            running_best_route = r

        is_best_so_far = (w <= running_best)
        route_line.set_color("#00ff88" if is_best_so_far else "#ff6b6b")
        route_line.set_linewidth(3.0 if is_best_so_far else 1.8)

        title_text.set_text(f"Searching... route {frame + 1} / {N_ROUTES}")
        path_str = " → ".join(r) + f" → {r[0]}"
        info_text.set_text(
            f"Current:  {path_str}  │  {w:.1f}h\n"
            f"Best:     {' → '.join(running_best_route)} → {running_best_route[0]}  │  {running_best:.1f}h"
        )
        phase_text.set_text("PHASE 1: BRUTE FORCE SEARCH")
        truck.set_data([], [])
        best_line.set_alpha(0.0)

    # ── PHASE 2: Show best route ──
    elif frame < SEARCH_FRAMES + PAUSE_FRAMES:
        sub = frame - SEARCH_FRAMES
        route_line.set_alpha(0.0)

        bx, by = get_route_xy(best_route)
        best_line.set_data(bx, by)
        best_line.set_alpha(min(1.0, sub / 10.0))  # fade in
        best_line.set_color("#00ff88")
        best_line.set_linewidth(3.5)

        title_text.set_text("✓ OPTIMAL ROUTE FOUND")
        info_text.set_text(
            f"Best:   {' → '.join(best_route)} → {start}  │  {best_weight:.1f}h\n"
            f"Worst:  {' → '.join(worst_route)} → {start}  │  {worst_weight:.1f}h\n"
            f"Saved:  {worst_weight - best_weight:.1f} hours"
        )
        phase_text.set_text("PHASE 2: RESULT")
        truck.set_data([], [])

    # ── PHASE 3: Truck animation ──
    elif frame < SEARCH_FRAMES + PAUSE_FRAMES + TRAVEL_STEPS:
        sub = frame - SEARCH_FRAMES - PAUSE_FRAMES
        progress = min(sub, len(interp_pts) - 1)

        # Draw the best route
        bx, by = get_route_xy(best_route)
        best_line.set_data(bx, by)
        best_line.set_alpha(1.0)
        best_line.set_color("#00ff88")

        # Also draw the "traveled" portion more brightly
        traveled_x = [p[0] for p in interp_pts[:progress + 1]]
        traveled_y = [p[1] for p in interp_pts[:progress + 1]]
        route_line.set_data(traveled_x, traveled_y)
        route_line.set_color("#ffd700")
        route_line.set_linewidth(4.0)
        route_line.set_alpha(0.9)

        # Move the truck
        tx, ty = interp_pts[progress]
        truck.set_data([tx], [ty])

        # Figure out which city leg we're on
        leg = min(progress // max(pts_per_leg, 1), len(best_route) - 1)
        current_city = best_route[leg] if leg < len(best_route) else start

        title_text.set_text(f"Traveling the optimal route... [{current_city}]")
        info_text.set_text(
            f"Route: {' → '.join(best_route)} → {start}\n"
            f"Total: {best_weight:.1f} hours  │  "
            f"Savings vs worst: {worst_weight - best_weight:.1f}h"
        )
        phase_text.set_text("PHASE 3: ROUTE TRAVERSAL")

    # ── PHASE 4: Hold final frame ──
    else:
        bx, by = get_route_xy(best_route)
        best_line.set_data(bx, by)
        best_line.set_alpha(1.0)
        route_line.set_alpha(0.0)

        tx, ty = coords[start]
        truck.set_data([tx], [ty])

        title_text.set_text("Route complete — back at San Luis Potosí")
        info_text.set_text(
            f"Optimal: {' → '.join(best_route)} → {start}  │  {best_weight:.1f}h\n"
            f"Worst:   {' → '.join(worst_route)} → {start}  │  {worst_weight:.1f}h\n"
            f"Savings: {worst_weight - best_weight:.1f} hours"
        )
        phase_text.set_text("")

    return route_line, best_line, truck, title_text, info_text, phase_text


# ── Create and save ──
print("Rendering animation...")
anim = animation.FuncAnimation(
    fig, animate, frames=TOTAL_FRAMES, interval=80, blit=True
)

output_path = "tsp_mexico.gif"
anim.save(output_path, writer="pillow", fps=15)
print(f"Saved to {output_path}")
plt.close()
