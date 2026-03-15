"""
generate_graphs.py
==================
Génère les graphiques du rapport en parsant directement les JSONs de seahorse.

Usage
-----
    # Tous les JSONs d'un dossier
    python generate_graphs.py benchmarks/results/json/

    # Ou des fichiers spécifiques
    python generate_graphs.py __REC_123.json __REC_456.json

Produit
-------
    benchmark_graphs.png
"""

import sys
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

C_SHANNON  = "#2E86AB"
C_DIJKSTRA = "#E84855"
C_BG       = "#F8F9FA"
C_GRID     = "#DEE2E6"
C_TEXT     = "#212529"

if len(sys.argv) < 2:
    print("Usage: python generate_graphs.py <dossier_json/> [ou fichiers .json ...]")
    sys.exit(1)

json_files = []
for arg in sys.argv[1:]:
    if os.path.isdir(arg):
        json_files += sorted(glob.glob(os.path.join(arg, "*.json")))
    elif os.path.isfile(arg):
        json_files.append(arg)

if not json_files:
    print("Aucun fichier JSON trouvé.")
    sys.exit(1)

print(f"Parsing {len(json_files)} fichier(s) JSON...")

SHANNON_KEYWORDS  = ["Electric", "Smarter", "Cachalot", "electric", "SmarterAgent"]
DIJKSTRA_KEYWORDS = ["Dijkstra", "dijkstra", "minimax", "Minimax", "DijkstraAgent"]

def identify_agents(players):
    shannon_p, dijkstra_p = None, None
    for p in players:
        name = p.get("name", "")
        if any(k in name for k in SHANNON_KEYWORDS):
            shannon_p = p
        elif any(k in name for k in DIJKSTRA_KEYWORDS):
            dijkstra_p = p
    if shannon_p is None:
        shannon_p = players[0]
    if dijkstra_p is None:
        dijkstra_p = players[1] if len(players) > 1 else players[0]
    return shannon_p, dijkstra_p

def parse_json(path):
    with open(path) as f:
        d = json.load(f)
    steps   = d["steps"]
    summary = d["final_summary"]
    winners = [str(w) for w in summary["winners_id"]]
    shannon_p, dijkstra_p = identify_agents(steps[0]["players"])
    s_id = str(shannon_p["id"])
    d_id = str(dijkstra_p["id"])
    winner = "shannon" if s_id in winners else ("dijkstra" if d_id in winners else "unknown")
    s_times, d_times = [], []
    for i in range(1, len(steps)):
        prev  = steps[i - 1]
        curr  = steps[i]
        actor = str(prev["active_player"]["id"])
        dt_s  = prev["remaining_time"][s_id] - curr["remaining_time"][s_id]
        dt_d  = prev["remaining_time"][d_id] - curr["remaining_time"][d_id]
        if actor == s_id:
            s_times.append(round(max(dt_s, 0), 4))
        else:
            d_times.append(round(max(dt_d, 0), 4))
    return {
        "file":              os.path.basename(path),
        "winner":            winner,
        "total_steps":       len(steps) - 1,
        "shannon_moves":     len(s_times),
        "dijkstra_moves":    len(d_times),
        "shannon_times":     s_times,
        "dijkstra_times":    d_times,
        "shannon_avg_time":  round(sum(s_times) / len(s_times), 4) if s_times else 0,
        "dijkstra_avg_time": round(sum(d_times) / len(d_times), 4) if d_times else 0,
        "shannon_end_time":  round(steps[-1]["remaining_time"][s_id], 2),
        "dijkstra_end_time": round(steps[-1]["remaining_time"][d_id], 2),
        "shannon_piece":     shannon_p.get("piece_type", "?"),
        "shannon_name":      shannon_p.get("name", "Shannon"),
        "dijkstra_name":     dijkstra_p.get("name", "Dijkstra"),
    }

rows = []
for path in json_files:
    try:
        r = parse_json(path)
        rows.append(r)
        print(f"  OK  {r['file']:45s} -> {r['winner']:8s} | {r['total_steps']} coups")
    except Exception as e:
        print(f"  ERR {path} -> {e}")

if not rows:
    print("Aucune partie parsee avec succes.")
    sys.exit(1)

N      = len(rows)
games  = list(range(1, N + 1))
s_wins = sum(1 for r in rows if r["winner"] == "shannon")
d_wins = sum(1 for r in rows if r["winner"] == "dijkstra")
print(f"\nResultats : Shannon {s_wins}V  Dijkstra {d_wins}V  ({N} parties)\n")

fig = plt.figure(figsize=(16, 18), facecolor=C_BG)
fig.suptitle(f"Shannon (Electric) vs Dijkstra - Comparaison sur {N} parties",
             fontsize=16, fontweight="bold", color=C_TEXT, y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                       top=0.94, bottom=0.05, left=0.08, right=0.97)

# 1. Donut
ax1 = fig.add_subplot(gs[0, 0])
wedge_vals = [s_wins, d_wins] if d_wins > 0 else [s_wins, 0.0001]
ax1.pie(wedge_vals, colors=[C_SHANNON, C_DIJKSTRA], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
ax1.text(0, 0, f"{s_wins}/{N}", ha="center", va="center",
         fontsize=22, fontweight="bold", color=C_TEXT)
ax1.legend(handles=[
    mpatches.Patch(color=C_SHANNON,  label=f"Shannon   {s_wins}V ({100*s_wins/N:.0f}%)"),
    mpatches.Patch(color=C_DIJKSTRA, label=f"Dijkstra  {d_wins}V ({100*d_wins/N:.0f}%)"),
], loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=10, framealpha=0.6)
ax1.set_title("Taux de victoire", fontsize=13, fontweight="bold", color=C_TEXT, pad=10)

# 2. Coups par partie
ax2 = fig.add_subplot(gs[0, 1])
bar_colors = [C_SHANNON if r["winner"] == "shannon" else C_DIJKSTRA for r in rows]
ax2.bar(games, [r["total_steps"] for r in rows],
        color=bar_colors, edgecolor="white", linewidth=1.2, width=0.6)
if s_wins:
    sm = np.mean([r["total_steps"] for r in rows if r["winner"] == "shannon"])
    ax2.axhline(sm, color=C_SHANNON, linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"Moy. Shannon ({sm:.0f})")
if d_wins:
    dm = np.mean([r["total_steps"] for r in rows if r["winner"] == "dijkstra"])
    ax2.axhline(dm, color=C_DIJKSTRA, linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"Moy. Dijkstra ({dm:.0f})")
ax2.set_xlabel("Partie", fontsize=10, color=C_TEXT)
ax2.set_ylabel("Nombre total de coups", fontsize=10, color=C_TEXT)
ax2.set_title("Duree des parties (nombre de coups)\nbleu = victoire Shannon  rouge = victoire Dijkstra",
              fontsize=11, fontweight="bold", color=C_TEXT)
ax2.set_facecolor(C_BG); ax2.grid(True, axis="y", color=C_GRID, linewidth=0.7)
ax2.spines[["top","right"]].set_visible(False); ax2.set_xticks(games)
if s_wins or d_wins: ax2.legend(fontsize=8.5, framealpha=0.6)

# 3. Temps moyen par coup
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(N); width = 0.38
b1 = ax3.bar(x-width/2, [r["shannon_avg_time"]  for r in rows], width,
             color=C_SHANNON,  label="Shannon (electrique)", alpha=0.9, edgecolor="white")
b2 = ax3.bar(x+width/2, [r["dijkstra_avg_time"] for r in rows], width,
             color=C_DIJKSTRA, label="Dijkstra",              alpha=0.9, edgecolor="white")
for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax3.text(bar.get_x()+bar.get_width()/2, h+0.1, f"{h:.1f}s",
             ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
for i, r in enumerate(rows):
    sym = "S gagne" if r["winner"]=="shannon" else "D gagne"
    col = C_SHANNON if r["winner"]=="shannon" else C_DIJKSTRA
    ax3.text(i, -1.5, sym, ha="center", fontsize=9, color=col, fontweight="bold")
ax3.set_xticks(x); ax3.set_xticklabels([f"P{g}" for g in games], fontsize=9)
ax3.set_ylabel("Temps moyen/coup (s)", fontsize=10, color=C_TEXT)
ax3.set_title("Temps de reflexion moyen par coup - Shannon vs Dijkstra",
              fontsize=12, fontweight="bold", color=C_TEXT)
ax3.set_facecolor(C_BG); ax3.grid(True, axis="y", color=C_GRID, linewidth=0.7)
ax3.spines[["top","right"]].set_visible(False)
ax3.legend(fontsize=10, framealpha=0.7); ax3.set_ylim(bottom=-2.5)

# 4. Violon distribution
ax4 = fig.add_subplot(gs[2, 0])
s_all = [t for r in rows for t in r["shannon_times"]  if t > 0.5]
d_all = [t for r in rows for t in r["dijkstra_times"] if t > 0.5]
parts = ax4.violinplot([s_all, d_all], positions=[1,2], showmedians=True, showextrema=True)
for pc, col in zip(parts["bodies"], [C_SHANNON, C_DIJKSTRA]):
    pc.set_facecolor(col); pc.set_alpha(0.7)
for key in ["cmedians","cbars","cmins","cmaxes"]:
    parts[key].set_color("white"); parts[key].set_linewidth(2)
for i, (di, col) in enumerate(zip([s_all, d_all], [C_SHANNON, C_DIJKSTRA]), start=1):
    med = float(np.median(di))
    ax4.text(i, med+0.3, f"med. {med:.1f}s", ha="center", fontsize=9, color=col, fontweight="bold")
ax4.set_xticks([1,2]); ax4.set_xticklabels(["Shannon\n(electrique)","Dijkstra"], fontsize=10)
ax4.set_ylabel("Temps par coup (s)", fontsize=10, color=C_TEXT)
ax4.set_title("Distribution des temps de reflexion\n(toutes parties)",
              fontsize=11, fontweight="bold", color=C_TEXT)
ax4.set_facecolor(C_BG); ax4.grid(True, axis="y", color=C_GRID, linewidth=0.7)
ax4.spines[["top","right"]].set_visible(False)

# 5. Temps restant
ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(x-width/2, [r["shannon_end_time"]  for r in rows], width,
        color=C_SHANNON,  alpha=0.9, edgecolor="white", label="Shannon")
ax5.bar(x+width/2, [r["dijkstra_end_time"] for r in rows], width,
        color=C_DIJKSTRA, alpha=0.9, edgecolor="white", label="Dijkstra")
ax5.axhline(15, color="#E63946", linestyle=":", linewidth=1.5, alpha=0.7, label="Seuil disqualif. (15s)")
ax5.fill_between([-0.5, N-0.5], 0, 15, color="#E63946", alpha=0.06)
ax5.set_xticks(x); ax5.set_xticklabels([f"P{g}" for g in games], fontsize=9)
ax5.set_ylabel("Temps restant (s)", fontsize=10, color=C_TEXT)
ax5.set_title("Temps restant en fin de partie\n(gestion du budget)",
              fontsize=11, fontweight="bold", color=C_TEXT)
ax5.set_facecolor(C_BG); ax5.grid(True, axis="y", color=C_GRID, linewidth=0.7)
ax5.spines[["top","right"]].set_visible(False); ax5.legend(fontsize=9, framealpha=0.7)

out = "benchmark_graphs.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Graphiques sauvegardes -> {out}")