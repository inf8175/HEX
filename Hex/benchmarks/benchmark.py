"""
run_benchmark.py
================
Lance N parties séquentielles Shannon vs Dijkstra et collecte les stats.

Usage
-----
    python run_benchmark.py --games 10 --output results/

Ce script :
  1. Lance `python main_hex.py -r -g -t local <agent1> <agent2>` pour chaque partie
  2. Récupère le JSON généré par seahorse
  3. Extrait les stats clés (gagnant, nb coups, temps par coup, profondeur)
  4. Sauvegarde un CSV global  →  results/benchmark_summary.csv
  5. Garde tous les JSONs      →  results/json/fight_XXX.json

Exigences
---------
  - Être dans le dossier racine du projet Hex (où se trouve main_hex.py)
  - Les deux agents doivent être dans ce dossier :
      * my_player.py        (Shannon / Electric)
      * my_player_dijkstra.py  (Dijkstra — voir note ci-dessous)

Note : créer my_player_dijkstra.py
-----------------------------------
C'est simplement une copie de ta V2 (Smart Agent V2 avec Dijkstra).
Renomme la classe en MyPlayerDijkstra pour éviter les conflits :

    class MyPlayerDijkstra(PlayerHex):
        ...

Puis dans le fichier, change aussi le nom par défaut :
    def __init__(self, piece_type: str, name: str = "DijkstraAgent"):
"""

import subprocess
import os
import sys
import json
import csv
import time
import argparse
import shutil
import glob
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────

# Noms des fichiers agents (modifie si besoin)
AGENT_SHANNON   = "agents/electric_base.py"
AGENT_DIJKSTRA  = "agents/minimax_djikstra.py"

# Seahorse sauvegarde les JSON dans le dossier racine (Hex/)
# Ce script doit donc être lancé depuis Hex/ :
#   python benchmarks/run_benchmark.py --games 10 --output benchmarks/results/ --swap
JSON_SAVE_DIR = "."  # dossier où seahorse crée les __REC_*.json

# Commande seahorse (adapte si tu utilises un venv)
PYTHON = sys.executable

# ─── Parse args ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Shannon vs Dijkstra — batch benchmark")
parser.add_argument("--games",  type=int, default=10,        help="Nombre de parties (défaut: 10)")
parser.add_argument("--output", type=str, default="results", help="Dossier de sortie")
parser.add_argument("--swap",   action="store_true",         help="Alterner les couleurs à chaque partie")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
os.makedirs(os.path.join(args.output, "json"), exist_ok=True)

SUMMARY_CSV = os.path.join(args.output, "benchmark_summary.csv")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_json(before_files):
    """Trouve le __REC_*.json créé par seahorse dans JSON_SAVE_DIR (racine Hex/)."""
    current = set(glob.glob(os.path.join(JSON_SAVE_DIR, "__REC_*.json")))
    new_files = current - before_files
    if new_files:
        return max(new_files, key=os.path.getmtime)
    return None


def parse_fight_json(path, shannon_keyword="Electric"):
    """
    Extrait les stats clés d'un fichier JSON de partie seahorse.

    Retourne un dict avec :
      - winner          : 'shannon' | 'dijkstra' | 'unknown'
      - total_steps     : nombre de demi-coups joués
      - shannon_moves   : coups joués par Shannon
      - dijkstra_moves  : coups joués par Dijkstra
      - shannon_times   : liste des temps par coup (Shannon)
      - dijkstra_times  : liste des temps par coup (Dijkstra)
      - shannon_end_time: temps restant en fin de partie (Shannon)
      - dijkstra_end_time: temps restant en fin de partie (Dijkstra)
      - shannon_piece   : 'R' ou 'B'
    """
    with open(path) as f:
        d = json.load(f)

    steps   = d["steps"]
    summary = d["final_summary"]
    winners = [str(w) for w in summary["winners_id"]]

    # Identifier les agents
    shannon_p, dijkstra_p = None, None
    for p in steps[0]["players"]:
        name = p.get("name", "")
        if any(k in name for k in [shannon_keyword, "Electric", "Smarter", "Cachalot", "my_player"]):
            shannon_p = p
        else:
            dijkstra_p = p

    # Fallback si nommage ambigu
    if shannon_p is None:
        shannon_p  = steps[0]["players"][0]
        dijkstra_p = steps[0]["players"][1]

    s_id = str(shannon_p["id"])
    d_id = str(dijkstra_p["id"])

    winner = "unknown"
    if s_id in winners:
        winner = "shannon"
    elif d_id in winners:
        winner = "dijkstra"

    # Temps par coup
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
        "dijkstra_piece":    dijkstra_p.get("piece_type", "?"),
        "shannon_name":      shannon_p.get("name", "Shannon"),
        "dijkstra_name":     dijkstra_p.get("name", "Dijkstra"),
    }


def write_summary_row(csv_path, game_idx, fight_id, stats, header_needed):
    """Ajoute une ligne au CSV de résumé."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow([
                "game", "fight_id",
                "winner",
                "total_steps", "shannon_moves", "dijkstra_moves",
                "shannon_avg_time_s", "dijkstra_avg_time_s",
                "shannon_end_time_s", "dijkstra_end_time_s",
                "shannon_piece", "dijkstra_piece",
                "shannon_times", "dijkstra_times",   # raw lists as strings
            ])
        writer.writerow([
            game_idx, fight_id,
            stats["winner"],
            stats["total_steps"], stats["shannon_moves"], stats["dijkstra_moves"],
            stats["shannon_avg_time"], stats["dijkstra_avg_time"],
            stats["shannon_end_time"], stats["dijkstra_end_time"],
            stats["shannon_piece"], stats["dijkstra_piece"],
            str(stats["shannon_times"]),
            str(stats["dijkstra_times"]),
        ])


# ─── Main loop ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  Shannon vs Dijkstra — {args.games} parties")
print(f"  Output : {os.path.abspath(args.output)}")
print(f"{'='*60}\n")

shannon_wins  = 0
dijkstra_wins = 0
header_needed = not os.path.exists(SUMMARY_CSV)

for game_idx in range(1, args.games + 1):

    # Alterner les couleurs si --swap (Rouge joue toujours en premier)
    if args.swap and game_idx % 2 == 0:
        p1, p2 = AGENT_DIJKSTRA, AGENT_SHANNON
        color_note = "Dijkstra=R  Shannon=B"
    else:
        p1, p2 = AGENT_SHANNON, AGENT_DIJKSTRA
        color_note = "Shannon=R  Dijkstra=B"

    print(f"[{game_idx:>2}/{args.games}] Partie {game_idx}  ({color_note})")

    # Snapshot des JSONs existants avant la partie
    before_jsons = set(glob.glob(os.path.join(JSON_SAVE_DIR, "__REC_*.json")))

    # Lance la partie (sans GUI, avec enregistrement JSON)
    # capture_output=False → logs seahorse + [ID] depth=X visibles en temps réel
    cmd = [PYTHON, "main_hex.py", "-r", "-g", "-t", "local", p1, p2]
    t0  = time.perf_counter()
    print(f"  ▶  Commande : {' '.join(cmd)}")
    print(f"  ⏳ Partie en cours...\n{'-'*50}")
    try:
        result = subprocess.run(cmd, timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"  ⚠  Partie {game_idx} : timeout (>20 min) — ignorée")
        continue
    elapsed_wall = time.perf_counter() - t0
    print(f"{'-'*50}")

    if result.returncode != 0:
        print(f"  ⚠  Partie {game_idx} : erreur seahorse (code {result.returncode})")
        continue

    # Trouve le JSON produit
    json_path = find_latest_json(before_jsons)
    if json_path is None:
        print(f"  ⚠  Partie {game_idx} : JSON introuvable")
        continue

    # Parse et sauvegarde
    try:
        stats    = parse_fight_json(json_path)
        fight_id = os.path.splitext(json_path)[0]
        dest     = os.path.join(args.output, "json", f"fight_{game_idx:03d}_{fight_id}.json")
        shutil.copy(json_path, dest)
        write_summary_row(SUMMARY_CSV, game_idx, fight_id, stats, header_needed)
        header_needed = False
    except Exception as e:
        print(f"  ⚠  Partie {game_idx} : erreur parsing JSON — {e}")
        continue

    # Mise à jour du score
    if stats["winner"] == "shannon":
        shannon_wins += 1
        result_str = "✓ Shannon"
    elif stats["winner"] == "dijkstra":
        dijkstra_wins += 1
        result_str = "✗ Dijkstra"
    else:
        result_str = "? inconnu"

    print(f"       → {result_str}  |  {stats['total_steps']} coups  "
          f"|  Shannon moy {stats['shannon_avg_time']:.1f}s/coup  "
          f"|  Dijkstra moy {stats['dijkstra_avg_time']:.1f}s/coup  "
          f"|  durée réelle {elapsed_wall:.0f}s\n")

# ─── Final summary ────────────────────────────────────────────────────────────

total_played = shannon_wins + dijkstra_wins
print(f"\n{'='*60}")
print(f"  Résultats finaux ({total_played} parties terminées)")
print(f"  Shannon   : {shannon_wins} victoires  ({100*shannon_wins/max(total_played,1):.0f}%)")
print(f"  Dijkstra  : {dijkstra_wins} victoires  ({100*dijkstra_wins/max(total_played,1):.0f}%)")
print(f"  CSV       : {os.path.abspath(SUMMARY_CSV)}")
print(f"{'='*60}\n")
print("Lance maintenant `python generate_graphs.py results/benchmark_summary.csv` pour les graphiques.")