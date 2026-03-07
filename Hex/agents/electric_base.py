"""
MyPlayer — Hex AI with:
  - Iterative Deepening (ID-alpha-beta) with a wall-clock time budget
  - Zobrist hashing + Transposition Table (exact / lower / upper bounds)
  - Theory-based opening book
  - Heuristic-ordered candidate moves (unchanged from original)

Iterative Deepening strategy
-----------------------------
We search depth 1, 2, 3, … until the time budget is exhausted or a
terminal state is proven.  The best move from depth d seeds move ordering
for depth d+1 (killer heuristic via candidate pre-sorting).

Time budget
-----------
seahorse passes remaining_time (total seconds left for the whole game) into
compute_action().  We derive a dynamic per-move budget:

    budget = clamp((remaining_time - SAFETY_MARGIN) / MOVES_REMAINING_ESTIMATE,
                   MIN_BUDGET_SECONDS, MAX_BUDGET_SECONDS)

This means early in the game (lots of time left) we spend ~15 s/move;
late in the game the budget shrinks proportionally, always keeping a 10 s
safety margin so we are never disqualified.

If a deeper ID iteration would exceed the budget, we return the best move
found so far.  This is safe because ID always has a complete result from
the previous depth.

Maximum depth cap
-----------------
At most 98 plies are available per player on a 14×14 board (196 cells / 2).
We cap at MAX_DEPTH_CAP = 12 to avoid pointless deep searches in endgame.
In practice the time budget cuts out far earlier.

Transposition Table integration (alpha-beta + TT)
--------------------------------------------------
Standard technique:
  - On entry: if TT hit at depth ≥ current depth, use stored bound.
  - On exit:  store EXACT if value is inside (alpha, beta),
              LOWER  if we caused a beta cut (value ≥ beta),
              UPPER  if value ≤ alpha (all moves failed low).
"""

import math
import time
from datetime import datetime
from typing import Optional

from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex

from colorama import Fore, Style

from superAlgos.heuristics import HexHeuristicDjikstra
from superAlgos.candidates import CandidateGenerator
from superAlgos.openings import Openings
from superAlgos.zobrist import ZobristTable, TranspositionTable, TTEntry
from superAlgos.zobrist import TT_EXACT, TT_LOWER, TT_UPPER

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]
DEBUG         = True

# Time management
# ---------------
# Total budget per game: 900 s.  A full game is ~50 moves per player.
# We derive a per-move budget from remaining_time (passed by seahorse):
#
#   per_move_budget = (remaining_time - SAFETY_MARGIN) / MOVES_REMAINING_ESTIMATE
#
# clamped to [MIN_BUDGET_SECONDS, MAX_BUDGET_SECONDS].
#
# Effective branching with width=12 + alpha-beta ≈ sqrt(12) ≈ 3.5.
# Dijkstra on 14x14 ≈ 0.1 ms, so:
#   depth 4 → ~150 nodes  → ~15 ms
#   depth 5 → ~525 nodes  → ~50 ms
#   depth 6 → ~1800 nodes → ~180 ms
#   depth 7 → ~6300 nodes → ~630 ms
# The time guard aborts any depth that overruns.

# Budget strategy:
#   900s / ~50 moves per game = 18s average.
#   We spend up to 20s early (lots of time), shrinking as clock drains.
#   The old depth=4 width=12 baseline costs ~0.1-0.5s/move — far too cheap.
#   With width=12 and alpha-beta, depth=5 costs ~0.05s and depth=6 ~0.2s,
#   depth=7 ~0.7s. Spending up to 20s means we reliably reach depth 7-8,
#   which is strictly deeper than the baseline's fixed depth=4.
MOVES_REMAINING_ESTIMATE = 50     # ~50 own moves in a full game
MIN_BUDGET_SECONDS       = 5.0    # floor: always spend at least 5s
MAX_BUDGET_SECONDS       = 20.0   # ceiling: up to 20s early game
SAFETY_MARGIN            = 15.0   # keep 15s in reserve (disqualification guard)
MAX_DEPTH_CAP            = 12     # hard depth ceiling
MIN_DEPTH                = 2      # always finish at least this depth


class MyPlayer(PlayerHex):
    """
    Minimax player for Hex with iterative deepening, Zobrist TT, and openings.
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._search_width    = 16
        self._board_dim       = 14
        self._opp_type        = "B" if self.piece_type == "R" else "R"
        self._heuristic       = HexHeuristicDjikstra(self.piece_type, board_dim=14)
        self._candidates      = CandidateGenerator(self.piece_type, board_dim=14, search_width=self._search_width)
        self._opening_book    = Openings(self.piece_type)
        self._zobrist         = ZobristTable(board_dim=14)
        self._tt              = TranspositionTable(max_size=1_000_000)
        self._last_opp_move   = None
        self._prev_positions  = set()
        self._deadline        = 0.0   # set at start of each compute_action()
        self._time_exceeded   = False

    # ------------------------------------------------------------------ #
    #  Entry point                                                        #
    # ------------------------------------------------------------------ #

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 900.0, **kwargs) -> Action:
        budget = self._compute_budget(remaining_time)
        self._deadline      = time.perf_counter() + budget
        self._time_exceeded = False

        self._track_opponent_move(current_state)

        # 1. Opening book
        opening_action = self._try_opening(current_state)
        if opening_action is not None:
            return opening_action

        if DEBUG:
            considered_set = self._candidates._get_considered_tiles(current_state)
            self.debug_print_candidates(current_state, considered_set)

        # 2. Iterative deepening
        return self._iterative_deepening(current_state)

    # ------------------------------------------------------------------ #
    #  Time budget                                                         #
    # ------------------------------------------------------------------ #

    def _compute_budget(self, remaining_time: float) -> float:
        """
        Derive a safe per-move time budget from the remaining game clock.

        Formula:
            spendable = remaining_time - SAFETY_MARGIN   (15 s reserve)
            budget    = spendable / MOVES_REMAINING_ESTIMATE (50 moves)
        Clamped to [MIN_BUDGET_SECONDS=5, MAX_BUDGET_SECONDS=20].

        Early game (900s left):  (900-15)/50 = 17.7s → 17.7s
        Mid game  (450s left):   (450-15)/50 =  8.7s →  8.7s
        Late game (100s left):   (100-15)/50 =  1.7s →  5.0s (floor)
        Critical  (≤16s left):   fast fallback → 0.5s

        With width=12 this reliably reaches depth 6-8, always deeper
        than the fixed depth=4 baseline.
        """
        spendable = remaining_time - SAFETY_MARGIN
        if spendable <= 1.0:
            return 0.5   # near disqualification: return fast
        budget = spendable / MOVES_REMAINING_ESTIMATE
        return max(MIN_BUDGET_SECONDS, min(MAX_BUDGET_SECONDS, budget))

    # ------------------------------------------------------------------ #
    #  Opponent tracking                                                   #
    # ------------------------------------------------------------------ #

    def _track_opponent_move(self, current_state: GameStateHex):
        step        = current_state.get_step()
        current_pos = set(current_state.rep.env.keys())
        if step > 0:
            new_pos = current_pos - self._prev_positions
            self._last_opp_move = new_pos.pop() if new_pos else None
        self._prev_positions = current_pos

    # ------------------------------------------------------------------ #
    #  Opening book                                                        #
    # ------------------------------------------------------------------ #

    def _try_opening(self, current_state: GameStateHex) -> Optional[Action]:
        step     = current_state.get_step()
        occupied = set(current_state.rep.env.keys())
        move     = self._opening_book.get_opening_move(
            step, self._last_opp_move, occupied
        )
        if move is None:
            return None
        for action in current_state.generate_possible_stateless_actions():
            if action.data["position"] == move:
                return action
        # Desired cell taken — exit opening book
        self._opening_book._in_opening = False
        return None

    # ------------------------------------------------------------------ #
    #  Iterative deepening                                                 #
    # ------------------------------------------------------------------ #

    def _iterative_deepening(self, current_state: GameStateHex) -> Action:
        """
        Search depth 1 … MAX_DEPTH_CAP, returning the best action found
        before the time budget expires.

        Move ordering across iterations:
          The candidate generator already sorts by heuristic value.
          After each completed depth, the best action is tried first at
          the next depth (we prepend it to the candidate list).
        """
        # Fallback: pick any candidate
        initial_candidates = list(self._get_candidates(current_state, is_maximizing=True))
        if not initial_candidates:
            return next(current_state.generate_possible_stateless_actions())

        best_action = initial_candidates[0][0]
        best_val    = -math.inf
        root_hash   = self._zobrist.hash_state(current_state)

        for depth in range(MIN_DEPTH, MAX_DEPTH_CAP + 1):
            if time.perf_counter() >= self._deadline:
                break   # budget gone before starting this depth

            self._time_exceeded = False
            iteration_best_action = best_action
            iteration_best_val    = -math.inf
            iter_start = time.perf_counter()

            # Re-generate candidates so the TT can reorder them
            candidates = list(self._get_candidates(current_state, is_maximizing=True))

            # Move the previous iteration's best to the front (root move ordering)
            candidates = self._prioritise_best(candidates, best_action)

            for action, next_state, _ in candidates:
                if time.perf_counter() >= self._deadline:
                    self._time_exceeded = True
                    break

                child_hash = self._zobrist.update(
                    root_hash,
                    self.piece_type,
                    action.data["position"],
                )
                val = self._minimax(
                    next_state, depth - 1,
                    -math.inf, math.inf,
                    is_maximizing=False,
                    h=child_hash,
                )
                if val > iteration_best_val:
                    iteration_best_val    = val
                    iteration_best_action = action

            # Only commit results from a *fully completed* iteration
            if not self._time_exceeded:
                best_action = iteration_best_action
                best_val    = iteration_best_val
                elapsed = time.perf_counter() - iter_start
                print(f"[ID] depth={depth} completed  val={best_val:.4f}  best={best_action.data['position']}  ({elapsed:.2f}s)")
            else:
                print(f"[ID] depth={depth} ABORTED (time)  returning depth={depth-1} result")

            # Early exit: proven win
            if best_val == math.inf:
                break

        return best_action

    @staticmethod
    def _prioritise_best(candidates, best_action):
        """Move best_action to the front of the candidate list."""
        for i, (action, state, val) in enumerate(candidates):
            if action is best_action or (
                action.data["position"] == best_action.data["position"]
            ):
                return [candidates[i]] + candidates[:i] + candidates[i + 1:]
        return candidates

    # ------------------------------------------------------------------ #
    #  Alpha-beta minimax with transposition table                        #
    # ------------------------------------------------------------------ #

    def _minimax(
        self,
        state:        GameStateHex,
        depth:        int,
        alpha:        float,
        beta:         float,
        is_maximizing: bool,
        h:            int,
    ) -> float:
        """
        Alpha-beta minimax with Zobrist TT.

        h is the Zobrist hash of `state` passed from the parent to avoid
        recomputing it from scratch at every node.
        """
        # --- Time check ---
        if time.perf_counter() >= self._deadline:
            self._time_exceeded = True
            return self.heuristic(state)

        # --- Terminal / leaf ---
        if state.is_done() or depth == 0:
            return self.heuristic(state)

        # --- Transposition table lookup ---
        orig_alpha = alpha
        orig_beta  = beta
        entry: Optional[TTEntry] = self._tt.lookup(h)
        if entry is not None and entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return entry.value
            elif entry.flag == TT_LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == TT_UPPER:
                beta  = min(beta,  entry.value)
            if alpha >= beta:
                return entry.value

        # --- Recurse ---
        # Use the scored (Dijkstra) generator — this gives good move ordering
        # which is critical for alpha-beta cutoffs. The TT cuts duplicate
        # subtrees; Dijkstra ordering makes alpha-beta effective within each subtree.
        candidates = self._get_candidates(state, is_maximizing)

        # Determine whose piece is being placed at this node.
        mover = self.piece_type if is_maximizing else self._opp_type

        if is_maximizing:
            best_val = -math.inf
            for action, next_state, _ in candidates:
                if time.perf_counter() >= self._deadline:
                    self._time_exceeded = True
                    break
                child_h = self._zobrist.update(h, mover, action.data["position"])
                val     = self._minimax(next_state, depth - 1, alpha, beta, False, child_h)
                best_val = max(best_val, val)
                alpha    = max(alpha, val)
                if beta <= alpha:
                    break
        else:
            best_val = math.inf
            for action, next_state, _ in candidates:
                if time.perf_counter() >= self._deadline:
                    self._time_exceeded = True
                    break
                child_h = self._zobrist.update(h, mover, action.data["position"])
                val     = self._minimax(next_state, depth - 1, alpha, beta, True, child_h)
                best_val = min(best_val, val)
                beta     = min(beta, val)
                if beta <= alpha:
                    break

        # --- Store in TT (only if time was not exceeded mid-search) ---
        if not self._time_exceeded:
            if best_val <= orig_alpha:
                flag = TT_UPPER
            elif best_val >= orig_beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt.store(h, best_val, depth, flag)

        return best_val

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_candidates(self, state, is_maximizing):
        return self._candidates.get_candidates(state, self.heuristic, is_maximizing)

    def heuristic(self, state):
        return self._heuristic.evaluate(state)

    # ------------------------------------------------------------------ #
    #  Debug                                                               #
    # ------------------------------------------------------------------ #

    def debug_print_candidates(self, state: GameStateHex, candidates_set: set):
        env   = state.rep.env
        dim   = state.rep.dimensions
        PURPLE = '\033[95m'
        RESET  = Style.RESET_ALL

        print("\n--- DEBUG: CANDIDATE TILES (Purple) ---")
        for i in range(dim[0]):
            print(" " * i, end="")
            for j in range(dim[1]):
                pos = (i, j)
                if pos in env:
                    char  = "⬢"
                    color = Fore.RED if env[pos].get_type() == "R" else Fore.BLUE
                    print(f"{color}{char}{RESET} ", end="")
                elif pos in candidates_set:
                    print(f"{PURPLE}c{RESET} ", end="")
                else:
                    print(". ", end="")
            print()
        print("---------------------------------------\n")