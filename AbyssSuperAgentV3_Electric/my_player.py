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


MOVES_REMAINING_ESTIMATE = 50    
MIN_BUDGET_SECONDS       = 5.0  
MAX_BUDGET_SECONDS       = 20.0 
SAFETY_MARGIN            = 15.0   
MAX_DEPTH_CAP            = 12    
MIN_DEPTH                = 2  


class MyPlayer(PlayerHex):

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
        self._deadline        = 0.0 
        self._time_exceeded   = False


    def compute_action(self, current_state: GameStateHex, remaining_time: float = 900.0, **kwargs) -> Action:
        budget = self._compute_budget(remaining_time)
        self._deadline      = time.perf_counter() + budget
        self._time_exceeded = False

        self._track_opponent_move(current_state)

        opening_action = self._try_opening(current_state)
        if opening_action is not None:
            return opening_action

        if DEBUG:
            considered_set = self._candidates._get_considered_tiles(current_state)
            self.debug_print_candidates(current_state, considered_set)

        return self._iterative_deepening(current_state)


    def _compute_budget(self, remaining_time: float) -> float:
        spendable = remaining_time - SAFETY_MARGIN
        if spendable <= 1.0:
            return 0.5  
        budget = spendable / MOVES_REMAINING_ESTIMATE
        return max(MIN_BUDGET_SECONDS, min(MAX_BUDGET_SECONDS, budget))


    def _track_opponent_move(self, current_state: GameStateHex):
        step        = current_state.get_step()
        current_pos = set(current_state.rep.env.keys())
        if step > 0:
            new_pos = current_pos - self._prev_positions
            self._last_opp_move = new_pos.pop() if new_pos else None
        self._prev_positions = current_pos

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
        self._opening_book._in_opening = False
        return None


    def _iterative_deepening(self, current_state: GameStateHex) -> Action:
        initial_candidates = list(self._get_candidates(current_state, is_maximizing=True))
        if not initial_candidates:
            return next(current_state.generate_possible_stateless_actions())

        best_action = initial_candidates[0][0]
        best_val    = -math.inf
        root_hash   = self._zobrist.hash_state(current_state)

        for depth in range(MIN_DEPTH, MAX_DEPTH_CAP + 1):
            if time.perf_counter() >= self._deadline:
                break 

            self._time_exceeded = False
            iteration_best_action = best_action
            iteration_best_val    = -math.inf
            iter_start = time.perf_counter()

            
            candidates = list(self._get_candidates(current_state, is_maximizing=True))

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

            if not self._time_exceeded:
                best_action = iteration_best_action
                best_val    = iteration_best_val
                elapsed = time.perf_counter() - iter_start
                print(f"[ID] depth={depth} completed  val={best_val:.4f}  best={best_action.data['position']}  ({elapsed:.2f}s)")
            else:
                print(f"[ID] depth={depth} ABORTED (time)  returning depth={depth-1} result")

            if best_val == math.inf:
                break

        return best_action

    @staticmethod
    def _prioritise_best(candidates, best_action):
        for i, (action, state, val) in enumerate(candidates):
            if action is best_action or (
                action.data["position"] == best_action.data["position"]
            ):
                return [candidates[i]] + candidates[:i] + candidates[i + 1:]
        return candidates


    def _minimax(
        self,
        state:        GameStateHex,
        depth:        int,
        alpha:        float,
        beta:         float,
        is_maximizing: bool,
        h:            int,
    ) -> float:
        if time.perf_counter() >= self._deadline:
            self._time_exceeded = True
            return self.heuristic(state)

        if state.is_done() or depth == 0:
            return self.heuristic(state)

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

        candidates = self._get_candidates(state, is_maximizing)

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

        if not self._time_exceeded:
            if best_val <= orig_alpha:
                flag = TT_UPPER
            elif best_val >= orig_beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt.store(h, best_val, depth, flag)

        return best_val

    def _get_candidates(self, state, is_maximizing):
        return self._candidates.get_candidates(state, self.heuristic, is_maximizing)

    def heuristic(self, state):
        return self._heuristic.evaluate(state)

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