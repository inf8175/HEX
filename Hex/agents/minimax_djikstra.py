import math
from heapq import heappush, heappop
from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex

from colorama import Fore, Back, Style

from algos.heuristics import HexHeuristicDjikstra
from algos.candidates import CandidateGenerator
from algos.openings import Openings

import os
import csv
import time
from datetime import datetime

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]
DEBUG = False

class MyPlayer(PlayerHex):

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._max_depth = 4
        self._search_width = 12
        self._board_dim = 14
        self._iteration = 0
        #os.makedirs("./benchmarks", exist_ok=True)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._opp_type = "B" if piece_type == "R" else "R"
        self._heuristic = HexHeuristicDjikstra(piece_type, board_dim=14)
        self._candidates = CandidateGenerator(piece_type, board_dim=14, search_width=self._search_width)
        self._opening_book = Openings(piece_type)
        self._last_opponent_move = None
        self._prev_positions = set()

    def compute_action(self, current_state: GameStateHex, **kwargs) -> Action:
        start_time = time.perf_counter()
        self._track_opponent_move(current_state)
        
        opening_action = self._try_opening(current_state)
        if opening_action is not None:
            #self._log_time(start_time)
            return opening_action
        
        if DEBUG:
            considered_set = self._candidates._get_considered_tiles(current_state)
            self.debug_print_candidates(current_state, considered_set)
    
        best_action = self._run_minimax(current_state)

        #self._log_time(start_time)
        return best_action
    
    def _track_opponent_move(self, current_state: GameStateHex):
        step = current_state.get_step()
        current_pos = set(current_state.rep.env.keys())
        if step > 0:
            new_pos = current_pos - self._prev_positions
            self._last_opponent_move = new_pos.pop() if new_pos else None
        self._prev_positions = current_pos
    
    def _try_opening(self, current_state: GameStateHex):
        step = current_state.get_step()
        opening_move = self._opening_book.get_opening_move(step, self._last_opponent_move)
        if opening_move is None:
            return None
        for action in current_state.generate_possible_stateless_actions():
            if action.data["position"] == opening_move:
                return action
        self._opening_book._in_opening = False
        return None
    
    def _run_minimax(self, current_state: GameStateHex) -> Action:
        best_action = next(current_state.generate_possible_stateless_actions(), None)
        best_val = -math.inf
    
        for action, next_state, _ in self._get_candidates(current_state, is_maximizing=True):
            value = self.minimax(next_state, self._max_depth - 1, -math.inf, math.inf, False)
            if value > best_val:
                best_val = value
                best_action = action
    
        return best_action

    def minimax(self, state: GameStateHex, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        if state.is_done() or depth == 0:
            return self.heuristic(state)

        candidates = self._get_candidates(state, is_maximizing)

        if is_maximizing:
            max_eval = -math.inf
            for _, next_state, _ in candidates:
                val = self.minimax(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for _, next_state, _ in candidates:
                val = self.minimax(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_eval
    
    def _get_candidates(self, state, is_maximizing):
        return self._candidates.get_candidates(state, self.heuristic, is_maximizing)

    def heuristic(self, state):
        return self._heuristic.evaluate(state)

    def _log_time(self, start_time):
       elapsed = time.perf_counter() - start_time
       self._iteration += 1

       filename = (
           f"benchmark_depth_{self._max_depth}"
           f"_width_{self._search_width}"
           f"_{self._timestamp}.csv"
       )

       filepath = os.path.join("benchmarks", filename)

       os.makedirs("benchmarks", exist_ok=True)

       file_exists = os.path.isfile(filepath)

       with open(filepath, "a", newline="") as f:
           writer = csv.writer(f)

           if not file_exists:
               writer.writerow(["iteration", "seconds"])

           writer.writerow([self._iteration, elapsed])

    def debug_print_candidates(self, state: GameStateHex, candidates_set: set):
        """
        Prints the board with candidate moves highlighted in Purple.
        """
        env = state.rep.env
        dim = state.rep.dimensions

        # Purple ANSI codes (some consoles prefer Magenta for Purple)
        PURPLE = '\033[95m'
        RESET = Style.RESET_ALL

        print("\n--- DEBUG: CANDIDATE TILES (Purple) ---")
        for i in range(dim[0]):
            # The 'Hex Slant'
            print(" " * i, end="")
            for j in range(dim[1]):
                pos = (i, j)
                if pos in env:
                    # Existing pieces
                    char = "⬢"
                    color = Fore.RED if env[pos].get_type() == "R" else Fore.BLUE
                    print(f"{color}{char}{RESET} ", end="")
                elif pos in candidates_set:
                    # THIS IS A CANDIDATE
                    # Using '?' or 'c' to represent a considered tile
                    print(f"{PURPLE}c{RESET} ", end="")
                else:
                    # Empty but not considered
                    print(". ", end="")
            print()  # Newline
        print("---------------------------------------\n")