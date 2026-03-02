"""
Heuristic module for Hex AI player.

Core philosophy: a human plays Hex by asking two questions:
  1. How close am I to winning?
  2. How close is the opponent to winning?

The answer to both is the Dijkstra shortest path distance.
The difference is the only signal that matters.

Everything else (bridges, edge control, connectivity bonuses) adds
noise that confuses the search more than it helps.

One exception: an URGENT THREAT bonus. If the opponent is 1 move from
winning, the defensive score spikes massively — mirroring how a human
immediately drops everything to block a winning threat.
"""

import math
from heapq import heappush, heappop
from game_state_hex import GameStateHex
import numpy as np

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]


class HexHeuristicDjikstra:
    """
    Evaluates a Hex board position as:

        score = opp_dist - my_dist

    Normalized to [-1, 1]. Positive = we are ahead. Negative = opponent ahead.

    Special case: if opponent is 1 move from winning, return a large
    negative value to force an immediate block regardless of our own progress.
    """

    def __init__(self, piece_type: str, board_dim: int = 14):
        self._piece_type = piece_type
        self._opp_type   = "B" if piece_type == "R" else "R"
        self._board_dim  = board_dim

    def evaluate(self, state: GameStateHex) -> float:
        """
        Returns a score in [-inf, inf].
        Positive = good for us. Negative = good for opponent.
        """
        if state.is_done():
            my_id = self._get_player_id(state, self._piece_type)
            return float('inf') if state.scores.get(my_id, 0) == 1 else float('-inf')

        my_dist  = self._dijkstra(state, self._piece_type)
        opp_dist = self._dijkstra(state, self._opp_type)

        # --- URGENT THREAT: opponent is 1 move from winning ---
        # A human ALWAYS blocks this. Score tanks to force a block.
        if opp_dist <= 1:
            return -0.9

        # --- WE ARE 1 MOVE FROM WINNING ---
        if my_dist <= 1:
            return 0.9

        # --- NORMAL CASE: path differential ---
        # opp_dist - my_dist: positive means we need fewer moves = we are ahead
        # Normalize by board_dim so result is in roughly [-1, 1]
        if my_dist == math.inf and opp_dist == math.inf:
            return 0.0
        if my_dist == math.inf:
            return -1.0
        if opp_dist == math.inf:
            return 1.0

        diff = opp_dist - my_dist
        # Normalize: max possible diff is board_dim in either direction
        return max(-1.0, min(1.0, diff / self._board_dim))

    # ------------------------------------------------------------------ #
    #  DIJKSTRA                                                          #
    #  Own pieces cost 0, empty cells cost 1, opponent = impassable.     #
    # ------------------------------------------------------------------ #

    def _dijkstra(self, state: GameStateHex, piece_type: str) -> float:
        dim = self._board_dim
        env = state.rep.env
        dist = {}
        heap = []

        if piece_type == "R":
            seeds      = [(0, j) for j in range(dim)]
            goal_check = lambda i, j: i == dim - 1
        else:
            seeds      = [(i, 0) for i in range(dim)]
            goal_check = lambda i, j: j == dim - 1

        for tile in seeds:
            cell = env.get(tile)
            if cell is not None and cell.get_type() != piece_type:
                continue
            cost = 0 if cell is not None else 1
            if cost < dist.get(tile, math.inf):
                dist[tile] = cost
                heappush(heap, (cost, tile))

        while heap:
            d, (i, j) = heappop(heap)
            if d > dist.get((i, j), math.inf):
                continue
            if goal_check(i, j):
                return d
            for di, dj in HEX_NEIGHBORS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < dim and 0 <= nj < dim):
                    continue
                cell = env.get((ni, nj))
                if cell is not None and cell.get_type() != piece_type:
                    continue
                step = 0 if cell is not None else 1
                new_dist = d + step
                if new_dist < dist.get((ni, nj), math.inf):
                    dist[(ni, nj)] = new_dist
                    heappush(heap, (new_dist, (ni, nj)))

        return math.inf

    # ------------------------------------------------------------------ #
    #  HELPER                                                              #
    # ------------------------------------------------------------------ #

    def _get_player_id(self, state: GameStateHex, piece_type: str):
        for player in state.players:
            if player.get_piece_type() == piece_type:
                return player.get_id()
        return None