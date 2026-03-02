"""
Heuristic module for Hex AI player.
Contains all evaluation logic: aggressive, defensive, and connective scores.
"""

import math
from heapq import heappush, heappop
from game_state_hex import GameStateHex

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]


class HexHeuristic:
    """
    Evaluates a Hex board state from the perspective of a given player.

    Combines three components:
    - Aggressive: how close YOU are to winning (shorter path = better)
    - Defensive:  how far OPPONENT is from winning (longer path = better)
    - Connective: how well YOUR pieces are linked together
    """

    def __init__(self, piece_type: str, board_dim: int = 14):
        self._piece_type = piece_type
        self._opp_type   = "B" if piece_type == "R" else "R"
        self._board_dim  = board_dim

    def evaluate(self, state: GameStateHex) -> float:
        """
        Main entry point. Returns a score in [-inf, inf]:
        - float('inf')  = player has won
        - float('-inf') = player has lost
        - values in [0, 1] otherwise
        """
        if state.is_done():
            my_score = state.scores.get(
                self._get_player_id(state, self._piece_type), 0
            )
            return float('inf') if my_score == 1 else float('-inf')

        w_aggressive = 0.5
        w_defensive  = 0.3
        w_connective = 0.2

        agg  = self._aggressive_score(state)
        defe = self._defensive_score(state)
        conn = self._connective_score(state)

        return w_aggressive * agg + w_defensive * defe + w_connective * conn

    # ------------------------------------------------------------------ #
    #  HEURISTIC 1 - AGGRESSIVE                                            #
    #  Shortest path for YOU. Score in [0,1]: 1 = already connected.      #
    # ------------------------------------------------------------------ #

    def _aggressive_score(self, state: GameStateHex) -> float:
        dist = self._dijkstra(state, self._piece_type)
        if dist == 0:
            return 1.0
        if dist == math.inf:
            return 0.0
        return 1.0 - (dist / self._board_dim)

    # ------------------------------------------------------------------ #
    #  HEURISTIC 2 - DEFENSIVE                                             #
    #  Shortest path for OPPONENT. Score in [0,1]: 1 = fully blocked.     #
    # ------------------------------------------------------------------ #

    def _defensive_score(self, state: GameStateHex) -> float:
        dist = self._dijkstra(state, self._opp_type)
        if dist == math.inf:
            return 1.0
        if dist == 0:
            return 0.0
        return dist / self._board_dim

    # ------------------------------------------------------------------ #
    #  HEURISTIC 3 - CONNECTIVE                                            #
    #  Fraction of YOUR pieces with at least one friendly neighbor.        #
    #  Score in [0,1]: 1 = all pieces connected to at least one other.    #
    # ------------------------------------------------------------------ #

    def _connective_score(self, state: GameStateHex) -> float:
        my_pieces = [
            tile for tile, piece in state.rep.env.items()
            if piece.get_type() == self._piece_type
        ]

        if not my_pieces:
            return 0.0

        connected_count = 0
        for tile in my_pieces:
            for di, dj in HEX_NEIGHBORS:
                neighbor = (tile[0] + di, tile[1] + dj)
                if (neighbor in state.rep.env
                        and state.rep.env[neighbor].get_type() == self._piece_type):
                    connected_count += 1
                    break  # count each piece at most once

        return connected_count / len(my_pieces)

    # ------------------------------------------------------------------ #
    #  DIJKSTRA                                                            #
    #  Own pieces cost 0, empty cells cost 1, opponent = impassable.      #
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
            cost = 0 if (cell is not None) else 1
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
                step = 0 if (cell is not None) else 1
                new_dist = d + step
                if new_dist < dist.get((ni, nj), math.inf):
                    dist[(ni, nj)] = new_dist
                    heappush(heap, (new_dist, (ni, nj)))

        return math.inf

    # ------------------------------------------------------------------ #
    #  HELPER                                                              #
    # ------------------------------------------------------------------ #

    def _get_player_id(self, state: GameStateHex, piece_type: str):
        """Get player ID matching the given piece type from the game state."""
        for player in state.players:
            if player.get_piece_type() == piece_type:
                return player.get_id()
        return None