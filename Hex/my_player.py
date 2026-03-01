import math
from heapq import heappush, heappop
from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]


class MyPlayer(PlayerHex):
    """
    Minimax player for Hex with:
    - Local considered_tiles per node (radius-2 expansion around all pieces)
    - Search width limiting with heuristic-based move ordering
    - Alpha-beta pruning
    - Three-component heuristic: aggressive, defensive, connective
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._max_depth = 4
        self._search_width = 6
        self._board_dim = 14
        self._opp_type = "B" if piece_type == "R" else "R"

    # ------------------------------------------------------------------ #
    #  ENTRY POINT                                                         #
    # ------------------------------------------------------------------ #

    def compute_action(self, current_state: GameStateHex, **kwargs) -> Action:
        best_val = -math.inf
        best_action = None

        candidates = self._get_candidates(current_state, is_maximizing=True)

        for action, next_state, _ in candidates:
            value = self.minimax(next_state, self._max_depth - 1, -math.inf, math.inf, False)
            if value > best_val:
                best_val = value
                best_action = action

        return best_action

    # ------------------------------------------------------------------ #
    #  MINIMAX WITH ALPHA-BETA                                             #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  CANDIDATE GENERATION                                                #
    # ------------------------------------------------------------------ #

    def _get_candidates(self, state: GameStateHex, is_maximizing: bool):
        """
        1. Compute considered tiles (radius-2 around all placed pieces).
        2. Filter framework-generated actions to only considered tiles.
        3. Evaluate heuristic on each resulting state.
        4. Sort best-first (maximizer: high to low, minimizer: low to high).
        5. Trim to search_width.

        Falls back to all actions on empty board.
        """
        considered = self._get_considered_tiles(state)
        scored = []

        for action in state.generate_possible_stateless_actions():
            # On empty board considered is empty so we take everything
            if considered and action.data["position"] not in considered:
                continue
            next_state = state.apply_action(action)
            h_val = self.heuristic(next_state)
            scored.append((action, next_state, h_val))

        scored.sort(key=lambda x: x[2], reverse=is_maximizing)
        return scored[:self._search_width]

    def _get_considered_tiles(self, state: GameStateHex) -> set:
        """
        Radius-2 expansion around every placed piece.
        Computed fresh at each node so branches don't contaminate each other.
        Only includes empty cells within board bounds.
        """
        considered = set()
        env = state.rep.env

        for tile in env.keys():
            for di, dj in HEX_NEIGHBORS:
                r1 = (tile[0] + di, tile[1] + dj)
                if self._in_bounds(r1) and r1 not in env:
                    considered.add(r1)
                for di2, dj2 in HEX_NEIGHBORS:
                    r2 = (r1[0] + di2, r1[1] + dj2)
                    if self._in_bounds(r2) and r2 not in env:
                        considered.add(r2)

        return considered

    def _in_bounds(self, tile) -> bool:
        return 0 <= tile[0] < self._board_dim and 0 <= tile[1] < self._board_dim

    # ------------------------------------------------------------------ #
    #  COMBINED HEURISTIC                                                  #
    # ------------------------------------------------------------------ #

    def heuristic(self, state: GameStateHex) -> float:
        if state.is_done():
            my_score = state.scores.get(self.get_id(), 0)
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
        dist = self._dijkstra(state, self.piece_type)
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
            if piece.get_type() == self.piece_type
        ]

        if not my_pieces:
            return 0.0

        connected_count = 0
        for tile in my_pieces:
            for di, dj in HEX_NEIGHBORS:
                neighbor = (tile[0] + di, tile[1] + dj)
                if (neighbor in state.rep.env
                        and state.rep.env[neighbor].get_type() == self.piece_type):
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
            seeds = [(0, j) for j in range(dim)]
            goal_check = lambda i, j: i == dim - 1
        else:
            seeds = [(i, 0) for i in range(dim)]
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

    def to_json(self) -> dict:
        return {}