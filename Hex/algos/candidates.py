"""
Candidate generation module for Hex AI player.
Handles considered tiles computation and move candidate selection.
"""

from game_state_hex import GameStateHex

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]


class CandidateGenerator:
    """
    Generates and ranks candidate moves for the minimax search.

    Strategy:
    - Expands radius-2 around own pieces (aggressive + connective moves)
    - Expands radius-1 around threatening opponent pieces only (defensive moves)
    - Filters and sorts candidates by heuristic value
    - Trims to search_width
    """

    def __init__(self, piece_type: str, board_dim: int = 14, search_width: int = 6):
        self._piece_type  = piece_type
        self._opp_type    = "B" if piece_type == "R" else "R"
        self._board_dim   = board_dim
        self._search_width = search_width

    def get_candidates(self, state: GameStateHex, heuristic_fn, is_maximizing: bool):
        """
        Generate, score, sort and trim candidate moves.

        Args:
            state:          Current game state.
            heuristic_fn:   Callable(state) -> float, used to score each candidate.
            is_maximizing:  True = sort highest first, False = sort lowest first.

        Returns:
            List of (action, next_state, heuristic_value) trimmed to search_width.
        """
        considered = self._get_considered_tiles(state)
        scored = []

        for action in state.generate_possible_stateless_actions():
            if considered and action.data["position"] not in considered:
                continue
            next_state = state.apply_action(action)
            h_val = heuristic_fn(next_state)
            scored.append((action, next_state, h_val))

        # Fallback: if filtering left nothing, use all available actions
        if not scored:
            for action in state.generate_possible_stateless_actions():
                next_state = state.apply_action(action)
                h_val = heuristic_fn(next_state)
                scored.append((action, next_state, h_val))

        scored.sort(key=lambda x: x[2], reverse=is_maximizing)
        return scored[:self._search_width]

    def _get_considered_tiles(self, state: GameStateHex) -> set:
        """
        Compute relevant candidate positions from the current board:
        - Ring 1+2 around OWN pieces (aggressive + connective play)
        - Ring 1 only around THREATENING opponent pieces (defensive play)

        A threatening opponent piece is one that is either:
        - Connected to another opponent piece, OR
        - Touching the opponent's goal edge

        Computed fresh at each node so branches don't contaminate each other.
        """
        considered = set()
        env = state.rep.env

        my_tiles  = [t for t, p in env.items() if p.get_type() == self._piece_type]
        opp_tiles = [t for t, p in env.items() if p.get_type() == self._opp_type]

        # Ring 1+2 around own pieces
        for tile in my_tiles:
            for di, dj in HEX_NEIGHBORS:
                r1 = (tile[0] + di, tile[1] + dj)
                if self._in_bounds(r1) and r1 not in env:
                    considered.add(r1)
                for di2, dj2 in HEX_NEIGHBORS:
                    r2 = (r1[0] + di2, r1[1] + dj2)
                    if self._in_bounds(r2) and r2 not in env:
                        considered.add(r2)

        # Ring 1 only around threatening opponent pieces
        for tile in opp_tiles:
            if self._is_threatening(tile, env):
                for di, dj in HEX_NEIGHBORS:
                    r1 = (tile[0] + di, tile[1] + dj)
                    if self._in_bounds(r1) and r1 not in env:
                        considered.add(r1)

        return considered

    def _is_threatening(self, tile, env) -> bool:
        """
        An opponent piece is threatening if it is part of a real chain:
        connected to another opponent piece OR touching its goal edge.
        Isolated pieces in irrelevant areas are ignored.
        """
        if self._opp_type == "R":
            if tile[0] == 0 or tile[0] == self._board_dim - 1:
                return True
        else:
            if tile[1] == 0 or tile[1] == self._board_dim - 1:
                return True

        for di, dj in HEX_NEIGHBORS:
            neighbor = (tile[0] + di, tile[1] + dj)
            if neighbor in env and env[neighbor].get_type() == self._opp_type:
                return True

        return False

    def _in_bounds(self, tile) -> bool:
        return 0 <= tile[0] < self._board_dim and 0 <= tile[1] < self._board_dim