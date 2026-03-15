import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from game_state_hex import GameStateHex

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]

_WIRE = 1e6
_SCALE = 0.5


class HexHeuristicResistance:
    """
    Evaluates a Hex board position using the Shannon electric resistance model.
    Positive score = good for us.  Negative = good for opponent.
    """

    def __init__(self, piece_type: str, board_dim: int = 14):
        self._piece_type      = piece_type
        self._opp_type        = "B" if piece_type == "R" else "R"
        self._board_dim       = board_dim
        self._n               = board_dim * board_dim
        self._my_rows  = (piece_type == "R")
        self._opp_rows = not self._my_rows

    def evaluate(self, state: GameStateHex) -> float:
        """Returns score in [-1, 1].  Positive = good for us."""
        if state.is_done():
            my_id = self._get_player_id(state, self._piece_type)
            return float('inf') if state.scores.get(my_id, 0) == 1 else float('-inf')

        env = state.rep.env
        my_pieces  = {t for t, p in env.items() if p.get_type() == self._piece_type}
        opp_pieces = {t for t, p in env.items() if p.get_type() == self._opp_type}

        r_me  = self._resistance(my_pieces,  opp_pieces, self._my_rows)
        r_opp = self._resistance(opp_pieces, my_pieces,  self._opp_rows)

        if r_opp <= 0.0 or r_opp == float('inf'):
            pass
        if r_me == float('inf') and r_opp == float('inf'):
            return 0.0
        if r_me == float('inf'):
            return -1.0
        if r_opp == float('inf'):
            return 1.0

        delta = r_opp - r_me
        return float(np.tanh(delta / _SCALE))

    def _resistance(
        self,
        own:  set,
        opp:  set,
        connects_rows: bool,
    ) -> float:
        dim  = self._board_dim
        N    = self._n

        r_idx  = []  
        c_idx  = [] 
        data   = []  
        b      = np.zeros(N, dtype=np.float64)

        def _add(r, c, val):
            r_idx.append(r); c_idx.append(c); data.append(float(val))

        def _node(i, j):
            return i * dim + j

        for i in range(dim):
            for j in range(dim):
                nd  = _node(i, j)
                pos = (i, j)

                if pos in opp:
                    _add(nd, nd, 1.0)
                    continue

                is_own = pos in own
                diag   = 0.0

                at_src = (i == 0)       if connects_rows else (j == 0)
                at_snk = (i == dim - 1) if connects_rows else (j == dim - 1)

                g_term = _WIRE if is_own else 1.0
                if at_src:
                    diag   += g_term
                    b[nd]  += g_term 
                if at_snk:
                    diag   += g_term 

                for di, dj in HEX_NEIGHBORS:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < dim and 0 <= nj < dim):
                        continue
                    nb = (ni, nj)
                    if nb in opp:
                        continue
                    g = _WIRE if (is_own or nb in own) else 1.0
                    diag += g
                    _add(nd, _node(ni, nj), -g)

                _add(nd, nd, diag if diag > 1e-12 else 1.0)

        A = sp.csr_matrix((data, (r_idx, c_idx)), shape=(N, N))

        try:
            v = spla.spsolve(A, b)
        except Exception:
            return float('inf')

        total_i = 0.0
        for k in range(dim):
            pos = (0, k) if connects_rows else (k, 0)
            if pos in opp:
                continue
            g   = _WIRE if pos in own else 1.0
            nd  = _node(pos[0], pos[1])
            v_n = v[nd] if np.isfinite(v[nd]) else 0.0
            total_i += g * (1.0 - v_n)

        return 1.0 / total_i if total_i > 1e-9 else float('inf')

    def _get_player_id(self, state: GameStateHex, piece_type: str):
        for player in state.players:
            if player.get_piece_type() == piece_type:
                return player.get_id()
        return None

HexHeuristicDjikstra = HexHeuristicResistance