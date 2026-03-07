"""
Zobrist hashing + Transposition Table for Hex AI.

Design
------
- Two 14×14 random bitfields: one per piece colour (R, B).
- Hash updated incrementally: hash ^= TABLE[colour][row][col]
- Transposition Table stores (value, depth, flag) per hash.
  flag ∈ {EXACT, LOWER, UPPER}  (for alpha-beta cut-off reuse)

Usage
-----
    zt = ZobristTable(board_dim=14)
    h  = zt.hash_state(state)          # full board hash (slow, use once)
    h  = zt.update(h, "R", (3, 7))    # incremental update (XOR in/out)

    tt = TranspositionTable()
    tt.store(h, value=0.42, depth=3, flag=TT_EXACT)
    entry = tt.lookup(h)               # None | TTEntry
"""

import random
from typing import Optional
from dataclasses import dataclass

# ------------------------------------------------------------------ #
#  Transposition table flags                                          #
# ------------------------------------------------------------------ #
TT_EXACT = 0   # Exact minimax value
TT_LOWER = 1   # Alpha cut-off  (value is a lower bound)
TT_UPPER = 2   # Beta  cut-off  (value is an upper bound)


@dataclass
class TTEntry:
    value: float
    depth: int
    flag:  int   # TT_EXACT | TT_LOWER | TT_UPPER


# ------------------------------------------------------------------ #
#  Zobrist Table                                                       #
# ------------------------------------------------------------------ #
class ZobristTable:
    """
    Pre-computes a random 64-bit integer for every (colour, row, col) triple.
    Thread-safe if constructed once and only read thereafter.
    """

    _COLOUR_INDEX = {"R": 0, "B": 1}

    def __init__(self, board_dim: int = 14, seed: int = 0xDEADBEEF):
        self._dim = board_dim
        rng = random.Random(seed)

        # table[colour_idx][row][col] → random 64-bit int
        self._table: list[list[list[int]]] = [
            [
                [rng.getrandbits(64) for _ in range(board_dim)]
                for _ in range(board_dim)
            ]
            for _ in range(2)          # 0 = R, 1 = B
        ]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def hash_state(self, state) -> int:
        """
        Compute a full hash for a game state from scratch.
        O(pieces) — call once per root, then use update() incrementally.
        """
        h = 0
        for (row, col), piece in state.rep.env.items():
            h ^= self._entry(piece.get_type(), row, col)
        return h

    def update(self, current_hash: int, colour: str, pos: tuple) -> int:
        """
        Incrementally toggle a cell in the hash (XOR in or out).
        Call once when a piece is *placed* (or *removed* — same operation).
        """
        row, col = pos
        return current_hash ^ self._entry(colour, row, col)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _entry(self, colour: str, row: int, col: int) -> int:
        return self._table[self._COLOUR_INDEX[colour]][row][col]


# ------------------------------------------------------------------ #
#  Transposition Table                                                 #
# ------------------------------------------------------------------ #
class TranspositionTable:
    """
    Fixed-size transposition table using a Python dict.

    Replacement policy: always-replace — a new entry always overwrites
    an older entry at the same hash (cheap, works well in practice).

    The table is intentionally *not* cleared between moves so that
    search results from previous iterations of IDA benefit the next.
    """

    def __init__(self, max_size: int = 1_000_000):
        self._max_size = max_size
        self._table: dict[int, TTEntry] = {}

    def store(self, h: int, value: float, depth: int, flag: int) -> None:
        """
        Store or overwrite an entry.
        Replacement policy: prefer deeper entries, evict if at capacity.
        """
        existing = self._table.get(h)
        if existing is not None and existing.depth > depth:
            return  # Keep the deeper (more accurate) entry

        if len(self._table) >= self._max_size:
            # Simple eviction: drop ~10 % of entries (random sample removal)
            keys_to_drop = list(self._table.keys())[:self._max_size // 10]
            for k in keys_to_drop:
                del self._table[k]

        self._table[h] = TTEntry(value=value, depth=depth, flag=flag)

    def lookup(self, h: int) -> Optional[TTEntry]:
        """Return the stored entry or None."""
        return self._table.get(h)

    def clear(self) -> None:
        self._table.clear()

    def __len__(self) -> int:
        return len(self._table)