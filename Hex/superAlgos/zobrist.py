import random
from typing import Optional
from dataclasses import dataclass

TT_EXACT = 0  
TT_LOWER = 1 
TT_UPPER = 2


@dataclass
class TTEntry:
    value: float
    depth: int
    flag:  int  


class ZobristTable:

    _COLOUR_INDEX = {"R": 0, "B": 1}

    def __init__(self, board_dim: int = 14, seed: int = 0xDEADBEEF):
        self._dim = board_dim
        rng = random.Random(seed)

        self._table: list[list[list[int]]] = [
            [
                [rng.getrandbits(64) for _ in range(board_dim)]
                for _ in range(board_dim)
            ]
            for _ in range(2)        
        ]

    def hash_state(self, state) -> int:
        h = 0
        for (row, col), piece in state.rep.env.items():
            h ^= self._entry(piece.get_type(), row, col)
        return h

    def update(self, current_hash: int, colour: str, pos: tuple) -> int:
        row, col = pos
        return current_hash ^ self._entry(colour, row, col)

    def _entry(self, colour: str, row: int, col: int) -> int:
        return self._table[self._COLOUR_INDEX[colour]][row][col]


class TranspositionTable:

    def __init__(self, max_size: int = 1_000_000):
        self._max_size = max_size
        self._table: dict[int, TTEntry] = {}

    def store(self, h: int, value: float, depth: int, flag: int) -> None:
        existing = self._table.get(h)
        if existing is not None and existing.depth > depth:
            return 

        if len(self._table) >= self._max_size:
            keys_to_drop = list(self._table.keys())[:self._max_size // 10]
            for k in keys_to_drop:
                del self._table[k]

        self._table[h] = TTEntry(value=value, depth=depth, flag=flag)

    def lookup(self, h: int) -> Optional[TTEntry]:
        return self._table.get(h)

    def clear(self) -> None:
        self._table.clear()

    def __len__(self) -> int:
        return len(self._table)