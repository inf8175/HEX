from typing import Optional, Tuple, List

BOARD_DIM    = 14
CENTER       = BOARD_DIM // 2       
OPENING_DEPTH = 4          

RED_OPENINGS: List[Tuple[int, int]] = [
    (7, 7), 
    (4, 7), 
    (9, 7),
    (2, 4),
    (2, 9),
]

BLUE_OPENINGS: List[Tuple[int, int]] = [
    (7, 7),
    (7, 4),  
    (7, 9), 
    (4, 2), 
    (9, 2),  
]


def _obtuse_mirror(move: Tuple[int, int], dim: int = BOARD_DIM) -> Tuple[int, int]:
    return (dim - 1 - move[0], dim - 1 - move[1])


class Openings:
    _CENTRE_ZONE = range(CENTER - 2, CENTER + 3)

    def __init__(self, piece_type: str):
        self._piece_type   = piece_type
        self._in_opening   = True
        self._my_moves:    List[Tuple[int, int]] = []
        self._sequence:    List[Tuple[int, int]] = (
            RED_OPENINGS if piece_type == "R" else BLUE_OPENINGS
        )
        self._responded_to_opp = False  

    def get_opening_move(
        self,
        step:          int,
        opponent_move: Optional[Tuple[int, int]] = None,
        occupied:      Optional[set]             = None,
    ) -> Optional[Tuple[int, int]]:
        if not self._in_opening:
            return None

        if step > OPENING_DEPTH:
            self._in_opening = False
            return None

        occupied = occupied or set()
        if (
            opponent_move is not None
            and not self._responded_to_opp
            and self._is_centre_zone(opponent_move)
        ):
            self._responded_to_opp = True
            mirror = _obtuse_mirror(opponent_move)
            if mirror not in occupied and mirror not in self._my_moves:
                self._my_moves.append(mirror)
                return mirror

        for move in self._sequence:
            if move not in self._my_moves and move not in occupied:
                self._my_moves.append(move)
                return move

        self._in_opening = False
        return None

    @property
    def in_opening(self) -> bool:
        return self._in_opening

    def _is_centre_zone(self, move: Tuple[int, int]) -> bool:
        """True if the move is in the strong central zone (rows/cols 5–9)."""
        return move[0] in self._CENTRE_ZONE and move[1] in self._CENTRE_ZONE