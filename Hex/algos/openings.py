from typing import Optional, Tuple

# For a 14x14 board
BOARD_DIM = 14
CENTER = BOARD_DIM // 2  # 7

class Openings:
    def __init__(self, piece_type: str):
        self._piece_type = piece_type
        self._in_opening = True
        self._my_moves = []

    def get_opening_move(
        self,
        step: int,
        opponent_move: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:

        if not self._in_opening:
            return None

        # After step 3 hand off to minimax
        if step > 3:
            self._in_opening = False
            return None

        if self._piece_type == "R":
            # Red connects top to bottom — extend vertically through center
            # Move sequence: anchor center, then push toward top and bottom edges
            sequence = [
                (6, 7),   # center anchor
                (4, 7),   # push toward top
                (9, 7),   # push toward bottom
            ]
        else:
            # Blue connects left to right — extend horizontally through center
            sequence = [
                (7, 6),   # center anchor
                (7, 4),   # push toward left
                (7, 9),   # push toward right
            ]

        # Pick the next move in sequence that isn't already occupied
        for move in sequence:
            if move not in self._my_moves:
                self._my_moves.append(move)
                return move

        # Sequence exhausted
        self._in_opening = False
        return None

    @property
    def in_opening(self) -> bool:
        return self._in_opening