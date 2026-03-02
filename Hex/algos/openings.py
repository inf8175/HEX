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
            sequence = [
                (6, 7),
                (7, 7),
            ]
        else:
            sequence = [
                (7, 6),  
                (7, 7),   
            ]

        for move in sequence:
            if move not in self._my_moves:
                self._my_moves.append(move)
                return move

        self._in_opening = False
        return None

    @property
    def in_opening(self) -> bool:
        return self._in_opening