"""
Opening table for 14x14 Hex.

Structure:
- FIRST_MOVE: what to play when you go first (step 0)
- RESPONSES: best response to opponent's move when you go second (step 1)
- FOLLOWUPS: best follow-up after the first exchange (step 2 and 3)

Board orientation:
- Red connects top (row 0) to bottom (row 13)
- Blue connects left (col 0) to right (col 13)
- Center is (6,7) or (7,6)

Regions (for 14x14):
- Center:      rows 5-8, cols 5-8
- Near-center: rows 4-9, cols 4-9
- Edge:        row/col 0-2 or 11-13
- Corner:      corners of the board
"""

from typing import Optional, Tuple

# ------------------------------------------------------------------ #
#  FIRST MOVE (you go first, step 0)                                  #
#  Always aim for the center of the board.                            #
# ------------------------------------------------------------------ #

FIRST_MOVE = (6, 7)  # Near-center, slightly biased toward your winning axis


# ------------------------------------------------------------------ #
#  RESPONSES (you go second, step 1)                                  #
#  Maps opponent's first move → your best response.                   #
#                                                                     #
#  Strategy:                                                          #
#  - Opponent takes center → play adjacent to contest it             #
#  - Opponent plays near-center → take center                        #
#  - Opponent plays edge → take center, ignore their edge piece      #
#  - Opponent plays corner → take center, corners are weak           #
# ------------------------------------------------------------------ #

def get_response(opponent_move: Tuple[int, int], my_piece_type: str) -> Optional[Tuple[int, int]]:
    """
    Get the best response to the opponent's first move.
    Returns None if no specific response is defined (fall back to minimax).
    """
    r, c = opponent_move

    # --- Opponent takes dead center ---
    if (r, c) == (6, 7):
        return (7, 6)
    if (r, c) == (7, 6):
        return (6, 7)
    if (r, c) == (6, 6):
        return (6, 7)
    if (r, c) == (7, 7):
        return (6, 7)

    # --- Opponent plays near-center (ring 1-2 around center) ---
    # Take center immediately
    if 5 <= r <= 8 and 5 <= c <= 8:
        return (6, 7)

    # --- Opponent plays the extended center zone (ring 3) ---
    if 4 <= r <= 9 and 4 <= c <= 9:
        return (6, 7)

    # --- Opponent plays an edge (row 0, row 13, col 0, col 13) ---
    # Edge moves are generally weak openings — take the center
    if r <= 1 or r >= 12:
        return (6, 7)
    if c <= 1 or c >= 12:
        return (6, 7)

    # --- Opponent plays a corner region ---
    if (r <= 2 and c <= 2) or (r <= 2 and c >= 11):
        return (6, 7)
    if (r >= 11 and c <= 2) or (r >= 11 and c >= 11):
        return (6, 7)

    # --- Opponent plays the left or right flanks (mid-rows, far columns) ---
    # For Red (top-bottom): flank columns are less relevant, take center
    # For Blue (left-right): flank rows are less relevant, take center
    if my_piece_type == "R":
        if c <= 2 or c >= 11:
            return (6, 7)
        # Opponent is in a relevant row zone — play adjacent to contest
        if abs(r - 6) <= 2:
            return (6, 7) if (r, c) != (6, 7) else (7, 6)
    else:
        if r <= 2 or r >= 11:
            return (6, 7)
        if abs(c - 7) <= 2:
            return (6, 7) if (r, c) != (6, 7) else (7, 6)

    # Default: take center
    return (6, 7)


# ------------------------------------------------------------------ #
#  FOLLOWUPS (step 2 and 3 — after first exchange)                    #
#  Maps (my_last_move, opponent_last_move) → next move               #
#                                                                     #
#  These are called for steps 2 and 3 only.                          #
#  After step 3 hand off to minimax — position is complex enough.    #
# ------------------------------------------------------------------ #

# Format: (my_move, opponent_response) -> followup
FOLLOWUPS = {
    # I played (6,7), opponent plays adjacent → extend toward goal
    ((6, 7), (7, 6)): (5, 7),
    ((6, 7), (5, 7)): (7, 6),
    ((6, 7), (6, 6)): (7, 7),
    ((6, 7), (7, 7)): (5, 7),
    ((6, 7), (5, 6)): (7, 7),
    ((6, 7), (8, 6)): (5, 7),
    ((6, 7), (6, 8)): (7, 6),
    ((6, 7), (5, 8)): (7, 6),

    # I played (7,6), opponent plays adjacent → extend
    ((7, 6), (6, 7)): (8, 5),
    ((7, 6), (8, 5)): (6, 7),
    ((7, 6), (6, 6)): (8, 6),
    ((7, 6), (8, 6)): (6, 6),
    ((7, 6), (7, 7)): (6, 6),
    ((7, 6), (6, 5)): (8, 6),

    # I played (6,7), opponent plays far → stay aggressive
    ((6, 7), (3, 3)): (7, 6),
    ((6, 7), (3, 10)): (7, 6),
    ((6, 7), (10, 3)): (5, 7),
    ((6, 7), (10, 10)): (5, 7),
    ((6, 7), (0, 6)): (7, 6),
    ((6, 7), (13, 6)): (5, 7),
    ((6, 7), (6, 0)): (7, 7),
    ((6, 7), (6, 13)): (7, 6),
}


def get_followup(my_move: Tuple[int, int], opponent_move: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Get the follow-up move after the first exchange.
    Returns None if no specific followup is defined (fall back to minimax).
    """
    return FOLLOWUPS.get((my_move, opponent_move), None)


# ------------------------------------------------------------------ #
#  MAIN INTERFACE                                                      #
# ------------------------------------------------------------------ #

class Openings:
    """
    Manages the opening sequence for the first few moves.
    Tracks move history to enable multi-move opening sequences.
    """

    def __init__(self, piece_type: str):
        self._piece_type   = piece_type
        self._my_last_move = None
        self._move_count   = 0
        self._in_opening   = True

    def get_opening_move(
        self,
        step: int,
        opponent_move: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Returns the opening move for the current step, or None to hand off to minimax.

        Args:
            step:          Current game step (0-indexed total moves).
            opponent_move: The opponent's last move (None on step 0).

        Returns:
            A move tuple, or None if the opening is over.
        """
        if not self._in_opening:
            return None

        # Step 0: we go first
        if step == 0:
            move = FIRST_MOVE
            self._my_last_move = move
            self._move_count += 1
            return move

        # Step 1: we go second, respond to opponent's first move
        if step == 1 and opponent_move is not None:
            move = get_response(opponent_move, self._piece_type)
            if move is None:
                self._in_opening = False
                return None
            self._my_last_move = move
            self._move_count += 1
            return move

        # Steps 2-3: follow up if we have a defined response
        if step in (2, 3) and opponent_move is not None and self._my_last_move is not None:
            move = get_followup(self._my_last_move, opponent_move)
            if move is None:
                self._in_opening = False
                return None
            self._my_last_move = move
            self._move_count += 1
            return move

        # Beyond step 3: hand off to minimax
        self._in_opening = False
        return None

    @property
    def in_opening(self) -> bool:
        return self._in_opening