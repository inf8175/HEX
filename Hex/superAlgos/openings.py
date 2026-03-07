"""
Opening book for 14×14 Hex, grounded in Hex theory.

Theory primer
-------------
- Red connects top↔bottom (row axis).   Blue connects left↔right (col axis).
- The centre cell (7,7) is the single strongest opening move on any board.
- The "acute corner" opening (e.g. (1,2) or (2,1) for Red) yields a fast
  edge-template that is very hard to stop — standard pro play.
- The "obtuse corner" region (e.g. (2,11) or (11,2)) is also strong.
- "Ladder escapes" near the edges (row/col 1-2 or 11-12) are valuable.
- Responding to a centre stone: play the opposite obtuse corner (mirror).
- Swap rule awareness: moves that are too strong invite a swap, so opening
  variety matters.

Red  (top ↔ bottom): prioritises rows.   Goal: connect row 0 → row 13.
Blue (left ↔ right): prioritises cols.   Goal: connect col 0 → col 13.

Opening sequences are expressed as lists of (row, col) moves.
The player plays them in order, skipping any that are already occupied.
After the book is exhausted (or after step > OPENING_DEPTH), hand off
to minimax.
"""

from typing import Optional, Tuple, List

BOARD_DIM    = 14
CENTER       = BOARD_DIM // 2          # 7
OPENING_DEPTH = 4                       # hand off after this many total steps


# ------------------------------------------------------------------ #
#  Theory-based opening sequences (own moves only)                    #
#  Each list: preferred first move, then preferred second move.       #
#  We try each in order and play the first unoccupied one.            #
# ------------------------------------------------------------------ #

# Red wants to build a "bridge" across the short dimension (rows).
# Best first moves (in priority order):
#   1. True centre (7,7) — universally strongest
#   2. Acute-corner ladder escape: (2,4) or (4,2)
#   3. Second-row edge template: (1,6)
RED_OPENINGS: List[Tuple[int, int]] = [
    (7, 7),   # centre — strongest single cell
    (4, 7),   # centre column, upper half — fast bridge toward row 0
    (9, 7),   # centre column, lower half — symmetric coverage
    (2, 4),   # acute corner template (top-left)
    (2, 9),   # acute corner template (top-right)
]

# Blue wants to bridge left↔right (columns).
BLUE_OPENINGS: List[Tuple[int, int]] = [
    (7, 7),   # centre
    (7, 4),   # centre row, left half
    (7, 9),   # centre row, right half
    (4, 2),   # acute corner template (top-left)
    (9, 2),   # acute corner template (bottom-left)
]

# ------------------------------------------------------------------ #
#  Swap responses                                                     #
#  If the opponent played near the centre we counter at the "obtuse   #
#  corner" diagonally opposite — strong standard response.            #
# ------------------------------------------------------------------ #
# Mapping: opponent_first_move → our_response
# Generated from Hex theory: respond to acute-corner openings by
# occupying the opposite obtuse corner cluster.

def _obtuse_mirror(move: Tuple[int, int], dim: int = BOARD_DIM) -> Tuple[int, int]:
    """Reflect a move through the centre of the board."""
    return (dim - 1 - move[0], dim - 1 - move[1])


# ------------------------------------------------------------------ #
#  Openings class                                                     #
# ------------------------------------------------------------------ #

class Openings:
    """
    Theory-based opening book for 14×14 Hex.

    Behaviour
    ---------
    - Plays a fixed sequence of strong opening moves (own moves only).
    - Aware of the opponent's first move: if they played near the centre
      or in a strong zone, respond at the "obtuse corner mirror."
    - After OPENING_DEPTH total board steps, always returns None
      (hands off to minimax).
    - Returns None if the desired cell is occupied (caller must handle).
    """

    # Zone definitions (inclusive) for "threatening centre" detection
    _CENTRE_ZONE = range(CENTER - 2, CENTER + 3)   # rows/cols 5..9

    def __init__(self, piece_type: str):
        self._piece_type   = piece_type
        self._in_opening   = True
        self._my_moves:    List[Tuple[int, int]] = []
        self._sequence:    List[Tuple[int, int]] = (
            RED_OPENINGS if piece_type == "R" else BLUE_OPENINGS
        )
        self._responded_to_opp = False   # did we already give a swap reply?

    # ------------------------------------------------------------------ #
    #  Public                                                             #
    # ------------------------------------------------------------------ #

    def get_opening_move(
        self,
        step:          int,
        opponent_move: Optional[Tuple[int, int]] = None,
        occupied:      Optional[set]             = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Return the next opening move, or None to signal hand-off to minimax.

        Args:
            step:          Current step number (total plies on the board).
            opponent_move: The last move the opponent made (if known).
            occupied:      Set of currently occupied cells (to skip conflicts).
        """
        if not self._in_opening:
            return None

        if step > OPENING_DEPTH:
            self._in_opening = False
            return None

        occupied = occupied or set()

        # --- Swap-aware response ---
        # If opponent just played in the strong centre zone and we haven't
        # responded with a counter-corner yet, do so now.
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

        # --- Standard sequence ---
        for move in self._sequence:
            if move not in self._my_moves and move not in occupied:
                self._my_moves.append(move)
                return move

        # Sequence exhausted
        self._in_opening = False
        return None

    @property
    def in_opening(self) -> bool:
        return self._in_opening

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _is_centre_zone(self, move: Tuple[int, int]) -> bool:
        """True if the move is in the strong central zone (rows/cols 5–9)."""
        return move[0] in self._CENTRE_ZONE and move[1] in self._CENTRE_ZONE