import math
from heapq import heappush, heappop
from game_state_hex import GameStateHex

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]


class HexHeuristicDjikstra:
    """
    Evaluates a Hex board position using Dijkstra shortest path.
    Positive = good for us. Negative = good for opponent.
    """

    def __init__(self, piece_type: str, board_dim: int = 14):
        self._piece_type = piece_type
        self._opp_type = "B" if piece_type == "R" else "R"
        self._board_dim = board_dim

    def evaluate(self, state: GameStateHex) -> float:
        """Returns score in [-inf, inf]. Positive = good for us."""
        if state.is_done():
            my_id = self._get_player_id(state, self._piece_type)
            return float('inf') if state.scores.get(my_id, 0) == 1 else float('-inf')

        my_dist = self._dijkstra(state, self._piece_type)
        opp_dist = self._dijkstra(state, self._opp_type)

        # Urgent threat detection
        if opp_dist <= 1:
            return -0.9
        
        if my_dist <= 1:
            return 0.9

        # Normal case: path differential
        if my_dist == math.inf and opp_dist == math.inf:
            return 0.0
        if my_dist == math.inf:
            return -1.0
        if opp_dist == math.inf:
            return 1.0

        diff = opp_dist - my_dist
        return max(-1.0, min(1.0, diff / self._board_dim))

    def _dijkstra(self, state: GameStateHex, piece_type: str) -> float:
        """Shortest path using Dijkstra. Own pieces cost 0, empty cost 1, opponent = blocked."""
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
            cost = 0 if cell is not None else 1
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
                step = 0 if cell is not None else 1
                new_dist = d + step
                if new_dist < dist.get((ni, nj), math.inf):
                    dist[(ni, nj)] = new_dist
                    heappush(heap, (new_dist, (ni, nj)))

        return math.inf

    def _get_player_id(self, state: GameStateHex, piece_type: str):
        for player in state.players:
            if player.get_piece_type() == piece_type:
                return player.get_id()
        return None