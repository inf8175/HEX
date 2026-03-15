import math
from heapq import heappush, heappop
from game_state_hex import GameStateHex
import numpy as np

HEX_NEIGHBORS = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]


class HexHeuristicDjikstra:

    def __init__(self, piece_type: str, board_dim: int = 14):
        self._piece_type = piece_type
        self._opp_type   = "B" if piece_type == "R" else "R"
        self._board_dim  = board_dim

    def evaluate(self, state: GameStateHex) -> float:
        if state.is_done():
            my_id = self._get_player_id(state, self._piece_type)
            return float('inf') if state.scores.get(my_id, 0) == 1 else float('-inf')

        my_dist  = self._dijkstra(state, self._piece_type)
        opp_dist = self._dijkstra(state, self._opp_type)

        if opp_dist <= 1:
            return -0.9

        if my_dist <= 1:
            return 0.9

        if my_dist == math.inf and opp_dist == math.inf:
            return 0.0
        if my_dist == math.inf:
            return -1.0
        if opp_dist == math.inf:
            return 1.0

        diff = opp_dist - my_dist
        return max(-1.0, min(1.0, diff / self._board_dim))


    def _dijkstra(self, state: GameStateHex, piece_type: str) -> float:
        dim = self._board_dim
        env = state.rep.env
        dist = {}
        heap = []

        if piece_type == "R":
            seeds      = [(0, j) for j in range(dim)]
            goal_check = lambda i, j: i == dim - 1
        else:
            seeds      = [(i, 0) for i in range(dim)]
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
    
class HexHeuristicHSearch(HexHeuristicDjikstra):
    def _get_2nd_order_connections(self, state: GameStateHex, piece_type: str):
        dim = self._board_dim
        env = state.rep.env
        bridges = {}

        bridge_offsets = [
            ((0, 1), (1, 0), (1, 1)),  
            ((1, 0), (1, -1), (0, -1)),
        ]

        return bridges

    def _dijkstra(self, state: GameStateHex, piece_type: str) -> float:
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
                if 0 <= ni < dim and 0 <= nj < dim:
                    cell = env.get((ni, nj))
                    if cell is not None and cell.get_type() != piece_type:
                        continue
                    
                    step = 0 if (cell is not None) else 1
                    
                    if d + step < dist.get((ni, nj), math.inf):
                        dist[(ni, nj)] = d + step
                        heappush(heap, (d + step, (ni, nj)))

            bridge_jumps = [(-1, 2), (1, 1), (2, -1), (1, -2), (-1, -1), (-2, 1)]
            for dj_i, dj_j in bridge_jumps:
                target = (i + dj_i, j + dj_j)
                if 0 <= target[0] < dim and 0 <= target[1] < dim:
                    target_cell = env.get(target)
                    if target_cell is not None and target_cell.get_type() != piece_type:
                        continue
                    
                    bridge_step = 0.2 if target_cell is None else 0.0
                    
                    if d + bridge_step < dist.get(target, math.inf):
                        dist[target] = d + bridge_step
                        heappush(heap, (d + bridge_step, target))

        return math.inf