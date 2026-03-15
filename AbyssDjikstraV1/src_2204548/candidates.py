import math
from heapq import heappush, heappop
from game_state_hex import GameStateHex

HEX_NEIGHBORS  = [(-1, 1), (-1, 0), (1, -1), (1, 0), (0, -1), (0, 1)]
BRIDGE_OFFSETS = [(-1, 2), (1, 1), (2, -1), (1, -2), (-1, -1), (-2, 1)]


class CandidateGenerator:

    def __init__(self, piece_type: str, board_dim: int = 14, search_width: int = 6):
        self._piece_type   = piece_type
        self._opp_type     = "B" if piece_type == "R" else "R"
        self._board_dim    = board_dim
        self._search_width = search_width

    def get_candidates(self, state: GameStateHex, heuristic_fn, is_maximizing: bool):
        considered = self._get_considered_tiles(state)
        scored = []

        for action in state.generate_possible_stateless_actions():
            if considered and action.data["position"] not in considered:
                continue
            next_state = state.apply_action(action)
            h_val = heuristic_fn(next_state)
            scored.append((action, next_state, h_val))

        if not scored:
            for action in state.generate_possible_stateless_actions():
                next_state = state.apply_action(action)
                h_val = heuristic_fn(next_state)
                scored.append((action, next_state, h_val))

        scored.sort(key=lambda x: x[2], reverse=is_maximizing)
        return scored[:self._search_width]


    def _get_considered_tiles(self, state: GameStateHex) -> set:
        considered = set()
        env = state.rep.env

        my_tiles  = [t for t, p in env.items() if p.get_type() == self._piece_type]
        opp_tiles = [t for t, p in env.items() if p.get_type() == self._opp_type]


        for tile in my_tiles:
            # Ring 1
            for di, dj in HEX_NEIGHBORS:
                r1 = (tile[0] + di, tile[1] + dj)
                if self._in_bounds(r1) and r1 not in env:
                    considered.add(r1)

            for di, dj in HEX_NEIGHBORS:
                for di2, dj2 in HEX_NEIGHBORS:
                    r2 = (tile[0] + di + di2, tile[1] + dj + dj2)
                    if self._in_bounds(r2) and r2 not in env:
                        considered.add(r2)

            for bi, bj in BRIDGE_OFFSETS:
                b = (tile[0] + bi, tile[1] + bj)
                if self._in_bounds(b) and b not in env:
                    considered.add(b)
        for tile in opp_tiles:
            if not self._is_threatening(tile, env):
                continue

            # Ring 1
            for di, dj in HEX_NEIGHBORS:
                r1 = (tile[0] + di, tile[1] + dj)
                if self._in_bounds(r1) and r1 not in env:
                    considered.add(r1)

            for di, dj in HEX_NEIGHBORS:
                for di2, dj2 in HEX_NEIGHBORS:
                    r2 = (tile[0] + di + di2, tile[1] + dj + dj2)
                    if self._in_bounds(r2) and r2 not in env:
                        considered.add(r2)

            for bi, bj in BRIDGE_OFFSETS:
                b = (tile[0] + bi, tile[1] + bj)
                if self._in_bounds(b) and b not in env:
                    considered.add(b)

        considered |= self._get_shortest_path_cells(state, self._opp_type)

        center = (self._board_dim // 2, self._board_dim // 2)
        if not considered and center not in env:
            considered.add(center)

        return considered

    def _is_threatening(self, tile, env) -> bool:
        if self._opp_type == "R":
            if tile[0] == 0 or tile[0] == self._board_dim - 1:
                return True
        else:
            if tile[1] == 0 or tile[1] == self._board_dim - 1:
                return True

        for di, dj in HEX_NEIGHBORS:
            neighbor = (tile[0] + di, tile[1] + dj)
            if neighbor in env and env[neighbor].get_type() == self._opp_type:
                return True

        if self._opp_type == "R":
            if 3 <= tile[0] <= self._board_dim - 3:
                return True
        else:
            if 3 <= tile[1] <= self._board_dim - 3:
                return True

        return False

    def _get_shortest_path_cells(self, state: GameStateHex, piece_type: str) -> set:
        dim = self._board_dim
        env = state.rep.env
        dist = {}
        prev = {}
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

        goal = None
        while heap:
            d, (i, j) = heappop(heap)
            if d > dist.get((i, j), math.inf):
                continue
            if goal_check(i, j):
                goal = (i, j)
                break
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
                    prev[(ni, nj)] = (i, j)
                    heappush(heap, (new_dist, (ni, nj)))

        path_cells = set()
        if goal is None:
            return path_cells

        current = goal
        while current in prev:
            if env.get(current) is None:
                path_cells.add(current)
            current = prev[current]

        return path_cells

    def _in_bounds(self, tile) -> bool:
        return 0 <= tile[0] < self._board_dim and 0 <= tile[1] < self._board_dim