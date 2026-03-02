from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_hex import GameStateHex

import math
import heapq
import time
from typing import Optional, Tuple


class MyPlayer(PlayerHex):
    """
    Minimax player with alpha-beta pruning and Dijkstra shortest-path heuristic for Hex.
    """

    def __init__(self, piece_type: str, name: str = "MinimaxPlayer"):
        super().__init__(piece_type, name)
        self._my_id = None
        self._my_type = None
        self._opp_type = None

    # =====================================================================
    # Public Interface
    # =====================================================================

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15 * 60, **kwargs) -> Action:
        """
        Main entry point: uses iterative deepening minimax to find best move.
        """
        self._my_id = self.get_id()
        self._my_type = self.get_piece_type()
        self._opp_type = "B" if self._my_type == "R" else "R"

        empty_cells = self._count_empty_cells(current_state)
        total_cells = current_state.get_rep().get_dimensions()[0] ** 2
        
        if empty_cells > total_cells * 0.9:
            time_budget = 5.0
        elif empty_cells < 20: 
            time_budget = 10.0
        else: 
            time_budget = 3.0

        return self._iterative_deepening(current_state, time_budget)

    # =====================================================================
    # Search Algorithm
    # =====================================================================

    def _iterative_deepening(self, state: GameStateHex, time_limit: float) -> Action:
        """
        Try increasing depths until time runs out, return best move found.
        """
        start_time = time.time()
        best_action = None
        best_value = -math.inf
        
        actions = list(state.get_possible_stateless_actions())
        if not actions:
            raise ValueError("No valid actions available")
        
        actions.sort(key=lambda a: self._move_priority(a, state))
        
        for depth in range(1, 20):
            if time.time() - start_time > time_limit * 0.85:
                break
            
            try:
                action, value = self._minimax(
                    state=state,
                    depth=depth,
                    is_maximizing=True,
                    alpha=-math.inf,
                    beta=math.inf,
                    start_time=start_time,
                    time_limit=time_limit
                )
                
                if action is not None:
                    best_action = action
                    best_value = value
                    
            except TimeoutError:
                break
        
        if best_action is None:
            best_action = actions[0]
        
        return best_action

    def _minimax(
        self,
        state: GameStateHex,
        depth: int,
        is_maximizing: bool,
        alpha: float,
        beta: float,
        start_time: float,
        time_limit: float
    ) -> Tuple[Optional[Action], float]:
        """
        Minimax with alpha-beta pruning.
        
        Returns: (best_action, value)
        """
        if time.time() - start_time > time_limit * 0.95:
            raise TimeoutError()
        
        if state.is_done():
            score = state.scores[self._my_id]
            if score > 0:
                return None, 1000.0 
            else:
                return None, -1000.0
        
        if depth == 0:
            return None, self._evaluate(state)
        
        actions = list(state.get_possible_stateless_actions())
        if not actions:
            return None, 0.0
        
        if depth >= 3:
            actions.sort(key=lambda a: self._move_priority(a, state))
        
        best_action = None
        
        if is_maximizing: 
            max_eval = -math.inf
            
            for action in actions:
                next_state = state.apply_action(action)
                _, eval = self._minimax(
                    next_state, depth - 1, False, alpha, beta,
                    start_time, time_limit
                )
                
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                
                alpha = max(alpha, eval)
                if beta <= alpha: 
                    break
            
            return best_action, max_eval
        
        else: 
            min_eval = math.inf
            
            for action in actions:
                next_state = state.apply_action(action)
                _, eval = self._minimax(
                    next_state, depth - 1, True, alpha, beta,
                    start_time, time_limit
                )
                
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            return best_action, min_eval

    # =====================================================================
    # Heuristic Evaluation (Dijkstra Shortest Path)
    # =====================================================================

    def _evaluate(self, state: GameStateHex) -> float:
        """
        Evaluate board state using Dijkstra shortest path.
        Positive = good for us, negative = good for opponent.
        """
        my_dist = self._shortest_path_length(state, self._my_type)
        opp_dist = self._shortest_path_length(state, self._opp_type)
        
        # Check for wins
        if my_dist == 0:
            return 1000.0
        if opp_dist == 0:
            return -1000.0
        
        dim = state.get_rep().get_dimensions()[0]
        normalized_diff = (opp_dist - my_dist) / dim
        
        return normalized_diff

    def _shortest_path_length(self, state: GameStateHex, piece_type: str) -> float:
        """
        Dijkstra's algorithm: find shortest path to connect player's edges.
        Cost: 0 for own pieces, 1 for empty cells, infinite for opponent pieces.
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        pq = []
        dist = {}
        
        if piece_type == "R": 
            for col in range(dim):
                pos = (0, col)
                cost = self._cell_cost(env, pos, piece_type)
                if cost < float('inf'):
                    heapq.heappush(pq, (cost, pos))
                    dist[pos] = cost
        
        else:
            for row in range(dim):
                pos = (row, 0)
                cost = self._cell_cost(env, pos, piece_type)
                if cost < float('inf'):
                    heapq.heappush(pq, (cost, pos))
                    dist[pos] = cost
        
        while pq:
            cost, (row, col) = heapq.heappop(pq)
            
            if piece_type == "R" and row == dim - 1:
                return cost
            if piece_type == "B" and col == dim - 1:
                return cost

            if dist.get((row, col), float('inf')) < cost:
                continue
            
            neighbors = state.get_neighbours(row, col)
            for _, (ptype, (nrow, ncol)) in neighbors.items():
                npos = (nrow, ncol)
                
                if ptype == piece_type:
                    new_cost = cost 
                elif ptype == "EMPTY":
                    new_cost = cost + 1
                else:
                    continue
                
                if new_cost < dist.get(npos, float('inf')):
                    dist[npos] = new_cost
                    heapq.heappush(pq, (new_cost, npos))
        
        return float('inf')

    def _cell_cost(self, env, pos: tuple, piece_type: str) -> float:
        """Cost to be at this cell (0 if we own it, 1 if empty, inf if blocked)."""
        piece = env.get(pos)
        if piece is None:
            return 1.0 
        if piece.get_type() == piece_type:
            return 0.0  
        return float('inf')  

    # =====================================================================
    # Helpers
    # =====================================================================

    def _count_empty_cells(self, state: GameStateHex) -> int:
        """Count empty cells on board."""
        return sum(1 for _ in state.get_rep().get_empty())

    def _move_priority(self, action: StatelessAction, state: GameStateHex) -> float:
        """
        Prioritize moves closer to center (better for alpha-beta pruning).
        Lower value = higher priority.
        """
        pos = action.data["position"]
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2
        
        dist = abs(pos[0] - center) + abs(pos[1] - center)
        return dist