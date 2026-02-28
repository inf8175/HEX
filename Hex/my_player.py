import math
from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex

class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.max_depth = 3  # Set your desired search depth here

    def compute_action(self, current_state: GameStateHex, **kwargs) -> Action:
        """
        Entry point for the framework.
        """
        best_val = -math.inf
        best_action = None
     
        for action in current_state.generate_possible_stateless_actions():
            next_state = current_state.apply_action(action)
            
            value = self.minimax(next_state, self.max_depth - 1, -math.inf, math.inf, False)
            
            if value > best_val:
                best_val = value
                best_action = action
                
        return best_action

    def minimax(self, state, depth, alpha, beta, is_maximizing):
        """
        Recursive minimax with Alpha-Beta pruning.
        """
        # 1. Base Case: Check for terminal state (win/loss) or depth limit
        if state.is_done() or depth == 0:
            return self.heuristic(state)

        if is_maximizing:
            max_eval = -math.inf
            for action in state.generate_possible_stateless_actions():
                eval = self.minimax(state.apply_action(action), depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Prune
            return max_eval
        
        else:
            min_eval = math.inf
            for action in state.generate_possible_stateless_actions():
                eval = self.minimax(state.apply_action(action), depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Prune
            return min_eval

    def heuristic(self, state: GameStateHex) -> float:
        """
        Write your heuristic evaluation here. 
        A positive value favors you (self.piece_type), 
        a negative value favors the opponent.
        """
        if state.is_done():
            my_score = state.scores.get(self.get_id(), 0)
            if my_score == 1:
                return float('inf')
            return float(-inf)
            
        return 0.0