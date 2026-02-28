from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_hex import GameStateHex

import math
import random
import time


class MCTSNode:
    def __init__(self, state: GameStateHex, parent=None, action=None, player_id=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.N = 0
        self.U = 0.0
        self.player_id = player_id  # Player who made the move to reach this state
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = list(self.state.get_possible_stateless_actions())
            random.shuffle(self._untried_actions)
        return self._untried_actions

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.is_done()

    def ucb1(self, c=math.sqrt(2)):
        if self.N == 0:
            return float('inf')
        exploitation = self.U / self.N
        parent_N = self.parent.N if self.parent else 1
        exploration = c * math.sqrt(math.log(parent_N) / self.N)
        return exploitation + exploration

    def best_child(self, c=math.sqrt(2)):
        return max(self.children, key=lambda child: child.ucb1(c))

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.apply_action(action)
        child = MCTSNode(
            state=next_state, 
            parent=self, 
            action=action,
            player_id=self.state.get_active_player().get_id()
        )
        self.children.append(child)
        return child

    def best_action_node(self):
        return max(self.children, key=lambda child: child.N)


class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MCTSPlayer"):
        super().__init__(piece_type, name)
        self._root = None
        self._my_id = None
        self._first_move_time = 300
        self._normal_move_time = 0.5

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15 * 60, **kwargs) -> Action:
        self._my_id = self.get_id()
        
        # Tree reuse: find current state in existing tree
        if self._root is not None:
            found = False
            for child in self._root.children:
                if self._states_equal(child.state, current_state):
                    self._root = child
                    self._root.parent = None  # Detach to save memory
                    found = True
                    break
            if not found:
                self._root = None
        
        # Create new root if needed
        if self._root is None:
            self._root = MCTSNode(state=current_state, player_id=None)
        
        # Determine time budget for this move
        step = current_state.get_step()
        if step <= 2:  # First two moves (Red plays, then Blue responds)
            time_budget = min(self._first_move_time, remaining_time * 0.6)
        else:
            time_budget = min(self._normal_move_time, remaining_time / 20)
        
        # Run MCTS for allocated time
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < time_budget - 0.05:  # Safety margin
            node = self._select(self._root)
            if not node.is_terminal():
                node = node.expand()
            result = self._simulate(node)
            self._backpropagate(node, result)
            iterations += 1
        
        print(f"Move {step}: {iterations} iterations in {time.time() - start_time:.2f}s, tree size: {self._tree_size(self._root)}")
        
        if not self._root.children:
            actions = list(current_state.get_possible_stateless_actions())
            return random.choice(actions)
        
        best_child = self._root.best_action_node()
        self._root = best_child  # Set up for next move reuse
        self._root.parent = None
        return best_child.action
    
    def _states_equal(self, s1: GameStateHex, s2: GameStateHex) -> bool:
        if s1.get_step() != s2.get_step():
            return False
        return s1.get_rep().get_env() == s2.get_rep().get_env()

    def _tree_size(self, node):
        """Count nodes in tree (for debugging)"""
        if node is None:
            return 0
        count = 1
        for child in node.children:
            count += self._tree_size(child)
        return count

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def _simulate(self, node: MCTSNode) -> float:
        state = node.state
        my_id = self._my_id
        steps = 0
        
        while not state.is_done():
            actions = list(state.get_possible_stateless_actions())
            if not actions:
                break
            
            action = random.choice(actions)
            
            state = state.apply_action(action)
            steps += 1
        
        scores = state.scores
        if scores[my_id] > 0:
            return 1.0
        else:
            return 0.0

    def _backpropagate(self, node: MCTSNode, result: float):
        while node is not None:
            node.N += 1
            if node.player_id is not None:
                if node.player_id == self._my_id:
                    node.U += result
                else:
                    node.U += (1.0 - result)
            node = node.parent