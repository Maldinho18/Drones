from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        num_agentes = state.get_num_agents()
        
        def terminal_profundidad(s: GameState, d: int) -> bool:
            return d == 0 or s.is_win() or s.is_lose()
        def valor(s: GameState, agente_index: int, d:int) -> float:
            if terminal_profundidad(s, d):
                return float(self.evaluation_function(s))
            
            acciones = s.get_legal_actions(agente_index)
            if not acciones:
                return float(self.evaluation_function(s))
            
            if agente_index == 0:
                v = float("-inf")
                for a in acciones:
                    succ = s.generate_successor(agente_index, a)
                    siguiente = (agente_index + 1) % num_agentes
                    siguiente_d = d - 1 if siguiente == 0 else d
                    v = max(v, valor(succ, siguiente, siguiente_d))
                return v
            
            v = float("inf")
            for a in acciones:
                succ = s.generate_successor(agente_index, a)
                siguiente = (agente_index + 1) % num_agentes
                siguiente_d = d - 1 if siguiente == 0 else d
                v = min(v, valor(succ, siguiente, siguiente_d))
            return v
        
        mejor_accion = None
        mejor_valor = float("-inf")
        
        for a in state.get_legal_actions(0):
            succ = state.generate_successor(0, a)
            siguiente = 1 % num_agentes
            siguiente_d = self.depth - 1 if siguiente == 0 else self.depth
            v = valor(succ, siguiente, siguiente_d)
            if v > mejor_valor:
                mejor_valor = v
                mejor_accion = a
        return mejor_accion
            


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        num_agentes = state.get_num_agents()

        def terminal_profundidad(s: GameState, d: int) -> bool:
            return d == 0 or s.is_win() or s.is_lose()

        def valor(s: GameState, agente_index: int, d: int, alpha: float, beta: float) -> float:

            if terminal_profundidad(s, d):
                return float(self.evaluation_function(s))

            acciones = s.get_legal_actions(agente_index)
            if not acciones:
                return float(self.evaluation_function(s))

            siguiente = (agente_index + 1) % num_agentes

    
            if agente_index == 0:
                v = float("-inf")

                for a in acciones:
                    succ = s.generate_successor(agente_index, a)
                    siguiente_d = d - 1 if siguiente == 0 else d

                    v = max(v, valor(succ, siguiente, siguiente_d, alpha, beta))

                    if v > beta:  
                        return v

                    alpha = max(alpha, v)

                return v

           
            v = float("inf")

            for a in acciones:
                succ = s.generate_successor(agente_index, a)
                siguiente_d = d - 1 if siguiente == 0 else d

                v = min(v, valor(succ, siguiente, siguiente_d, alpha, beta))

                if v < alpha: 
                    return v

                beta = min(beta, v)

            return v

        mejor_accion = None
        mejor_valor = float("-inf")

        alpha = float("-inf")
        beta = float("inf")

        for a in state.get_legal_actions(0):
            succ = state.generate_successor(0, a)

            siguiente = 1 % num_agentes
            siguiente_d = self.depth - 1 if siguiente == 0 else self.depth

            v = valor(succ, siguiente, siguiente_d, alpha, beta)

            if v > mejor_valor:
                mejor_valor = v
                mejor_accion = a

            alpha = max(alpha, mejor_valor)

        return mejor_accion


class ExpectimaxAgent(MultiAgentSearchAgent):

    def get_action(self, state: GameState) -> Directions | None:
        
        num_agents = state.get_num_agents()
        p = self.prob  # probabilidad de random uniforme en hunters

        def terminal_or_depth(s: GameState, d: int) -> bool:
            return d == 0 or s.is_win() or s.is_lose()

        def value(s: GameState, agent_index: int, d: int) -> float:
            if terminal_or_depth(s, d):
                return float(self.evaluation_function(s))

            actions = s.get_legal_actions(agent_index)
            if not actions:
                return float(self.evaluation_function(s))

            if agent_index == 0:
                v = float("-inf")
                for a in actions:
                    succ = s.generate_successor(agent_index, a)
                    next_agent = (agent_index + 1) % num_agents
                    next_d = d - 1 if next_agent == 0 else d
                    v = max(v, value(succ, next_agent, next_d))
                return v

            child_vals = []
            for a in actions:
                succ = s.generate_successor(agent_index, a)
                next_agent = (agent_index + 1) % num_agents
                next_d = d - 1 if next_agent == 0 else d
                child_vals.append(value(succ, next_agent, next_d))

            worst_case = min(child_vals)
            mean_case = sum(child_vals) / len(child_vals)
            return (1.0 - p) * worst_case + p * mean_case

        best_action = None
        best_val = float("-inf")

        for a in state.get_legal_actions(0):
            succ = state.generate_successor(0, a)
            next_agent = 1 % num_agents
            next_d = self.depth - 1 if next_agent == 0 else self.depth
            v = value(succ, next_agent, next_d)

            if v > best_val:
                best_val = v
                best_action = a

        return best_action
        
