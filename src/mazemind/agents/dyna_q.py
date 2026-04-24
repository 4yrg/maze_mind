"""Dyna-Q agent: model-based RL with planning from simulated experiences (Prioritized Sweeping).

Optimization Algorithm Details:
-------------------------------
This agent utilizes the **Prioritized Sweeping** optimization algorithm. 

Standard Dyna-Q relies on indiscriminately extracting simulated experiences out of an ever-growing transition model via uniform random sampling. This evaluates states inherently unassociated with critical value changes (a massive waste of computation).
Instead of uniform sampling, this codebase has been refactored mathematically:
1. Dynamic priority assessments track structural Bellman errors (absolute magnitude difference between projected target rewards and the current Q-table estimates).
2. Any calculated transition yielding a priority above threshold `theta` logs securely to Python's minimized `heapq` queue.
3. Once an initial transition is updated, traversing back propagates algorithmic updates toward their verified chronological predecessor coordinates dynamically stored in the array logic. Thus, propagating value iterations smartly inward only across known pathways drastically shrinking planning horizons inside complex architectures.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
import numpy as np

from mazemind.agents.base_agent import BaseAgent


class DynaQAgent(BaseAgent):
    def __init__(
        self,
        n_states: int = 256,
        n_actions: int = 4,
        n_planning_steps: int = 10,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        theta: float = 1e-4,
    ):
        super().__init__(n_states, n_actions, epsilon, epsilon_min, epsilon_decay)
        self.n_planning_steps = n_planning_steps
        self.theta = theta
        self.model: dict[tuple[int, int], tuple[float, int, bool]] = {}
        self.pq = []
        self.predecessors = defaultdict(set)

    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state_index: int,
        alpha: float,
        gamma: float,
        done: bool,
        **kwargs,
    ) -> None:
        self.model[(state_index, action)] = (reward, next_state_index, done)
        self.predecessors[next_state_index].add((state_index, action))

        current_q = self.q_table[state_index, action]

        if done:
            target = reward
        else:
            max_future_q = np.max(self.q_table[next_state_index])
            target = reward + gamma * max_future_q
            
        p = abs(target - current_q)
        if p > self.theta:
            heapq.heappush(self.pq, (-p, state_index, action))

        for _ in range(self.n_planning_steps):
            if not self.pq:
                break
                
            _, sim_s, sim_a = heapq.heappop(self.pq)
            sim_r, sim_ns, sim_done = self.model[(sim_s, sim_a)]
            sim_current_q = self.q_table[sim_s, sim_a]
            
            if sim_done:
                sim_target = sim_r
            else:
                sim_target = sim_r + gamma * np.max(self.q_table[sim_ns])
                
            self.q_table[sim_s, sim_a] += alpha * (sim_target - sim_current_q)
            
            for p_s, p_a in self.predecessors[sim_s]:
                p_r, p_ns, p_done = self.model[(p_s, p_a)]
                p_current_q = self.q_table[p_s, p_a]
                if p_done:
                    p_tgt = p_r
                else:
                    p_tgt = p_r + gamma * np.max(self.q_table[p_ns])
                
                pred_p = abs(p_tgt - p_current_q)
                if pred_p > self.theta:
                    heapq.heappush(self.pq, (-pred_p, p_s, p_a))

    def reset(self):
        super().reset()
        self.model = {}
        self.pq = []
        self.predecessors.clear()
