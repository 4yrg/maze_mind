"""Classical pathfinding algorithms for maze navigation.

This module provides optimal (non-learned) pathfinding algorithms:
- A*: A* search with Manhattan distance heuristic
- Dijkstra: Uniform cost search
- BFS: Breadth-first search (shortest path in unweighted)
- Flood Fill: Classic micromouse exploration algorithm
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set

from mazemind.envs.maze_parser import MazeData, ACTION_DELTAS, ACTION_NAMES


@dataclass
class PathResult:
    path: List[Tuple[int, int]]
    cost: int
    found: bool
    algorithm: str


class ClassicalSolver:
    def __init__(self, maze: MazeData):
        self.maze = maze

    def _get_neighbors(
        self, state: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], int]]:
        row, col = state
        neighbors = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = row + dr, col + dc
            direction = ACTION_NAMES[action]
            if self.maze.has_wall(row, col, direction):
                continue
            if 0 <= nr < self.maze.size and 0 <= nc < self.maze.size:
                neighbors.append(((nr, nc), 1))
        return neighbors

    def _manhattan_distance(
        self, a: Tuple[int, int], b: Tuple[int, int]
    ) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_closest_goal(self, state: Tuple[int, int]) -> Tuple[int, int]:
        return min(
            self.maze.goals,
            key=lambda g: self._manhattan_distance(state, g)
        )

    def solve_astar(self) -> PathResult:
        start = self.maze.start
        goals = self.maze.goals
        
        open_set = []
        heapq.heappush(open_set, (0, start, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_set:
            _, current, _ = heapq.heappop(open_set)
            
            if current in goals:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    path=path,
                    cost=g_score[current],
                    found=True,
                    algorithm="A*"
                )
            
            for neighbor, cost in self._get_neighbors(current):
                tentative_g = g_score.get(current, float('inf')) + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    goal = self._get_closest_goal(neighbor)
                    f_score = tentative_g + self._manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor, current))
        
        return PathResult(path=[], cost=0, found=False, algorithm="A*")

    def solve_dijkstra(self) -> PathResult:
        start = self.maze.start
        goals = self.maze.goals
        
        pq = []
        heapq.heappush(pq, (0, start, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        dist: Dict[Tuple[int, int], float] = {start: 0}
        
        while pq:
            distance, current, _ = heapq.heappop(pq)
            
            if current in goals:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    path=path,
                    cost=int(distance),
                    found=True,
                    algorithm="Dijkstra"
                )
            
            if distance > dist.get(current, float('inf')):
                continue
            
            for neighbor, cost in self._get_neighbors(current):
                new_dist = dist.get(current, float('inf')) + cost
                
                if neighbor not in dist or new_dist < dist[neighbor]:
                    came_from[neighbor] = current
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor, current))
        
        return PathResult(path=[], cost=0, found=False, algorithm="Dijkstra")

    def solve_bfs(self) -> PathResult:
        start = self.maze.start
        goals = self.maze.goals
        
        queue = deque([(start, start)])
        visited: Set[Tuple[int, int]] = {start}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while queue:
            current, prev = queue.popleft()
            
            if current in goals:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    path=path,
                    cost=len(path) - 1,
                    found=True,
                    algorithm="BFS"
                )
            
            for neighbor, _ in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append((neighbor, current))
        
        return PathResult(path=[], cost=0, found=False, algorithm="BFS")

    def solve_flood_fill(self) -> PathResult:
        start = self.maze.start
        goals = self.maze.goals
        
        queue = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while queue:
            current = queue.popleft()
            
            if current in goals:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    path=path,
                    cost=len(path) - 1,
                    found=True,
                    algorithm="Flood Fill"
                )
            
            for neighbor, _ in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        return PathResult(path=[], cost=0, found=False, algorithm="Flood Fill")

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def solve_all(self) -> List[PathResult]:
        return [
            self.solve_astar(),
            self.solve_dijkstra(),
            self.solve_bfs(),
            self.solve_flood_fill(),
        ]