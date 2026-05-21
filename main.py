"""
AI Pac-Man
==============================
Ghost AI Algorithms:
  • Random      — ghosts wander randomly
  • A* Chase    — optimal shortest-path to Pac-Man
  • Dijkstra    — cost-uniform shortest-path (no heuristic)
  • Greedy BFS  — heuristic-only, no cost tracking (fast but suboptimal)
  • Manhattan   — pure Manhattan-distance steering (no graph search)

Pac-Man Control Modes:
  • Manual      — arrow-key control
  • RL Agent    — Q-learning agent trained live; run N parallel sims,
                  pick the best policy each generation, show progress

The point: watch how A* / Dijkstra trap Pac-Man while Greedy and Manhattan
ghosts can be exploited by tight loops.  Then see whether the RL agent
learns to exploit those patterns.
"""

import pygame
import sys
import math
import random
import heapq
import threading
import time
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from collections import deque
import numpy as np


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

TILE_SIZE   = 28
GRID_COLS   = 21
GRID_ROWS   = 21
SCREEN_W    = GRID_COLS * TILE_SIZE          # 588
SCREEN_H    = GRID_ROWS * TILE_SIZE + 100   # 688  (+100 HUD)
FPS         = 60

PAC_SPEED   = 3
GHOST_SPEED = 2

# Colors
BLACK        = (0,   0,   0)
DARK_BLUE    = (8,   8,  35)
WALL_COLOR   = (40,  80, 220)
PELLET_CLR   = (255, 220, 140)
PAC_COLOR    = (255, 230,   0)
GHOST_COLORS = [
    (255,  60,  60),
    (255, 180, 255),
    (  0, 220, 255),
    (255, 165,   0),
]
WHITE        = (255, 255, 255)
TEXT_COLOR   = (220, 220, 255)
HUD_BG       = (10,  10,  30)
DIM          = (120, 120, 160)
GREEN        = (60,  200,  60)
YELLOW       = (255, 230,   0)
ORANGE       = (255, 140,   0)
RED_CLR      = (255,  60,  60)
CYAN         = (0,   200, 220)

# Algorithm button accent colors
BTN_COLORS = {
    "Random":   ((50, 180,  80), (30, 120,  50)),
    "A* Chase": ((60, 120, 220), (30,  70, 160)),
    "Dijkstra": ((180, 90, 220), (110, 50, 150)),
    "Greedy":   ((200, 130,  40), (140,  80, 20)),
    "Manhattan":((40, 160, 160), (20,  100, 100)),
}

# Tile types
WALL   = 1
PELLET = 2
EMPTY  = 0
POWER  = 3

FRIGHTENED_DURATION    = 5.0
FRIGHTENED_FLASH_START = 3.0
GHOST_EAT_SCORE        = 200
GHOST_FRIGHTENED_COLOR = (40,  40, 160)
GHOST_FRIGHTENED_FLASH = (240, 240, 255)

DIR_UP    = ( 0, -1)
DIR_DOWN  = ( 0,  1)
DIR_LEFT  = (-1,  0)
DIR_RIGHT = ( 1,  0)
ALL_DIRS  = [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]

# RL constants
RL_PARALLEL_SIMS  = 12      # parallel simulations per training batch
RL_SIM_STEPS      = 1000     # steps per simulation (more board coverage)
RL_TRAIN_INTERVAL = 15      # frames between training batches
RL_ALPHA          = 0.20    # learning rate (decays with epsilon)
RL_ALPHA_MIN      = 0.02    # floor for alpha decay
RL_GAMMA          = 0.97    # higher = values distant pellets more
RL_EPSILON_START  = 1.0     # start fully random
RL_EPSILON_MIN    = 0.08    # keep some exploration forever
RL_EPSILON_DECAY  = 0.99   # slower decay = more thorough exploration

# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------

DEFAULT_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1],
    [1,3,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,3,1],
    [1,2,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,2,1,2,1,1,1,1,1,1,1,2,1,2,1,1,2,1],
    [1,2,2,2,2,1,2,2,2,2,1,2,2,2,2,1,2,2,2,2,1],
    [1,1,1,1,2,1,1,1,0,0,0,0,0,1,1,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,0,0,1,0,1,0,0,0,1,2,1,1,1,1],
    [1,1,1,1,2,0,0,1,1,0,0,0,1,1,0,0,2,1,1,1,1],
    [0,0,0,0,2,0,0,1,0,0,0,0,0,1,0,0,2,0,0,0,0],
    [1,1,1,1,2,0,0,1,1,1,1,1,1,1,0,0,2,1,1,1,1],
    [1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,1],
    [1,1,1,1,2,1,0,1,1,1,1,1,1,1,0,1,2,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,2,1],
    [1,3,2,1,2,2,2,2,2,2,0,2,2,2,2,2,2,1,2,3,1],
    [1,1,2,1,2,1,2,1,1,1,1,1,1,1,2,1,2,1,2,1,1],
    [1,2,2,2,2,1,2,2,2,2,1,2,2,2,2,1,2,2,2,2,1],
    [1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]


class Map:
    TUNNEL_ROW = 10

    def __init__(self, layout: List[List[int]]):
        self.grid = [row[:] for row in layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

    def is_wall(self, col: int, row: int) -> bool:
        if row == self.TUNNEL_ROW and (col < 0 or col >= self.cols):
            return False
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True
        return self.grid[row][col] == WALL

    def consume_pellet(self, col: int, row: int) -> Tuple[int, bool]:
        tile = self.grid[row][col]
        if tile == PELLET:
            self.grid[row][col] = EMPTY
            return 10, False
        if tile == POWER:
            self.grid[row][col] = EMPTY
            return 50, True
        return 0, False

    def remaining_pellets(self) -> int:
        return sum(c in (PELLET, POWER) for row in self.grid for c in row)

    def total_pellets(self) -> int:
        return sum(c in (PELLET, POWER) for row in DEFAULT_MAP for c in row)

    def draw(self, surface: pygame.Surface) -> None:
        for r, row in enumerate(self.grid):
            for c, tile in enumerate(row):
                x = c * TILE_SIZE
                y = r * TILE_SIZE
                if tile == WALL:
                    pygame.draw.rect(surface, WALL_COLOR,
                        pygame.Rect(x+1, y+1, TILE_SIZE-2, TILE_SIZE-2), border_radius=4)
                elif tile == PELLET:
                    pygame.draw.circle(surface, PELLET_CLR,
                        (x + TILE_SIZE//2, y + TILE_SIZE//2), 3)
                elif tile == POWER:
                    pygame.draw.circle(surface, PELLET_CLR,
                        (x + TILE_SIZE//2, y + TILE_SIZE//2), 7)


# ---------------------------------------------------------------------------
# PATHFINDING UTILITIES
# ---------------------------------------------------------------------------

def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(start: Tuple[int,int], goal: Tuple[int,int], game_map: Map,
          max_iter: int = 300) -> Optional[Tuple[int,int]]:
    """Returns first-step direction toward goal, or None if unreachable."""
    if start == goal:
        return None
    open_set = [(0, start)]
    came_from: Dict = {}
    g = {start: 0}
    iterations = 0
    while open_set and iterations < max_iter:
        iterations += 1
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Backtrack to find first step
            node = current
            while came_from.get(node) != start:
                node = came_from[node]
                if node not in came_from:
                    break
            dc = node[0] - start[0]
            dr = node[1] - start[1]
            return (max(-1, min(1, dc)), max(-1, min(1, dr)))
        for d in ALL_DIRS:
            nb = (current[0]+d[0], current[1]+d[1])
            if game_map.is_wall(*nb):
                continue
            ng = g[current] + 1
            if ng < g.get(nb, float('inf')):
                came_from[nb] = current
                g[nb] = ng
                h = heuristic(nb, goal)
                heapq.heappush(open_set, (ng+h, nb))
    return None


def dijkstra(start: Tuple[int,int], goal: Tuple[int,int], game_map: Map,
             max_iter: int = 300) -> Optional[Tuple[int,int]]:
    """Dijkstra (uniform cost, no heuristic) — first-step direction toward goal."""
    if start == goal:
        return None
    open_set = [(0, start)]
    came_from: Dict = {}
    dist = {start: 0}
    iterations = 0
    while open_set and iterations < max_iter:
        iterations += 1
        cost, current = heapq.heappop(open_set)
        if current == goal:
            node = current
            while came_from.get(node) != start:
                node = came_from[node]
                if node not in came_from:
                    break
            dc = node[0] - start[0]
            dr = node[1] - start[1]
            return (max(-1, min(1, dc)), max(-1, min(1, dr)))
        for d in ALL_DIRS:
            nb = (current[0]+d[0], current[1]+d[1])
            if game_map.is_wall(*nb):
                continue
            nd = dist[current] + 1
            if nd < dist.get(nb, float('inf')):
                came_from[nb] = current
                dist[nb] = nd
                heapq.heappush(open_set, (nd, nb))
    return None


def greedy_bfs(start: Tuple[int,int], goal: Tuple[int,int], game_map: Map,
               max_iter: int = 300) -> Optional[Tuple[int,int]]:
    """Greedy best-first — heuristic only, no cost tracking."""
    if start == goal:
        return None
    open_set = [(heuristic(start, goal), start)]
    came_from: Dict = {}
    visited = {start}
    iterations = 0
    while open_set and iterations < max_iter:
        iterations += 1
        _, current = heapq.heappop(open_set)
        if current == goal:
            node = current
            while came_from.get(node) != start:
                node = came_from[node]
                if node not in came_from:
                    break
            dc = node[0] - start[0]
            dr = node[1] - start[1]
            return (max(-1, min(1, dc)), max(-1, min(1, dr)))
        for d in ALL_DIRS:
            nb = (current[0]+d[0], current[1]+d[1])
            if game_map.is_wall(*nb) or nb in visited:
                continue
            visited.add(nb)
            came_from[nb] = current
            heapq.heappush(open_set, (heuristic(nb, goal), nb))
    return None


# ---------------------------------------------------------------------------
# MOVE STRATEGIES
# ---------------------------------------------------------------------------

class MoveStrategy(ABC):
    @abstractmethod
    def choose_direction(self, entity: "Entity", game_map: Map,
                         target_pos: Optional[Tuple[int,int]] = None) -> Tuple[int,int]:
        ...


class RandomMoveStrategy(MoveStrategy):
    def choose_direction(self, entity, game_map, target_pos=None):
        col, row = entity.grid_col, entity.grid_row
        reverse  = (-entity.direction[0], -entity.direction[1])
        options  = [d for d in ALL_DIRS
                    if d != reverse and not game_map.is_wall(col+d[0], row+d[1])]
        if options:
            return random.choice(options)
        if not game_map.is_wall(col+reverse[0], row+reverse[1]):
            return reverse
        return entity.direction


class AStarStrategy(MoveStrategy):
    """
    Optimal shortest-path chase.
    Weakness: predictable — Pac-Man can loop it into a corridor,
    but it WILL catch up eventually since it never wastes a step.
    """
    def choose_direction(self, entity, game_map, target_pos=None):
        if target_pos is None:
            return RandomMoveStrategy().choose_direction(entity, game_map)
        start = (entity.grid_col, entity.grid_row)
        result = astar(start, target_pos, game_map)
        if result:
            return result
        return RandomMoveStrategy().choose_direction(entity, game_map)


class DijkstraStrategy(MoveStrategy):
    """
    Uniform-cost search — identical path to A* on uniform grid,
    but explores more nodes (no heuristic to guide it).
    Weakness: same as A* in result, slightly slower in practice.
    On uniform grids this equals A*; the difference shows in weighted mazes.
    """
    def choose_direction(self, entity, game_map, target_pos=None):
        if target_pos is None:
            return RandomMoveStrategy().choose_direction(entity, game_map)
        start = (entity.grid_col, entity.grid_row)
        result = dijkstra(start, target_pos, game_map)
        if result:
            return result
        return RandomMoveStrategy().choose_direction(entity, game_map)


class GreedyStrategy(MoveStrategy):
    """
    Greedy Best-First — only looks at heuristic, ignores actual cost.
    Weakness: can get trapped in loops around U-shaped walls,
    spinning in place while Pac-Man escapes. Easily exploited by
    driving into any concave obstacle.
    """
    def choose_direction(self, entity, game_map, target_pos=None):
        if target_pos is None:
            return RandomMoveStrategy().choose_direction(entity, game_map)
        start = (entity.grid_col, entity.grid_row)
        result = greedy_bfs(start, target_pos, game_map)
        if result:
            return result
        return RandomMoveStrategy().choose_direction(entity, game_map)


class ManhattanStrategy(MoveStrategy):
    """
    Pure Manhattan steering — no graph search at all.
    Each turn, pick the open neighbour that minimises Manhattan distance to target.
    Weakness: can't navigate around walls; will oscillate against obstacles,
    allowing Pac-Man to hide behind any wall for indefinite safety.
    """
    def choose_direction(self, entity, game_map, target_pos=None):
        if target_pos is None:
            return RandomMoveStrategy().choose_direction(entity, game_map)
        col, row = entity.grid_col, entity.grid_row
        reverse  = (-entity.direction[0], -entity.direction[1])
        options  = [d for d in ALL_DIRS
                    if d != reverse and not game_map.is_wall(col+d[0], row+d[1])]
        if not options:
            options = [d for d in ALL_DIRS
                       if not game_map.is_wall(col+d[0], row+d[1])]
        if not options:
            return entity.direction
        tc, tr = target_pos
        return min(options, key=lambda d: abs((col+d[0])-tc) + abs((row+d[1])-tr))



class FleeMoveStrategy(MoveStrategy):
    """
    Used when a ghost is frightened — runs AWAY from Pac-Man.
    Picks the open non-reversing neighbour that MAXIMISES Manhattan
    distance from target_pos (Pac-Man).
    """
    def choose_direction(self, entity, game_map, target_pos=None):
        col, row = entity.grid_col, entity.grid_row
        reverse  = (-entity.direction[0], -entity.direction[1])
        options  = [d for d in ALL_DIRS
                    if d != reverse and not game_map.is_wall(col+d[0], row+d[1])]
        if not options:
            options = [d for d in ALL_DIRS
                       if not game_map.is_wall(col+d[0], row+d[1])]
        if not options:
            return entity.direction
        if target_pos is None:
            return random.choice(options)
        tc, tr = target_pos
        return max(options, key=lambda d: abs((col+d[0])-tc) + abs((row+d[1])-tr))


# ---------------------------------------------------------------------------
# RL / Q-LEARNING AGENT
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent for Pac-Man.

    State space: (pac_col, pac_row, nearest_ghost_dir_4, power_active)
                 where nearest_ghost_dir_4 encodes octant of closest ghost.
    Action space: 4 directions (UP, DOWN, LEFT, RIGHT)

    Training: parallel_sims independent sim threads each episode,
              best scoring run's experiences update the shared Q-table.
    """

    ACTIONS = [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]

    def __init__(self):
        self.q_table: Dict[tuple, np.ndarray] = {}
        self.epsilon   = RL_EPSILON_START
        self.generation = 0
        self.best_score = 0
        self.training   = True
        self.last_train_scores: List[float] = []
        self._lock = threading.Lock()

        # Stats for HUD
        self.current_sim_scores: List[float] = [0.0] * RL_PARALLEL_SIMS
        self.training_in_progress = False

    def get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def _encode_state(self, pac_col, pac_row, ghost_positions, power_active,
                       game_map=None) -> tuple:
        """
        Relative-only state — NO absolute position.
        (ghost_dir8, ghost_dist6, dot_dir8, dot_dist6,
         wall_N, wall_S, wall_E, wall_W, power_active)

        Dropping pac_col/pac_row means what the agent learns at one spot
        transfers to every similar situation elsewhere on the board.
        State space: ~150k vs ~2.7M before — converges ~18x faster.
        """
        # ── Nearest ghost ─────────────────────────────────────────────
        if ghost_positions:
            gc, gr = min(ghost_positions,
                         key=lambda p: abs(p[0]-pac_col)+abs(p[1]-pac_row))
            g_dist = abs(gc-pac_col) + abs(gr-pac_row)
            g_dir  = int((math.atan2(gr-pac_row, gc-pac_col) + math.pi)
                         / (2*math.pi) * 8) % 8
        else:
            g_dist, g_dir = 99, 0
        g_dist_b = min(g_dist // 2, 7)   # 0-7 buckets, finer near ghost

        # ── Nearest dot ───────────────────────────────────────────────
        dot_dir, dot_dist_b = 0, 7
        if game_map is not None:
            best = 9999
            bdc, bdr = 1, 0
            for dr in range(-10, 11):
                for dc in range(-10, 11):
                    d = abs(dc) + abs(dr)
                    if d >= best:
                        continue
                    tc = (pac_col + dc) % game_map.cols
                    tr = pac_row + dr
                    if tr < 0 or tr >= game_map.rows:
                        continue
                    if game_map.grid[tr][tc] in (PELLET, POWER):
                        best, bdc, bdr = d, dc, dr
            if best < 9999:
                dot_dir   = int((math.atan2(bdr, bdc) + math.pi) / (2*math.pi) * 8) % 8
                dot_dist_b = min(best // 2, 7)

        # ── Immediate wall sensors (1 tile in each cardinal direction) ─
        if game_map is not None:
            wN = int(game_map.is_wall(pac_col,   pac_row-1))
            wS = int(game_map.is_wall(pac_col,   pac_row+1))
            wE = int(game_map.is_wall(pac_col+1, pac_row))
            wW = int(game_map.is_wall(pac_col-1, pac_row))
        else:
            wN = wS = wE = wW = 0

        return (g_dir, g_dist_b, dot_dir, dot_dist_b, wN, wS, wE, wW,
                int(power_active))

    def choose_action(self, state: tuple, stuck_override: float = 0.0) -> Tuple[int,int]:
        """
        stuck_override: extra exploration probability injected when the agent
        hasn't eaten a dot recently.  Decays back to epsilon once it moves.
        """
        eff_epsilon = max(self.epsilon, stuck_override)
        if random.random() < eff_epsilon:
            return random.choice(self.ACTIONS)
        q = self.get_q(state)
        return self.ACTIONS[int(np.argmax(q))]

    @property
    def alpha(self):
        """Learning rate decays with epsilon — aggressive early, careful late."""
        t = max(0.0, (self.epsilon - RL_EPSILON_MIN) / (RL_EPSILON_START - RL_EPSILON_MIN))
        return RL_ALPHA_MIN + (RL_ALPHA - RL_ALPHA_MIN) * t

    def update(self, state, action_idx, reward, next_state, done):
        q   = self.get_q(state)
        q_n = self.get_q(next_state)
        target = reward + (0 if done else RL_GAMMA * np.max(q_n))
        q[action_idx] += self.alpha * (target - q[action_idx])

    def run_parallel_training(self, map_layout, ghost_strategy_name):
        """Spawn RL_PARALLEL_SIMS threads; best run updates Q-table."""
        results = [None] * RL_PARALLEL_SIMS
        threads = []
        for i in range(RL_PARALLEL_SIMS):
            t = threading.Thread(target=self._run_sim,
                                 args=(i, map_layout, ghost_strategy_name, results),
                                 daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Each result is (score, experiences) — unpack correctly
        valid = [(r[0], r[1]) for r in results if r is not None]
        if not valid:
            self.training_in_progress = False
            return

        scores = [v[0] for v in valid]
        best_score = max(scores)
        self.current_sim_scores = [r[0] if r is not None else 0 for r in results]
        self.last_train_scores  = self.current_sim_scores[:]

        if best_score > self.best_score:
            self.best_score = best_score

        # Learn from ALL sims — higher-scoring runs get more weight.
        # This uses all 12 sims instead of throwing away 11 of them.
        min_score = max(1, min(scores))
        with self._lock:
            for (score, experiences) in valid:
                # Weight: best sim = 1.0, worst sim = ~0.25 (still learns)
                w = 0.25 + 0.75 * (score / max(best_score, 1))
                # Prioritise recent experiences in each episode (they're
                # less noisy than early random steps)
                n = len(experiences)
                for idx, (s, ai, rew, ns, done) in enumerate(experiences):
                    recency = 0.5 + 0.5 * (idx / max(n-1, 1))
                    # Temporarily scale alpha by weight*recency
                    orig = self.alpha
                    q   = self.get_q(s)
                    q_n = self.get_q(ns)
                    target = rew + (0 if done else RL_GAMMA * np.max(q_n))
                    q[ai] += orig * w * recency * (target - q[ai])

        self.generation += 1
        self.epsilon = max(RL_EPSILON_MIN, self.epsilon * RL_EPSILON_DECAY)
        self.training_in_progress = False

    def _run_sim(self, sim_idx, map_layout, ghost_strategy_name, results):
        """Single simulation run — no pygame rendering."""
        try:
            game_map = Map(map_layout)
            pac_col, pac_row = 10, 16
            pac_dir = DIR_RIGHT
            pac_score = 0
            pac_lives = 1

            # Ghost positions (simplified sim)
            ghost_starts = [(9,9), (10,9), (11,9), (10,10)]
            ghost_cols   = [g[0] for g in ghost_starts]
            ghost_rows   = [g[1] for g in ghost_starts]
            ghost_dirs   = [DIR_RIGHT] * 4
            frightened   = False
            fright_timer = 0.0

            experiences = []
            cls, _ = ALGORITHM_REGISTRY[ghost_strategy_name]
            ghost_strats = [cls() for _ in range(4)]
            steps_since_dot = 0          # stuck detection

            # Minimal entity stub for strategy interface
            class FakeEntity:
                def __init__(self, c, r, d):
                    self.grid_col, self.grid_row, self.direction = c, r, d

            for step in range(RL_SIM_STEPS):
                # Break early if completely stuck — saves sim time
                if steps_since_dot > 60:
                    break

                # Build state — pass game_map so dot direction is encoded
                ghost_pos = list(zip(ghost_cols, ghost_rows))
                state = self._encode_state(pac_col, pac_row, ghost_pos, frightened,
                                           game_map)

                # Stuck override: spike exploration if no dot eaten recently
                stuck_eps = min(steps_since_dot * 0.008, 0.95)
                action = self.choose_action(state, stuck_override=stuck_eps)
                action_idx = self.ACTIONS.index(action)

                # Pac moves
                nc, nr = pac_col + action[0], pac_row + action[1]
                if not game_map.is_wall(nc, nr):
                    pac_col, pac_row = nc % GRID_COLS, nr
                    pac_dir = action

                # Eat pellets
                pts, power = game_map.consume_pellet(pac_col, pac_row)
                pac_score += pts
                if power:
                    frightened = True
                    fright_timer = FRIGHTENED_DURATION

                # Move ghosts (simplified tile-snapping)
                for i in range(4):
                    fe = FakeEntity(ghost_cols[i], ghost_rows[i], ghost_dirs[i])
                    pac_target = (pac_col, pac_row)
                    if frightened:
                        # Flee — pick direction away from pac
                        reverse = (-ghost_dirs[i][0], -ghost_dirs[i][1])
                        opts = [d for d in ALL_DIRS
                                if d != reverse
                                and not game_map.is_wall(ghost_cols[i]+d[0], ghost_rows[i]+d[1])]
                        if opts:
                            gd = max(opts, key=lambda d: abs((ghost_cols[i]+d[0])-pac_col)+abs((ghost_rows[i]+d[1])-pac_row))
                        else:
                            gd = ghost_dirs[i]
                    else:
                        gd = ghost_strats[i].choose_direction(fe, game_map, pac_target)
                    ghost_dirs[i] = gd
                    gnc = (ghost_cols[i] + gd[0]) % GRID_COLS
                    gnr = ghost_rows[i] + gd[1]
                    if not game_map.is_wall(gnc, gnr):
                        ghost_cols[i], ghost_rows[i] = gnc, gnr

                # Update fright
                if frightened:
                    fright_timer -= 1.0/FPS
                    if fright_timer <= 0:
                        frightened = False

                # Check collisions
                done = False

                # --- Reward shaping ---
                if pts == 10:
                    reward = 10.0
                    steps_since_dot = 0
                elif pts == 50:
                    reward = 30.0
                    steps_since_dot = 0
                else:
                    steps_since_dot += 1
                    # Escalating idle penalty
                    reward = -0.1 - min(steps_since_dot * 0.03, 2.0)
                    # Small proximity bonus: reward moving toward nearest dot
                    # This gives a gradient signal between pellets so the agent
                    # doesn't have to stumble randomly into every dot
                    if game_map is not None:
                        best_dot = 9999
                        for dr in range(-6, 7):
                            for dc in range(-6, 7):
                                tc = (pac_col + dc) % game_map.cols
                                tr = pac_row + dr
                                if 0 <= tr < game_map.rows:
                                    if game_map.grid[tr][tc] in (PELLET, POWER):
                                        best_dot = min(best_dot, abs(dc)+abs(dr))
                        if best_dot < 9999:
                            reward += max(0.0, (6 - best_dot) * 0.15)

                for i in range(4):
                    if ghost_cols[i] == pac_col and ghost_rows[i] == pac_row:
                        if frightened:
                            pac_score += GHOST_EAT_SCORE
                            reward += 20.0   # eating a scared ghost is great
                        else:
                            # Death is catastrophic — large enough to outweigh
                            # any plausible pellet-eating run (~240 pellets * 10 = 2400)
                            reward -= 500.0
                            done = True
                            break

                # Clearing the board: massive bonus
                rem = game_map.remaining_pellets()
                if rem == 0:
                    reward += 1000.0
                    done = True

                # Next state
                ghost_pos2 = list(zip(ghost_cols, ghost_rows))
                next_state = self._encode_state(pac_col, pac_row, ghost_pos2, frightened,
                                                game_map)
                experiences.append((state, action_idx, reward, next_state, done))
                if done:
                    break

            results[sim_idx] = (pac_score, experiences)
        except Exception as e:
            results[sim_idx] = (0, [])


class RLMoveStrategy(MoveStrategy):
    """Wraps a QLearningAgent for use as a Pac-Man MoveStrategy."""

    def __init__(self, agent):
        self.agent             = agent
        self._ghost_positions  = []
        self._power_active     = False
        self._steps_since_dot  = 0   # stuck detection for live play

    def notify_ate_dot(self):
        """Call this from GameEngine whenever Pac-Man eats a pellet."""
        self._steps_since_dot = 0

    def choose_direction(self, entity, game_map, target_pos=None):
        self._steps_since_dot += 1
        ghost_pos    = self._ghost_positions
        power_active = self._power_active
        state = self.agent._encode_state(
            entity.grid_col, entity.grid_row, ghost_pos, power_active, game_map)
        stuck_eps = min(self._steps_since_dot * 0.008, 0.95)
        with self.agent._lock:
            action = self.agent.choose_action(state, stuck_override=stuck_eps)
        return action


class ManualControl(MoveStrategy):
    def __init__(self):
        self.queued: Tuple[int,int] = DIR_RIGHT

    def queue_direction(self, d: Tuple[int,int]) -> None:
        self.queued = d

    def choose_direction(self, entity, game_map, target_pos=None):
        col, row = entity.grid_col, entity.grid_row
        if not game_map.is_wall(col+self.queued[0], row+self.queued[1]):
            return self.queued
        if not game_map.is_wall(col+entity.direction[0], row+entity.direction[1]):
            return entity.direction
        return entity.direction


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY = {
    "Random":    (RandomMoveStrategy, True),
    "A* Chase":  (AStarStrategy,      True),
    "Dijkstra":  (DijkstraStrategy,   True),
    "Greedy":    (GreedyStrategy,     True),
    "Manhattan": (ManhattanStrategy,  True),
}

PACMAN_MODES = ["Manual", "RL Agent"]


# ---------------------------------------------------------------------------
# ENTITY
# ---------------------------------------------------------------------------

class Entity(ABC):
    def __init__(self, start_col, start_row, speed, move_strategy, color):
        self.grid_col      = start_col
        self.grid_row      = start_row
        self.x = float(start_col * TILE_SIZE + TILE_SIZE // 2)
        self.y = float(start_row * TILE_SIZE + TILE_SIZE // 2)
        self.speed         = speed
        self.direction: Tuple[int,int] = DIR_RIGHT
        self.move_strategy = move_strategy
        self.color         = color

    def update(self, game_map: Map, target_pos=None):
        tile_cx = self.grid_col * TILE_SIZE + TILE_SIZE // 2
        tile_cy = self.grid_row * TILE_SIZE + TILE_SIZE // 2
        dx = tile_cx - self.x
        dy = tile_cy - self.y
        dist = math.hypot(dx, dy)
        if dist <= self.speed:
            self.x = float(tile_cx)
            self.y = float(tile_cy)
            self.direction = self.move_strategy.choose_direction(self, game_map, target_pos)
            nc = self.grid_col + self.direction[0]
            nr = self.grid_row + self.direction[1]
            if not game_map.is_wall(nc, nr):
                self.grid_col = nc % game_map.cols
                self.grid_row = nr
                if nc != self.grid_col:
                    self.x = float(self.grid_col * TILE_SIZE + TILE_SIZE // 2)
        else:
            self.x += self.speed * dx / dist
            self.y += self.speed * dy / dist

    def pixel_pos(self): return int(self.x), int(self.y)

    @abstractmethod
    def draw(self, surface): ...


class PacMan(Entity):
    RADIUS = TILE_SIZE // 2 - 3

    def __init__(self, start_col, start_row, move_strategy):
        super().__init__(start_col, start_row, PAC_SPEED, move_strategy, PAC_COLOR)
        self.score  = 0
        self.lives  = 3
        self._mouth = 30
        self._mouth_dir = -1

    def update(self, game_map: Map, target_pos=None) -> bool:
        super().update(game_map)
        points, power_eaten = game_map.consume_pellet(self.grid_col, self.grid_row)
        self.score += points
        self._mouth += self._mouth_dir * 4
        if self._mouth <= 4:
            self._mouth_dir = 1
        elif self._mouth >= 42:
            self._mouth_dir = -1
        return power_eaten

    def draw(self, surface):
        cx, cy = self.pixel_pos()
        angle_map = {DIR_RIGHT:0, DIR_LEFT:180, DIR_UP:90, DIR_DOWN:270}
        base = angle_map.get(self.direction, 0)
        pts  = [(cx, cy)]
        a_start = base + self._mouth
        a_end   = base - self._mouth + 360
        for i in range(33):
            a = math.radians(a_start + (a_end - a_start) * i / 32)
            pts.append((cx + self.RADIUS * math.cos(a),
                        cy - self.RADIUS * math.sin(a)))
        if len(pts) > 2:
            pygame.draw.polygon(surface, self.color, pts)
        ea = math.radians(base + 65)
        pygame.draw.circle(surface, BLACK,
            (int(cx + self.RADIUS*0.5*math.cos(ea)),
             int(cy - self.RADIUS*0.5*math.sin(ea))), 2)


class Ghost(Entity):
    RADIUS = TILE_SIZE // 2 - 3

    def __init__(self, start_col, start_row, move_strategy, color, name="Ghost"):
        super().__init__(start_col, start_row, GHOST_SPEED, move_strategy, color)
        self.name           = name
        self.start_col      = start_col
        self.start_row      = start_row
        self.base_color     = color
        self.chase_strategy = move_strategy      # saved for mode-switching
        self.flee_strategy  = FleeMoveStrategy()
        self.frightened     = False
        self._fright_timer  = 0.0
        self._eaten         = False

    def frighten(self):
        self.frightened    = True
        self._fright_timer = FRIGHTENED_DURATION
        self._eaten        = False
        self.move_strategy = self.flee_strategy   # ghosts run away

    def recover(self):
        self.frightened    = False
        self._fright_timer = 0.0
        self.move_strategy = self.chase_strategy  # back to chasing

    def update_fright(self, dt):
        if self.frightened and not self._eaten:
            self._fright_timer -= dt
            if self._fright_timer <= 0:
                self.recover()

    def respawn(self):
        """Return ghost to home tile — called after Pac-Man loses a life."""
        self.grid_col = self.start_col
        self.grid_row = self.start_row
        self.x = float(self.start_col * TILE_SIZE + TILE_SIZE // 2)
        self.y = float(self.start_row * TILE_SIZE + TILE_SIZE // 2)
        self.direction = DIR_RIGHT
        self.recover()
        self._eaten = False

    def draw(self, surface):
        cx, cy = self.pixel_pos()
        r = self.RADIUS
        if self.frightened and not self._eaten:
            if self._fright_timer <= FRIGHTENED_FLASH_START:
                speed = 6 + (FRIGHTENED_FLASH_START - self._fright_timer) * 3
                flash_on = int(pygame.time.get_ticks() / (1000 / speed)) % 2 == 0
                body_color = GHOST_FRIGHTENED_FLASH if flash_on else GHOST_FRIGHTENED_COLOR
            else:
                body_color = GHOST_FRIGHTENED_COLOR
        else:
            body_color = self.base_color
        if self._eaten:
            return
        pygame.draw.circle(surface, body_color, (cx, cy - r//5), r)
        pygame.draw.rect(surface, body_color,
                         pygame.Rect(cx-r, cy-r//5, r*2, r+2))
        wave_w  = (r*2)/4
        skirt_y = cy + r - 3
        for i in range(4):
            wx = int(cx - r + wave_w*i + wave_w/2)
            pygame.draw.circle(surface, DARK_BLUE, (wx, skirt_y), int(wave_w/2)+1)
        if not self.frightened:
            ey = cy - r//3
            for ex in (cx - r//3, cx + r//3):
                pygame.draw.circle(surface, WHITE, (ex, ey), 4)
                pygame.draw.circle(surface, (0,0,180), (ex+1, ey+1), 2)
        else:
            mouth_y = cy + r//4
            pts = []
            for i in range(7):
                mx = cx - r + i*(r*2//6)
                my = mouth_y + (4 if i%2==0 else -4)
                pts.append((mx, my))
            if len(pts) >= 2:
                pygame.draw.lines(surface, WHITE, False, pts, 2)


# ---------------------------------------------------------------------------
# START SCREEN
# ---------------------------------------------------------------------------

class Button:
    W, H = 100, 52

    def __init__(self, label, center, colors, locked=False):
        self.label  = label
        self.rect   = pygame.Rect(0,0,self.W,self.H)
        self.rect.center = center
        self.colors = colors
        self.locked = locked
        self._hover = False

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and not self.locked:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface, font, small):
        col = (70,70,70) if self.locked else (
            self.colors[1] if self._hover else self.colors[0])
        pygame.draw.rect(surface, col, self.rect, border_radius=10)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=10)
        label_col = DIM if self.locked else WHITE
        lbl = font.render(self.label, True, label_col)
        surface.blit(lbl, lbl.get_rect(center=self.rect.center))


class ModeButton:
    W, H = 160, 44

    def __init__(self, label, center, active_color):
        self.label        = label
        self.rect         = pygame.Rect(0,0,self.W,self.H)
        self.rect.center  = center
        self.active_color = active_color
        self.active       = False
        self._hover       = False

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface, font):
        if self.active:
            col = self.active_color
        elif self._hover:
            col = tuple(min(255,c+40) for c in self.active_color)
        else:
            col = (40,40,80)
        pygame.draw.rect(surface, col, self.rect, border_radius=8)
        pygame.draw.rect(surface, WHITE if self.active else DIM, self.rect, 2, border_radius=8)
        lbl = font.render(self.label, True, WHITE)
        surface.blit(lbl, lbl.get_rect(center=self.rect.center))


class StartScreen:
    def __init__(self, screen, clock):
        self.screen     = screen
        self.clock      = clock
        self.font_title = pygame.font.SysFont("monospace", 44, bold=True)
        self.font_sub   = pygame.font.SysFont("monospace", 16)
        self.font_btn   = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12)
        self._t = 0

        # Ghost algorithm buttons
        names = list(ALGORITHM_REGISTRY.keys())
        n = len(names)
        btn_y = int(SCREEN_H * 0.52)
        spacing = Button.W + 16
        start_x = SCREEN_W//2 - (n-1)*spacing//2
        self.algo_buttons: List[Tuple[str,Button]] = []
        for i, name in enumerate(names):
            _, impl = ALGORITHM_REGISTRY[name]
            colors = BTN_COLORS.get(name, ((70,70,70),(45,45,45)))
            btn = Button(name, (start_x + i*spacing, btn_y), colors, locked=not impl)
            self.algo_buttons.append((name, btn))

        # Pac-Man mode buttons
        mode_y = int(SCREEN_H * 0.70)
        self.mode_buttons: List[Tuple[str,ModeButton]] = []
        pac_colors = [(80,180,80), (60,120,220)]
        for i, mode in enumerate(PACMAN_MODES):
            mx = SCREEN_W//2 + (i - (len(PACMAN_MODES)-1)/2) * (ModeButton.W+20)
            mbtn = ModeButton(mode, (int(mx), mode_y), pac_colors[i])
            if mode == "Manual":
                mbtn.active = True
            self.mode_buttons.append((mode, mbtn))

        self.selected_algo = "Random"
        self.selected_mode = "Manual"

        # Start button
        self.start_rect = pygame.Rect(0,0,180,52)
        self.start_rect.center = (SCREEN_W//2, int(SCREEN_H*0.85))
        self._start_hover = False

    def run(self) -> Tuple[str,str]:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEMOTION:
                    self._start_hover = self.start_rect.collidepoint(event.pos)
                # Algo buttons
                for name, btn in self.algo_buttons:
                    if btn.handle_event(event):
                        self.selected_algo = name
                # Mode buttons
                for mode, mbtn in self.mode_buttons:
                    if mbtn.handle_event(event):
                        self.selected_mode = mode
                        for _, b in self.mode_buttons:
                            b.active = False
                        mbtn.active = True
                # Start button
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.start_rect.collidepoint(event.pos):
                        return self.selected_algo, self.selected_mode
            self._draw()
            self.clock.tick(FPS)

    def _draw(self):
        self._t += 1
        self.screen.fill(DARK_BLUE)
        for r in range(0, SCREEN_H, TILE_SIZE):
            for c in range(0, SCREEN_W, TILE_SIZE):
                pygame.draw.circle(self.screen, (18,18,52),(c,r),2)

        # Animated pac-man
        pac_x = (self._t * 3) % (SCREEN_W + 80) - 40
        pac_y = 78
        mouth = abs(math.sin(self._t*0.14)) * 36
        pts = [(pac_x, pac_y)]
        for i in range(33):
            a = math.radians(mouth + (360-2*mouth)*i/32)
            pts.append((pac_x+22*math.cos(a), pac_y-22*math.sin(a)))
        if len(pts) > 2:
            pygame.draw.polygon(self.screen, PAC_COLOR, pts)
        for gi, gc in enumerate(GHOST_COLORS):
            gx = pac_x - 55 - gi*50
            gy = pac_y
            r  = 18
            pygame.draw.circle(self.screen, gc, (gx, gy-r//5), r)
            pygame.draw.rect(self.screen, gc, pygame.Rect(gx-r,gy-r//5,r*2,r+2))
            for wi in range(4):
                wx = int(gx - r + (r*2/4)*wi + r/4)
                pygame.draw.circle(self.screen, DARK_BLUE, (wx,gy+r-3), int(r/4)+1)

        # Title
        title = self.font_title.render("PAC-MAN  AI SHOWROOM", True, PAC_COLOR)
        self.screen.blit(title, title.get_rect(center=(SCREEN_W//2,155)))

        # Section: Ghost AI
        lbl = self.font_sub.render("— Ghost AI Algorithm —", True, TEXT_COLOR)
        self.screen.blit(lbl, lbl.get_rect(center=(SCREEN_W//2, int(SCREEN_H*0.43))))

        # Highlight selected algo
        for name, btn in self.algo_buttons:
            is_sel = (name == self.selected_algo)
            if is_sel:
                sel_rect = btn.rect.inflate(8,8)
                pygame.draw.rect(self.screen, YELLOW, sel_rect, 2, border_radius=12)
            btn.draw(self.screen, self.font_btn, self.font_small)

        # Section: Pac-Man mode
        lbl2 = self.font_sub.render("— Pac-Man Control Mode —", True, TEXT_COLOR)
        self.screen.blit(lbl2, lbl2.get_rect(center=(SCREEN_W//2, int(SCREEN_H*0.62))))
        for mode, mbtn in self.mode_buttons:
            mbtn.draw(self.screen, self.font_btn)

        # Algorithm weakness hint
        weaknesses = {
            "Random":   "Weaknesses: unpredictable, hard to exploit",
            "A* Chase": "Weaknesses: predictable path, exploitable corridors",
            "Dijkstra": "Weaknesses: same as A* on uniform grid, high node expansion",
            "Greedy":   "Weaknesses: U-shaped walls cause loops, easily trapped",
            "Manhattan":"Weaknesses: cannot navigate walls, oscillates indefinitely",
        }
        hint = weaknesses.get(self.selected_algo, "")
        h = self.font_small.render(hint, True, (180,180,100))
        self.screen.blit(h, h.get_rect(center=(SCREEN_W//2, int(SCREEN_H*0.58))))

        # RL hint
        if self.selected_mode == "RL Agent":
            rl_hint = self.font_small.render(
                f"RL trains live: {RL_PARALLEL_SIMS} parallel sims/batch — learns to exploit chosen AI!",
                True, (100,220,100))
            self.screen.blit(rl_hint, rl_hint.get_rect(center=(SCREEN_W//2, int(SCREEN_H*0.78))))

        # Start button
        sc = (60,160,60) if self._start_hover else (40,120,40)
        pygame.draw.rect(self.screen, sc, self.start_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, self.start_rect, 2, border_radius=10)
        st = self.font_sub.render("▶  START GAME", True, WHITE)
        self.screen.blit(st, st.get_rect(center=self.start_rect.center))

        ctrl = self.font_small.render(
            "Arrow Keys = Move   |   R = Restart   |   Q = Quit", True, DIM)
        self.screen.blit(ctrl, ctrl.get_rect(center=(SCREEN_W//2, SCREEN_H-12)))

        pygame.display.flip()


# ---------------------------------------------------------------------------
# GAME ENGINE
# ---------------------------------------------------------------------------

class GameEngine:
    def __init__(self, ghost_strategy_name: str, pacman_mode: str):
        self.ghost_strategy_name = ghost_strategy_name
        self.pacman_mode         = pacman_mode
        self.font_big   = pygame.font.SysFont("monospace", 19, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12)
        self.font_tiny  = pygame.font.SysFont("monospace", 11)
        self.rl_agent     = QLearningAgent() if pacman_mode == "RL Agent" else None
        self._train_timer = 0
        self.fast_mode    = False
        self._end_timer   = 0
        self._speed_btn   = pygame.Rect(SCREEN_W//2 - 55, GRID_ROWS*TILE_SIZE + 70, 110, 26)
        self.reset()

    def _new_ghost_strategy(self):
        cls, _ = ALGORITHM_REGISTRY[self.ghost_strategy_name]
        return cls()

    def reset(self):
        self.game_map    = Map(DEFAULT_MAP)
        if self.pacman_mode == "Manual":
            self.manual_ctrl = ManualControl()
            pac_strat = self.manual_ctrl
        else:
            self.manual_ctrl = None
            self.rl_strat    = RLMoveStrategy(self.rl_agent)
            pac_strat        = self.rl_strat

        self.pacman = PacMan(10, 16, pac_strat)
        ghost_starts = [(9,9,"Blinky"),(10,9,"Pinky"),(11,9,"Inky"),(10,10,"Clyde")]
        self.ghosts: List[Ghost] = [
            Ghost(c, r, self._new_ghost_strategy(), GHOST_COLORS[i%4], name)
            for i, (c, r, name) in enumerate(ghost_starts)
        ]
        self.game_over      = False
        self.victory        = False
        self._fright_active = False
        self._frame         = 0
        self._end_timer     = 0

    def run(self, screen, clock):
        running = True
        while running:
            running = self._handle_events()
            steps = self._steps_per_frame()
            for _ in range(steps):
                if self.game_over or self.victory:
                    self._end_timer += 1
                    # Auto-restart: brief pause so you can see the result
                    pause = 6 if self.fast_mode else 75
                    if self._end_timer >= pause:
                        self.reset()
                    break
                self._update()
                self._check_collisions()
            self._render(screen)
            clock.tick(0 if self.fast_mode else FPS)

    def _steps_per_frame(self):
        if not self.fast_mode:
            return 1
        gen = self.rl_agent.generation if self.rl_agent else 0
        return min(20 + gen * 4, 120)

    def _handle_events(self) -> bool:
        key_to_dir = {
            pygame.K_UP: DIR_UP, pygame.K_DOWN: DIR_DOWN,
            pygame.K_LEFT: DIR_LEFT, pygame.K_RIGHT: DIR_RIGHT,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if self.manual_ctrl and event.key in key_to_dir:
                    self.manual_ctrl.queue_direction(key_to_dir[event.key])
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_s:
                    self.fast_mode = not self.fast_mode
                if event.key == pygame.K_q:
                    return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self._speed_btn.collidepoint(event.pos):
                    self.fast_mode = not self.fast_mode
        return True

    def _update(self):
        self._frame += 1

        # Fast mode: snap entities to tile centres so pixel interpolation
        # doesn't eat up steps sliding across sub-tile distances
        if self.fast_mode:
            for ent in [self.pacman, *self.ghosts]:
                ent.x = float(ent.grid_col * TILE_SIZE + TILE_SIZE // 2)
                ent.y = float(ent.grid_row * TILE_SIZE + TILE_SIZE // 2)

        # RL training: fire a new batch as soon as the last one finishes
        if self.rl_agent and not self.rl_agent.training_in_progress:
            self._train_timer += 1
            if self._train_timer >= RL_TRAIN_INTERVAL:
                self._train_timer = 0
                self.rl_agent.training_in_progress = True
                threading.Thread(
                    target=self.rl_agent.run_parallel_training,
                    args=(DEFAULT_MAP, self.ghost_strategy_name),
                    daemon=True).start()

        # Pass ghost positions to RL strategy
        if self.rl_agent and hasattr(self, 'rl_strat'):
            self.rl_strat._ghost_positions = [
                (g.grid_col, g.grid_row) for g in self.ghosts if not g._eaten]
            self.rl_strat._power_active = self._fright_active

        prev_score = self.pacman.score
        power_eaten = self.pacman.update(self.game_map)
        if self.pacman.score > prev_score and hasattr(self, 'rl_strat'):
            self.rl_strat.notify_ate_dot()   # reset stuck counter on any pellet
        if power_eaten:
            for ghost in self.ghosts:
                if not ghost._eaten:
                    ghost.frighten()
            self._fright_active = True

        pac_pos = (self.pacman.grid_col, self.pacman.grid_row)
        dt = 1.0 / FPS
        for ghost in self.ghosts:
            ghost.update(self.game_map, target_pos=pac_pos)
            ghost.update_fright(dt)

        self._fright_active = any(g.frightened and not g._eaten for g in self.ghosts)
        if self.game_map.remaining_pellets() == 0:
            self.victory = True

    def _check_collisions(self):
        px, py = self.pacman.pixel_pos()
        for ghost in self.ghosts:
            if ghost._eaten:
                continue
            gx, gy = ghost.pixel_pos()
            if math.hypot(px-gx, py-gy) < PacMan.RADIUS + Ghost.RADIUS - 4:
                if ghost.frightened:
                    ghost._eaten = True
                    ghost.recover()
                    self.pacman.score += GHOST_EAT_SCORE
                else:
                    self.pacman.lives -= 1
                    if self.pacman.lives <= 0:
                        self.game_over = True
                    else:
                        # Reset Pac-Man to spawn
                        self.pacman.grid_col = 10
                        self.pacman.grid_row = 16
                        self.pacman.x = float(10*TILE_SIZE + TILE_SIZE//2)
                        self.pacman.y = float(16*TILE_SIZE + TILE_SIZE//2)
                        self.pacman.direction = DIR_RIGHT
                        # Send all ghosts back to their home tiles
                        for g in self.ghosts:
                            g.respawn()
                        self._fright_active = False

    def _render(self, surface):
        surface.fill(DARK_BLUE)
        self.game_map.draw(surface)
        self.pacman.draw(surface)
        for ghost in self.ghosts:
            ghost.draw(surface)

        # Draw ghost AI labels above each ghost
        for ghost in self.ghosts:
            if not ghost._eaten:
                gx, gy = ghost.pixel_pos()
                label = self.font_tiny.render(ghost.name, True, ghost.base_color)
                surface.blit(label, label.get_rect(midbottom=(gx, gy-Ghost.RADIUS-2)))

        # HUD
        hud_y = GRID_ROWS * TILE_SIZE
        pygame.draw.rect(surface, HUD_BG, pygame.Rect(0, hud_y, SCREEN_W, 100))

        # Row 1: Score + Lives
        surface.blit(self.font_big.render(
            f"SCORE  {self.pacman.score:>6}", True, TEXT_COLOR), (12, hud_y+6))
        surface.blit(self.font_big.render(
            f"LIVES  {'♥ ' * self.pacman.lives}", True, (255,80,80)), (12, hud_y+26))

        # Row 2: Algorithm info
        surface.blit(self.font_small.render(
            f"Ghost AI: {self.ghost_strategy_name}", True, CYAN), (12, hud_y+50))

        # Pac-Man mode
        mode_col = (100,220,100) if self.pacman_mode=="RL Agent" else (220,220,100)
        surface.blit(self.font_small.render(
            f"Pac: {self.pacman_mode}", True, mode_col), (12, hud_y+66))

        # Algorithm weakness reminder
        weaknesses = {
            "Random":   "Exploit: unpredictable",
            "A* Chase": "Exploit: corridor loops",
            "Dijkstra": "Exploit: same as A*",
            "Greedy":   "Exploit: U-shaped walls",
            "Manhattan":"Exploit: hide behind walls",
        }
        w = weaknesses.get(self.ghost_strategy_name, "")
        surface.blit(self.font_tiny.render(w, True, (200,200,80)), (12, hud_y+82))

        # Right side: RL stats or controls
        rx = SCREEN_W - 200
        if self.rl_agent:
            self._draw_rl_hud(surface, rx, hud_y)
        else:
            surface.blit(self.font_small.render("R=Restart  Q=Quit", True, DIM),
                         (rx+20, hud_y+10))
            surface.blit(self.font_small.render("Arrow Keys = Move", True, DIM),
                         (rx+20, hud_y+28))

        # Frightened timer
        if self._fright_active:
            max_t = max((g._fright_timer for g in self.ghosts
                         if g.frightened and not g._eaten), default=0.0)
            secs = math.ceil(max_t)
            tc = RED_CLR if max_t <= FRIGHTENED_FLASH_START else GHOST_FRIGHTENED_FLASH
            ts = self.font_big.render(f"POWER  {secs:>2}s", True, tc)
            surface.blit(ts, ts.get_rect(center=(SCREEN_W//2, hud_y+28)))

        # Speed toggle button — centre of HUD
        btn_lbl = "⚡ FAST" if not self.fast_mode else "👁  SLOW"
        btn_col = (35, 55, 130) if not self.fast_mode else (130, 55, 35)
        if self._speed_btn.collidepoint(pygame.mouse.get_pos()):
            btn_col = tuple(min(255, c+45) for c in btn_col)
        pygame.draw.rect(surface, btn_col, self._speed_btn, border_radius=6)
        pygame.draw.rect(surface, WHITE, self._speed_btn, 1, border_radius=6)
        bs = self.font_small.render(btn_lbl, True, WHITE)
        surface.blit(bs, bs.get_rect(center=self._speed_btn.center))
        hs = self.font_tiny.render("S key", True, DIM)
        surface.blit(hs, hs.get_rect(
            center=(self._speed_btn.centerx, self._speed_btn.bottom + 7)))

        # In fast mode skip the blocking overlay — auto-restart handles it
        if self.fast_mode and self.rl_agent:
            if self.game_over or self.victory:
                pass  # just let auto-restart fire; screen keeps showing last frame
        elif self.game_over:
            self._draw_overlay(surface, "GAME OVER", RED_CLR)
        elif self.victory:
            self._draw_overlay(surface, "YOU WIN!", PAC_COLOR)

        pygame.display.flip()

    def _draw_rl_hud(self, surface, rx, hud_y):
        """Draw RL training stats in HUD."""
        agent = self.rl_agent
        gen_col = GREEN if agent.generation > 0 else DIM
        spd = f" ×{self._steps_per_frame()}" if self.fast_mode else " ×1"
        surface.blit(self.font_small.render(
            f"Gen:{agent.generation} ε:{agent.epsilon:.2f} α:{agent.alpha:.2f}{spd}", True, gen_col),
            (rx, hud_y+6))
        surface.blit(self.font_small.render(
            f"Best: {agent.best_score:>5}  Q-states: {len(agent.q_table):>4}",
            True, (180,220,255)), (rx, hud_y+22))

        # Mini bar chart of last batch sim scores
        if agent.last_train_scores:
            bar_x = rx
            bar_y = hud_y + 42
            max_s = max(agent.last_train_scores) if max(agent.last_train_scores)>0 else 1
            bar_w = 180 // RL_PARALLEL_SIMS - 2
            for i, sc in enumerate(agent.last_train_scores):
                bh = max(2, int(sc / max_s * 28))
                col = GREEN if sc == max(agent.last_train_scores) else (100,140,200)
                pygame.draw.rect(surface, col,
                    pygame.Rect(bar_x + i*(bar_w+2), bar_y+28-bh, bar_w, bh))
            surface.blit(self.font_tiny.render(
                "Parallel sims (best=green)", True, DIM), (rx, hud_y+74))
        else:
            surface.blit(self.font_small.render("Training...", True, ORANGE),
                         (rx, hud_y+44))

        if agent.training_in_progress:
            t = int(pygame.time.get_ticks()/200)
            dots = "." * (t%4)
            surface.blit(self.font_tiny.render(f"Training{dots}", True, ORANGE),
                         (rx, hud_y+62))

    def _draw_overlay(self, surface, text, color):
        overlay = pygame.Surface((SCREEN_W, GRID_ROWS*TILE_SIZE), pygame.SRCALPHA)
        overlay.fill((0,0,0,160))
        surface.blit(overlay,(0,0))
        big = pygame.font.SysFont("monospace",48,bold=True)
        txt = big.render(text, True, color)
        surface.blit(txt, txt.get_rect(center=(SCREEN_W//2, GRID_ROWS*TILE_SIZE//2)))
        sub = self.font_big.render("Auto-restarting...  Q=Quit", True, WHITE)
        surface.blit(sub, sub.get_rect(
            center=(SCREEN_W//2, GRID_ROWS*TILE_SIZE//2+60)))
        if self.rl_agent:
            rl_note = self.font_small.render(
                f"RL Agent — Gen {self.rl_agent.generation}, Best Score {self.rl_agent.best_score}",
                True, (100,220,100))
            surface.blit(rl_note, rl_note.get_rect(
                center=(SCREEN_W//2, GRID_ROWS*TILE_SIZE//2+90)))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Pac-Man — AI Showroom")
    clock  = pygame.time.Clock()

    while True:
        chosen_algo, chosen_mode = StartScreen(screen, clock).run()
        GameEngine(ghost_strategy_name=chosen_algo,
                   pacman_mode=chosen_mode).run(screen, clock)

    pygame.quit()
    sys.exit()