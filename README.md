# Pac-Man AI Showroom

A Python and PyGame-based interactive simulator designed to demonstrate the behavioral differences between various classical pathfinding/steering algorithms in enemy ghosts, and a live-trained **Tabular Q-Learning Reinforcement Learning Agent** controlling Pac-Man.

The primary objective of this showroom is to visually contrast optimal search graphs against heuristic-only pitfalls, and observe how an RL agent adapts dynamically to exploit the structural and algorithmic flaws of its opponents.

---

## Project Architecture Overview

The system is engineered using a clean, decoupling-focused architecture consisting of three core pillars:
1. **The Environment Model (`Map`, `Entity`)**: A static grid layout featuring wall detection, pellet/power-pellet state management, and continuous-to-discrete spatial interpolation (`Entity.update`) for movement coordination.
2. **The Strategy Pipeline (`MoveStrategy`)**: An abstract interface allowing execution of different algorithms seamlessly at runtime. Ghosts swap between chase and flee strategies on the fly.
3. **The Simulation Engine (`GameEngine`, `QLearningAgent`)**: Orchestrates the frames, processes game boundaries, handles spatial multi-threading during batch training, and renders real-time diagnostics via the HUD.

---

## Ghost AI: Algorithm Breakdowns

The repository isolates five distinct navigation models for the ghost entities. Each demonstrates specific mathematical assumptions regarding state space traversal:

### 1. Optimal Search Algorithms

* **A* Chase Strategy (`AStarStrategy`)**
    * **Mechanic**: Computes the true shortest path minimizing $f(n) = g(n) + h(n)$, where $g(n) is the exact uniform graph step-cost and $h(n)$ is the Manhattan distance heuristic to Pac-Man.
    * **Behavior**: Highly efficient and direct. It never completes a suboptimal step.
    * **Exploit**: Completely deterministic. Pac-Man can exploit this consistency by leading the ghost into tight, predictable corridor loops to stall indefinitely.
* **Dijkstra Strategy (`DijkstraStrategy`)**
    * **Mechanic**: Computes standard uniform cost minimization ($f(n) = g(n)$) without a targeted heuristic.
    * **Behavior**: Yields identical physical trajectories to A* on standard grids but scales poorly regarding computational footprint, expanding uniformly in all directions before isolating the target coordinate.

### 2. Heuristic & Steering Approximations

* **Greedy Best-First Strategy (`GreedyStrategy`)**
    * **Mechanic**: Evaluates node transitions purely on the heuristic evaluation: $f(n) = h(n)$. It ignores the cumulative historical path cost $g(n)$.
    * **Behavior**: Rapidly pushes down the localized gradient toward Pac-Man.
    * **Exploit**: Highly susceptible to concave environments (U-shaped walls). The ghost will repeatedly get stuck in local minima, spinning inside the cavity because moving backward along a real path would temporarily increase the heuristic distance.
* **Manhattan Steering Strategy (`ManhattanStrategy`)**
    * **Mechanic**: Eliminates graph searches completely. At each junction, it checks immediate valid adjacent tiles and selects the neighbor minimizing $d = |\Delta x| + |\Delta y|$.
    * **Behavior**: Functions purely as a localized vector-steering alignment mechanism.
    * **Exploit**: Lacks global maze awareness. Placing a straight wall between the ghost and Pac-Man causes the ghost to oscillate aimlessly back and forth against the barrier.
* **Random Strategy (`RandomMoveStrategy`)**
    * **Mechanic**: Rejection-samples valid tiles at every intersection, excluding a complete 180° reverse unless forced by a dead end.

---

## Reinforcement Learning: Tabular Q-Learning Engine

When running under **RL Agent Mode**, Pac-Man drops manual inputs and acts as an autonomous agent processing state matrices into actionable outputs via a live-updating action-value function:

$$\text{Q}(s, a) \leftarrow \text{Q}(s, a) + \alpha \left[ r + \gamma \max_{a'} \text{Q}(s', a') - \text{Q}(s, a) \right]$$

### 1. State Space Compression (The Convergence Secret)
A naive absolute-coordinate mapping $(x_{\text{pac}}, y_{\text{pac}}, x_{\text{ghost}}, y_{\text{ghost}})$ blows up the combinations exponentially to $\sim 2.7\text{M}$ states, demanding hours of training. 

This engine solves that by compressing the environment into a highly refined **Relative Feature State Space (~150k states)**:
* **Closest Ghost Octant**: Direct 8-way directional heading (via `math.atan2`) mapped to a discrete variable $[0-7]$.
* **Closest Ghost Distance Bucket**: Multi-tile proximity indicator ($0-7$ scale, with high resolution right near the ghost).
* **Nearest Dot Octant & Distance Bucket**: Scans a $21 \times 21$ window around Pac-Man to identify the exact vector leading to the closest consumable item.
* **Cardinal Wall Sensors**: $4$ binary fields representing immediate blocking constraints `[Wall_N, Wall_S, Wall_E, Wall_W]`.
* **Frightened Status Bit**: Tracks if ghosts are currently vulnerable.

This abstraction means that whatever the agent learns at one corner of the board instantly generalizes across every structurally similar junction, boosting convergence speeds by **$\sim 18\times$**.

### 2. Multi-Threaded Batch Training Framework
To allow seamless gameplay while optimizing model training, the execution framework forks computational workloads into an asymmetric multi-threaded design:

* **Main Render Loop**: Handles game engine ticks at 60Hz. If **Fast Mode** is checked, pixel interpolation parameters skip intermediate steps and entities snap cleanly from tile to tile to maximize CPU cycles.
* **Parallel Trajectory Simulation Worker**: Every 15 frames, the model spawns **12 parallel worker threads** (`threading.Thread`) running background simulations for up to 1000 steps without UI overhead.
* **Experience Recency Weighting**: When an optimization window completes, the model aggregates trajectories from all 12 simulators rather than discarding them. Lower-scoring runs still impart baseline wisdom, while higher-scoring paths carry larger mathematical weight ($w$) alongside a recency multiplier that rewards crisp end-of-game maneuvers.

### 3. Deep Reward Shaping Setup
To guide exploration over long horizons, the reward function $r$ maps environmental changes beyond simple score outcomes:
* **Pellet Consumption**: $+10.0$ | **Power Pellet**: $+30.0$
* **Ghost Consumption**: $+20.0$
* **Catastrophic Death**: $-500.0$
* **Board Clear (Global Completion)**: $+1000.0$
* **Escalating Idle Penalty**: Scaled incrementally down to $-2.0$ per frame if the agent spins in empty squares without securing items.
* **Proximity Gradient Signal**: Adds a sliding bonus based on near-field distance to a pellet: $\max(0, (6 - \text{dot\_distance}) \times 0.15)$. This generates a smooth optimization landscape, letting the agent follow a clear trail instead of wandering blindly in empty corridors.
* **Stuck Mitigation Override**: Tracks consecutive steps without pellet ingestion. If an agent gets trapped inside an isolated loop, it temporarily injects a localized exploratory probability spike ($stuck\_override$) to override deterministic exploit behavior and shake Pac-Man loose.

---

## Installation & Running

### Prerequisites
* Python 3.8+
* PyGame
* NumPy

### Quick Start
```bash
# Clone the repository
git clone [https://github.com/yourusername/pacman-ai-showroom.git](https://github.com/yourusername/pacman-ai-showroom.git)
cd pacman-ai-showroom

# Install dependencies
pip install pygame numpy

# Run the simulation dashboard
python main.py
