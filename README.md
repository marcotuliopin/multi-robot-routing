
# **IC-Robotics**

This project addresses the **Team Orienteering Problem (TOP)** with the added complexity of maintaining a distance constraint between two agents. A **multi-objective Genetic Algorithm (GA)** is used to optimize both the rewards collected by the agents and the total distance traveled. The approach incorporates clustering, path initialization, and advanced visualization tools.

## **Features**

- **Multi-objective Genetic Algorithm**: Optimizes rewards collected and distance traveled by two agents.
- **Distance Constraint**: Ensures the agents remain within a specified maximum distance from each other during task execution.
- **Clustering Initialization**: Improves path efficiency by grouping rewards into clusters for task assignment.
- **Visualization Tools**:
  - Path plotting for both agents.
  - Distance monitoring between agents over time.
  - Real-time animation of the agents’ movement.
- **Simulation Integration**: Compatible with CoppeliaSim for realistic robot movement.

---

## **Setup**

### **Requirements**
To run the project, ensure the following Python packages are installed:
- `numpy`
- `matplotlib`
- `matplotlib.animation`
- `deap`
- `sklearn.neighbors`
- `scipy.spatial`

## **Usage**

Run the script from the command line with the desired options:

```bash
python main.py [OPTIONS]
```

### **Arguments**

| Argument             | Description                                                                                     | Default          |
|----------------------|-------------------------------------------------------------------------------------------------|------------------|
| `--seed <value>`     | Sets the random seed for reproducibility.                                                       | `42`             |
| `--plot-path`        | Plots the best paths found by the GA for both agents.                                           | Disabled         |
| `--plot-distances`   | Plots the distance between the two agents over the entire path.                                 | Disabled         |
| `--save-plot <file>` | Saves the plotted paths or distance plots to the specified file.                                | Disabled         |
| `--run-animation`    | Runs an animation showing the agents moving along their respective paths in real time.          | Disabled         |

### **Examples**

- **Run with default settings**:
  ```bash
  python main.py
  ```

- **Run with a specific seed and save the path plot**:
  ```bash
  python main.py --seed 123 --plot-path --save-plot path_plot.png
  ```

- **Run with animation enabled**:
  ```bash
  python main.py --run-animation
  ```

---

## **Algorithm Overview**

1. **Input**: A set of rewards distributed over a 2D grid and constraints for agent collaboration.
3. **Genetic Algorithm**:
   - **Representation**: Each individual represents two paths, one for each agent.
   - **Fitness Function**:
     - Maximizes the rewards collected by both agents.
     - Minimizes the total distance traveled by the agents.
     - Penalizes solutions where agents exceed the maximum distance constraint.
   - **Operators**: Includes PMX crossover and mutation designed for permutation problems.
4. **Output**: The best paths for both agents and their corresponding performance metrics.

---

## **Visualization**

### **1. Path Plotting**
Visualizes the paths for both agents, the rewards collected, and the clusters. Use the `--plot-path` argument to enable.

### **2. Distance Monitoring**
Plots the distance between the two agents during the execution of their paths. Use the `--plot-distances` argument to enable.

### **3. Animation**
Simulates the movement of both agents in real time. Requires CoppeliaSim to be running with the appropriate scene. Use the `--run-animation` argument to enable.
