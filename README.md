
# **Team Orienteering Problem With Communication Constraints**

- **Multi-objective optimization** with Pareto front evolution tracking
- **Wireless connectivity modeling** with realistic RSSI calculations
- **Flexible agent configuration** supporting variable numbers of agents
- **Comprehensive visualization tools**:
  - 3D Pareto front plotting and animation
  - Agent path visualization with reward collection
  - Real-time connectivity monitoring
  - Solution convergence analysis
- **Benchmarking support** with performance comparison tools
- **Reproducible experiments** with configurable random seedsraints**

This project addresses the **Team Orienteering Problem (TOP)** with wireless connectivity constraints between agents. A **Multi-Objective Variable Neighborhood Search (MOVNS)** algorithm is used to simultaneously optimize three objectives: reward collection, wireless signal strength (RSSI), and path efficiency.

## **Problem Description**

The Multi-Agent Team Orienteering Problem with Connectivity Constraints involves:

- **Multiple autonomous agents** navigating in a 2D environment
- **Reward points** distributed across the environment that agents must collect
- **Wireless connectivity constraints** requiring agents to maintain communication quality (RSSI) above a threshold
- **Budget constraints** limiting the total travel distance for each agent
- **Multi-objective optimization** balancing reward collection, connectivity, and travel efficiency

This problem is particularly relevant for:
- Search and rescue operations requiring coordinated team communication
- Environmental monitoring with distributed sensor networks
- Exploration missions where agents must maintain contact with base stations
- Any scenario where autonomous agents must work collaboratively while maintaining reliable communication

## **Motivation**

Traditional path planning approaches often optimize single objectives and fail to account for real-world communication constraints. In many robotic applications, maintaining wireless connectivity between agents is crucial for:

1. **Coordination**: Agents need to share information and coordinate actions
2. **Safety**: Communication enables emergency response and fault detection
3. **Efficiency**: Shared knowledge improves overall mission performance
4. **Reliability**: Redundant communication paths increase system robustness

This project addresses the gap by providing a multi-objective optimization framework that explicitly considers wireless signal strength alongside traditional objectives like reward collection and path length.

## **Solution Approach**

### **Multi-Objective Variable Neighborhood Search (MOVNS)**

The solution implements a sophisticated metaheuristic algorithm with the following components:

1. **Three-Objective Optimization**:
   - **Reward Maximization**: Total value collected from visited points
   - **RSSI Maximization**: Wireless signal strength between agent pairs
   - **Path Length Minimization**: Total travel distance (energy efficiency)

2. **Variable Neighborhood Search**:
   - Multiple neighborhood operators for solution exploration
   - Local search procedures for solution improvement
   - Perturbation mechanisms to escape local optima

3. **Pareto Front Management**:
   - Non-dominated solution archive with crowding distance selection
   - Dynamic archive size management (default: 40 solutions)
   - Probabilistic selection from Pareto front vs dominated solutions

4. **Advanced Evaluation**:
   - RSSI calculation based on distance
   - Connectivity constraint enforcement
   - Budget constraint validation

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
Install the following Python packages:

```bash
# Create conda environment
conda create -n topcc python=3.12
conda activate topcc

# Install required packages
conda install numpy matplotlib scipy scikit-learn tqdm numba
```

Or use the provided environment file:
```bash
conda env create -f environment.yml
conda activate topcc
```

---

## **Usage**

### **Basic Execution**

Run the optimization with default settings:
```bash
python main.py
```

### **Command Line Arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--map <file>` | Problem instance map file | Required |
| `--seed <value>` | Random seed for reproducibility | `42` |
| `--total-time <seconds>` | Maximum execution time | `60` |
| `--algorithm <name>` | Algorithm variant (e.g., "unique_vis") | `"unique_vis"` |
| `--out <directory>` | Output directory for results | `"out/"` |
| `--budget <values>` | Agent budget constraints | From map file |
| `--speeds <values>` | Agent speed parameters | From map file |

### **Examples**

**Run with specific map and time limit:**
```bash
python main.py --map maps/1.txt --total-time 120
```

**Extended optimization run:**
```bash
python main.py --map maps/1.txt --total-time 300 --algorithm unique_vis
```

---

## **Project Structure**

```
├── main.py                 # Main execution script
├── src/                    # Core algorithm implementation
│   ├── movns.py           # MOVNS algorithm
│   ├── evaluation.py      # Multi-objective evaluation functions
│   ├── operators.py       # Neighborhood operators and local search
│   └── entities.py        # Data structures (Solution, Neighborhood)
├── plot.py                # Visualization utilities
├── utils.py               # Helper functions
├── experiments/           # Experimental analysis notebooks
├── benchmarks/            # Standard benchmark instances
├── data/                  # Problem instance data
└── docs/                  # Algorithm documentation
```

---

## **Algorithm Performance**

See `docs/complexity_analysis.md` for detailed complexity analysis.

---

## **Contributing**

This project is part of ongoing research in multi-agent systems and wireless robotics. For questions or contributions, please refer to the experimental notebooks in `experiments/` for detailed analysis examples.
