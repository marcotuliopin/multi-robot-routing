# Complexity Analysis of Multi-Objective Variable Neighborhood Search (MOVNS)

## Overview

This document provides a thorough complexity analysis of the Multi-Objective Variable Neighborhood Search algorithm for multi-agent path planning with connectivity constraints.

## Problem Parameters

- **n**: Number of reward points
- **k**: Number of agents  
- **T**: Maximum execution time
- **I**: Maximum iterations
- **A**: Archive size (default: 40)
- **N**: Number of neighborhood operators (fixed value)

## Time Complexity Analysis

### 1. Solution Initialization

**Single Solution Creation**: O(n²)
- Path matrix initialization: O(kn)
- Path bounding for all agents: O(k × n²) due to distance calculations
- Evaluation: O(k³n) for reward calculation and RSSI computation

**Archive Initialization**: O(k × n⁴)
- For each neighborhood operator, generates O(n²) neighbors per agent: O(k × n²) total neighbors
- Each neighbor evaluation: O(k³ x n) 
- Total evaluation cost: O(k × n² × (k³n)) = O(k⁴ × n³)
- Archive update: O(A²) for dominance checking
- Since N is constant: **Dominant term is O(k⁴ × n³)**

### 2. Main Loop Components

#### Solution Selection: O(A)
- Candidate filtering: O(A)
- Probability calculation: O(A)
- Selection: O(1)

#### Perturbation Operators: O(kn)
- `invert_points_all_agents_unique`: O(kn)
- `two_opt_all_paths`: O(kn)

#### Local Search Operators (per agent): O(n²)
- `move_point`: O(n²) - generates O(n²) neighbors
- `swap_points`: O(n²) - generates O(n²) neighbors  
- `two_opt`: O(n²) - generates O(n²) neighbors
- `invert_single_point_unique`: O(n) - generates O(n) neighbors
- `add_and_move`: O(n²) - generates O(n²) neighbors

#### Evaluation Function: O(k³n)
- Reward calculation: O(n)
- Time calculation: O(kn) 
- Position interpolation: O(k²n log n)
- RSSI calculation: O(k³n) - **dominant term**
- Path length: O(kn)

**Three Objectives Being Optimized:**
1. **Reward Maximization**: Total reward collected from visited points
2. **Connectivity Maximization**: RSSI signal strength between agents (communication quality)
3. **Path Length Minimization**: Total travel distance (negated for maximization)

#### Archive Update: O(A²)
- Dominance checking: O(A²) where each comparison involves 3 objectives (constant)
- Non-dominated sorting: O(A²) using fast dominance-based sorting
- Crowding distance: O(A × log(A)) due to sorting by each objective

### 3. Overall Time Complexity

**Per Iteration (Original)**: 
- Solution selection: O(A)
- For each neighborhood (N iterations, where N is constant):
  - Perturbation: O(kn)
  - Neighbor generation: O(kn²) per neighborhood
  - Neighbor evaluation: O(kn² × (k³n)) per neighborhood
  - Archive update: O(A²)

**Total per iteration**: O(A + N × (kn + kn² + kn² × (k³n)) + A²)
**Simplified**: O(A + N × k⁴ × n³ + A²)
**Since N is constant**: O(k⁴ × n³ + A²)

**Total Algorithm**: O(I × (k⁴ × n³ + A²))
**Dominant Term**: O(I × k⁴n³)

## Space Complexity Analysis

### 1. Solution Representation
- Path matrix: O(kn) per solution
- Evaluation scores: O(1) per solution (3 objectives)
- Additional metadata: O(1) per solution

### 2. Archive and Populations
- Archive: O(A × kn)
- Neighbors per iteration: O(N x k × n² × kn) = O(k² × n³) (since N is constant)
- Working memory for operators: O(kn)

### 3. Problem Data
- Distance matrix: O(n²)
- Reward positions: O(n)
- Reward values: O(n)

**Total Space Complexity**: O(k² × n³ + n²)

## Complexity Bottlenecks

### 1. Critical Operations
1. **Neighbor Generation**: O(k × n²) per iteration (since N is constant)
2. **Solution Evaluation**: O(k³n) for RSSI calculation  

### 2. Scalability Limits
- **Number of Rewards (n)**: Cubic growth in worst case
- **Number of Agents (k)**: Quartic growth when k ≈ n
- **Archive Size (A)**: Quadratic impact on dominance checking

## Conclusion

The MOVNS algorithm exhibits polynomial complexity in most practical scenarios, with cubic dependence on the number of rewards being the primary scalability concern. The multi-objective nature and constraint handling add significant computational overhead but provide valuable solution diversity. For large-scale problems (n > 100, k > 10), algorithmic and implementation optimizations become critical for practical application.
