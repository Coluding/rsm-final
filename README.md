# Optimizing State Machine Replication with Reinforcement Learning and Graph Neural Networks

## Abstract 
Replicated state machines (RSMs) are a core abstraction for building fault-
tolerant distributed services, but their performance in wide-area deployments is highly sen-
sitive to replica placement. Static placement strategies fail to adapt to dynamic and geo-
graphically shifting client demand, while existing dynamic approaches rely on greedy, myopic
reconfiguration heuristics with non-negligible runtime overhead.
In this work, we study dynamic replica placement as a stochastic location-allocation problem
and formalize it within an optimization framework related to dynamic facility location. We
propose a reinforcement learning-based solution that learns replica reconfiguration policies
offline from simulated system trajectories and enables constant-time decision making at run-
time. The approach combines spatio-temporal graph representations with policy optimization
to capture global communication effects and long-term trade-offs. Experimental results using
a full-system simulation show that the learned policy consistently outperforms both static
placement and the existing greedy baseline, with increasing benefits for longer horizons and
larger configuration spaces. These findings demonstrate that learning-based optimization is
a practical and effective alternative for dynamic replica management in wide-area replicated
systems

This repository contains the implementation of a research project that leverages **Reinforcement Learning (RL)** and **Graph Neural Networks (GNNs)** to optimize distributed **Byzantine Fault-Tolerant (BFT) State Machine Replication** systems. The system trains RL agents to make intelligent reconfiguration decisions (e.g., swapping active and passive replica locations) to minimize client perceived latency.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Java Connection Details](#java-connection-details)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Run Configurations](#run-configurations)
7. [Running the Code](#running-the-code)
8. [Monitoring and Logging](#monitoring-and-logging)

## Project Overview

The core objective of this project is to automate the reconfiguration of distributed systems. Replicas in a BFT system can be in different states (Active or Passive) and located in different geographical regions. The RL agent observes the current state of the system (network latencies, request loads, current topology) as a graph and outputs actions to swap locations of active and passive replicas.

Key Technologies:
- **Python**: Primary language for RL and GNN.
- **PyTorch & PyTorch Geometric**: Frameworks for neural networks and graph processing.
- **Java**: The underlying BFT-SMaRt simulator is written in Java.
- **JPype**: Bridges Python and Java, allowing the RL agent to interact directly with the Java simulator in memory.

## Architecture

The system consists of three main components:

1.  **Fluidity Environment (`src/environment`)**: A Gym-compliant environment that wraps the Java simulator. It converts raw simulation data into graph observations (`torch_geometric.data.Data`).
2.  **Java Simulator**: A custom extension of BFT-SMaRt that simulates network conditions and protocol execution.
3.  **RL Agent (`src/algorithm`)**: PPO, DQN, or REINFORCE agents that learn policies to optimize the environment's reward (latency).

## Java Connection Details

The connection between the Python RL environment and the Java BFT-SMaRt simulator is established using **JPype**. This allows for high-performance, in-process communication avoiding the overhead of external sockets or HTTP requests.

### implementation
The connection logic is encapsulated in `src/environment/java_conn.py`.
1.  **JVM Initialization**: The script starts a Java Virtual Machine (JVM) within the Python process using `jpype.startJVM`.
2.  **Classpath**: It includes the compiled simulator JAR file (`ressources/jars/simulator-xmr-*-jar-with-dependencies.jar`).
3.  **Class Loading**: Python proxies for Java classes (e.g., `FluiditySimulator`, `FluiditySimulationConfiguration`) are created using `jpype.JClass`.
4.  **Interaction**:
    -   The `JavaSimulator` class instantiates the simulation engine.
    -   `step()` calls are forwarded to the Java method `execution.executeStep()`.
    -   Java objects (like Layouts, Latency Maps) are returned to Python and converted into native Python dictionaries/lists for the Gym environment.

**Important Note**: The JVM can only be started once per process. If you encounter JVM errors during development (e.g., when reloading modules in a notebook), you may need to restart the Python kernel.

## Project Structure

```
├── README.md                   # Project documentation
├── ressources/                 # Non-code resources
│   ├── jars/                   # Compiled Java simulator JARs
│   └── run_configs/            # Simulation configurations (scenarios)
├── src/                        # Source code
│   ├── algorithm/              # RL implementations (PPO, DQN, REINFORCE, Dreamer)
│   ├── environment/            # Gym environment and Java bridge
│   ├── models/                 # Neural network architectures (GNNs, Transformers)
│   ├── training.py             # Main entry point for training
│   └── utils/                  # Logging and helper utilities
└── runs/                       # TensorBoard log directories
```

## Installation

Ensure you have a Python environment set up (Python 3.9+ recommended).
You will need a Java Runtime Environment (JRE) installed to run the BFT-SMaRt simulator.

1.  **Install Dependencies**:
    ```bash
    pip install -r reqs.txt
    ```
    *Note: Ensure `jpype1` is installed successfully.*

2.  **Java Requirements**:
    Ensure `java` is in your system PATH. The simulator currently targets Java 11 or higher.

## Run Configurations

The simulation scenarios are defined in `ressources/run_configs/`. Each configuration folder contains specific settings for the distributed system, such as:
-   Number of replicas
-   Network topology (latency matrix)
-   Workload definition (client request patterns)

**Available Configurations:**
-   **`easy/`**: Simplified scenarios ideal for quick debugging and testing.
-   **`40_steps/`**: Scenarios designed for short episodes (40 simulation steps).
-   **`400_steps/`**: Longer episodes (400 simulation steps), more realistic for convergence testing.
- **1500_steps/**: Extended scenarios (1500 simulation steps) for in-depth training with 15 locations instead of just 8.
To use a specific configuration, pass the folder name (relative to `ressources/run_configs/`) to the `--run_config` argument.

## Running the Code

The main entry point is `src/training.py`. Below are examples for running different RL algorithms.

### 1. Proximal Policy Optimization (PPO) - Recommended
PPO is generally the most stable algorithm for this task.

```bash
python src/training.py \
  --algorithm ppo \
  --action_space cross_product \
  --run_config easy/ \
  --num_episodes 1000 \
  --batch_size 30 \
  --train_every 10 \
  --update_epochs 12 \
  --reduced_state_space True
```

### 2. Deep Q-Network (DQN)
Train a value-based agent.

```bash
python src/training.py \
  --algorithm dqn \
  --epsilon 1.0 \
  --epsilon_decay 0.999 \
  --batch_size 32 \
  --run_config 400_steps/
```

### 3. REINFORCE
A simpler policy gradient method.

```bash
python src/training.py \
  --algorithm reinforce \
  --action_space separate \
  --run_config 40_steps/
```

### Key Arguments Explained

-   `--algorithm`: The RL algorithm to use (`ppo`, `dqn`, `reinforce`).
-   `--run_config`: The specific simulation scenario folder in `ressources/run_configs/` (e.g., `easy/`, `40_steps/`).
-   `--action_space`:
    -   `separate`: The agent outputs two separate actions (location to add, location to remove).
    -   `cross_product`: The agent outputs a single action from a flattened action space (all valid combinations). **Recommended** for better learning stability.
-   `--num_episodes`: Total number of training episodes.
-   `--reduced_state_space`: If `True`, simplifies the observation features to focus on critical metrics.
-   `--load_path`: Path to a `.pth` file to resume training from a specific checkpoint.

## Monitoring and Logging

The training script automatically logs metrics to two services:

1.  **TensorBoard**:
    -   Logs are saved locally in the `runs/` directory.
    -   To view: `tensorboard --logdir runs/`

2.  **Weights & Biases (WandB)**:
    -   Remote logging is enabled by default.
    -   Project name: `replictated-state-machines-rl`.
    -   Ensure you have logged in via `wandb login` if you wish to track your experiments online.
