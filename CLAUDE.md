# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project that combines **Reinforcement Learning (RL)** with **Graph Neural Networks (GNNs)** to optimize distributed **Byzantine Fault-Tolerant (BFT) State Machine Replication** systems. The system trains RL agents to make intelligent reconfiguration decisions (location swaps) by learning which server locations should be active or passive replicas to minimize latency.

**Core technology stack:**
- Python with PyTorch 2.5.1 + PyTorch Geometric 2.6.1 (GNN framework)
- Java-based BFT-SMaRt simulator (connected via JPype bridge)
- Stable Baselines3, TorchRL for RL implementations
- WandB and TensorBoard for experiment tracking

## Training Commands

**Main entry point:** `src/training.py`

Train a PPO agent (recommended):
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

Train a DQN agent:
```bash
python src/training.py \
  --algorithm dqn \
  --epsilon 1.0 \
  --epsilon_decay 0.999 \
  --batch_size 32 \
  --run_config 400_steps/
```

Train a REINFORCE agent:
```bash
python src/training.py \
  --algorithm reinforce \
  --action_space separate \
  --run_config 40_steps/
```

Load and continue training from checkpoint:
```bash
python src/training.py \
  --algorithm ppo \
  --load_path src/best_policy/best_policy_40.pth
```

**Available run configurations** (in `ressources/run_configs/`):
- `easy/`: Simplified scenarios for quick testing
- `40_steps/`: 40-step short simulation runs
- `400_steps/`: Full 400-step simulation runs

## Key Architecture Components

### 1. Environment (`src/environment/fluidity_environment.py`)

The core Gym environment represents the distributed system as a **graph**:
- **Nodes**: Locations (servers/clients) with labels (0=inactive, 1=active replica, 2=passive replica)
- **Edges**: Network links with latency weights
- **State**: PyTorch Geometric `Data` object with graph structure
- **Actions**: Swap active/passive replica locations (separate mode or cross-product mode)
- **Reward**: Negative mean latency (lower latency = higher reward)

**Important wrappers:**
- `TorchGraphObservationWrapper`: Converts graph observations to PyTorch tensors
- `StackedBatchObservationWrapper`: Stacks temporal observations (default 4 timesteps)
- `FilteredActionSamplingWrapper`: Filters invalid actions

**Configuration options:**
- `reduced_state_space`: If True, simplifies state representation
- `use_reconfig_costs`: If True, adds penalty for reconfiguration actions

### 2. Java Bridge (`src/environment/java_conn.py`)

The `JavaSimulator` class communicates with the BFT-SMaRt Java simulator via JPype:
- Loads configuration from `ressources/run_configs/*/server0/xmr/config/`
- Executes simulation steps and returns metrics (latency matrices, request quantities, mean delay)
- **Performance note:** Java simulator is a bottleneck; training speed is limited by JVM communication

### 3. Neural Network Models (`src/models/`)

**Encoders** (`encoder.py`):
- `GCNEncoder`: Graph Convolutional Networks (2 layers)
- `GATEncoder`: Graph Attention Networks (multi-head, 3+ layers) - recommended for better performance

**Action Mappers** (`aggregator.py`):
- `SwapActionMapper`: Outputs 2D logits for separate add/remove actions
- `CrossProductSwapActionMapper`: Outputs 1D logits for cross-product action space (64 actions for 8 locations)

**Complete Models** (`model.py`):
- `StandardModel`: GCN/GAT encoder + action head (for separate action space)
- `StandardCrossProductModel`: Extended for cross-product action space
- `StandardValueModel`: Policy + value head for PPO
- `RSMDecisionTransformer`: Transformer-based decision transformer

### 4. RL Algorithms (`src/algorithm/`)

**DQN** (`off_policy.py`):
- Double DQN with target network
- Experience replay buffer (optional priority sampling)
- Epsilon-greedy exploration (epsilon decays over time)

**PPO** (`on_policy.py`):
- Proximal Policy Optimization with clipped surrogate objective
- Separate policy and value networks
- GAE (Generalized Advantage Estimation) for advantage computation
- Entropy regularization for exploration

**REINFORCE** (`on_policy.py`):
- Policy gradient with baseline
- Simpler than PPO but higher variance

**Dreamer** (`dreamer.py`):
- Model-based RL with RSSM (Recurrent State Space Model)
- Learns dynamics, reward, and decoder models
- More sample-efficient but experimental

### 5. Action Spaces

**Separate mode** (`--action_space separate`):
- Two independent actions: (1) add location to active set, (2) remove location from active set
- Output: 2 logit vectors of size `num_locations`

**Cross-product mode** (`--action_space cross_product`):
- Single combined action representing (add, remove) pairs
- 64 pre-computed valid actions stored in `src/data/action_space.json`
- Output: 1 logit vector of size 64
- **Recommended** for better action coordination

## Important Training Parameters

**Model hyperparameters:**
- `--type_embedding_dim`: Node type embedding dimension (default: 32)
- `--hidden_dim`: GNN hidden dimension (default: 256)
- `--num_heads`: Number of attention heads for GAT (default: 2)
- `--num_locations`: Number of server locations (fixed: 8)

**RL hyperparameters:**
- `--lr`: Policy learning rate (default: 3e-5)
- `--lr_value_fn`: Value function learning rate for PPO (default: 4e-3)
- `--gamma`: Discount factor (default: 0.97)
- `--batch_size`: Training batch size (default: 30)
- `--train_every`: Train every N episodes (default: 10)
- `--update_epochs`: Number of epochs per training update (default: 12)

**PPO-specific:**
- `--clip_epsilon`: PPO clipping parameter (default: 0.05)
- `--entropy_coeff`: Entropy regularization coefficient (default: 0.08)
- `--value_coeff`: Value loss coefficient (default: 1.5)

**DQN-specific:**
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon_decay`: Epsilon decay factor (default: 0.999)
- `--target_update`: Target network update frequency (default: 810 episodes)

**Environment options:**
- `--stack_states`: Number of temporal states to stack (default: 4)
- `--reduced_state_space`: Use simplified state representation (default: True)
- `--use_reconfig_costs`: Add reconfiguration penalty (default: False)
- `--run_config`: Simulation configuration directory (default: "easy/")

## Code Organization Principles

### Graph-First Design
All observations are PyTorch Geometric `Data` objects. Models must process graph-structured inputs:
```python
# Example observation structure
data = Data(
    x=node_features,        # [num_nodes, feature_dim]
    edge_index=edge_index,  # [2, num_edges]
    edge_attr=edge_weights  # [num_edges, 1]
)
```

### Action Masking
Invalid actions (e.g., swapping locations already in correct state) should be filtered:
- Use `FilteredActionSamplingWrapper` in environment
- Or implement masking in model's forward pass

### Temporal Context
The system uses temporal stacking (default 4 timesteps) to capture sequential dynamics:
- Observations are batched across time via `StackedBatchObservationWrapper`
- Models receive shape `[batch_size * stack_size, ...]`

### Experiment Tracking
All training runs are logged to both TensorBoard and WandB:
- TensorBoard logs: `runs/` directory
- WandB project: `replictated-state-machines-rl`
- Use `MetricLogger` utility (`src/utils/metric_logger.py`) for unified logging

## Common Development Patterns

### Adding a New RL Algorithm
1. Create new file in `src/algorithm/`
2. Implement agent class with `train()` method
3. Add configuration dataclass for hyperparameters
4. Update `src/training.py` to add new algorithm option
5. Ensure compatibility with graph observations

### Modifying the Neural Network Architecture
1. Edit encoder in `src/models/encoder.py` (e.g., add layers, change attention mechanism)
2. Update aggregator in `src/models/aggregator.py` if changing graph aggregation
3. Modify complete model in `src/models/model.py`
4. Adjust `input_dim` and `embedding_dim` parameters in training command

### Changing the Environment
1. Edit `src/environment/fluidity_environment.py`
2. Key methods: `reset()`, `step()`, `_build_graph()`, `_calculate_reward()`
3. Update `FluidityEnvironmentConfig` for new parameters
4. Test with `src/environment/test.py`

### Working with Java Simulator
1. Java JAR location: `ressources/jars/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar`
2. Configuration files: `ressources/run_configs/<run_config>/server0/xmr/config/`
3. Modify `src/environment/java_conn.py` to change Java communication
4. **Warning:** JVM must be started only once per process (JPype limitation)

## Model Checkpoints

Saved models are located in:
- `src/best_policy/best_policy_40.pth`: Best policy network (40-step config)
- `src/best_value_40.pth`: Corresponding value network
- Root directory may contain additional checkpoints from recent training

Load models using `--load_path` argument.

## Debugging Tips

1. **Java connection issues:** Check JVM options in `FluidityEnvironmentConfig.jvm_options`
2. **CUDA out of memory:** Reduce `--batch_size` or `--hidden_dim`
3. **Poor training performance:** Try cross-product action space, increase `--update_epochs`, or adjust learning rates
4. **Exploration issues (DQN):** Increase `--epsilon_decay` or `--epsilon_min`
5. **Check logs:** Application logs in `.log` files, TensorBoard in `runs/`, WandB in dashboard

## Recent Development Focus

Based on recent git commits, active areas of development:
- Dreamer (model-based RL) implementation
- RSSM (Recurrent State Space Model) integration
- Offline RL from recorded trajectories
- Cross-product action space improvements
- Timestep context embeddings for transformers
