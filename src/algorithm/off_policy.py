import os.path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import gym
from dataclasses import dataclass
from typing import Literal, Optional
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import datetime

from src.models.model import BaseSwapModel, BaseValueModel, SwapActionMapper, CrossProductSwapActionMapper
from src.algorithm.replay_buffer import OffPolicyReplayBuffer
from src.environment import initialize_logger
from src.environment.utils import _estimate_mean_total_latency_from_dict, _sample_passive_rotation
from src.utils import MetricLogger


class RunningMeanStd:
    """Streaming mean/variance with Welford updates (kept on a specific device)."""
    def __init__(self, epsilon: float = 1e-4, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var  = torch.ones(shape,  dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.float()
        batch_mean = x.mean()
        batch_var  = x.var(unbiased=False) if x.numel() > 1 else torch.zeros((), device=x.device)
        batch_count = torch.tensor(x.numel(), dtype=torch.float32, device=x.device)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)

        self.mean  = new_mean
        self.var   = M2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)


@dataclass()
class DQNAgentConfig:
    policy_net: BaseSwapModel
    target_net: BaseSwapModel
    env: gym.Env
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 10000
    target_update: int = 10
    priority: bool = False
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.1
    reward_scaling: bool = False
    eval_every_episode: int = 10
    cross_product_action_space: CrossProductSwapActionMapper = None
    use_timestep_context: bool = False
    update_epochs: int = 10
    train_every_steps: int = 100
    temporal_size: int = 4
    # Logging configuration
    use_tensorboard: bool = True  # Enable/disable tensorboard logging
    use_wandb: bool = True  # Enable/disable wandb logging
    wandb_project: Optional[str] = "replictated-state-machines-rl"
    wandb_entity: Optional[str] = "coluding"  # Wandb entity/team name
    wandb_run_name: Optional[str] = None


logger = initialize_logger("dqn.log")

class DQNAgent:
    def __init__(self, config: DQNAgentConfig):
        self.policy_net = config.policy_net
        self.target_net = config.target_net
        self.env = config.env
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.target_update = config.target_update
        self.priority = config.priority
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

        self.reward_scaling = config.reward_scaling

        self.device = self.env.config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = OffPolicyReplayBuffer(self.buffer_size, self.priority)

        self.steps = 0
        self.eval_every = config.eval_every_episode

        self.cross_product_action_space = config.cross_product_action_space
        self.use_timestep_context = config.use_timestep_context
        self.update_epochs = config.update_epochs
        self.train_every_steps = config.train_every_steps
        self.temporal_size = config.temporal_size

        # Running mean/std for reward normalization
        self.rew_rms = RunningMeanStd(device=self.device)

        self.save_path = f"max_steps={self.env.max_steps}_use_timestep_context={self.use_timestep_context}"
        self.step = 0

        # Initialize unified metric logger
        wandb_config = {
            "algorithm": "DQN",
            "reconfig_costs": self.env.use_reconfig_costs,
            "lr": self.lr,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "target_update": self.target_update,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "reward_scaling": self.reward_scaling,
            "update_epochs": self.update_epochs,
            "train_every_steps": self.train_every_steps,
            "temporal_size": self.temporal_size,
            "use_timestep_context": self.use_timestep_context,
        }

        if config.wandb_run_name is None:
            config.wandb_run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.logger = MetricLogger(
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb,
            tensorboard_comment="dqn" + str(self.env.max_steps) + "_" + f"use_reconfig_costs={self.env.use_reconfig_costs}",
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            wandb_run_name="dqn" + str(self.env.max_steps) + "_" + f"use_reconfig_costs={self.env.use_reconfig_costs}",
            wandb_config=wandb_config
        )

    def save_model(self, directory: str):
        path = os.path.join(directory + self.save_path)
        policy_path = os.path.join(path, "policy.pth")
        torch.save(self.policy_net.state_dict(), policy_path)

    def load_model(self, directory: str = None, path: str = None):
        if path is None:
            if directory is None:
                raise ValueError("Either path or directory must be specified")
            path = os.path.join(directory + self.save_path)
            policy_path = os.path.join(path, "policy.pth")
        else:
            policy_path = os.path.join(path, "policy.pth") if os.path.isdir(path) else path

        self.policy_net.load_state_dict(torch.load(policy_path))
        self.policy_net.to(self.device)
        logger.info(f"Model loaded from {policy_path}")

    def select_action(self, state, timestep = None, deterministic=False):
        if random.random() < self.epsilon and not deterministic:
            sampled_action = self.env.sample_action(self.cross_product_action_space is not None)

            # Convert tuple action to cross-product index if needed
            if self.cross_product_action_space is not None:
                # sampled_action is (add, remove) tuple, convert to cross-product index
                # Find the index in cross_product_action_space that matches this tuple
                for idx, action_tuple in enumerate(self.cross_product_action_space.action_space):
                    if action_tuple == sampled_action:
                        return idx
                # If not found, return random valid action
                return random.randint(0, len(self.cross_product_action_space.action_space) - 1)
            else:
                return sampled_action

        with torch.no_grad():
            if isinstance(self.policy_net.action_mapper, SwapActionMapper):
                state = state.to(self.device)
                add_q_vals, remove_q_vals = self.policy_net(state, self.temporal_size)
                return torch.argmax(add_q_vals).item(), torch.argmax(remove_q_vals).item()
            elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
                state = state.to(self.device)
                timestep = torch.tensor(timestep, dtype=torch.long, device=self.device).unsqueeze(0) if timestep is not None else None
                q_vals = self.policy_net(state, timestep, self.temporal_size)
                mask = self._process_mask(state)
                q_vals = q_vals + mask

                return torch.argmax(q_vals).item()

    def _process_mask(self, batched_states, B=1):
        location_indices = torch.where(batched_states.label != 0)[0]
        loc_ids = batched_states.name[location_indices].view(B, self.temporal_size, len(self.env.loc_mapping))[:, -1, :]
        mask = batched_states.add_mask[location_indices].view(B, self.temporal_size, len(self.env.loc_mapping))[:, -1,:]

        zero_mask_index = mask == 0
        removable_locations = loc_ids[zero_mask_index].view(B, -1).tolist()
        addable_locations = loc_ids[~zero_mask_index].view(-1, len(self.env.active_locations)).tolist()
        final_add_mask = [self.cross_product_action_space.build_add_action_mask(x) for x in addable_locations]
        final_remove_mask = [self.cross_product_action_space.build_remove_action_mask(x) for x in removable_locations]

        final_mask = torch.tensor(final_add_mask) + torch.tensor(final_remove_mask)

        return final_mask.to(self.device)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.full_training_runs += 1
        epoch_losses = []
        iterator = tqdm(range(self.update_epochs), desc="Training DQN...")
        for _ in iterator:
            batch = self.replay_buffer.sample(self.batch_size)

            loss = self._compute_q_vals_and_loss(batch) if self.cross_product_action_space is None else self._compute_q_vals_and_loss_cross_product(batch)

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(self.policy_net.state_dict(), "best_q_model.pt")

            self.optimizer.zero_grad()
            loss.backward()

            # Log gradient norm
            grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.logger.add_scalar("Policy Grad Norm", grad_norm, self.training_step)

            self.optimizer.step()
            self.training_step += 1
            self.logger.add_scalar("Loss", loss.item(), self.training_step)
            epoch_losses.append(loss.item())

            iterator.set_postfix({"Loss": loss.item()})

        self.logger.add_scalar("Epoch Loss", np.mean(epoch_losses), self.full_training_runs)

    def _compute_q_vals_and_loss(self, batch):
        states, actions, rewards, next_states, dones, timesteps = zip(*batch)

        batched_states = Batch.from_data_list(states).to(self.device)
        batched_next_states = Batch.from_data_list(next_states).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        if self.reward_scaling:
            with torch.no_grad():
                self.rew_rms.update(rewards)
            rewards = (rewards - self.rew_rms.mean) / (self.rew_rms.std + 1e-8)

        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        add_q_values, remove_q_values = self.policy_net(batched_states)
        add_q_values = add_q_values.gather(1, actions[:, 0].unsqueeze(1))
        remove_q_values = remove_q_values.gather(1, actions[:, 1].unsqueeze(1))

        with torch.no_grad():
            next_add_q_values, next_remove_q_values = self.target_net(batched_next_states)
            next_add_q_values = next_add_q_values.max(1)[0].detach().unsqueeze(1)
            next_remove_q_values = next_remove_q_values.max(1)[0].detach().unsqueeze(1)

        target_add_q_values = rewards + (1 - dones) * self.gamma * next_add_q_values
        target_remove_q_values = rewards + (1 - dones) * self.gamma * next_remove_q_values

        loss_add = nn.functional.mse_loss(add_q_values, target_add_q_values)
        loss_remove = nn.functional.mse_loss(remove_q_values, target_remove_q_values)
        loss = loss_add + loss_remove

        if self.training_step % 100 == 0:
            logger.info(f"Loss: {loss.item()}")
            logger.info(f"Q values example: {add_q_values[0].tolist()} {remove_q_values[0].tolist()}")
            logger.info(f"Corresponding rewards {rewards.tolist()}")

        return loss

    def _compute_q_vals_and_loss_cross_product(self, batch):
        states, actions, rewards, next_states, dones, timesteps = zip(*batch)

        batched_states = Batch.from_data_list(states).to(self.device)
        batched_next_states = Batch.from_data_list(next_states).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        timesteps = torch.tensor(timesteps, dtype=torch.long, device=self.device)

        if self.reward_scaling:
            with torch.no_grad():
                self.rew_rms.update(rewards)
            rewards = (rewards - self.rew_rms.mean) / (self.rew_rms.std + 1e-8)

        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.policy_net(batched_states, timesteps, self.temporal_size)
        q_vals = q_vals.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_vals = self.target_net(batched_next_states, timesteps, self.temporal_size)
            next_q_vals = next_q_vals.max(1)[0].detach().unsqueeze(1)

        target_q_vals = rewards + (1 - dones) * self.gamma * next_q_vals
        loss = nn.functional.mse_loss(q_vals, target_q_vals)

        if self.training_step % 100 == 0:
            logger.info(f"Loss: {loss.item()}")
            logger.info(f"Q values example: {q_vals.tolist()}")
            logger.info(f"Corresponding rewards {rewards.tolist()}")

        return loss

    def update_target(self, episode):
        if episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.to(self.device)

    def train(self, num_episodes=500):
        self.training_step = 0
        self.best_loss = float("inf")
        self.full_training_runs = 0
        self.step = 0
        self.evaluation_step = 0
        action_history = []
        all_rewards = []

        iterator = tqdm(range(num_episodes), desc="Training DQN...", unit="episode")

        for episode in iterator:
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            counter = 0
            rewards = []
            episode_actions = []
            raw_actions = []
            pure_latencies = []

            starting_locs = self.env.active_locations

            while not done:
                action = self.select_action(state, counter)

                if not isinstance(action, tuple):
                    env_action = self.cross_product_action_space[action]
                else:
                    env_action = action

                next_state, reward, done, _, _ = self.env.step(env_action)
                next_state = next_state.to(self.device)
                self.step += 1
                self.replay_buffer.push(state, action, reward, next_state, done, counter)
                state = next_state
                total_reward += reward

                rewards.append(reward)
                all_rewards.append(reward)
                episode_actions.append(action)
                raw_actions.append(env_action)
                pure_latencies.append(self.env.last_latency)

                counter += 1

                if self.step % self.train_every_steps == 0:
                    self.train_step()

                if self.step % self.target_update == 0:
                    self.update_target(episode)

            logger.info("Episode [{}] Average Reward: {:.4f}".format(episode, np.mean(rewards)))

            action_history.extend(episode_actions)
            steps_per_ep = len(rewards)

            # Periodic baseline evaluation
            if episode % max((3000 // steps_per_ep), 15) == 0:

                # --- Run baselines ---
                fluidity_baseline_rewards, fluidity_baseline_latencies, _, fluidity_time = self.run_fluidity_baseline()
                baseline_rewards, pure_baseline_latencies = self.run_static_baseline()
                random_baseline_rewards, random_pure_latencies = self.run_random_baseline()

                # --- Compute averages ---
                avg_policy_reward = np.mean(rewards)
                avg_bl_reward = np.mean(baseline_rewards)
                avg_random_reward = np.mean(random_baseline_rewards)
                avg_flu_reward = np.mean(fluidity_baseline_rewards)

                avg_policy_latency = np.mean(pure_latencies)
                avg_bl_latency = np.mean(pure_baseline_latencies)
                avg_random_latency = np.mean(random_pure_latencies)
                avg_flu_latency = np.mean(fluidity_baseline_latencies)

                # --- Reward plot ---
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(rewards, label=f"Policy Reward (avg: {avg_policy_reward:.3f})")
                ax.plot(baseline_rewards, label=f"Static Baseline (avg: {avg_bl_reward:.3f})")
                ax.plot(fluidity_baseline_rewards, label=f"Fluidity Baseline (avg: {avg_flu_reward:.3f})",
                        linestyle='--', color='orange')
                ax.set_xlabel("Step")
                ax.set_ylabel("Reward")
                ax.set_title(f"Episode {episode} Reward")
                ax.legend()
                self.logger.add_figure(f"Episode {episode} Reward", fig, episode)

                # --- Latency plot ---
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(pure_latencies, label=f"Policy Latency (avg: {avg_policy_latency:.3f})")
                ax.plot(pure_baseline_latencies, label=f"Static Baseline (avg: {avg_bl_latency:.3f})")
                ax.plot(fluidity_baseline_latencies, label=f"Fluidity Baseline (avg: {avg_flu_latency:.3f})",
                        linestyle='--', color='orange')
                ax.set_xlabel("Step")
                ax.set_ylabel("Latency")
                ax.set_title(f"Episode {episode} Latency")
                ax.legend()
                self.logger.add_figure(f"Episode {episode} Latency", fig, episode)

                if not isinstance(action, tuple):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(episode_actions, bins=np.arange(min(episode_actions) - 0.5, max(episode_actions) + 1.5, 1),
                            edgecolor="black")
                    ax.set_xlabel("Action")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Action Selection Histogram (Last 10 Episodes)")
                    self.logger.add_figure(f"Episode {episode} Action Histogram", fig, episode)
                    action_history.clear()

                # --- Multi-line scalar plots for WandB/TensorBoard ---
                for i_step in range(len(rewards)):
                    # Log Rewards
                    reward_scalars = {
                        "Policy": rewards[i_step],
                        "Static": baseline_rewards[i_step] if i_step < len(baseline_rewards) else baseline_rewards[-1],
                        "Random": random_baseline_rewards[i_step] if i_step < len(random_baseline_rewards) else random_baseline_rewards[-1],
                        "Fluidity": fluidity_baseline_rewards[i_step] if i_step < len(fluidity_baseline_rewards) else fluidity_baseline_rewards[-1]
                    }
                    self.logger.add_scalars(f"baseline_eval_step_{episode}/Reward", reward_scalars, i_step)

                    # Log Latencies
                    latency_scalars = {
                        "Policy": pure_latencies[i_step],
                        "Static": pure_baseline_latencies[i_step] if i_step < len(pure_baseline_latencies) else pure_baseline_latencies[-1],
                        "Random": random_pure_latencies[i_step] if i_step < len(random_pure_latencies) else random_pure_latencies[-1],
                        "Fluidity": fluidity_baseline_latencies[i_step] if i_step < len(fluidity_baseline_latencies) else fluidity_baseline_latencies[-1]
                    }
                    self.logger.add_scalars(f"baseline_eval_step_{episode}/Latency", latency_scalars, i_step)

                self.logger.add_text(f"Starting Locations episode {episode}", str(starting_locs), episode)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            logger.info(f"Epsilon: {self.epsilon}")
            logger.info(f"Buffer Size: {len(self.replay_buffer)}")

            self.logger.add_text("Episode Actions", str(raw_actions), episode)
            self.logger.add_scalar("Episode Reward", total_reward, episode)
            self.logger.add_scalar("Average Reward", np.mean(rewards), episode)
            self.logger.add_scalar("Rolling Average Reward", np.mean(all_rewards[-20:]), episode)
            self.logger.add_scalar("Epsilon", self.epsilon, episode)
            self.logger.add_scalar("Buffer Size", len(self.replay_buffer), episode)

        # Close logger at the end of training
        self.logger.close()


    def run_static_baseline(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        all_rewards = []
        pure_latencies = []

        while not done:
            next_state, reward, done, _, _ = self.env.step((-1, -1))
            pure_latencies.append(self.env.last_latency)
            steps += 1
            total_reward += reward
            all_rewards.append(reward)
            self.step += 1

        return all_rewards, pure_latencies

    def run_random_baseline(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        all_rewards = []
        pure_latencies = []
        while not done:
            next_state, reward, done, _, _ = self.env.step(self.env.sample_action(crss_product=self.cross_product_action_space is not None))
            pure_latencies.append(self.env.last_latency)
            steps += 1
            total_reward += reward
            all_rewards.append(reward)
            self.step += 1

        return all_rewards, pure_latencies

    def run_fluidity_baseline(self, log_interval: int = 5):
        """
        Runs a Fluidity-inspired baseline across a full episode.
        Uses the env's latency dict at each step to approximate optimal reconfigurations.
        """

        env = self.env.env.env
        env.graph = False
        env.reduced_state_space = False
        raw_obs, _ = env.reset()
        done = False

        rewards, latencies, times = [], [], []
        total_reward = 0.0
        step_count = 0

        start_total = time.perf_counter()

        while not done:
            step_start = time.perf_counter()
            current_latency = env.last_latency if env.last_latency > 0 else np.inf
            current_active = set(env.active_locations)
            current_passive = set(env.passive_locations)
            best_latency = current_latency
            best_action = (-1, -1)

            # --- (1) Evaluate all (add, remove) combinations ---
            num_candidates = 1
            random_candidates = random.sample(list(env.available_locations.union(env.passive_locations)),
                                              num_candidates)
            options = list(current_active.union(random_candidates))
            for add_loc in random_candidates:
                for remove_loc in list(current_active):
                    if add_loc == remove_loc:
                        continue

                    est_latency = _estimate_mean_total_latency_from_dict(
                        raw_obs, current_active, current_passive, add_loc, remove_loc
                    )

                    if est_latency < best_latency:
                        best_latency = est_latency
                        best_action = (env.inv_loc_mapping[add_loc], env.inv_loc_mapping[remove_loc])

            # --- (2) Execute if better, else no-op ---
            if best_latency < current_latency:
                action = best_action
            else:
                action = (-1, -1)

            # --- (4) Step environment ---
            obs, reward, done, _, _ = env.step(action)

            # --- (5) Logging ---
            step_time = time.perf_counter() - step_start
            times.append(step_time)
            total_reward += reward
            rewards.append(reward)
            latencies.append(env.last_latency)
            step_count += 1

        total_time = time.perf_counter() - start_total

        env.graph = True
        env.reduced_state_space = True

        return rewards, latencies, times, total_time

    def evaluate(self):
        state, _ = self.env.reset()
        self.evaluation_step += 1
        state = state.to(self.device)
        total_reward = 0
        done = False
        steps = 0
        action_history = []
        reward_history = []
        while not done:
            action = self.select_action(state, timestep=steps, deterministic=True)

            if not isinstance(action, tuple):
                env_action = self.cross_product_action_space[action]

            action_history.append(action)
            next_state, reward, done, _, _ = self.env.step(env_action)
            next_state = next_state.to(self.device)
            steps += 1
            state = next_state
            total_reward += reward
            reward_history.append(reward)

        logger.info(f"Average Reward = {total_reward / steps: .4f}")
        self.logger.add_scalar("Evaluation reward Reward", total_reward / steps)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(reward_history)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title(f"Reward development of Evaluation {self.evaluation_step}")
        self.logger.add_figure(f"Evaluation {self.evaluation_step} Reward", fig, self.evaluation_step)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(action_history, bins=np.arange(min(action_history) - 0.5, max(action_history) + 1.5, 1),
                edgecolor="black")
        ax.set_xlabel("Action")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Action Selection Histogram of Evaluation {self.evaluation_step}")
        self.logger.add_figure(f"Evaluation {self.evaluation_step} Action Histogram", fig, self.evaluation_step)
        action_history.clear()