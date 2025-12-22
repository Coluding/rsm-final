import os.path

import time
import torch.nn as nn
import torch.optim as optim
import torch_geometric.data
import torch_geometric.nn as gnn
import numpy as np
import random
import tqdm
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal, Optional
import gym
import datetime

from src.algorithm import OnPolicyReplayBuffer, OnPolicyTrajectoryBuffer
from src.models.model import BaseSwapModel, BaseValueModel, SwapActionMapper, CrossProductSwapActionMapper
from src.algorithm.replay_buffer import *
from src.environment import initialize_logger
from src.environment.utils import _estimate_mean_total_latency_from_dict, _sample_passive_rotation
from src.utils import MetricLogger


@dataclass()
class OnPolicyAgentConfig:
    algorithm: Literal["PPO", "REINFORCE"]
    policy_net: BaseSwapModel
    env: gym.Env
    value_net: BaseValueModel = None
    lr: float = 3e-4
    lr_value_fn: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 32
    buffer_size: int = 10000
    clip_epsilon: float = 0.3
    entropy_coeff: float = 0.01
    value_coeff: float = 2
    update_epochs: int = 10
    reward_scaling: bool = True
    train_every: int = 50  # Train every n steps
    temporal_size: int = 4
    cross_product_action_space: CrossProductSwapActionMapper = None
    use_timestep_context: bool = False
    use_gae: bool = True
    baseline_beta: float = 0.01  # EMA for REINFORCE baseline
    normalize_advantages: bool = False  # PPO advantage whitening (running if reward_scaling)
    # Logging configuration
    use_tensorboard: bool = True  # Enable/disable tensorboard logging
    use_wandb: bool = True  # Enable/disable wandb logging
    wandb_project: Optional[str] = "replictated-state-machines-rl"
    wandb_entity: Optional[str] = "coluding"  # Wandb entity/team name
    wandb_run_name: Optional[str] =  None

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


torch.autograd.set_detect_anomaly(True)

logger = initialize_logger("ppo.log")



class OnPolicyAgent:
    def __init__(self, config: OnPolicyAgentConfig):
        self.algorithm = config.algorithm
        self.policy_net = config.policy_net
        self.value_net = config.value_net
        self.env = config.env
        self.lr = config.lr
        self.value_lr = config.lr_value_fn
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.clip_epsilon = config.clip_epsilon
        self.entropy_coeff = config.entropy_coeff
        self.value_coeff = config.value_coeff
        self.update_epochs = config.update_epochs
        self.reward_scaling = config.reward_scaling
        self.train_every = config.train_every
        self.cross_product_action_space = config.cross_product_action_space
        self.temporal_size = config.temporal_size
        self.use_timestep_context = config.use_timestep_context
        self.use_gae = config.use_gae

        assert not (self.cross_product_action_space is None and isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper)), "Cross Product Action Space is required for CrossProductSwapActionMapper"

        self.device = self.env.config.device
        self.policy_net.to(self.device)
        self.value_net.to(self.device) if self.value_net is not None else None

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr) if self.value_net is not None else None
        self.replay_buffer = OnPolicyReplayBuffer(self.buffer_size)
        self.rew_rms = RunningMeanStd(device=self.device)  # for raw rewards (PPO)
        self.ret_rms = RunningMeanStd(device=self.device)  # for returns (REINFORCE)
        self.adv_rms = RunningMeanStd(device=self.device)  # for advantages (PPO)
        self.return_baseline = torch.zeros((), device=self.device)  # EMA of returns for REINFORCE
        self.baseline_beta = config.baseline_beta
        self.normalize_advantages = config.normalize_advantages

        self.save_path = f"max_steps={self.env.max_steps}_use_timestep_context={self.use_timestep_context}"
        self.step = 0

        # Initialize unified metric logger
        wandb_config = {
            "algorithm": self.algorithm,
            "reconfig_costs": self.env.use_reconfig_costs,
            "lr": self.lr,
            "lr_value_fn": self.value_lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coeff": self.entropy_coeff,
            "value_coeff": self.value_coeff,
            "update_epochs": self.update_epochs,
            "reward_scaling": self.reward_scaling,
            "train_every": self.train_every,
            "temporal_size": self.temporal_size,
            "use_timestep_context": self.use_timestep_context,
            "use_gae": self.use_gae,
            "baseline_beta": self.baseline_beta,
            "normalize_advantages": self.normalize_advantages,
        }

        if config.wandb_run_name is None:
            config.wandb_run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.logger = MetricLogger(
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb,
            tensorboard_comment="ppo" + str(self.env.max_steps) + "_" + f"use_reconfig_costs={self.env.use_reconfig_costs}" if self.algorithm == "PPO" else "reinforce" + str(self.env.max_steps) + "_" + f"reduced_state_space={self.env.config.reduced_state_space}",
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            wandb_run_name="ppo" + str(self.env.max_steps) + "_" + f"use_reconfig_costs={self.env.use_reconfig_costs}" if self.algorithm == "PPO" else "reinforce" + str(self.env.max_steps) + "_" + f"reduced_state_space={self.env.config.reduced_state_space}",
            wandb_config=wandb_config
        )

    def select_action(self, state, timestep=None):
        state = state.to(self.device)
        with torch.no_grad():
            if isinstance(self.policy_net.action_mapper, SwapActionMapper):
                add_logits, remove_logits = self.policy_net(state, self.temporal_size)
                add_dist = torch.distributions.Categorical(logits=add_logits)
                remove_dist = torch.distributions.Categorical(logits=remove_logits)
                add_action = add_dist.sample()
                remove_action = remove_dist.sample()
                add_log_prob = add_dist.log_prob(add_action)
                remove_log_prob = remove_dist.log_prob(remove_action)

                return (add_action.item(), remove_action.item()), (add_log_prob.item(), remove_log_prob.item())

            elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
                logits = self.policy_net(state, timestep, self.temporal_size)
                mask = self._process_mask(state)
                logits = logits + mask
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                return action.item(), log_prob.item()

    def _process_mask(self, batched_states, B=1):
        location_indices = torch.where(batched_states.label != 0)[0]
        loc_ids = batched_states.name[location_indices].view(B, self.temporal_size, len(self.env.loc_mapping))[:, -1, :]
        mask = batched_states.add_mask[location_indices].view(B, self.temporal_size, len(self.env.loc_mapping))[:, -1,:]

        zero_mask_index = mask == 0
        removable_locations = loc_ids[zero_mask_index].view(B, -1).tolist()
        addable_locations = loc_ids[~zero_mask_index].view(B, -1).tolist()
        final_add_mask = [self.cross_product_action_space.build_add_action_mask(x) for x in addable_locations]
        final_remove_mask = [self.cross_product_action_space.build_remove_action_mask(x) for x in removable_locations]

        final_mask = torch.tensor(final_add_mask) + torch.tensor(final_remove_mask)

        return final_mask.to(self.device)

    def compute_gae(self, rewards, values, dones):
        """
        Computes the generalized advantage estimate
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float)

        last_advantage = 0
        for t in reversed(range(len(rewards))):
            terminal = 1 - dones[t].int()
            next_value = values[t + 1] if (t + 1 < len(rewards) and not dones[t]) else 0
            delta = terminal * (rewards[t] + self.gamma * next_value - values[t])
            advantages[t] = delta + self.gamma * self.gae_lambda * terminal * last_advantage
            last_advantage = advantages[t]

        return advantages

    def compute_reward_to_go(rewards, gamma=0.99):
        """
        Computes reward-to-go for each timestep t:
        G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^(T-t) r_T
        """
        G = torch.zeros_like(rewards, dtype=torch.float)
        future_return = 0
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + gamma * future_return
            G[t] = future_return
        return G

    def compute_reward_to_go_with_baseline(self, rewards):
        """
        Computes reward-to-go for each timestep t and subtracts baseline (mean return).
        G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^(T-t) r_T
        Baseline: mean of all G_t values in the episode.
        """
        G = torch.zeros_like(rewards, dtype=torch.float)
        future_return = 0
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            G[t] = future_return

        baseline = G.mean()

        return G - baseline

    def compute_advantage_normal(self, rewards, values, dones):
        """
        Computes the normal one-step advantage: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float)

        for t in range(len(rewards)):
            next_value = values[t + 1] if (t + 1 < len(rewards) and not dones[t]) else 0
            advantages[t] = rewards[t] + self.gamma * next_value - values[t]

        return advantages

    def compute_clipped_advantage(self, rewards, values, dones):
        """
        Computes the clipped advamtage to ensure smooth training
        """

    def train_step_reinforce(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        iterator = tqdm.tqdm(range(self.update_epochs), desc="Running epoch training...", unit="epoch")
        for _ in iterator:
            epoch_loss = 0
            for batch in self.replay_buffer(self.batch_size, shuffle=False):
                states, actions, _, rewards, _, dones, timesteps = zip(*batch)

                batched_states = Batch.from_data_list(states).to(self.device)
                actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                timesteps = torch.tensor(timesteps, dtype=torch.long, device=self.device) if self.use_timestep_context else None
                # Compute baseline-subtracted reward-to-go
                reward_to_go = self.compute_reward_to_go_with_baseline(rewards)

                # Compute policy loss
                logits = self.policy_net(batched_states, timesteps)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                policy_loss = -torch.mean(log_probs * reward_to_go)  # Policy gradient update

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                epoch_loss += policy_loss.item()
                self.logger.add_scalar("Policy Loss", policy_loss.item(), self.training_step)
                iterator.set_postfix({"Policy Loss": policy_loss.item()})

                self.training_step += 1

                if policy_loss < self.best_loss:
                    self.best_loss = policy_loss
                    torch.save(self.policy_net.state_dict(), "best_policy_re.pth")

            self.logger.add_scalar("Epoch Loss", epoch_loss)

        self.replay_buffer.clear()

    def save_model(self, directory: str):
        path = os.path.join(directory + self.save_path)
        policy_path = os.path.join(path, "policy.pth")
        value_path = os.path.join(path, "value.pth")
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.value_net.state_dict(), value_path)

    def load_model(self, directory: str = None, path: str = None):
        if path is None:
            if directory is None:
                raise ValueError("Either path or directory must be specified")
            path = os.path.join(directory + self.save_path)
            policy_path = os.path.join(path, "policy.pth")
            value_path = os.path.join(path, "value.pth")
        else:
            policy_path = os.path.join(path, "policy.pth")
            value_path = os.path.join(path, "value.pth")

        self.policy_net.load_state_dict(torch.load(policy_path))
        self.value_net.load_state_dict(torch.load(value_path))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        #self.replay_buffer.normalize_rewards() if self.reward_scaling else None
        iterator = tqdm.tqdm(self.replay_buffer(self.batch_size), desc="Running epoch training...", unit="epoch",
                             total=self.update_epochs)
        for i, batch in enumerate(iterator):
            if batch is None:
                logger.info("Not ready to train yet")
                return
            states, actions, old_log_probs, rewards, next_states, dones, timesteps = zip(*batch)

            batched_states = Batch.from_data_list(states).to(self.device)
            batched_next_states = Batch.from_data_list(next_states).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=self.device) if self.use_timestep_context else None

            rewards_proc = rewards

            if self.reward_scaling:
                with torch.no_grad():
                    self.rew_rms.update(rewards_proc)
                rewards_proc = (rewards_proc - self.rew_rms.mean) / (self.rew_rms.std + 1e-8)

            value_loss, advantages = self._value_loss(batched_states, rewards_proc, dones, batched_next_states, timesteps)

            if self.normalize_advantages:
                if self.reward_scaling:
                    with torch.no_grad():
                        self.adv_rms.update(advantages)
                    advantages = (advantages - self.adv_rms.mean) / (self.adv_rms.std + 1e-8)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = self._ppo_loss_logic(batched_states, actions, old_log_probs, advantages, timesteps)

            total_loss = policy_loss + value_loss

            self.logger.add_scalar("Policy Loss", policy_loss.item(), self.training_step)
            self.logger.add_scalar("Value Loss", value_loss.item(), self.training_step)

            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            total_loss.backward()

            # log grad norm
            policy_grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            value_grad_norm = nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

            self.logger.add_scalar("Policy Grad Norm", policy_grad_norm, self.training_step)
            self.logger.add_scalar("Value Grad Norm", value_grad_norm, self.training_step)

            self.optimizer.step()
            self.value_optimizer.step()

            self.training_step += 1

            if total_loss < self.best_loss:
                self.best_loss = total_loss
                torch.save(self.policy_net.state_dict(), "best_policy.pth")
                torch.save(self.value_net.state_dict(), "best_value.pth")

            iterator.set_postfix({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item()})

            if i >= self.update_epochs:
                break

    def _value_loss(self, batched_states: torch_geometric.data.Data, rewards: torch.Tensor,
                    dones: torch.Tensor, batched_next_states: torch_geometric.data.Data,
                    timesteps: torch.Tensor = None
                    ):
        values = self.value_net(batched_states, timesteps, self.temporal_size).squeeze()
        next_timesteps = (timesteps + 1) if timesteps is not None else None
        next_values = self.value_net(batched_next_states, next_timesteps ,self.temporal_size).squeeze()
        advantages = self.compute_gae(rewards, values, dones) if self.use_gae else self.compute_advantage_normal(rewards, values, dones)
        value_loss = nn.functional.mse_loss(values, (rewards + self.gamma * next_values * (1 - dones)))
        value_loss = value_loss * self.value_coeff

        # log example values
        #if self.training_step % 200 == 0:
            #logger.info(f"Example Values: {str(values.tolist())}")
            #logger.info("Corresponding rewards: ", str(rewards.tolist()))
           # logger.info("Next Values: ", str(next_values.tolist()))

        return value_loss, advantages

    def _ppo_loss_logic(self, batched_states: torch_geometric.data.Data, actions: torch.Tensor,
                         old_log_probs: torch.Tensor, advantages: torch.Tensor, timesteps : torch.Tensor = None):

        if isinstance(self.policy_net.action_mapper, SwapActionMapper):
            add_logits, remove_logits = self.policy_net(batched_states, self.temporal_size)
            add_dist = torch.distributions.Categorical(logits=add_logits)
            remove_dist = torch.distributions.Categorical(logits=remove_logits)
            add_log_probs = add_dist.log_prob(actions[:, 0])
            remove_log_probs = remove_dist.log_prob(actions[:, 1])
            entropy = (add_dist.entropy() + remove_dist.entropy()).mean()

            ratio_add = torch.exp(add_log_probs - old_log_probs[:, 0])
            ratio_remove = torch.exp(remove_log_probs - old_log_probs[:, 1])
            clipped_ratio_add = torch.clamp(ratio_add, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            clipped_ratio_remove = torch.clamp(ratio_remove, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio_add * advantages, clipped_ratio_add * advantages).mean()
            policy_loss = policy_loss - torch.min(ratio_remove * advantages, clipped_ratio_remove * advantages).mean()
            policy_loss = policy_loss - self.entropy_coeff * entropy

            self.logger.add_scalar("Entropy", entropy, self.training_step)

        elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
            logits = self.policy_net(batched_states, timesteps, self.temporal_size)
            mask = self._process_mask(batched_states, actions.shape[0])
            logits = logits + mask
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            policy_loss = policy_loss - self.entropy_coeff * entropy

            self.logger.add_scalar("Entropy", entropy, self.training_step)
            self.logger.add_scalar("Ratio", ratio.mean(), self.training_step)

        else:
            raise NotImplementedError(f"Action mapper of type {type(self.policy_net.action_mapper)} not supported")

        if self.step % 1000 == 0:
            if isinstance(self.policy_net.action_mapper, SwapActionMapper):
                logger.info(f"Example Probabilities: {add_dist.probs.tolist()[0]}, {remove_dist.probs.tolist()[0]}")
            elif isinstance(self.policy_net.action_mapper, CrossProductSwapActionMapper):
                logger.info(f"Example Probabilities: {dist.probs.tolist()[0]}")


        return policy_loss

    def train(self, num_episodes=500):
        iterator = tqdm.tqdm(range(num_episodes), desc="Training PPO...", unit="episode")
        all_rewards = []
        self.training_step = 0
        self.best_loss = float("inf")

        action_history = []  # Store actions per episode

        for episode in iterator:
            state, _ = self.env.reset()
            state = state.to(self.device)
            total_reward = 0
            done = False
            steps = 0
            episode_buffer = []
            rewards = []
            episode_actions = []
            raw_actions = []
            pure_latencies = []

            starting_locs = self.env.active_locations

            while not done:
                action, log_prob = self.select_action(state, torch.tensor(steps, dtype=torch.long, device=self.device).unsqueeze(0) if self.use_timestep_context else None)

                if not isinstance(action, tuple):
                    env_action = self.cross_product_action_space[action]

                else:
                    env_action = action

                next_state, reward, done, _, _ = self.env.step(env_action)
                next_state = next_state.to(self.device)
                steps += 1
                self.replay_buffer.push(state, action, log_prob, reward, next_state, done, steps)
                state = next_state
                total_reward += reward
                rewards.append(reward)
                pure_latencies.append(self.env.last_latency)
                all_rewards.append(reward)
                self.step += 1

                episode_actions.append(action)
                raw_actions.append(env_action)

                if self.step % self.train_every == 0:
                    match self.algorithm:
                        case "PPO":
                            self.train_step()
                        case "REINFORCE":
                            self.train_step_reinforce()

            logger.info("Episode [{}]Average Reward: {:.4f}".format(episode, np.mean(rewards)))

            action_history.extend(episode_actions)
            steps_per_ep = len(rewards)
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
               # ax.plot(random_baseline_rewards, label=f"Random Baseline (avg: {avg_random_reward:.3f})")
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
                #ax.plot(random_pure_latencies, label=f"Random Baseline (avg: {avg_random_latency:.3f})")
                ax.plot(fluidity_baseline_latencies, label=f"Fluidity Baseline (avg: {avg_flu_latency:.3f})",
                        linestyle='--', color='orange')
                ax.set_xlabel("Step")
                ax.set_ylabel("Latency")
                ax.set_title(f"Episode {episode} Latency")
                ax.legend()
                self.logger.add_figure(f"Episode {episode} Latency", fig, episode)

                if not isinstance(action, tuple) :
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(episode_actions, bins=np.arange(min(episode_actions) - 0.5, max(episode_actions) + 1.5, 1),
                            edgecolor="black")
                    ax.set_xlabel("Action")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Action Selection Histogram (Last 10 Episodes)")
                    self.logger.add_figure(f"Episode {episode} Action Histogram", fig, episode)
                    action_history.clear()

                self.logger.add_text(f"Starting Locations episode {episode}", str(starting_locs), episode)

            self.logger.add_text("Episode Actions", str(raw_actions), episode)
            self.logger.add_scalar("Episode Reward", total_reward, episode)
            self.logger.add_scalar("Average Reward", np.mean(rewards), episode)
            self.logger.add_scalar("Rolling Average Reward", np.mean(all_rewards[-20:]), episode)

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
            next_state, reward, done, _, _ = self.env.step((-1,-1))
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

