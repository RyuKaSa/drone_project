"""
PPO with Custom Policy Matching DAgger Architecture

Uses Stable Baselines3 PPO but with a custom policy that has LayerNorm,
exactly matching the DAgger BCPolicy. This allows direct weight transfer.

Usage:
    python train_ppo_custom.py                    # Train from DAgger
    python train_ppo_custom.py --resume           # Resume training
    python train_ppo_custom.py --eval-only        # Evaluate
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import time
import signal
import sys
import os

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.distributions import DiagGaussianDistribution

from drone_env_simple import quat_to_euler, quat_rotate_inverse, quat_rotate


SHUTDOWN_REQUESTED = False


# ============================================================================
# ENVIRONMENT
# ============================================================================

CARDINAL_DIRECTIONS = {
    'forward':  np.array([ 1.0,  0.0,  0.0]),
    'backward': np.array([-1.0,  0.0,  0.0]),
    'left':     np.array([ 0.0,  1.0,  0.0]),
    'right':    np.array([ 0.0, -1.0,  0.0]),
    'up':       np.array([ 0.0,  0.0,  1.0]),
    'down':     np.array([ 0.0,  0.0, -1.0]),
}

COMBINED_AXES = [
    ('forward', 'left'), ('forward', 'right'),
    ('forward', 'up'), ('forward', 'down'),
    ('backward', 'left'), ('backward', 'right'),
    ('backward', 'up'), ('backward', 'down'),
    ('left', 'up'), ('left', 'down'),
    ('right', 'up'), ('right', 'down'),
]


class PPODroneEnv(gym.Env):
    """Drone environment with curriculum support."""
    
    def __init__(self, model_path="model.xml", easy_ratio=0.9,
                 min_dist=3.0, max_dist=7.0):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        self.easy_ratio = easy_ratio
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.target_pos = np.array([5.0, 0.0, 3.0])
        self.target_radius = 0.5
        self.prev_distance = 5.0
        
        self.max_steps = 2000
        self.steps = 0
        self.targets_reached = 0
        
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[2] = self.np_random.uniform(2.0, 6.0)
        self.data.qpos[3] = 1.0
        mujoco.mj_forward(self.model, self.data)
        
        self._spawn_target()
        self.steps = 0
        self.targets_reached = 0
        
        return self._get_obs(), {}
    
    def _spawn_target(self):
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7]
        dist = self.np_random.uniform(self.min_dist, self.max_dist)
        
        if self.np_random.random() < self.easy_ratio:
            # Cardinal direction
            name = self.np_random.choice(list(CARDINAL_DIRECTIONS.keys()))
            direction = CARDINAL_DIRECTIONS[name]
            if name in ['up', 'down']:
                direction_world = direction
            else:
                direction_world = quat_rotate(quat, direction)
                direction_world[2] = 0
                direction_world /= (np.linalg.norm(direction_world) + 1e-6)
        else:
            # Combined direction
            ax1, ax2 = COMBINED_AXES[self.np_random.integers(len(COMBINED_AXES))]
            d1 = CARDINAL_DIRECTIONS[ax1]
            d2 = CARDINAL_DIRECTIONS[ax2]
            w = self.np_random.uniform(0.3, 0.7)
            combined = w * d1 + (1-w) * d2
            combined /= (np.linalg.norm(combined) + 1e-6)
            direction_world = quat_rotate(quat, combined)
        
        self.target_pos = pos + direction_world * dist
        self.target_pos[2] = np.clip(self.target_pos[2], 0.5, 15.0)
        self.prev_distance = np.linalg.norm(self.target_pos - pos)
        
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = self.target_pos
    
    def step(self, action):
        motors = (np.array(action) + 1.0) * 6.0
        motors = np.clip(motors, 0.0, 12.0)
        
        self.data.ctrl[:4] = motors
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        pos = self.data.qpos[:3]
        ang_vel = self.data.qvel[3:6]
        distance = np.linalg.norm(self.target_pos - pos)
        
        # Reward
        progress = self.prev_distance - distance
        reward = progress * 2.0
        reward -= distance * 0.01
        reward -= np.linalg.norm(ang_vel) * 0.02
        reward -= np.mean(np.abs(action)) * 0.01
        self.prev_distance = distance

        # === PRECISION REWARDS ===
        to_target = self.target_pos - pos
        to_target_norm = to_target / (distance + 1e-6)
        vel_world = self.data.qvel[:3]
        speed = np.linalg.norm(vel_world)
        vel_norm = vel_world / (speed + 1e-6)

        # 1. Velocity alignment (move toward target)
        alignment = np.dot(vel_norm, to_target_norm)
        reward += alignment * 0.05

        # 2. Perpendicular velocity penalty (no circling)
        vel_parallel = np.dot(vel_world, to_target_norm) * to_target_norm
        vel_perp = vel_world - vel_parallel
        reward -= np.linalg.norm(vel_perp) * 0.01

        # Target reached
        if distance < self.target_radius:
            reward += 10.0
            self.targets_reached += 1
            self._spawn_target()
        
        # Crash
        terminated = pos[2] < 0.1
        if terminated:
            reward -= 10.0
        
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {
            'distance': distance,
            'targets_reached': self.targets_reached,
            'target_reached': distance < self.target_radius,
        }
    
    def _get_obs(self):
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel_world = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        to_target = self.target_pos - pos
        distance = np.linalg.norm(to_target)
        to_target_body = quat_rotate_inverse(quat, to_target)
        target_dir = to_target_body / (distance + 1e-6)
        
        vel_body = quat_rotate_inverse(quat, vel_world)
        gravity_body = quat_rotate_inverse(quat, np.array([0, 0, -1]))
        
        return np.concatenate([
            target_dir,
            [distance / 10.0],
            vel_body / 5.0,
            ang_vel / 5.0,
            gravity_body,
        ]).astype(np.float32)
    
    def set_easy_ratio(self, ratio):
        self.easy_ratio = np.clip(ratio, 0.0, 1.0)


# ============================================================================
# CUSTOM POLICY WITH LAYERNORM (matches DAgger exactly)
# ============================================================================

class DAggerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with LayerNorm matching DAgger's BCPolicy.
    
    DAgger architecture:
        Linear(13, 256) -> LayerNorm -> ReLU ->
        Linear(256, 256) -> LayerNorm -> ReLU
    
    This extracts 256-dim features, then action_net adds final Linear(256, 4).
    """
    
    def __init__(self, observation_space: spaces.Box):
        # Output 256 features
        super().__init__(observation_space, features_dim=256)
        
        obs_dim = observation_space.shape[0]
        
        # Match DAgger: net.0, net.1, net.2, net.3, net.4, net.5
        self.layer1 = nn.Linear(obs_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.layer1(observations)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        return x


class DAggerActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using DAgger's architecture.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Remove net_arch if present - we use custom extractor
        kwargs.pop('net_arch', None)
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DAggerFeaturesExtractor,
            features_extractor_kwargs={},
            net_arch=[],  # Empty - features extractor does all hidden layers
            activation_fn=nn.ReLU,
            **kwargs
        )


def load_dagger_into_custom_policy(policy, dagger_path="dagger_policy.pt"):
    """
    Load DAgger weights into custom policy with matching architecture.
    
    DAgger state_dict keys:
        net.0.weight, net.0.bias  -> layer1
        net.1.weight, net.1.bias  -> ln1
        net.3.weight, net.3.bias  -> layer2
        net.4.weight, net.4.bias  -> ln2
        net.6.weight, net.6.bias  -> action_net
    """
    print(f"\nLoading DAgger weights from {dagger_path}...")
    
    dagger_state = torch.load(dagger_path, map_location='cpu')
    policy_state = policy.state_dict()
    
    # Print for debugging
    print("\nDAgger keys:")
    for k, v in dagger_state.items():
        print(f"  {k}: {v.shape}")
    
    print("\nPolicy keys (features_extractor + action_net):")
    for k, v in policy_state.items():
        if 'features_extractor' in k or 'action_net' in k:
            print(f"  {k}: {v.shape}")
    
    # Mapping DAgger -> Custom Policy
    mapping = {
        # First layer
        'net.0.weight': 'features_extractor.layer1.weight',
        'net.0.bias': 'features_extractor.layer1.bias',
        # First LayerNorm
        'net.1.weight': 'features_extractor.ln1.weight',
        'net.1.bias': 'features_extractor.ln1.bias',
        # Second layer
        'net.3.weight': 'features_extractor.layer2.weight',
        'net.3.bias': 'features_extractor.layer2.bias',
        # Second LayerNorm
        'net.4.weight': 'features_extractor.ln2.weight',
        'net.4.bias': 'features_extractor.ln2.bias',
        # Action output layer
        'net.6.weight': 'action_net.weight',
        'net.6.bias': 'action_net.bias',
    }
    
    loaded = 0
    for dagger_key, policy_key in mapping.items():
        if dagger_key in dagger_state and policy_key in policy_state:
            d_shape = dagger_state[dagger_key].shape
            p_shape = policy_state[policy_key].shape
            
            if d_shape == p_shape:
                policy_state[policy_key] = dagger_state[dagger_key].clone()
                print(f"  ✓ {dagger_key} -> {policy_key}")
                loaded += 1
            else:
                print(f"  ✗ Shape mismatch: {dagger_key} {d_shape} vs {policy_key} {p_shape}")
        else:
            if dagger_key not in dagger_state:
                print(f"  ✗ Missing in DAgger: {dagger_key}")
            if policy_key not in policy_state:
                print(f"  ✗ Missing in Policy: {policy_key}")
    
    if loaded > 0:
        policy.load_state_dict(policy_state)
        print(f"\n✓ Loaded {loaded}/10 tensors from DAgger")
    else:
        print(f"\n✗ No weights transferred!")
    
    return loaded >= 8  # Need at least the main layers


# ============================================================================
# CALLBACKS
# ============================================================================

class CurriculumCallback(BaseCallback):
    def __init__(self, start=0.9, end=0.1, total_steps=1_000_000, verbose=1):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total = total_steps
    
    def _on_step(self):
        progress = min(self.num_timesteps / self.total, 1.0)
        ratio = self.start + (self.end - self.start) * progress
        
        for env in self.training_env.envs:
            if hasattr(env, 'set_easy_ratio'):
                env.set_easy_ratio(ratio)
        
        if self.verbose and self.num_timesteps % 20000 == 0:
            print(f"  [Curriculum] Step {self.num_timesteps}: easy_ratio = {ratio:.2f}")
        
        return True


class GracefulExitCallback(BaseCallback):
    def __init__(self, save_path="ppo_drone", verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
    
    def _on_step(self):
        global SHUTDOWN_REQUESTED
        if SHUTDOWN_REQUESTED:
            print(f"\n[Ctrl+C] Saving to {self.save_path}_interrupted.zip...")
            self.model.save(f"{self.save_path}_interrupted")
            return False
        return True


class TargetTracker(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.targets = []
    
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'targets_reached' in info:
                self.targets.append(info['targets_reached'])
        
        if self.verbose and self.num_timesteps % 10000 == 0 and self.targets:
            recent = self.targets[-100:] if len(self.targets) > 100 else self.targets
            print(f"  [Step {self.num_timesteps}] Avg targets (recent): {np.mean(recent):.2f}")
        
        return True


# ============================================================================
# TRAINING
# ============================================================================

def make_env(easy_ratio=0.9):
    def _init():
        return PPODroneEnv(easy_ratio=easy_ratio)
    return _init


def test_policy(model, n_episodes=5, easy_ratio=1.0):
    """Quick test of policy performance."""
    env = PPODroneEnv(easy_ratio=easy_ratio)
    
    total_targets = 0
    total_reward = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_targets = 0
        ep_reward = 0
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if info.get('target_reached'):
                ep_targets += 1
            if term or trunc:
                break
        
        print(f"  Test ep {ep+1}: {ep_targets} targets, reward={ep_reward:.1f}")
        total_targets += ep_targets
        total_reward += ep_reward
    
    avg_targets = total_targets / n_episodes
    avg_reward = total_reward / n_episodes
    print(f"  Average: {avg_targets:.1f} targets, {avg_reward:.1f} reward")
    
    return avg_targets, avg_reward


def train(args):
    global SHUTDOWN_REQUESTED
    
    print("=" * 60)
    print("PPO TRAINING WITH DAGGER ARCHITECTURE")
    print("=" * 60)
    
    # Signal handler
    def handler(sig, frame):
        global SHUTDOWN_REQUESTED
        if SHUTDOWN_REQUESTED:
            sys.exit(1)
        print("\n\n[Ctrl+C] Will save and exit...")
        SHUTDOWN_REQUESTED = True
    
    signal.signal(signal.SIGINT, handler)
    
    # Create env
    print(f"\nCreating {args.n_envs} environments...")
    env = DummyVecEnv([make_env(args.easy_start) for _ in range(args.n_envs)])
    
    if args.resume:
        print(f"\nResuming from {args.resume_path}...")
        model = PPO.load(args.resume_path, env=env)
    else:
        print("\nCreating PPO with DAgger-compatible architecture...")
        
        model = PPO(
            DAggerActorCriticPolicy,  # Custom policy with LayerNorm
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )
        
        # Load DAgger weights
        if args.from_dagger and os.path.exists(args.dagger_path):
            success = load_dagger_into_custom_policy(model.policy, args.dagger_path)
            
            if success:
                print("\n" + "="*60)
                print("TESTING LOADED POLICY")
                print("="*60)
                avg_targets, _ = test_policy(model, n_episodes=5, easy_ratio=1.0)
                
                if avg_targets < 0.5:
                    print("\n⚠ Warning: Low target count. Something may be wrong.")
                else:
                    print(f"\n✓ Policy working! ({avg_targets:.1f} targets/ep)")
            else:
                print("\n⚠ Weight loading failed. Training from scratch.")
    
    # Callbacks
    callbacks = [
        CurriculumCallback(args.easy_start, args.easy_end, args.total_timesteps),
        GracefulExitCallback(args.save_path),
        TargetTracker(),
    ]
    
    if args.save_freq > 0:
        os.makedirs("ppo_checkpoints", exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=args.save_freq,
            save_path="./ppo_checkpoints/",
            name_prefix="ppo_dagger"
        ))
    
    print(f"\nSettings:")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Entropy coef: {args.ent_coef}")
    print(f"  N envs: {args.n_envs}")
    print(f"  Curriculum: {args.easy_start} -> {args.easy_end}")
    
    print(f"\n{'='*60}")
    print("Training... (Ctrl+C to save and exit)")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        pass
    
    print(f"\nSaving to {args.save_path}.zip...")
    model.save(args.save_path)
    
    # Final test
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    test_policy(model, n_episodes=10, easy_ratio=0.5)
    
    return model


def evaluate(args):
    print("=" * 60)
    print("EVALUATING PPO POLICY")
    print("=" * 60)
    
    model = PPO.load(args.eval_model)
    
    env = PPODroneEnv(easy_ratio=args.eval_easy_ratio)
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    total_targets = 0
    total_crashes = 0
    
    print(f"\nRunning {args.eval_episodes} episodes (easy_ratio={args.eval_easy_ratio})...")
    
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        ep_targets = 0
        ep_reward = 0
        
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            
            if info.get('target_reached'):
                ep_targets += 1
            
            if viewer.is_running():
                viewer.sync()
                time.sleep(env.dt)
            
            if term or trunc:
                break
        
        total_targets += ep_targets
        if term and env.data.qpos[2] < 0.1:
            total_crashes += 1
            status = "CRASH"
        else:
            status = "OK"
        
        print(f"  Ep {ep+1}: {ep_targets} targets, reward={ep_reward:.1f} [{status}]")
    
    viewer.close()
    
    print(f"\n{'='*60}")
    print(f"Total targets: {total_targets}")
    print(f"Avg targets/ep: {total_targets/args.eval_episodes:.2f}")
    print(f"Crashes: {total_crashes}/{args.eval_episodes}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    
    # Mode
    parser.add_argument("--eval-only", action="store_true")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    
    # Curriculum
    parser.add_argument("--easy-start", type=float, default=0.9)
    parser.add_argument("--easy-end", type=float, default=0.1)
    
    # DAgger
    parser.add_argument("--from-dagger", action="store_true", default=True)
    parser.add_argument("--no-dagger", action="store_true")
    parser.add_argument("--dagger-path", type=str, default="dagger_policy.pt")
    
    # Save/Load
    parser.add_argument("--save-path", type=str, default="ppo_dagger")
    parser.add_argument("--save-freq", type=int, default=50000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default="ppo_dagger.zip")
    
    # Eval
    parser.add_argument("--eval-model", type=str, default="ppo_dagger.zip")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-easy-ratio", type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.no_dagger:
        args.from_dagger = False
    
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()