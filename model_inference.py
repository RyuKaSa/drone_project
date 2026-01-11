"""
Drone PPO Inference with Professional Camera Work

Standalone visualization script with smooth tracking camera
that follows the drone-target midpoint with lerp transitions.
Camera orientation is locked - only position moves.

Usage:
    python drone_inference.py                          # Run with defaults
    python drone_inference.py --model ppo_dagger.zip   # Specify model
    python drone_inference.py --episodes 10            # Number of episodes
    python drone_inference.py --slow                   # Slower playback
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import time
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO


# ============================================================================
# QUATERNION UTILITIES
# ============================================================================

def quat_rotate(quat, vec):
    """Rotate vector by quaternion (wxyz format)."""
    w, x, y, z = quat
    vx, vy, vz = vec
    
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    
    return np.array([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx
    ])


def quat_rotate_inverse(quat, vec):
    """Rotate vector by inverse quaternion."""
    w, x, y, z = quat
    return quat_rotate([w, -x, -y, -z], vec)


# ============================================================================
# ENVIRONMENT (Simplified for inference)
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


class DroneEnv(gym.Env):
    """Drone environment for inference."""
    
    def __init__(self, model_path="model.xml", easy_ratio=0.5,
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
        
        self.max_steps = 10000
        self.steps = 0
        self.targets_reached = 0
        
        # Flag for camera to detect target changes
        self.target_just_changed = False
        
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
        self.target_just_changed = True
        
        return self._get_obs(), {}
    
    def _spawn_target(self):
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7]
        dist = self.np_random.uniform(self.min_dist, self.max_dist)
        
        if self.np_random.random() < self.easy_ratio:
            name = self.np_random.choice(list(CARDINAL_DIRECTIONS.keys()))
            direction = CARDINAL_DIRECTIONS[name]
            if name in ['up', 'down']:
                direction_world = direction
            else:
                direction_world = quat_rotate(quat, direction)
                direction_world[2] = 0
                direction_world /= (np.linalg.norm(direction_world) + 1e-6)
        else:
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
        
        self.target_just_changed = True
    
    def step(self, action):
        self.target_just_changed = False
        
        motors = (np.array(action) + 1.0) * 6.0
        motors = np.clip(motors, 0.0, 12.0)
        
        self.data.ctrl[:4] = motors
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        pos = self.data.qpos[:3]
        ang_vel = self.data.qvel[3:6]
        distance = np.linalg.norm(self.target_pos - pos)
        
        progress = self.prev_distance - distance
        reward = progress * 2.0
        reward -= distance * 0.01
        reward -= np.linalg.norm(ang_vel) * 0.02
        reward -= np.mean(np.abs(action)) * 0.01
        self.prev_distance = distance
        
        target_reached = False
        if distance < self.target_radius:
            reward += 10.0
            self.targets_reached += 1
            target_reached = True
            self._spawn_target()
        
        terminated = pos[2] < 0.1
        if terminated:
            reward -= 10.0
        
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {
            'distance': distance,
            'targets_reached': self.targets_reached,
            'target_reached': target_reached,
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
    
    @property
    def drone_pos(self):
        return self.data.qpos[:3].copy()


# ============================================================================
# FIXED-ORIENTATION TRACKING CAMERA
# ============================================================================

class TrackingCamera:
    """
    Smooth tracking camera with FIXED orientation.
    
    Only the lookat position moves - camera angles are locked.
    Smooth lerp when target changes.
    """
    
    def __init__(self, viewer, distance=15.0, azimuth=135.0, elevation=-30.0):
        self.viewer = viewer
        
        # Camera position (what we're looking at)
        self.lookat = np.array([0.0, 0.0, 3.0])
        
        # FIXED camera orientation - set once, never changes
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        
        # Lerp speeds
        self.normal_speed = 0.08
        self.transition_speed = 0.025  # Slower during target change
        
        # Transition state
        self.in_transition = False
        self.transition_frames = 0
        self.transition_duration = 50  # frames
        
        self._apply_camera()
    
    def _apply_camera(self):
        """Apply camera state to viewer."""
        self.viewer.cam.lookat[:] = self.lookat
        self.viewer.cam.distance = self.distance
        self.viewer.cam.azimuth = self.azimuth
        self.viewer.cam.elevation = self.elevation
    
    def update(self, drone_pos, target_pos, target_changed=False):
        """
        Update camera position to track drone and target.
        
        Only moves the lookat point - orientation stays fixed.
        """
        if target_changed:
            self.in_transition = True
            self.transition_frames = self.transition_duration
        
        # Target: midpoint between drone and target
        desired_lookat = (drone_pos + target_pos) * 0.5
        
        # Choose lerp speed based on transition state
        if self.in_transition:
            # Cubic ease-out for smooth deceleration
            progress = 1.0 - (self.transition_frames / self.transition_duration)
            ease = 1.0 - (1.0 - progress) ** 3
            
            speed = self.transition_speed + (self.normal_speed - self.transition_speed) * ease
            
            self.transition_frames -= 1
            if self.transition_frames <= 0:
                self.in_transition = False
        else:
            speed = self.normal_speed
        
        # Lerp position only
        self.lookat = self.lookat + (desired_lookat - self.lookat) * speed
        
        self._apply_camera()
    
    def reset(self, drone_pos, target_pos):
        """Hard reset camera position for new episode."""
        self.lookat = (drone_pos + target_pos) * 0.5
        self.in_transition = False
        self.transition_frames = 0
        self._apply_camera()


# ============================================================================
# INFERENCE RUNNER
# ============================================================================

def run_inference(model_path, num_episodes=10, max_steps=2000, easy_ratio=0.5,
                  speed=1.0, cam_distance=15.0, cam_azimuth=135.0, cam_elevation=-30.0):
    """Run inference with visualization."""
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print("Creating environment...")
    env = DroneEnv(easy_ratio=easy_ratio)
    
    print("Launching viewer...")
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    camera = TrackingCamera(viewer, cam_distance, cam_azimuth, cam_elevation)
    
    # Stats
    total_targets = 0
    total_crashes = 0
    episode_stats = []
    
    print("\n" + "=" * 60)
    print("DRONE INFERENCE - TRACKING CAMERA")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Speed: {speed}x")
    print(f"Camera: distance={cam_distance}, azimuth={cam_azimuth}°, elevation={cam_elevation}°")
    print("Press Ctrl+C to exit")
    print("=" * 60 + "\n")
    
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            camera.reset(env.drone_pos, env.target_pos)
            
            ep_targets = 0
            ep_reward = 0.0
            ep_steps = 0
            crashed = False
            
            print(f"Episode {ep + 1}/{num_episodes}")
            
            for step in range(max_steps):
                if not viewer.is_running():
                    print("\nViewer closed.")
                    break
                
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                ep_reward += reward
                ep_steps += 1
                
                if info.get('target_reached'):
                    ep_targets += 1
                    print(f"  ✓ Target {ep_targets} reached!")
                
                # Update camera position (orientation stays fixed)
                camera.update(
                    env.drone_pos,
                    env.target_pos,
                    target_changed=env.target_just_changed
                )
                
                # Sync viewer
                viewer.sync()
                
                # Control playback speed
                time.sleep(env.dt / speed)
                
                if terminated:
                    crashed = True
                    break
                if truncated:
                    break
            
            if not viewer.is_running():
                break
            
            # Episode summary
            status = "CRASH" if crashed else "TIMEOUT" if ep_steps >= max_steps else "OK"
            print(f"  → Targets: {ep_targets} | Reward: {ep_reward:.1f} | Steps: {ep_steps} | {status}\n")
            
            episode_stats.append({
                'targets': ep_targets,
                'reward': ep_reward,
                'steps': ep_steps,
                'crashed': crashed
            })
            
            total_targets += ep_targets
            if crashed:
                total_crashes += 1
            
            # Brief pause between episodes
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        viewer.close()
    
    # Final stats
    if episode_stats:
        num_eps = len(episode_stats)
        avg_targets = total_targets / num_eps
        avg_reward = sum(e['reward'] for e in episode_stats) / num_eps
        avg_steps = sum(e['steps'] for e in episode_stats) / num_eps
        crash_rate = total_crashes / num_eps * 100
        
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Episodes completed:    {num_eps}")
        print(f"Total targets reached: {total_targets}")
        print(f"Average targets/ep:    {avg_targets:.2f}")
        print(f"Average reward/ep:     {avg_reward:.1f}")
        print(f"Average steps/ep:      {avg_steps:.0f}")
        print(f"Crash rate:            {crash_rate:.1f}%")
        print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drone PPO Inference with Tracking Camera"
    )
    
    parser.add_argument(
        "--model", type=str, default="ppo_dagger.zip",
        help="Path to trained PPO model"
    )
    parser.add_argument(
        "--episodes", type=int, default=2,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--easy-ratio", type=float, default=0.3,
        help="Ratio of easy (cardinal direction) targets (0-1)"
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Run at 0.5x speed for better viewing"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier"
    )
    
    # Camera settings (fixed orientation)
    parser.add_argument(
        "--cam-distance", type=float, default=8.0,
        help="Camera distance from lookat point"
    )
    parser.add_argument(
        "--cam-azimuth", type=float, default=135.0,
        help="Camera azimuth angle (degrees) - FIXED"
    )
    parser.add_argument(
        "--cam-elevation", type=float, default=-30.0,
        help="Camera elevation angle (degrees) - FIXED"
    )
    
    args = parser.parse_args()
    
    speed = 0.5 if args.slow else args.speed
    
    run_inference(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        easy_ratio=args.easy_ratio,
        speed=speed,
        cam_distance=args.cam_distance,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
    )


if __name__ == "__main__":
    main()