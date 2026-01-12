"""
Parallel Visual Benchmark: PID vs DAgger vs PPO
Same position, same targets, trajectory trails.
With smooth tracking camera following all drones.
"""

import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
import argparse
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO

from drone_env_simple import quat_rotate_inverse, quat_rotate
from train_dagger import BCPolicy
from pid_controller import SimplePIDController


# ============================================================================
# TRACKING CAMERA FOR MULTIPLE DRONES
# ============================================================================

class MultiDroneTrackingCamera:
    """
    Smooth tracking camera with FIXED orientation for multiple drones.
    
    Computes lookat as weighted centroid of all drone and target positions.
    Uses smooth exponential lerp with extra smoothing during target transitions.
    """
    
    def __init__(self, viewer, distance=10.0, azimuth=135.0, elevation=-20.0):
        self.viewer = viewer
        
        # Camera position (what we're looking at)
        self.lookat = np.array([0.0, 0.0, 3.0])
        
        # FIXED camera orientation - set once, never changes
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        
        # Lerp configuration
        self.normal_speed = 0.04          # Slower base tracking
        self.transition_speed = 0.012     # Much slower during transitions
        
        # Transition state
        self.in_transition = False
        self.transition_frames = 0
        self.transition_duration = 80     # Longer transition period
        
        # Velocity-based smoothing (for extra smoothness)
        self.lookat_velocity = np.zeros(3)
        self.velocity_damping = 0.85      # How much velocity carries over
        
        self._apply_camera()
    
    def _apply_camera(self):
        """Apply camera state to viewer."""
        self.viewer.cam.lookat[:] = self.lookat
        self.viewer.cam.distance = self.distance
        self.viewer.cam.azimuth = self.azimuth
        self.viewer.cam.elevation = self.elevation
    
    def _compute_centroid(self, drone_positions, target_positions):
        """
        Compute weighted centroid of all drones and targets.
        
        Drones get slightly more weight to keep them more centered.
        Only includes non-crashed drones.
        """
        points = []
        weights = []
        
        # Add drone positions (higher weight)
        for name, pos in drone_positions.items():
            if pos is not None:
                points.append(pos)
                weights.append(1.2)
        
        for name, pos in target_positions.items():
            if pos is not None:
                points.append(pos)
                weights.append(0.8)
        
        if not points:
            return self.lookat  # Fallback to current position
        
        points = np.array(points)
        weights = np.array(weights)
        weights /= weights.sum()
        
        return np.average(points, axis=0, weights=weights)
    
    def update(self, drone_positions, target_positions, any_target_changed=False):
        """
        Update camera position to track all drones and targets.
        
        Uses velocity-based smoothing for extra fluid motion.
        """
        if any_target_changed:
            self.in_transition = True
            self.transition_frames = self.transition_duration
            # Dampen velocity on target change for smoother transition
            self.lookat_velocity *= 0.3
        
        # Compute desired lookat (centroid of all positions)
        desired_lookat = self._compute_centroid(drone_positions, target_positions)
        
        # Choose lerp speed based on transition state
        if self.in_transition:
            # Smooth ease-out curve (quintic for extra smoothness)
            progress = 1.0 - (self.transition_frames / self.transition_duration)
            ease = 1.0 - (1.0 - progress) ** 5  # Quintic ease-out
            
            speed = self.transition_speed + (self.normal_speed - self.transition_speed) * ease
            
            self.transition_frames -= 1
            if self.transition_frames <= 0:
                self.in_transition = False
        else:
            speed = self.normal_speed
        
        # Compute target displacement
        displacement = desired_lookat - self.lookat
        
        # Velocity-based smoothing: blend current velocity with new direction
        target_velocity = displacement * speed
        self.lookat_velocity = (self.lookat_velocity * self.velocity_damping + 
                                target_velocity * (1.0 - self.velocity_damping))
        
        # Apply velocity
        self.lookat = self.lookat + self.lookat_velocity
        
        self._apply_camera()
    
    def reset(self, drone_positions, target_positions):
        """Hard reset camera position for new episode."""
        self.lookat = self._compute_centroid(drone_positions, target_positions)
        self.lookat_velocity = np.zeros(3)
        self.in_transition = False
        self.transition_frames = 0
        self._apply_camera()


# ============================================================================
# TRAJECTORY TRAIL RENDERER
# ============================================================================

class TrajectoryTrails:
    """Manages colored trajectory trails using spheres."""
    
    def __init__(self, max_points=2000):
        self.max_points = max_points
        self.trails = {
            'PID': deque(maxlen=max_points),
            'DAgger': deque(maxlen=max_points),
            'PPO': deque(maxlen=max_points),
        }
        self.colors = {
            'PID': np.array([0.2, 0.9, 0.2, 0.9]),
            'DAgger': np.array([0.2, 0.2, 0.9, 0.9]),
            'PPO': np.array([0.9, 0.2, 0.2, 0.9]),
        }
        self.record_interval = 50
        self.step_count = 0
    
    def reset(self):
        for trail in self.trails.values():
            trail.clear()
        self.step_count = 0
    
    def update(self, positions):
        self.step_count += 1
        if self.step_count % self.record_interval == 0:
            for name, pos in positions.items():
                if pos is not None:
                    self.trails[name].append(pos.copy())
    
    def render(self, viewer):
        """Render trails as small spheres."""
        if not hasattr(viewer, 'user_scn'):
            return
        
        viewer.user_scn.ngeom = 0
        
        for name, trail in self.trails.items():
            if len(trail) < 1:
                continue
            
            color = self.colors[name]
            points = list(trail)
            
            for i, p in enumerate(points):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
                    break
                
                # Fade: older = more transparent
                age = i / len(points)
                alpha = 0.15 + 0.85 * age
                
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                
                # Initialize as sphere
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.03, 0, 0]),  # size (radius)
                    p,  # position
                    np.eye(3).flatten(),  # rotation matrix
                    np.array([color[0], color[1], color[2], alpha * color[3]])
                )
                
                viewer.user_scn.ngeom += 1


# ============================================================================
# CARDINAL DIRECTIONS
# ============================================================================

CARDINAL_DIRECTIONS = {
    'forward':  np.array([ 1.0,  0.0,  0.0]),
    'backward': np.array([-1.0,  0.0,  0.0]),
    'left':     np.array([ 0.0,  1.0,  0.0]),
    'right':    np.array([ 0.0, -1.0,  0.0]),
    'up':       np.array([ 0.0,  0.0,  1.0]),
    'down':     np.array([ 0.0,  0.0, -1.0]),
}


# ============================================================================
# DRONE STATE & ENVIRONMENT
# ============================================================================

class DroneState:
    def __init__(self, name, qpos_start, qvel_start, ctrl_start, mocap_id):
        self.name = name
        self.qpos_slice = slice(qpos_start, qpos_start + 7)
        self.qvel_slice = slice(qvel_start, qvel_start + 6)
        self.ctrl_slice = slice(ctrl_start, ctrl_start + 4)
        self.mocap_id = mocap_id
        
        self.target_pos = np.array([0., 0., 3.])
        self.targets_reached = 0
        self.total_reward = 0.
        self.crashed = False
        self.steps = 0
        self.target_index = 0

class TripleDroneEnv:
    """Environment with 3 superimposed drones, SHARED target sequence."""
    
    def __init__(self, model_path="model_benchmark.xml", target_distance=7.0):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.target_distance = target_distance
        self.target_radius = 0.5
        
        self.drones = {
            'PID': DroneState('PID', qpos_start=0, qvel_start=0, ctrl_start=0, mocap_id=0),
            'DAgger': DroneState('DAgger', qpos_start=7, qvel_start=6, ctrl_start=4, mocap_id=1),
            'PPO': DroneState('PPO', qpos_start=14, qvel_start=12, ctrl_start=8, mocap_id=2),
        }
        
        self.rng = np.random.default_rng()
        
        # Pre-generated target sequence (shared by all drones)
        self.target_sequence = []  # List of (direction_name, distance)
        self.max_targets = 200  # Pre-generate this many targets
        
    def _generate_target_sequence(self):
        """Pre-generate ABSOLUTE world positions for targets."""
        self.target_sequence = []
        direction_names = list(CARDINAL_DIRECTIONS.keys())
        
        # Start from initial position, simulate a reference path
        ref_pos = np.array([0., 0., 3.])
        ref_yaw = 0.0  # Reference orientation (no rotation)
        
        for _ in range(self.max_targets):
            direction_name = self.rng.choice(direction_names)
            distance = self.rng.uniform(self.target_distance * 0.8, self.target_distance * 1.2)
            direction_body = CARDINAL_DIRECTIONS[direction_name]
            
            if direction_name in ['up', 'down']:
                # Vertical: just add to z
                target = ref_pos + direction_body * distance
            else:
                # Horizontal: rotate body direction by reference yaw
                cos_yaw = np.cos(ref_yaw)
                sin_yaw = np.sin(ref_yaw)
                
                # 2D rotation: body frame -> world frame
                world_x = direction_body[0] * cos_yaw - direction_body[1] * sin_yaw
                world_y = direction_body[0] * sin_yaw + direction_body[1] * cos_yaw
                direction_world = np.array([world_x, world_y, 0.0])
                
                target = ref_pos + direction_world * distance
                target[2] = ref_pos[2]  # Maintain altitude
            
            target[2] = np.clip(target[2], 0.5, 15.0)
            self.target_sequence.append(target.copy())
            
            # Move reference to this target for next iteration
            ref_pos = target.copy()

    def _spawn_target_for_drone(self, drone):
        """Spawn target at pre-computed ABSOLUTE world position."""
        if drone.target_index >= len(self.target_sequence):
            drone.target_index = len(self.target_sequence) - 1
        
        # Just use the pre-computed absolute position
        target = self.target_sequence[drone.target_index].copy()
        drone.target_pos = target
        self.data.mocap_pos[drone.mocap_id] = target
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Pre-generate target sequence for this episode
        self._generate_target_sequence()
        
        # All drones start at same position
        start_pos = np.array([0., 0., 3.])
        
        for name, drone in self.drones.items():
            self.data.qpos[drone.qpos_slice][:3] = start_pos
            self.data.qpos[drone.qpos_slice][3:7] = [1, 0, 0, 0]
            drone.targets_reached = 0
            drone.total_reward = 0.
            drone.crashed = False
            drone.steps = 0
            drone.target_index = 0  # Track which target in sequence
        
        mujoco.mj_forward(self.model, self.data)
        
        # Spawn first target for all drones
        for drone in self.drones.values():
            self._spawn_target_for_drone(drone)
    
    def get_obs(self, drone_name):
        drone = self.drones[drone_name]
        
        pos = self.data.qpos[drone.qpos_slice][:3]
        quat = self.data.qpos[drone.qpos_slice][3:7]
        vel_world = self.data.qvel[drone.qvel_slice][:3]
        ang_vel = self.data.qvel[drone.qvel_slice][3:6]
        
        to_target = drone.target_pos - pos
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
    
    def get_state(self, drone_name):
        drone = self.drones[drone_name]
        return {
            'pos': self.data.qpos[drone.qpos_slice][:3].copy(),
            'quat': self.data.qpos[drone.qpos_slice][3:7].copy(),
            'vel': self.data.qvel[drone.qvel_slice][:3].copy(),
            'ang_vel': self.data.qvel[drone.qvel_slice][3:6].copy(),
            'target': drone.target_pos.copy(),
        }
    
    def get_positions(self):
        """Get all drone positions (for trail rendering)."""
        positions = {}
        for name, drone in self.drones.items():
            if not drone.crashed:
                positions[name] = self.data.qpos[drone.qpos_slice][:3].copy()
            else:
                positions[name] = None
        return positions
    
    def get_target_positions(self):
        """Get all target positions (for camera tracking)."""
        targets = {}
        for name, drone in self.drones.items():
            if not drone.crashed:
                targets[name] = drone.target_pos.copy()
            else:
                targets[name] = None
        return targets
    
    def apply_action(self, drone_name, action_normalized):
        drone = self.drones[drone_name]
        motors = (action_normalized + 1.0) * 6.0
        motors = np.clip(motors, 0.0, 12.0)
        self.data.ctrl[drone.ctrl_slice] = motors
    
    def step(self):
        mujoco.mj_step(self.model, self.data)
        
        results = {}
        
        for name, drone in self.drones.items():
            if drone.crashed:
                results[name] = {'crashed': True, 'target_reached': False, 'reward': 0}
                continue
            
            drone.steps += 1
            pos = self.data.qpos[drone.qpos_slice][:3]
            distance = np.linalg.norm(drone.target_pos - pos)
            
            reward = -distance * 0.01
            target_reached = False
            
            if distance < self.target_radius:
                reward = 10.0
                drone.targets_reached += 1
                target_reached = True
                
                # Move to next target in shared sequence
                drone.target_index += 1
                self._spawn_target_for_drone(drone)
            
            if pos[2] < 0.1:
                drone.crashed = True
                reward = -10.0
            
            drone.total_reward += reward
            
            results[name] = {
                'crashed': drone.crashed,
                'target_reached': target_reached,
                'reward': reward,
                'distance': distance,
            }
        
        return results


# ============================================================================
# POLICIES
# ============================================================================

class PIDPolicy:
    def __init__(self, dt=0.005):
        self.pid = SimplePIDController(dt=dt)
    
    def reset(self):
        self.pid.reset()
    
    def get_action(self, state):
        pos, quat = state['pos'], state['quat']
        vel, ang_vel = state['vel'], state['ang_vel']
        target = state['target']
        
        to_target = target - pos
        to_target_body = quat_rotate_inverse(quat, to_target)
        distance = np.linalg.norm(to_target)
        
        aggression = np.clip(distance / 3.0, 0.5, 2.0)
        cmd_vx = np.clip(to_target_body[0] * aggression, -3.0, 3.0)
        cmd_vy = np.clip(to_target_body[1] * aggression, -3.0, 3.0)
        cmd_vz = np.clip(to_target[2] * 1.0, -1.0, 1.0)
        
        motors = self.pid.compute(pos, quat, vel, ang_vel,
                                   cmd_vx=cmd_vx, cmd_vy=cmd_vy,
                                   cmd_vz=cmd_vz, cmd_yaw=0)
        return (motors / 6.0) - 1.0


class DAggerPolicy:
    def __init__(self, path="dagger_policy.pt"):
        self.model = BCPolicy()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
    
    def reset(self):
        pass
    
    def get_action(self, obs):
        return self.model.get_action(obs)


class PPOPolicy:
    def __init__(self, path="ppo_dagger.zip"):
        self.model = PPO.load(path)
    
    def reset(self):
        pass
    
    def get_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# ============================================================================
# BENCHMARK
# ============================================================================

def run_parallel_benchmark(args):
    print("=" * 60)
    print("PARALLEL BENCHMARK: PID (green) vs DAgger (blue) vs PPO (red)")
    print("=" * 60)
    
    writer = SummaryWriter(log_dir=f"./runs/parallel_{int(time.time())}")
    
    env = TripleDroneEnv(model_path=args.model_path, target_distance=args.distance)
    trails = TrajectoryTrails(max_points=args.trail_length)
    
    print("\nLoading policies...")
    policies = {
        'PID': PIDPolicy(dt=env.dt),
        'DAgger': DAggerPolicy(args.dagger_path),
        'PPO': PPOPolicy(args.ppo_path),
    }
    print("  ✓ PID (green), DAgger (blue), PPO (red)")
    
    all_stats = {name: defaultdict(list) for name in policies}
    
    viewer = None
    camera = None
    if args.visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        camera = MultiDroneTrackingCamera(
            viewer,
            distance=args.cam_distance,
            azimuth=args.cam_azimuth,
            elevation=args.cam_elevation
        )
    
    np.random.seed(args.master_seed)
    episode_seeds = np.random.randint(0, 100000, size=args.episodes)
    
    print(f"\nSettings:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Target distance: {args.distance}m")
    print(f"  Trail length: {args.trail_length} points")
    if camera:
        print(f"  Camera: distance={args.cam_distance}, azimuth={args.cam_azimuth}°, elevation={args.cam_elevation}°")
    print()
    
    global_step = 0
    
    for ep_idx, seed in enumerate(episode_seeds):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{args.episodes} (seed={seed})")
        print(f"{'='*60}")
        
        env.reset(seed=int(seed))
        trails.reset()
        for policy in policies.values():
            policy.reset()
        
        # Reset camera to initial positions
        if camera:
            camera.reset(env.get_positions(), env.get_target_positions())
        
        step = 0
        episode_done = False
        
        while step < args.max_steps and not episode_done:
            # Compute actions
            for name in ['PID', 'DAgger', 'PPO']:
                drone = env.drones[name]
                if drone.crashed:
                    continue
                
                if name == 'PID':
                    state = env.get_state(name)
                    action = policies[name].get_action(state)
                else:
                    obs = env.get_obs(name)
                    action = policies[name].get_action(obs)
                
                env.apply_action(name, action)
            
            results = env.step()
            step += 1
            global_step += 1
            
            # Update trails
            trails.update(env.get_positions())
            
            # Check if any target was reached (for camera transition)
            any_target_changed = any(res['target_reached'] for res in results.values())
            
            # Log events
            for name, res in results.items():
                if res['target_reached']:
                    targets = env.drones[name].targets_reached
                    print(f"  [{step:5d}] {name}: Target {targets}!")
                    writer.add_scalar(f"targets/{name}", targets, global_step)
            
            all_crashed = all(d.crashed for d in env.drones.values())
            if all_crashed:
                print(f"  [{step:5d}] All drones crashed!")
                episode_done = True
            
            # Render
            if viewer and viewer.is_running():
                # Update tracking camera
                if camera:
                    camera.update(
                        env.get_positions(),
                        env.get_target_positions(),
                        any_target_changed=any_target_changed
                    )
                
                trails.render(viewer)
                viewer.sync()
                if args.slow_factor > 0:
                    time.sleep(env.dt * args.slow_factor)
            elif viewer and not viewer.is_running():
                episode_done = True
        
        # Episode summary
        print(f"\nEpisode {ep_idx + 1} Results:")
        for name, drone in env.drones.items():
            status = "CRASH" if drone.crashed else "OK"
            color = {'PID': '🟢', 'DAgger': '🔵', 'PPO': '🔴'}[name]
            print(f"  {color} {name:8s}: {drone.targets_reached:2d} targets, "
                  f"reward={drone.total_reward:7.1f}, {drone.steps:4d} steps [{status}]")
            
            all_stats[name]['targets'].append(drone.targets_reached)
            all_stats[name]['rewards'].append(drone.total_reward)
            all_stats[name]['crashes'].append(drone.crashed)
            
            writer.add_scalar(f"episode_targets/{name}", drone.targets_reached, ep_idx)
            writer.add_scalar(f"episode_reward/{name}", drone.total_reward, ep_idx)
    
    if viewer:
        viewer.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'':3}{'Policy':<10} {'Targets/Ep':>12} {'Reward/Ep':>12} {'Crash %':>10}")
    print("-" * 55)
    
    for name in ['PID', 'DAgger', 'PPO']:
        stats = all_stats[name]
        avg_t = np.mean(stats['targets'])
        std_t = np.std(stats['targets'])
        avg_r = np.mean(stats['rewards'])
        crash_pct = np.mean(stats['crashes']) * 100
        color = {'PID': '🟢', 'DAgger': '🔵', 'PPO': '🔴'}[name]
        
        print(f"{color} {name:<10} {avg_t:>8.2f}±{std_t:<4.1f} {avg_r:>12.1f} {crash_pct:>9.1f}%")
    
    writer.close()
    
    np.savez("benchmark_parallel_results.npz", **{
        f"{name}_{metric}": np.array(vals)
        for name, stats in all_stats.items()
        for metric, vals in stats.items()
    })
    
    print(f"\n✓ Tensorboard: tensorboard --logdir=./runs/")
    print(f"✓ Results: benchmark_parallel_results.npz")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-path", default="model_benchmark.xml")
    parser.add_argument("--dagger-path", default="dagger_policy.pt")
    parser.add_argument("--ppo-path", default="ppo_dagger.zip")
    
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--distance", type=float, default=7.0)
    parser.add_argument("--master-seed", type=int, default=42)
    
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--slow-factor", type=float, default=0.0)
    parser.add_argument("--trail-length", type=int, default=40,
                        help="Max points per trail")
    
    # Camera settings
    parser.add_argument("--cam-distance", type=float, default=13.0,
                        help="Camera distance from lookat point")
    parser.add_argument("--cam-azimuth", type=float, default=135.0,
                        help="Camera azimuth angle (degrees) - FIXED")
    parser.add_argument("--cam-elevation", type=float, default=-23.0,
                        help="Camera elevation angle (degrees) - FIXED")
    
    args = parser.parse_args()
    
    if args.no_vis:
        args.visualize = False
    
    run_parallel_benchmark(args)


if __name__ == "__main__":
    main()