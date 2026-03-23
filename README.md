# Drone Final

A reinforcement learning project for drone control using MuJoCo simulation environment.

## Overview

This repository implements drone control policies trained with reinforcement learning algorithms including PPO and DAgger (Dataset Aggregation). The project covers the full pipeline: PID expert baseline, imitation learning via DAgger, PPO reinforcement learning with curriculum, reward-shaping finetuning, and a sim-to-real deployment attempt on an ESP-Drone V2.0.

## Key Components

- **drone_env_simple.py** / **drone_env_cardinal.py**: MuJoCo-based drone environments (13D body-frame observations, 4D continuous motor actions)
- **train_dagger.py**: DAgger imitation learning from PID expert with dataset aggregation
- **train_ppo.py**: PPO training with custom LayerNorm policy (matching DAgger architecture for weight transfer), curriculum scheduling, and precision reward shaping
- **pid_controller.py**: Pure PID controller baseline (altitude + attitude + velocity cascaded loops)
- **drone_inference.py**: Inference visualization with smooth tracking camera
- **bench_parallel.py**: Parallel multi-agent benchmarking

## Setup

```bash
conda env create -f environment.yml
conda activate mujoco
```

## Results

### Training Visualization

This short video shows the capabilities of the PPO model after training for 50 000 000 steps and 8 envs, CPU time approximately 10 hours. We can note that it overshoots and drifts perpendicularly to the target direction, but is able to recover and reach the target without crashing.

![Training](images/GIF.gif)

### Benchmark Comparison

To demonstrate the finetuning of the PPO model, we compare it to the PID controller and the DAgger model, with the exact same targets. After only 1 000 000 additional training steps with precision rewards (velocity alignment bonus, perpendicular drift penalty), the PPO model further improves and reaches targets in straight-line trajectories. This showcases that once bulk training for stability and basic cardinal control is done, we can relatively quickly finetune the PPO model for more specific tasks.

![Benchmark](images/4drone_demo.gif)

🔴 PPO RL model (finetuned with precision rewards)

🟡 PPO RL model (bulk training, simple rewards)

🔵 DAgger model

🟢 PID Controller

## Sim-to-Real

A deployment was attempted on an ESP-Drone V2.0 (ESP32, Crazyflie firmware port) over WiFi/CRTP. Achieved ~64 Hz telemetry and autonomous altitude hold, but identified hard blockers (firmware abstraction layer, WiFi latency) preventing direct motor-level policy execution. See [DEPLOYMENT.md](DEPLOYMENT.md) for details.