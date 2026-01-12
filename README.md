# Drone Final

A reinforcement learning project for drone control using MuJoCo simulation environment.

## Overview

This repository implements drone control policies trained with reinforcement learning algorithms including PPO and DAGGER. The project includes environment simulation, model training, and benchmarking utilities.

## Key Components

- **drone_env_simple.py**: MuJoCo-based drone environment
- **train_ppo.py**: PPO algorithm training implementation
- **train_dagger.py**: DAGGER (Dataset Aggregation) training
- **model_inference.py**: Inference engine for trained models
- **pid_controller.py**: PID controller baseline
- **bench_parallel.py**: Parallel benchmarking utilities

## Results

### Training Visualization

This short video shows the capabilities of the PPO model after training for 50 000 000 steps and 8 envs, CPU time approximately 10 hours. We can note that is overshoots, and drifst perpendicularly to the target direction, but is able to recover and reach the target without failing.
![Training](images/GIF.gif)

### Benchmark Comparison

To demosntrate the finetuning of the PPO model, we compare it to the extremely stable PID controller and the DAgger model, with the exact same targets. After only 1 000 000 additional of training steps, and new rewards to force more precise movements, and no overshoot, no perpendicular drift, the PPO model which was already outperforming the 2 other models, manages to further improve its performances, and reach target in a strict manner. This shocases that once the bulk training for learning stability, and basic cardinal control is done, we can relatively quickly finetune the PPO model for different more specific tasks.
![Benchmark](images/benchmark_GIF.gif)
