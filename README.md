# RL Course Project

## Introduction

An RL course project of Tsinghua University, 2024 fall. We try to apply what we learnt in class into practice, which is probably just a toy example. Based on Webots simulation platform and Python, we attempt to teach an NAO robot to walk. We implemented REINFORCE, AC, PPO and DDPG algorithm taught in class combined with DRL. We got relatively descent result using PPO, which makes the robot to walk some distance. Though pretty much time spent in hyperparameters finetuning and reward design, it's still not enough to get such a satisfying outcome. However we learnt a lot when trying all these things in practice.

## Members

Liu Ziheng, Cao Xinyuan, Chen Yitao.

## Usage

1. Download Webots software and load the world file `World/nao.wbt` . 
2. Make sure `Tools->Preferences->Python command` is configured correctly according to your Python path. You can use Python package managers like `anaconda` or `venv` here. 
3. Run the simulation and you can see the training process.
4. At the newest commit of branch PPO_DDPG, we implemented PPO. At the newest commit of branch AC_REINFORCE, we implemented AC and REINFORCE. We also add support to SPOT robot and AIBO robot at corresponding branches.

## Files

- `controllers/walking/walking.py` : The entrance function of the controller.
- `controllers/walking/walkingRobot.py` : Defining the `WalkingRobot` class.
- `controllers/walking/agent.py` : The NN used to approximate the policy function and value function used in `walkingRobot.py` .

## Division

- Liu Ziheng: Basic structure of controller including walking.py and walkingRobot.py, RL algorithms including AC and PPO, NAO's interactions with the environment and designing part of reward function.
- Cao Xinyuan: Finetuning the hyperparameters, helping with PPO and REINFORCE, designing part of reward function.
- Chen Yitao: Helping with REINFORCE and trying implmenting DDPG and running some experiments.