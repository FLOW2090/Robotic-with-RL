# RL Course Project

## Introduction

An RL course project of Tsinghua University, 2024 fall. We try to apply what we learnt in class into practice, which is probably just a toy example. Based on Webots simulation platform and Python, we attempt to teach an NAO robot to walk even to kick football. We plan to try AC and PPO algorithm taught in class combined with DRL.

## Members

Liu Ziheng, Cao Xinyuan, Chen Yitao.

## Usage

Firstly download Webots software and load the world file `nao.wbt` . Make sure `Tools->Preferences->Python command` is configured correctly according to your python path. You can use Python package managers like `anaconda` or `venv` here. Run the simulation and you can see the training process.

## Files

- `controllers/walking/walking.py` : The entrance function of the controller.
- `controllers/walking/walkingRobot.py` : Defining the `WalkingRobot` class.
- `controllers/walking/agent.py` : The NN used to approximate the policy function and value funciton used in `walkingRobot.py` .

## Division

- Liu Ziheng: Basic structure of controller, including the RL algorithm, interacion with the environment and part of reward function design.
- Cao Xinyuan: 
- Chen Yitao:

## TODO

- Redesign the reward function.
- Introduce PPO based on the AC algorithm applied currently.
- Finetune the parameters, including lr, batch method, DOF and so on.
- Show more loggings, illustrate the loss and return.
- If our robot has learnt how to walk, try teaching it to kick the ball which is currently put outside the playground.