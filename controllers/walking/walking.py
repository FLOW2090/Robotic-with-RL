from controller import Supervisor
from walkingRobot import WalkingRobot
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

walkingRobot = WalkingRobot(Supervisor(), device)
timestep = int(walkingRobot.timestep)
step = 0
interval = 4
episode = 0

while walkingRobot.robot.step(timestep) != -1:

    # At initial state, only to observe s and choose a
    if step == 0:
        walkingRobot.updateState()
        walkingRobot.act()
        step += 1
        continue

    # Observe reward and add record (s, a, r) to trajectory
    walkingRobot.observeReward()
    walkingRobot.addRecord()

    # Observe state
    walkingRobot.updateState()

    # Terminate, update and reset
    if walkingRobot.isTerminal(step):
        walkingRobot.update()
        walkingRobot.reset()
        step = 0
        episode += 1
        if episode % 500 == 0:
            pass
            # walkingRobot.plot(episode)
        continue

    # Update agent after having sampled a slice
    if step % interval == 0:
        walkingRobot.update()

    # Choose action
    walkingRobot.act()
    step += 1