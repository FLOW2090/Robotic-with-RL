from controller import Supervisor
from walkingRobot import WalkingRobot
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

walkingRobot = WalkingRobot(Supervisor(), device)
timestep = int(walkingRobot.timestep)
step = 0
interval = 2
episode = 0

while walkingRobot.robot.step(timestep) != -1:
    walkingRobot.updateState()
    if step % interval == 0:
        walkingRobot.act()
        if step != 0:
            walkingRobot.update(step)
    if step != 0:
        walkingRobot.accumulateReward()
    if walkingRobot.isTerminal(step):
        episode += 1
        print(f"Episode {episode} finished with reward {walkingRobot.cumulatedReward} and step {step}")
        walkingRobot.reset()
        step = 0
    else:
        step += 1