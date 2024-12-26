import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from controller import Supervisor
from walkingRobot import WalkingRobot
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

walkingRobot = WalkingRobot(Supervisor(), device)
timestep = int(walkingRobot.timestep)
step = 0
interval = 8
episode = 0

while walkingRobot.robot.step(timestep) != -1:

    # At initial state, only to observe s and choose a
    if step == 0:
        walkingRobot.updateState()
        walkingRobot.act(episode)
        step += 1
        continue

    # Observe reward and add record (s, a, r) to trajectory
    walkingRobot.observeReward()
    walkingRobot.addRecord()

    # Observe state
    walkingRobot.updateState()

    # Terminate, update and reset
    if walkingRobot.isTerminal(step):
        # 输出训练信息
        walkingRobot.observeReward()
        walkingRobot.addRecord()
        print(f"Episode {episode} finished with reward {walkingRobot.reward} and step {step}")
        valueLossLen = len(walkingRobot.agent.valueLosses)
        policyLossLen = len(walkingRobot.agent.policyLosses)
        print(f"Value Loss: {torch.tensor(walkingRobot.agent.valueLosses[valueLossLen - step//interval:]).mean()}, Policy Loss: {torch.tensor(walkingRobot.agent.policyLosses[policyLossLen - step//interval:]).mean()}")
        walkingRobot.update(episode)
        walkingRobot.reset()
        step = 0
        episode += 1
        if episode % 200 == 0:
            # pass
            walkingRobot.plot(episode)
        continue

    # Update agent after having sampled a slice
    if step % interval == 0:
        walkingRobot.update(episode)

    # Choose action
    walkingRobot.act(episode)
    step += 1