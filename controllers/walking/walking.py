from controller import Supervisor
from walkingRobot import WalkingRobot
import torch
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

walkingRobot = WalkingRobot(Supervisor(), device)
timestep = int(walkingRobot.timestep)
step = 0
interval = 2
episode = 0
rewards = []
forwardRewards = []
sidewardPenalties = []
stableRewards = []
fallPenalties = []
rescaleActionPenalties = []
aliveRewards = []

while walkingRobot.robot.step(timestep) != -1:
    walkingRobot.updateState()
    if walkingRobot.isTerminal(step):
        episode += 1
        print(f"Episode {episode} finished with reward {walkingRobot.cumulatedReward} and step {step}")
        rewards.append(walkingRobot.cumulatedReward)
        forwardRewards.append(walkingRobot.cumulatedForwardReward)
        sidewardPenalties.append(walkingRobot.cumulatedSidewardPenalty)
        stableRewards.append(walkingRobot.cumulatedStableReward)
        fallPenalties.append(walkingRobot.cumulatedFallPenalty)
        rescaleActionPenalties.append(walkingRobot.cumulatedRescaleActionPenalty)
        aliveRewards.append(walkingRobot.cumulatedAliveReward)
        if episode % 100 == 0:
            os.makedirs(f'image/{episode}', exist_ok=True)
            # os.makedirs(f'model/{episode}', exist_ok=True)

            # # 保存模型参数
            # torch.save(walkingRobot.agent.policyNet.state_dict(), f'model/{episode}/policyNet.pth')
            # torch.save(walkingRobot.agent.valueNet.state_dict(), f'model/{episode}/valueNet.pth')

            # # 绘制奖励曲线并保存
            plt.figure()
            plt.plot(rewards, label='Reward')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'image/{episode}/reward_curve.png')
            plt.close()
            
            plt.figure()
            plt.plot(walkingRobot.agent.valueLossList, label='Value Loss')
            plt.plot(walkingRobot.agent.policyLossList, label='Policy Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'image/{episode}/loss_curve.png')
            plt.close()

            plt.figure()
            plt.plot(forwardRewards, label='Forward Reward')
            plt.plot(sidewardPenalties, label='Sideward Penalty')
            plt.plot(stableRewards, label='Stable Reward')
            plt.plot(fallPenalties, label='Fall Penalty')
            plt.plot(rescaleActionPenalties, label='Rescale Action Penalty')
            plt.plot(aliveRewards, label='Alive Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward/Penalty')
            plt.legend()
            plt.savefig(f'image/{episode}/detailed_reward_curve.png')
            plt.close()
            walkingRobot.agent.valueLossList = []
            walkingRobot.agent.policyLossList = []
        walkingRobot.reset()
        step = 0
    else:
        if step % interval == 0:
            walkingRobot.act()
            if step != 0:
                walkingRobot.update(step)
        if step != 0:
            walkingRobot.accumulateReward()
        step += 1