import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from controller import Supervisor
from walkingRobot import WalkingRobot
import torch
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
actionSmoothnessPenalties = []
balanceRewards = []

while walkingRobot.robot.step(timestep) != -1:
    walkingRobot.updateState()
    if step % interval == 0:
        walkingRobot.act()
        if step != 0:
            walkingRobot.update(step)
    if step != 0:
        walkingRobot.accumulateReward(step)
    if walkingRobot.isTerminal(step):
        walkingRobot.update(step)
        episode += 1
        rewards.append(walkingRobot.cumulatedReward)
        forwardRewards.append(walkingRobot.cumulatedForwardReward)
        sidewardPenalties.append(walkingRobot.cumulatedSidewardPenalty)
        stableRewards.append(walkingRobot.cumulatedStableReward)
        fallPenalties.append(walkingRobot.cumulatedFallPenalty)
        rescaleActionPenalties.append(walkingRobot.cumulatedRescaleActionPenalty)
        aliveRewards.append(walkingRobot.cumulatedAliveReward)
        actionSmoothnessPenalties.append(walkingRobot.cumulatedActionSmoothnessPenalty)
        balanceRewards.append(walkingRobot.cumulatedBalanceReward)
        print(f"Episode {episode} finished with reward {walkingRobot.cumulatedReward} and step {step}")
        walkingRobot.reset()
        step = 0

        # 每1000个episode保存一次图像和模型参数
        if episode % 1000 == 0:
            # 创建目录
            os.makedirs(f'image/{episode}', exist_ok=True)
            os.makedirs(f'model/{episode}', exist_ok=True)

            # 保存模型参数
            torch.save(walkingRobot.agent.policyNet.state_dict(), f'model/{episode}/policyNet.pth')
            torch.save(walkingRobot.agent.valueNet.state_dict(), f'model/{episode}/valueNet.pth')

            # 绘制奖励曲线并保存
            plt.figure()
            plt.plot([reward.detach().cpu().numpy() for reward in rewards], label='Reward')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'image/{episode}/reward_curve.png')
            plt.close()
            
            plt.figure()
            plt.plot([forwardReward.detach().cpu().numpy() for forwardReward in forwardRewards], label='Forward Reward')
            plt.plot([sidewardPenalty.detach().cpu().numpy() for sidewardPenalty in sidewardPenalties], label='Sideward Penalty')
            plt.plot([stableReward.detach().cpu().numpy() for stableReward in stableRewards], label='Stable Reward')
            plt.plot([fallPenalty.detach().cpu().numpy() for fallPenalty in fallPenalties], label='Fall Penalty')
            plt.plot([rescaleActionPenalty.detach().cpu().numpy() for rescaleActionPenalty in rescaleActionPenalties], label='Rescale Action Penalty')
            plt.plot([aliveReward.detach().cpu().numpy() for aliveReward in aliveRewards], label='Alive Reward')
            plt.plot([actionSmoothnessPenalty.detach().cpu().numpy() for actionSmoothnessPenalty in actionSmoothnessPenalties], label='Action Smoothness Penalty')
            plt.plot([balanceReward.detach().cpu().numpy() for balanceReward in balanceRewards], label='Balance Reward')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'image/{episode}/detailed_reward_curve.png')
            plt.close()
    else:
        step += 1