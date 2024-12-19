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
        rewards.append(walkingRobot.cumulatedReward)
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
    else:
        step += 1