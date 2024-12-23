from agent_ac import Agent_AC
from agent_ppo import Agent_PPO
import torch
import os
import matplotlib.pyplot as plt

class Record:

    def __init__(self, stateVec, actionVec=None, reward=None):
        self.stateVec = stateVec
        self.actionVec = actionVec
        self.reward = reward

class WalkingRobot:

    def __init__(self, robot, device):
        self.robot = robot
        self.device = device
        self.timestep = int(self.robot.getBasicTimeStep())

        # Hyperparameters
        self.maxStep = 512
        self.gamma = 0.99
        self.lmd = 0.97
        self.policyLR = 3e-4
        self.valueLR = 1e-3

        # Data for training
        self.position = None
        self.prevPosition = None
        self.actionVec = None
        self.prevActionVec = None
        self.trajectory = []
        self.reward = 0

        # Data for plot
        self.rewards = []
        self.forwardRewards = []
        self.fallPenalties = []
        self.movementPenalties = []
        self.rescaleActionPenalties = []
        self.aliveRewards = []
        self.valueLossList = []
        self.policyLossList = []

        # Position sensors
        self.accelerometer = self.robot.getDevice('accelerometer')
        self.accelerometer.enable(self.timestep)
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.inertialUnit = self.robot.getDevice('inertial unit')
        self.inertialUnit.enable(self.timestep)

        # Basic motors
        motorNames = [
            'LAnklePitch', 'RAnklePitch', 'LKneePitch', 'RKneePitch', 'LHipPitch', 'RHipPitch',
            'LShoulderPitch', 'RShoulderPitch',
            ]
        # motorNames = [
        #     'LAnklePitch', 'RAnklePitch', 'LKneePitch', 'RKneePitch', 'LHipPitch', 'RHipPitch',
        #     'LAnkleRoll', 'RAnkleRoll', 'LHipRoll', 'RHipRoll', 'LHipYawPitch', 'RHipYawPitch',
        #     'LElbowRoll', 'RElbowRoll', 'LElbowYaw', 'RElbowYaw', 'LShoulderPitch', 'RShoulderPitch',
        #     'LElbowYaw', 'RElbowYaw', 'LShoulderRoll', 'RShoulderRoll', 'LWristYaw', 'RWristYaw'
        #     ]
        self.motors = []
        self.actionBounds = []
        for motorName in motorNames:
            motor = self.robot.getDevice(motorName)
            self.motors.append(motor)
            self.actionBounds.append([motor.getMinPosition(), motor.getMaxPosition()])
        self.actionBounds = torch.tensor(self.actionBounds, dtype=torch.float32, device=self.device)

        # Basic motor position sensors
        motorSensorNames = [
            'LAnklePitchS', 'RAnklePitchS', 'LKneePitchS', 'RKneePitchS', 'LHipPitchS', 'RHipPitchS',
            'LShoulderPitchS', 'RShoulderPitchS',
            ]
        self.motorSensors = []
        for motorSensorName in motorSensorNames:
            motorSensor = self.robot.getDevice(motorSensorName)
            motorSensor.enable(self.timestep)
            self.motorSensors.append(motorSensor)

        # Initialize agent
        stateDim = 8 + len(self.motorSensors)
        actionDim = len(self.motors)
        self.agent = Agent_PPO(stateDim, actionDim, self.gamma, self.lmd, self.policyLR, self.valueLR, self.device)
        
        # # 加载模型参数
        # try:
        #     self.agent.policyNet.load_state_dict(torch.load('model/61000/policyNet.pth'))
        #     self.agent.valueNet.load_state_dict(torch.load('model/61000/valueNet.pth'))
        #     print("Model parameters loaded successfully.")
        # except FileNotFoundError:
        #     print("Model parameters not found. Training from scratch.")

    def reset(self):
        self.robot.simulationReset()
        self.position = None
        self.prevPosition = None
        self.actionVec = None
        self.prevActionVec = None
        self.trajectory = []
        self.reward = 0

    def isTerminal(self, step):
        if step >= self.maxStep:
            return True
        if self.robot.getFromDef('Robot').getField('translation').getSFVec3f()[2] < 0.25:
            return True
        return False

    # Generate new action based on current state
    def act(self):
        self.actionVec = self.agent.genActionVec(self.stateVec)
        rescaledActionVec = self.rescaleActionVec(self.actionVec)
        self.takeAction(rescaledActionVec)

    def update(self):
        # Add only a state to the trajectory
        self.trajectory.append(Record(self.stateVec))
        # Update agent
        self.agent.update(self.trajectory)
        self.trajectory = []

    # Update state and save previous state and action
    def updateState(self):
        self.prevPosition = self.position
        self.prevActionVec = self.actionVec
        self.stateVec = self.genStateVec()
        self.position = self.robot.getFromDef('Robot').getField('translation').getSFVec3f()

    def addRecord(self):
        self.trajectory.append(Record(self.stateVec, self.actionVec, self.reward))

    def genStateVec(self):
        stateVec = torch.tensor([], dtype=torch.float32, device=self.device)
        stateVec = torch.cat((stateVec, torch.tensor(self.accelerometer.getValues(), dtype=torch.float32, device=self.device)))
        stateVec = torch.cat((stateVec, torch.tensor(self.gyro.getValues()[0:2], dtype=torch.float32, device=self.device)))
        stateVec = torch.cat((stateVec, torch.tensor(self.inertialUnit.getRollPitchYaw(), dtype=torch.float32, device=self.device)))
        for motorSensor in self.motorSensors:
            stateVec = torch.cat((stateVec, torch.tensor([motorSensor.getValue()], dtype=torch.float32, device=self.device)))
        return stateVec

    def rescaleActionVec(self, actionVec):
        return torch.clip(actionVec, self.actionBounds[:, 0] + 1e-3, self.actionBounds[:, 1] - 1e-3)

    def takeAction(self, actionVec):
        for i, motor in enumerate(self.motors):
            motor.setPosition(actionVec[i].item())

    def observeReward(self):
        # Encourage to move forward
        forwardReward = 200 * (self.position[1] - self.prevPosition[1]) if self.prevPosition is not None else 0
        # Penalty for falling
        fallPenalty = 50 * (self.position[2] < 0.25)
        # Penalty for too large clipping in motor movement
        rescaleActionPenalty = 0.05 * torch.norm(self.actionVec - self.rescaleActionVec(self.actionVec)).item()
        # Encourage to stay alive
        aliveReward = 0.75
        # Penalty for too large change in action
        movementPenalty = 0.2 * torch.norm(self.rescaleActionVec(self.actionVec) - self.rescaleActionVec(self.prevActionVec)).item() if self.prevActionVec is not None else 0
        reward = forwardReward - movementPenalty - fallPenalty - rescaleActionPenalty + aliveReward
        reward /= 20
        self.reward = reward
        # self.rewards.append(reward)
        # self.forwardRewards.append(forwardReward)
        # self.fallPenalties.append(fallPenalty)
        # self.rescaleActionPenalties.append(rescaleActionPenalty)
        # self.aliveRewards.append(aliveReward)
        # self.movementPenalties.append(movementPenalty)

    def plot(self, episode):
        os.makedirs(f'image/{episode}', exist_ok=True)
        # os.makedirs(f'model/{episode}', exist_ok=True)

        # # 保存模型参数
        # torch.save(self.agent.policyNet.state_dict(), f'model/{episode}/policyNet.pth')
        # torch.save(self.agent.valueNet.state_dict(), f'model/{episode}/valueNet.pth')

        # 绘制奖励曲线并保存
        plt.figure()
        plt.plot(self.rewards, label='Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(f'image/{episode}/reward_curve.png')
        plt.close()
        
        plt.figure()
        plt.plot(self.valueLossList, label='Value Loss')
        plt.plot(self.policyLossList, label='Policy Loss')
        plt.xlabel('Interval')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig(f'image/{episode}/loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(self.forwardRewards, label='Forward Reward')
        plt.plot(self.fallPenalties, label='Fall Penalty')
        plt.plot(self.movementPenalties, label='Movement Penalty')
        plt.plot(self.rescaleActionPenalties, label='Rescale Action Penalty')
        plt.plot(self.aliveRewards, label='Alive Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward/Penalty')
        plt.legend()
        # plt.show()
        plt.savefig(f'image/{episode}/detailed_reward_curve.png')
        plt.close()