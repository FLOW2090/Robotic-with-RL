from agent_ac import Agent_AC
from agent_ppo import Agent_PPO
from agent_reinforce import Agent_REINFORCE
import torch

class WalkingRobot:
    def __init__(self, robot, device):
        self.robot = robot
        self.device = device
        self.timestep = int(self.robot.getBasicTimeStep())
        self.stateVec = None
        self.prevStateVec = None
        self.actionVec = None
        self.preActionVec = None
        self.position = None
        self.prevPosition = None
        self.maxStep = 512
        self.gamma = 0.99
        self.policyLR = 1e-7
        self.valueLR = 1e-6
        self.reward = 0
        self.cumulatedReward = 0
        self.cumulatedForwardReward = 0
        self.cumulatedSidewardPenalty = 0
        self.cumulatedStableReward = 0
        self.cumulatedFallPenalty = 0
        self.cumulatedRescaleActionPenalty = 0
        self.cumulatedAliveReward = 0
        self.cumulatedActionSmoothnessPenalty = 0
        self.cumulatedBalanceReward = 0
        self.cumulatedNoMovementPenalty = 0

        # Position sensors
        self.accelerometer = self.robot.getDevice('accelerometer')
        self.accelerometer.enable(self.timestep)
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.inertialUnit = self.robot.getDevice('inertial unit')
        self.inertialUnit.enable(self.timestep)

        # Basic motors
        motorNames = [
            'LAnklePitch', 'RAnklePitch', 'LKneePitch', 'RKneePitch', 
            'LHipPitch', 'RHipPitch', 'LShoulderPitch', 'RShoulderPitch'
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
            'LAnklePitchS', 'RAnklePitchS', 'LKneePitchS', 'RKneePitchS', 
            'LHipPitchS', 'RHipPitchS', 'LShoulderPitchS', 'RShoulderPitchS'
            ]
        self.motorSensors = []
        for motorSensorName in motorSensorNames:
            motorSensor = self.robot.getDevice(motorSensorName)
            motorSensor.enable(self.timestep)
            self.motorSensors.append(motorSensor)

        # Initialize agent
        stateDim = 8 + len(self.motorSensors)
        actionDim = len(self.motors)
        # self.agent = Agent_PPO(stateDim, actionDim, self.gamma, self.policyLR, self.valueLR, self.device)
        self.agent = Agent_REINFORCE(stateDim, actionDim, self.gamma, self.policyLR, self.device)
        
        # # 加载模型参数
        # try:
        #     self.agent.policyNet.load_state_dict(torch.load('model/61000/policyNet.pth'))
        #     self.agent.valueNet.load_state_dict(torch.load('model/61000/valueNet.pth'))
        #     print("Model parameters loaded successfully.")
        # except FileNotFoundError:
        #     print("Model parameters not found. Training from scratch.")

    def reset(self):
        self.robot.simulationReset()
        self.reward = 0
        self.prevPosition = None
        self.position = None
        self.prevStateVec = None
        self.stateVec = None
        self.prevActionVec = None
        self.actionVec = None
        self.cumulatedReward = 0
        self.cumulatedForwardReward = 0
        self.cumulatedSidewardPenalty = 0
        self.cumulatedStableReward = 0
        self.cumulatedFallPenalty = 0
        self.cumulatedRescaleActionPenalty = 0
        self.cumulatedAliveReward = 0
        self.cumulatedActionSmoothnessPenalty = 0
        self.cumulatedBalanceReward = 0
        self.cumulatedNoMovementPenalty = 0

    def isTerminal(self, step):
        if step >= self.maxStep:
            return True
        if self.robot.getFromDef('Robot').getField('translation').getSFVec3f()[2] < 0.20:
            return True
        return False

    def act(self):
        self.actionVec = self.agent.genActionVec(self.stateVec)
        rescaledActionVec = self.rescaleActionVec(self.actionVec)
        self.takeAction(rescaledActionVec)

    def update(self, step):
        self.agent.update(self.reward, self.prevStateVec, self.stateVec, self.actionVec, step, self.isTerminal(step))
        self.reward = 0

    def updateState(self):
        self.prevStateVec = self.stateVec
        self.prevPosition = self.position
        self.prevActionVec = self.actionVec
        self.stateVec = self.genStateVec()
        self.position = self.robot.getFromDef('Robot').getField('translation').getSFVec3f()

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

    def accumulateReward(self, step):
        # Encourage to move forward
        forwardReward = 500 * (self.position[1] - self.prevPosition[1])
        # Penalty for moving sideward
        # sidewardPenalty = 5 * abs(self.position[0] - self.prevPosition[0])
        # Encourage to stay stable
        # stableReward = 10 * (1 - abs(self.position[2] - self.prevPosition[2]))
        # Penalty for falling
        fallPenalty = 25 * (self.position[2] < 0.20)
        # Penalty for too large clipping in motor movement
        # rescaleActionPenalty = 5 * torch.norm(self.actionVec - self.rescaleActionVec(self.actionVec))
        # Encourage to stay alive
        aliveReward = 1
        # 平滑动作惩罚：鼓励动作的连续性和平滑性
        # actionSmoothnessPenalty = 5 * torch.norm(self.actionVec - (self.prevActionVec if self.prevActionVec is not None else self.actionVec))
        # 平衡奖励：利用陀螺仪数据鼓励机器人保持平衡
        # balanceReward = 10 * (1 - abs(self.gyro.getValues()[0]))
        # 原地不动惩罚
        noMovementPenalty = 3 * (abs(self.position[1] - self.prevPosition[1]) < 1e-3)
        # 汇总奖励
        reward = forwardReward - fallPenalty + aliveReward - noMovementPenalty
        # reward = (forwardReward - sidewardPenalty + stableReward - fallPenalty
        #         - rescaleActionPenalty + aliveReward - actionSmoothnessPenalty
        #         + balanceReward)
        # reward = forwardReward - sidewardPenalty + stableReward - fallPenalty - rescaleActionPenalty + aliveReward
        reward /= 80
        self.reward += reward
        self.cumulatedReward += reward * self.gamma ** step
        self.cumulatedForwardReward += forwardReward * self.gamma ** step
        # self.cumulatedSidewardPenalty += sidewardPenalty * self.gamma ** step
        # self.cumulatedStableReward += stableReward * self.gamma ** step
        self.cumulatedFallPenalty += fallPenalty * self.gamma ** step
        # self.cumulatedRescaleActionPenalty += rescaleActionPenalty * self.gamma ** step
        self.cumulatedAliveReward += aliveReward * self.gamma ** step
        # self.cumulatedActionSmoothnessPenalty += actionSmoothnessPenalty * self.gamma ** step
        # self.cumulatedBalanceReward += balanceReward * self.gamma ** step
        self.cumulatedNoMovementPenalty += noMovementPenalty * self.gamma ** step
        