from agent import Agent
import torch

class WalkingRobot:
    def __init__(self, robot, device):
        self.robot = robot
        self.device = device
        self.timestep = int(self.robot.getBasicTimeStep())
        self.stateVec = None
        self.prevStateVec = None
        self.actionVec = None
        self.prevActionVec = None
        self.position = None
        self.prevPosition = None
        self.maxStep = 512
        self.gamma = 0.99
        self.policyLR = 3e-4
        self.valueLR = 1e-3
        self.reward = 0
        self.cumulatedReward = 0
        self.cumulatedForwardReward = 0
        self.cumulatedSidewardPenalty = 0
        self.cumulatedStableReward = 0
        self.cumulatedFallPenalty = 0
        self.cumulatedRescaleActionPenalty = 0
        self.cumulatedAliveReward = 0
        self.cumulatedMovementPenalty = 0

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
        self.agent = Agent(stateDim, actionDim, self.gamma, self.policyLR, self.valueLR, self.device)

    def reset(self):
        self.robot.simulationReset()
        self.reward = 0
        self.prevPosition = None
        self.position = None
        self.prevStateVec = None
        self.stateVec = None
        self.actionVec = None
        self.cumulatedReward = 0
        self.cumulatedForwardReward = 0
        self.cumulatedSidewardPenalty = 0
        self.cumulatedStableReward = 0
        self.cumulatedFallPenalty = 0
        self.cumulatedRescaleActionPenalty = 0
        self.cumulatedAliveReward = 0
        self.cumulatedMovementPenalty = 0

    def isTerminal(self, step):
        if step >= self.maxStep:
            return True
        if self.robot.getFromDef('Robot').getField('translation').getSFVec3f()[2] < 0.25:
            return True
        return False

    def act(self):
        self.prevActionVec = self.actionVec
        self.actionVec = self.agent.genActionVec(self.stateVec)
        rescaledActionVec = self.rescaleActionVec(self.actionVec)
        self.takeAction(rescaledActionVec)

    def update(self, step):
        self.agent.update(self.reward, self.prevStateVec, self.stateVec, self.actionVec, step)
        self.reward = 0

    def updateState(self):
        self.prevStateVec = self.stateVec
        self.prevPosition = self.position
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
        forwardReward = 200 * (self.position[1] - self.prevPosition[1])
        # # Penalty for moving sideward
        # sidewardPenalty = 10 * abs(self.position[0] - self.prevPosition[0])
        # # Encourage to stay stable
        # stableReward = 0.3 * (1 - abs(self.position[2] - self.prevPosition[2]))
        # Penalty for falling
        fallPenalty = 50 * (self.position[2] < 0.25)
        # Penalty for too large clipping in motor movement
        rescaleActionPenalty = 0.05 * torch.norm(self.actionVec - self.rescaleActionVec(self.actionVec)).item()
        # Encourage to stay alive
        aliveReward = 0.75
        # Penalty for too large change in action
        movementPenalty = 0.2 * torch.norm(self.rescaleActionVec(self.actionVec) - (self.rescaleActionVec(self.prevActionVec) if self.prevActionVec is not None else self.rescaleActionVec(self.actionVec))).item()
        reward = forwardReward - movementPenalty - fallPenalty - rescaleActionPenalty + aliveReward
        reward /= 20
        self.reward += reward
        self.cumulatedReward += self.gamma ** step * reward
        self.cumulatedForwardReward += self.gamma ** step * forwardReward
        self.cumulatedFallPenalty += self.gamma ** step * fallPenalty
        self.cumulatedRescaleActionPenalty += self.gamma ** step * rescaleActionPenalty
        self.cumulatedAliveReward += self.gamma ** step * aliveReward
        self.cumulatedMovementPenalty += self.gamma ** step * movementPenalty