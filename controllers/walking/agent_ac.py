import torch
import math

# actor
class PolicyNet(torch.nn.Module):

    def __init__(self, stateDim, actionDim, actionBounds):
        super(PolicyNet, self).__init__()
        self.mufc1 = torch.nn.Linear(stateDim, 64)
        self.mufc2 = torch.nn.Linear(64, 64)
        self.mufc3 = torch.nn.Linear(64, actionDim)
        self.sigmafc1 = torch.nn.Linear(stateDim, 64)
        self.sigmafc2 = torch.nn.Linear(64, 64)
        self.sigmafc3 = torch.nn.Linear(64, actionDim)
        self.actionBounds = actionBounds

    def forward(self, state, episode):
        mu = torch.relu(self.mufc1(state))
        mu = torch.relu(self.mufc2(mu))
        mu = torch.tanh(self.mufc3(mu))
        # sigma1 = torch.tanh(self.sigmafc1(state))
        # sigma2 = torch.tanh(self.sigmafc2(sigma1))
        # sigma = torch.exp(self.sigmafc3(sigma2))
        sigma = math.exp(-episode/500) * self.actionBounds.max(1)[0]
        return mu, sigma

# critic
class ValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(stateDim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# actor-critic agent
class Agent_AC:

    def __init__(self, stateDim, actionDim, gamma, policyLR, valueLR, actionBounds, device):
        self.policyNet = PolicyNet(stateDim, actionDim, actionBounds).to(device)
        self.valueNet = ValueNet(stateDim).to(device)
        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=policyLR)
        self.valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=valueLR)
        self.gamma = gamma
        self.policyLR = policyLR
        self.valueLR = valueLR
        self.device = device
        self.valueLossList = []
        self.policyLossList = []
        self.sigmaList = []

    def genActionVec(self, stateVec, episode):
        assert(not torch.isnan(stateVec).any())
        mu, sigma = self.policyNet(stateVec, episode)
        self.sigmaList.append(sigma.mean().item())
        actionVec = torch.normal(mu, sigma).to(self.device)
        return actionVec

    def genValue(self, stateVec):
        return self.valueNet(stateVec)

    def genLogProb(self, actionVec, stateVec, episode):
        mu, sigma = self.policyNet(stateVec)
        return torch.distributions.Normal(mu, sigma).log_prob(actionVec)

    def update(self, reward, prevStateVec, stateVec, actionVec, step, episode):
        with torch.no_grad():
            delta = (reward + self.gamma * self.genValue(stateVec) - self.genValue(prevStateVec))
        valueLoss = -delta * self.genValue(prevStateVec)
        self.valueLossList.append(valueLoss.item())
        self.valueOptimizer.zero_grad()
        valueLoss.backward()
        self.valueOptimizer.step()
        policyLoss = -delta * self.gamma ** step * self.genLogProb(actionVec, prevStateVec, episode).mean()
        self.policyLossList.append(policyLoss.item())
        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        self.policyOptimizer.step()
