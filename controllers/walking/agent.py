import torch

class PolicyNet(torch.nn.Module):

    def __init__(self, stateDim, actionDim):
        super(PolicyNet, self).__init__()
        self.mufc1 = torch.nn.Linear(stateDim, 64)
        self.mufc2 = torch.nn.Linear(64, 64)
        self.mufc3 = torch.nn.Linear(64, actionDim)
        self.sigmafc1 = torch.nn.Linear(stateDim, 64)
        self.sigmafc2 = torch.nn.Linear(64, 64)
        self.sigmafc3 = torch.nn.Linear(64, actionDim)

    def forward(self, state):
        mu = torch.relu(self.mufc1(state))
        mu = torch.relu(self.mufc2(mu))
        mu = self.mufc3(mu)
        sigma = torch.relu(self.sigmafc1(state))
        sigma = torch.relu(self.sigmafc2(sigma))
        sigma = torch.exp(self.sigmafc3(sigma))
        # Sometimes it will encounter nan due to exp for some unknown reason...
        if not torch.all(sigma > 0):
            print(sigma)
        return mu, sigma

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

class Agent:

    def __init__(self, stateDim, actionDim, gamma, policyLR, valueLR, device):
        self.policyNet = PolicyNet(stateDim, actionDim).to(device)
        self.valueNet = ValueNet(stateDim).to(device)
        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=policyLR)
        self.valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=valueLR)
        self.gamma = gamma
        self.policyLR = policyLR
        self.valueLR = valueLR
        self.device = device
        self.valueLossList = []
        self.policyLossList = []

    def genActionVec(self, stateVec):
        assert(not torch.isnan(stateVec).any())
        mu, sigma = self.policyNet(stateVec)
        actionVec = torch.normal(mu, sigma).to(self.device)
        return actionVec

    def genValue(self, stateVec):
        return self.valueNet(stateVec)

    def genLogProb(self, actionVec, stateVec):
        mu, sigma = self.policyNet(stateVec)
        return torch.distributions.Normal(mu, sigma).log_prob(actionVec)

    def update(self, reward, prevStateVec, stateVec, actionVec, step):
        with torch.no_grad():
            delta = (reward + self.gamma * self.genValue(stateVec) - self.genValue(prevStateVec))
        valueLoss = -delta * self.genValue(prevStateVec)
        self.valueLossList.append(valueLoss.item())
        self.valueOptimizer.zero_grad()
        valueLoss.backward()
        self.valueOptimizer.step()
        policyLoss = -delta * self.gamma ** step * self.genLogProb(actionVec, prevStateVec).mean()
        self.policyLossList.append(policyLoss.item())
        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        self.policyOptimizer.step()
