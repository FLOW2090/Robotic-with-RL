import torch

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim):
        super(PolicyNet, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.mufc1 = torch.nn.Linear(stateDim, 64)
        self.mufc2 = torch.nn.Linear(64, 64)
        self.mufc3 = torch.nn.Linear(64, actionDim)
        self.sigmafc1 = torch.nn.Linear(stateDim, 64)
        self.sigmafc2 = torch.nn.Linear(64, 64)
        self.sigmafc3 = torch.nn.Linear(64, actionDim)

        torch.nn.init.xavier_normal_(self.mufc1.weight)
        torch.nn.init.xavier_normal_(self.mufc2.weight)
        torch.nn.init.xavier_normal_(self.mufc3.weight)
        torch.nn.init.xavier_normal_(self.sigmafc1.weight)
        torch.nn.init.xavier_normal_(self.sigmafc2.weight)
        torch.nn.init.xavier_normal_(self.sigmafc3.weight)

    def forward(self, state):
        mu1 = torch.tanh(self.mufc1(state))
        mu2 = torch.tanh(self.mufc2(mu1))
        mu = self.mufc3(mu2)
        sigma1 = torch.tanh(self.sigmafc1(state))
        sigma2 = torch.tanh(self.sigmafc2(sigma1))
        sigma = torch.exp(self.sigmafc3(sigma2))
        return mu, sigma


class ValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(stateDim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return torch.clamp(x, -1e3, 1e3) # 限制范围 [-1e3, 1e3]，防止梯度爆炸/消失

class Agent_PPO:

    def __init__(self, stateDim, actionDim, gamma, lmd, policyLR, valueLR, device):
        self.policyNet = PolicyNet(stateDim, actionDim).to(device)
        self.valueNet = ValueNet(stateDim).to(device)
        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=policyLR)
        self.valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=valueLR)
        self.gamma = gamma
        self.lmd = lmd
        self.policyLR = policyLR
        self.valueLR = valueLR
        self.device = device

    def genActionVec(self, stateVec):
        mu, sigma = self.policyNet(stateVec)
        actionVec = torch.normal(mu, sigma).to(self.device)
        return actionVec

    def genValue(self, stateVec):
        return self.valueNet(stateVec)

    def genLogProb(self, actionVec, stateVec):
        mu, sigma = self.policyNet(stateVec)
        return torch.distributions.Normal(mu, sigma).log_prob(actionVec)
    
    def update(self, trajectory):
        # Calculate old policy
        with torch.no_grad():
            oldLogProbs = [self.genLogProb(record.actionVec, record.stateVec) for record in trajectory[:-1]]

        # Update
        for index in range(len(trajectory)-1):
            # Calculate advantage
            advantage = 0
            with torch.no_grad():
                for t in range(index, len(trajectory)-1):
                    delta = trajectory[t].reward + self.gamma * self.genValue(trajectory[t+1].stateVec) - self.genValue(trajectory[t].stateVec)
                    advantage += (self.gamma * self.lmd) ** (t - index) * delta
            # Calculate ratio
            oldLogProb = oldLogProbs[index]
            logProb = self.genLogProb(trajectory[index].actionVec, trajectory[index].stateVec)
            ratio = torch.exp(logProb - oldLogProb)
            # Calculate policy loss
            policyLoss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.lmd, 1 + self.lmd) * advantage).mean()
            self.policyOptimizer.zero_grad()
            policyLoss.backward()
            self.policyOptimizer.step()
            # Calculate value loss
            valueLoss = -advantage * self.genValue(trajectory[index].stateVec)
            self.valueOptimizer.zero_grad()
            valueLoss.backward()
            self.valueOptimizer.step()
