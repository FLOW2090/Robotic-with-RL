import torch

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(stateDim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mean_layer = torch.nn.Linear(64, actionDim)  # 输出均值
        self.std_layer = torch.nn.Linear(64, actionDim)  # 输出标准差

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.std_layer(x))
        return mean, std


class Agent_REINFORCE:
    def __init__(self, stateDim, actionDim, gamma, policyLR, device):
        self.policyNet = PolicyNet(stateDim, actionDim).to(device)
        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=policyLR)
        self.gamma = gamma
        self.device = device
        self.policyLossList = []
        self.valueLossList = []
        self.rewardList = []
        self.actionLogProbList = []

    def genActionVec(self, stateVec):
        assert (not torch.isnan(stateVec).any())
        mu, sigma = self.policyNet(stateVec)
        actionVec = torch.normal(mu, sigma).to(self.device)
        return actionVec

    def genLogProb(self, actionVec, stateVec):
        mu, sigma = self.policyNet(stateVec)

        return torch.distributions.Normal(mu, sigma).log_prob(actionVec)

    def update(self, reward, prevStateVec, stateVec, actionVec, step, isTerminal=False):
        self.rewardList.append(reward)
        action_log_prob = self.genLogProb(actionVec, prevStateVec)
        self.actionLogProbList.append(action_log_prob)
        if isTerminal:
            G = 0
            returns = []
            for r in self.rewardList[::-1]:
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            policyLoss = []
            for log_prob, R in zip(self.actionLogProbList, returns):
                policyLoss.append(-log_prob * R)
            policyLoss = torch.cat(policyLoss).sum()
            self.policyLossList.append(policyLoss.item())
            self.policyOptimizer.zero_grad()
            policyLoss.backward()
            self.policyOptimizer.step()

            self.rewardList = []
            self.actionLogProbList = []

