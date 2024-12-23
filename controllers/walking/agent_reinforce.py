import torch

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim):
        super(PolicyNet, self).__init__()
        # 共享特征提取层
        self.shared_fc1 = torch.nn.Linear(stateDim, 1024)
        self.shared_fc2 = torch.nn.Linear(1024, 256)
        self.shared_fc3 = torch.nn.Linear(256, 64)

        # 均值（mu）分支
        self.mu_fc = torch.nn.Linear(64, actionDim)
        
        # 标准差（sigma）分支
        self.sigma_fc = torch.nn.Linear(64, actionDim)
        
        # 初始化权重
        torch.nn.init.xavier_uniform_(self.shared_fc1.weight)
        torch.nn.init.xavier_uniform_(self.shared_fc2.weight)
        torch.nn.init.xavier_uniform_(self.shared_fc3.weight)
        torch.nn.init.xavier_uniform_(self.mu_fc.weight)
        torch.nn.init.xavier_uniform_(self.sigma_fc.weight)

    def forward(self, state):
        # 提取共享特征
        shared = torch.relu(self.shared_fc1(state))
        shared = torch.relu(self.shared_fc2(shared))
        shared = torch.relu(self.shared_fc3(shared))
        
        mu = self.mu_fc(shared)
        sigma = torch.exp(self.sigma_fc(shared))
        
        return mu, sigma


class Agent_REINFORCE:
    def __init__(self, stateDim, actionDim, gamma, policyLR, device):
        self.policyNet = PolicyNet(stateDim, actionDim).to(device)
        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=policyLR)
        self.gamma = gamma
        self.device = device
        self.policyLossList = []

    def genActionVec(self, stateVec):
        assert not torch.isnan(stateVec).any()
        mu, sigma = self.policyNet(stateVec)
        actionVec = torch.normal(mu, sigma).to(self.device)
        return actionVec

    def genLogProb(self, actionVec, stateVec):
        mu, sigma = self.policyNet(stateVec)
        return torch.distributions.Normal(mu, sigma).log_prob(actionVec).sum(dim=-1)  # Sum over action dimensions

    def computeReturns(self, rewards):
        """
        计算每个时间步的累积回报（Return）。
        """
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, states, actions, rewards):
        """
        使用REINFORCE算法更新策略网络。
        - states: 每个时间步的状态
        - actions: 每个时间步执行的动作
        - rewards: 每个时间步的即时奖励
        """
        # 计算累积回报
        returns = self.computeReturns(rewards)

        # 计算每个时间步的log概率和策略损失
        policyLoss = 0
        for state, action, G in zip(states, actions, returns):
            log_prob = self.genLogProb(action, state)
            policyLoss += -log_prob * G  # REINFORCE策略梯度公式

        # 优化策略网络
        policyLoss /= len(states)  # 取平均，稳定梯度
        self.policyLossList.append(policyLoss.item())
        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        self.policyOptimizer.step()
