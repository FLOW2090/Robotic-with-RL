import torch

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim, actionDim):
        super(PolicyNet, self).__init__()
        # 共享特征提取层
        self.shared_fc1 = torch.nn.Linear(stateDim, 128)
        self.shared_fc2 = torch.nn.Linear(128, 64)

        # 均值（mu）分支
        self.mu_fc = torch.nn.Linear(64, actionDim)
        
        # 标准差（sigma）分支
        self.sigma_fc = torch.nn.Linear(64, actionDim)

    def forward(self, state):
        # 提取共享特征
        shared = torch.relu(self.shared_fc1(state))
        shared = torch.relu(self.shared_fc2(shared))
        
        # 计算均值（mu）
        mu = self.mu_fc(shared)
        
        # 计算标准差（sigma），并确保数值稳定
        sigma = torch.exp(torch.clamp(self.sigma_fc(shared), -2, 2))
        
        # 防止 sigma 太小
        sigma = torch.clamp(sigma, min=1e-3)
        return mu, sigma


class ValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(stateDim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        
        # 使用 He 初始化
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.clamp(x, -1e3, 1e3) # 限制范围 [-1e3, 1e3]，防止梯度爆炸/消失

class Agent_PPO:

    def __init__(self, stateDim, actionDim, gamma, policyLR, valueLR, device):
        self.policyNet = PolicyNet(stateDim, actionDim).to(device)
        self.valueNet = ValueNet(stateDim).to(device)
        self.policyOptimizer = torch.optim.Adam(self.policyNet.parameters(), lr=policyLR)
        self.valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=valueLR)
        self.gamma = gamma
        self.policyLR = policyLR
        self.valueLR = valueLR
        self.device = device

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
        # Critic 更新
        delta = (reward + self.gamma * self.genValue(stateVec).detach() - self.genValue(prevStateVec))
        valueLoss = delta ** 2
        self.valueOptimizer.zero_grad()
        valueLoss.backward(retain_graph=True)  # 单独的计算图
        self.valueOptimizer.step()

        # Actor 更新
        mu, sigma = self.policyNet(prevStateVec)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(actionVec)
        ratio = torch.exp(log_prob - self.genLogProb(actionVec, prevStateVec))

        # PPO 剪辑
        epsilon = 0.2
        advantage = delta.detach()  # 使用 TD 误差作为 Advantage
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        policyLoss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        self.policyOptimizer.zero_grad()
        policyLoss.backward(retain_graph=True)  # 单独的计算图
        self.policyOptimizer.step()
