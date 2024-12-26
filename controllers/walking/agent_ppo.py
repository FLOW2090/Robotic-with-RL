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
        
        # 使用 He 初始化
        torch.nn.init.xavier_normal_(self.shared_fc1.weight)
        torch.nn.init.xavier_normal_(self.shared_fc2.weight)
        torch.nn.init.xavier_normal_(self.shared_fc3.weight)
        torch.nn.init.xavier_normal_(self.mu_fc.weight)
        

    def forward(self, state):
        # 提取共享特征
        shared = torch.relu(self.shared_fc1(state))
        shared = torch.relu(self.shared_fc2(shared))
        shared = torch.relu(self.shared_fc3(shared))
        
        mu = self.mu_fc(shared)
        sigma = torch.exp(self.sigma_fc(shared))
        
        return mu, sigma


class ValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(stateDim, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, 1)
        
        # 使用 He 初始化
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
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
        self.valueLossList = []
        self.policyLossList = []
        self.freeze_actor = True  # 初始阶段冻结Actor网络

    def genActionVec(self, stateVec):
        assert(not torch.isnan(stateVec).any())
        mu, sigma = self.policyNet(stateVec)
        # print(f"mu: {mu}, sigma: {sigma}")
        actionVec = torch.normal(mu, sigma).to(self.device)
        # 打印生成的动作向量
        # print(f"Generated action vector: {actionVec}")
        return actionVec

    def genValue(self, stateVec):
        return self.valueNet(stateVec)

    def genLogProb(self, actionVec, stateVec):
        mu, sigma = self.policyNet(stateVec)
        
        return torch.distributions.Normal(mu, sigma).log_prob(actionVec)
    
    def update(self, reward, prevStateVec, stateVec, actionVec, step, isTerminal=False):
        # Critic 更新
        # delta = (reward + self.gamma * self.genValue(stateVec).detach() - self.genValue(prevStateVec))
        # valueLoss = delta ** 2
        with torch.no_grad():
            delta = (reward + self.gamma * self.genValue(stateVec) - self.genValue(prevStateVec))
        valueLoss = -delta * self.genValue(prevStateVec)
        
        self.valueLossList.append(valueLoss.item())
        self.valueOptimizer.zero_grad()
        valueLoss.backward(retain_graph=True)
        self.valueOptimizer.step()

        # 如果冻结Actor网络，则不更新Actor网络
        if self.freeze_actor:
            return
        
        # Actor 更新
        mu, sigma = self.policyNet(prevStateVec)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(actionVec)
        ratio = torch.exp(log_prob - self.genLogProb(actionVec, prevStateVec))
        # new_log_prob = self.genLogProb(actionVec, stateVec)
        # old_log_prob = self.genLogProb(actionVec, prevStateVec)
        # ratio = torch.exp(new_log_prob - old_log_prob)

        # PPO 剪辑
        epsilon = 0.2
        advantage = delta.detach()  # 使用 TD 误差作为 Advantage
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        policyLoss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        
        self.policyLossList.append(policyLoss.item())
        self.policyOptimizer.zero_grad()
        policyLoss.backward(retain_graph=True)
        self.policyOptimizer.step()
        
        # print(f"Step {step} reward: {reward} Value loss: {valueLoss.item()}, Policy loss: {policyLoss.item()}")
