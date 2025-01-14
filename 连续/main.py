from mujoco_py import MujocoException

import time
from collections import deque

import numpy as np
import torch
import gym  # gym==0.19.0对应atari-py==0.2.6

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.distributions import MultivariateNormal, Categorical
from memory import Memory
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler


import pybullet_envs

#torch.cuda.is_available = lambda: False

class RewardScaling:
    def __init__(self, shape, gamma, clip_state=False):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
        self.clip_state = clip_state

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
        if self.clip_state:
            x = np.clip(x, -5, 5)# 基本没用，因为几乎没有离群数据，即使离群了也会因为数据量足够大而不导致梯度爆炸
        return x

    def reset(self):
        self.R = np.zeros(self.shape)


# 移动均值方差
class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_size=512, discrete=False, activate_func=nn.ReLU()):  #mujoco是512
        super(ActorNetwork, self).__init__()
        self.discrete = discrete  # 用于标记是否是离散空间
        if not self.discrete:
            self.scale = (action_high - action_low) / 2.0
            self.shift = (action_high + action_low) / 2.0

        # 网络层
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            activate_func,
            nn.Linear(hidden_size, hidden_size),
            activate_func,
            nn.Linear(hidden_size, hidden_size),
            activate_func,
            nn.Linear(hidden_size, hidden_size),
            activate_func,
        )

        # 如果是离散空间，使用logits输出
        if self.discrete:
            self.logits_layer = nn.Linear(hidden_size, action_dim)  # 离散动作空间
        else:
            # 如果是连续空间，输出均值和标准差
            #self.log_std = nn.Parameter(torch.ones(1, action_dim))

            self.mu_layer = nn.Sequential(
                nn.Linear(hidden_size, action_dim),
                #activate_func,
            )
            self.sigma_layer = nn.Linear(hidden_size, action_dim)

        for layer in self.actor:
            if isinstance(layer, nn.Linear):  # 只对线性层应用初始化
                orthogonal_init(layer)

        if self.discrete:
            orthogonal_init(self.logits_layer, gain=0.01)
        else:
            for layer in self.mu_layer:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer, gain=0.01)
            #orthogonal_init(self.log_sigma_layer, gain=1)

    def forward(self, state):
        x = self.actor(state)
        if self.discrete:
            logits = self.logits_layer(x)  # 离散空间输出logits
            return logits
        else:
            mu = self.mu_layer(x)  # 连续空间输出均值
            #mu = self.shift + self.scale * mu
            sigma = self.sigma_layer(x)
            sigma = torch.nn.functional.softplus(sigma) + 1e-6
            return mu, sigma


# 定义 Critic 网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=512, activate_func=nn.ReLU()):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            activate_func,
            nn.Linear(hidden_size, hidden_size),
            activate_func,
            nn.Linear(hidden_size, hidden_size),
            activate_func,
            nn.Linear(hidden_size, hidden_size),
            activate_func,
            nn.Linear(hidden_size, 1)
        )
        for layer in self.critic:
            if isinstance(layer, nn.Linear):  # 只对线性层应用初始化
                orthogonal_init(layer)

    def forward(self, state):
        return self.critic(state)


def debug_print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            # Compute gradient statistics
            grad_mean = grad.mean()
            grad_std = grad.std()
            grad_max = grad.max()
            grad_min = grad.min()
            grad_l2_norm = torch.norm(grad, p=2)
            # Check for NaNs in gradients
            has_nans = "Yes" if torch.isnan(grad).any() else "No"
            # Print all gradient information
            print(f'Layer {name}, '
                  f'Gradient mean: {grad_mean}, '
                  f'std: {grad_std}, '
                  f'max: {grad_max}, '
                  f'min: {grad_min}, '
                  f'L2 norm: {grad_l2_norm}, '
                  f'NaNs: {has_nans}')
    print("-------------------------")



class PPO:
    def __init__(self, env, state_dim, action_dim, action_low, action_high, lr=3e-4, gamma=0.99, GAElambda=0.95,
                 eps_clip=0.2, k_epochs=8, device="cuda", discrete=False, window_size=32, batch_size=2048,
                 mini_batch_size=512, use_clip=True, target_kl=0.01):#window_size=32

        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(self.device)
        self.env_name = env.spec.id

        self.actor = ActorNetwork(state_dim, action_dim,action_low=action_low, action_high=action_high, discrete=discrete)
        self.critic = CriticNetwork(state_dim)
        self.learning_rate_actor = lr
        self.learning_rate_critic = lr
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor, eps=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic, eps=1e-5)

        self.gamma = gamma
        self.GAE_lambda = GAElambda  # GAE的平滑因子，GAE是一种用于估计优势函数的方法
        self.eps_clip = eps_clip  #裁剪限制更新幅度
        self.k_epochs = k_epochs  #一批数据，对应k_epochs次网络梯度更新
        self.mse_loss = nn.MSELoss()  # 均方误差损失，用于 critic 更新
        self.state_normalizer = Normalization(shape=state_dim)
        self.max_episode_steps = env.spec.max_episode_steps
        self.reward_scaling = RewardScaling(shape=1, gamma=gamma)

        self.discrete = discrete

        self.window_size = window_size
        self.state_sequence = deque(maxlen=window_size)
        self.state_dim = state_dim

        self.memory= Memory()
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.use_clip=use_clip
        self.target_kl=target_kl

    def select_action(self, state, test_mode=False):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            if self.discrete:
                action_logits = self.actor(state)  # 从 actor 获取动作分布
                dist = Categorical(logits=action_logits)
                if test_mode:
                    action = torch.argmax(action_logits, dim=-1)  # 正是因为我同时实现了用于测试的代码，所以不能包装dist的获取
                else:
                    action = dist.sample()
                action_logprob = dist.log_prob(action)
            else:
                action_mean, std = self.actor(state)  # 从 actor 获取动作分布
                cov_matrix = torch.diag_embed(std ** 2)
                dist = MultivariateNormal(action_mean, cov_matrix)
                if test_mode:
                    action = action_mean
                else:
                    action = dist.sample()  # 从高斯分布中采样动作
                #action=clamp(action, self.action_low, self.action_high)
                action_logprob = dist.log_prob(action)  # 计算动作的 log 概率
        return action.cpu().numpy(), action_logprob.cpu().numpy()#之前unsqueeze现在忘记squeeze了

    def update(self, memory_data):
        states = torch.FloatTensor(np.array(memory_data["states"])).to(self.device)
        next_states = torch.FloatTensor(np.array(memory_data["next_states"])).to(self.device)
        actions = torch.FloatTensor(np.array(memory_data["actions"])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(memory_data["logprobs"])).to(self.device)
        rewards = torch.FloatTensor(np.array(memory_data["rewards"])).to(self.device)
        dones = torch.FloatTensor(np.array(memory_data["dones"])).to(self.device)
        dead_end = torch.FloatTensor(np.array(memory_data["dead_end"])).to(self.device)

        # GAE（Generalized Advantage Estimation，广义优势估计）
        v_targets, advantages = self.compute_vtarget_advantages(states, next_states, rewards, dones, dead_end)

        # 标准化优势函数，提升算法稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        exact_batch_size = len(states)

        if not self.use_clip:
            self.kl_coef = 0.2  # 可调整初始值

        # PPO更新，但是裁剪更新幅度
        for _ in range(self.k_epochs):
            #sampler = SubsetRandomSampler(range(exact_batch_size))
            #batch_sampler = BatchSampler(sampler, self.mini_batch_size, drop_last=False)
            #batch_sampler = BatchSampler(sampler, exact_batch_size, drop_last=False)
            sampler = SequentialSampler(range(exact_batch_size))  # 使用 SequentialSampler 代替 SubsetRandomSampler
            batch_sampler = BatchSampler(sampler, exact_batch_size, drop_last=False)
            for indices in batch_sampler:
                # Select mini-batch data
                exact_mini_batch_size = len(indices)
                #scale_factor = exact_mini_batch_size / self.mini_batch_size
                states_batch = states[indices]
                actions_batch = actions[indices]
                old_logprobs_batch = old_logprobs[indices]
                advantages_batch = advantages[indices]
                v_targets_batch = v_targets[indices]

                # Compute current policy distribution and new log probs
                dist, new_logprobs = self.get_logprobs(states_batch, actions_batch)
                entropy = dist.entropy().mean()

                # Compute state values
                state_values = self.critic(states_batch).squeeze(-1)

                # Compute ratios
                ratios = torch.exp(new_logprobs - old_logprobs_batch)

                critic_loss = self.mse_loss(state_values, v_targets_batch)

                surr1 = ratios * advantages_batch  # 无裁剪

                if not self.use_clip:
                    # 直接用old_logprobs_batch来算KL散度，而非使用旧策略的mean动作（最大概率动作）
                    # 尽管我们默认高斯分布，高斯分布的KL散度直接用旧策略的mean动作可以得到闭式解，但是注意：本项目还一并实现了离散版本的PPO
                    # 离散版本的PPO不是高斯分布动作采样，这时无法通过求闭式解得到KL散度而只能采样
                    # 所以，为了代码实现的简便和统一，直接用采样得到的old_logprobs_batch吧，相信大数定律，只要采样够多就是无偏的
                    kl_div = (old_logprobs_batch - new_logprobs).mean()
                    loss = -surr1.mean() + self.kl_coef * kl_div + 0.5 * critic_loss - 0.01 * entropy
                    # 根据KL散度调整Lagrangian乘数
                    if kl_div > self.target_kl:
                        self.kl_coef *= 1.5  # 如果KL超过阈值，增加惩罚
                    elif kl_div < self.target_kl * 0.5:
                        self.kl_coef *= 0.5  # 如果KL低于阈值的一半，减少惩罚
                    # 就目前情况而言，clip比kl散度好，kl散度训练双节倒立杆居然要三十万步，是clip的五倍
                else:
                    # 替代损失（surrogate loss），其中的surr2即为PPO算法的核心，也即clip裁剪更新幅
                    # 两个替代损失中更小的那个，更小的更新幅度，牺牲训练迭代次数换取稳定性
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch  # 裁剪
                    loss = -torch.min(surr1, surr2).mean() + 0.5 * critic_loss - 0.01 * entropy

                #loss*=scale_factor
                #critic_loss*=scale_factor

                self.optimizer_actor.zero_grad()
                loss.backward(retain_graph=True)  # 只需要一次反向传播，retain_graph=True TODO: 其实这里可能要改
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_critic.step()

        #print("kl_coef"+str(self.kl_coef))




    def compute_vtarget_advantages(self, states, next_states, rewards, dones, dead_end):
        # 初始化GAE计算
        v_targets = []
        advantages = []
        G = 0  # 初始化累积奖励
        adv = 0  # 初始化优势
        #  全部反转避免重复计算相同的因子（因为如果不反转，下一条乘式就是上一条乘式的前缀）
        with torch.no_grad():
            for reward, done, dead_end, value, next_value in zip(rewards.flip(0), dones.flip(0), dead_end.flip(0),
                                                                 self.critic(states).flip(0),
                                                                 self.critic(next_states).flip(0)):
                # 计算TD误差
                delta = reward + self.gamma * next_value * (
                        1 - dead_end) - value

                # GAE优势计算
                adv = delta + self.gamma * self.GAE_lambda * (1 - done) * adv
                advantages.insert(0, adv)

                G = value + adv
                v_targets.insert(0, G)

        # 返回计算出的回报和优势
        return torch.FloatTensor(v_targets).to(self.device), torch.FloatTensor(advantages).to(self.device)

    def get_logprobs(self, states, actions):
        if self.discrete:
            # 离散空间，使用Categorical分布
            logits = self.actor(states)  # 获取logits
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(actions)  # 计算log概率
            return dist, logprobs
        else:
            # 连续空间，使用MultivariateNormal分布
            action_means, std = self.actor(states)  # 获取均值和标准差
            cov_matrix = torch.diag_embed(std ** 2)
            dist = MultivariateNormal(action_means, cov_matrix)
            # dist = torch.distributions.Normal(action_means, std)
            logprobs = dist.log_prob(actions)  # 计算log概率
            dist = torch.distributions.Normal(action_means, std)
            return dist, logprobs

        #actions不是之前的平均，这里的actions是一组样本（n个样本），每个样本的动作都是一维向量，
        #对于每个样本，求它在这个高斯分布被取到的联合概率，给这个概率再取对数，一个样本（一维向量）对应一个概率（标量）
        #由于有n个样本，所以最后的输出是有n个元素的一维向量

    def learning_rate_decay(self, total_steps, max_timesteps):
        learning_rate_actor_now = self.learning_rate_actor * (1 - total_steps / max_timesteps)
        learning_rate_critic_now = self.learning_rate_critic * (1 - total_steps / max_timesteps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = learning_rate_actor_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = learning_rate_critic_now

    def train(self, max_timesteps, smooth_window=100):
        self.actor.train()
        self.critic.train()
        state = self.env.reset()
        if self.discrete:
            state = state.astype(np.int16)
        # 无符号难以状态标准化，有时小数一相减就成大数了
        timestep = 0
        episode_rewards = []
        timesteps_list = []
        start_time = time.time()  # 记录开始时间
        last_update_timestep = 0  # 记录上次更新的时间戳
        debug_episode_reward = 0
        debug_episode_counter = 0
        state = self.state_normalizer(state)
        self.reward_scaling.reset()
        episode_steps = 0
        self.actor.to("cpu")  # 第一次采样也在CPU

        while timestep < max_timesteps:
            action, action_logprob = self.select_action(state, test_mode=False)
            try:
                next_state, reward, done, _ = self.env.step(action)
            except MujocoException as e:
                print(f"Error occurred at timestep {timestep}")
                raise
            if self.discrete:
                next_state = next_state.astype(np.int16)
            next_state = self.state_normalizer(next_state)

            debug_episode_reward += reward  #不计入我自己新加的reward
            timestep += 1
            episode_steps += 1

            if done and episode_steps != self.max_episode_steps:
                dead_end = True
            else:
                dead_end = False

            reward = self.reward_scaling(reward)

            self.memory.add(state, next_state, action, action_logprob, reward, done, dead_end)

            state = next_state

            if done and (timestep - last_update_timestep) >= self.batch_size:
                print(
                    f"Updating policy at timestep {timestep}/{max_timesteps}, last episode rewards: {debug_episode_reward:.2f}, episode dones: {debug_episode_counter}")
                self.actor.to(self.device)  #网络训练前加载到GPU
                self.critic.to(self.device)
                memory_data = self.memory.get()
                self.update(memory_data)
                self.actor.to("cpu")  # CPU采样；不用管self.critic，它只有在update才有用
                last_update_timestep = timestep  # 更新最后一次更新的时间戳
                self.memory.clear()
                self.learning_rate_decay(timestep, max_timesteps)

            if timestep % (max_timesteps//2) == 0:
                elapsed_time = time.time() - start_time
                print(f"Progress: {timestep}/{max_timesteps}, Time elapsed: {elapsed_time:.2f}s")
                self.save_model(timestep)

            if done:
                state = self.env.reset()
                if self.discrete:
                    state = state.astype(np.int16)
                self.reward_scaling.reset()
                state = self.state_normalizer(state)
                episode_rewards.append(debug_episode_reward)
                debug_episode_reward = 0
                debug_episode_counter += 1
                episode_steps = 0
                timesteps_list.append(timestep)

        # 平滑 reward（使用滑动平均）
        smoothed_rewards = self.smooth_rewards(episode_rewards, smooth_window)

        # 绘制折线图
        #self.plot_rewards(episode_rewards, smoothed_rewards, )
        self.plot_rewards(timesteps_list, episode_rewards, smoothed_rewards, smooth_window)
        print("Training complete!")
        self.save_model(timestep)  # 训练完成后保存模型

    def test(self, num_episodes=10, render=False):
        self.actor.eval()
        self.critic.eval()
        total_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            if self.discrete:
                state = state.astype(np.int16)
            state = self.state_normalizer(state, update=False)
            done = False
            episode_reward = 0
            while not done:
                if render:
                    self.env.render(mode='human')
                action, _ = self.select_action(state, test_mode=True)
                next_state, reward, done, _ = self.env.step(action)
                if self.discrete:
                    next_state = next_state.astype(np.int16)
                next_state = self.state_normalizer(next_state, update=False)
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Rewards: {episode_reward}")

        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    # 保存整个agent（模型+状态归一化+优化器状态等）
    def save_model(self, timestep):
        model_filename = f"{self.env_name}_ppo_model_{timestep}.pth"
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'normalizer_mean': self.state_normalizer.running_ms.mean,
            'normalizer_std': self.state_normalizer.running_ms.std,
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }, model_filename)
        print(f"Model saved at timestep {timestep} as {model_filename}")

    def load_model(self, model_filename):
        checkpoint = torch.load(model_filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.state_normalizer.running_ms.mean = checkpoint['normalizer_mean']
        self.state_normalizer.running_ms.std = checkpoint['normalizer_std']
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        print(f"Model loaded from {model_filename}")

    def smooth_rewards(self, rewards, window_size):
        return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    def plot_rewards(self, timesteps_list, raw_rewards, smoothed_rewards, window_size):
        start_index = window_size - 1
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps_list, raw_rewards, label="Episode Rewards", alpha=0.3)
        plt.plot(timesteps_list[start_index:], smoothed_rewards, label="Smoothed Rewards", alpha=1)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title(self.env_name+' Reward Curve')
        plt.show()


def set_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU 上的随机种子
    torch.cuda.manual_seed(seed)  # CUDA 随机种子
    torch.cuda.manual_seed_all(seed)  # 所有设备的CUDA种子
    env.seed(seed)


def load_train(env_name, train_steps=1000000, discrete=False, test_mark=False, window_size=32, use_clip=True):
    env = gym.make(env_name)
    set_seed(21307352, env)
    if not discrete:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        action_low = None
        action_high = None
    #ppo = PPO(env, state_dim, action_dim, action_low, action_high, device="cpu", discrete=discrete)
    ppo = PPO(env, state_dim, action_dim, action_low, action_high, device="cuda", discrete=discrete, window_size=window_size, use_clip=use_clip)
    #隐藏层64不如cpu；隐藏层512时CPU==CPU采样+CUDA训练；隐藏层1024时CPU采样+CUDA训练性能优势明显
    if test_mark:
        ppo.load_model(env.spec.id + "_ppo_model_"+str(train_steps)+".pth")
    else:
        ppo.train(max_timesteps=train_steps)
    return ppo

def pybullet_envs_load_train(env_name, train_steps=1000000, discrete=False, test_mark=False, window_size=32, use_clip=True):
    env = pybullet_envs.make(env_name)
    set_seed(21307352, env)
    if not discrete:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        action_low = None
        action_high = None
    ppo = PPO(env, state_dim, action_dim, action_low, action_high, device="cuda", discrete=discrete, window_size=window_size, use_clip=use_clip)
    if test_mark:
        ppo.load_model(env.spec.id + "_ppo_model_"+str(train_steps)+".pth")
    else:
        ppo.train(max_timesteps=train_steps)
    return ppo

if __name__ == "__main__":
    #choose_envs=["HalfCheetah","Hopper", "InvertedDoublePendulum", "InvertedPendulum","Reacher","Swimmer","Walker"]
    #choose_envs=["InvertedDoublePendulum"]
    choose_envs=["Humanoid",]
    #choose_envs=["HumanoidFlagrunBulletEnv", "HumanoidFlagrunHarderBulletEnv"]
    #test_mark = True
    test_mark = False
    for choose_env in choose_envs:
        if choose_env == "InvertedDoublePendulum":
            ppo = load_train("InvertedDoublePendulum-v2", train_steps=1000000, test_mark=test_mark, window_size=4, use_clip=True)
        elif choose_env == "Humanoid":
            ppo = load_train("Humanoid-v2", train_steps=7000000, test_mark=test_mark)
        elif choose_env == "InvertedPendulum":
            ppo = load_train('InvertedPendulum-v2', train_steps=1000000, test_mark=test_mark)
        elif choose_env == "Walker":
            ppo = load_train("Walker2d-v2", test_mark=test_mark, train_steps=1000000)
        elif choose_env == "HalfCheetah":
            ppo = load_train("HalfCheetah-v2", test_mark=test_mark)
        elif choose_env == "Hopper":  #可能这个环境改用tanh会更好
            ppo = load_train("Hopper-v2", test_mark=test_mark)
        elif choose_env == "Swimmer":
            ppo = load_train("Swimmer-v2", test_mark=test_mark)
        elif choose_env=="Reacher":
            ppo = load_train("Reacher-v2", test_mark=test_mark)
        elif choose_env == "Alien":
            ppo = load_train("Alien-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
            # 不进行状态标准化的五百万步Average reward over 100 episodes: 1105.2
            # 状态标准化的五百万步Average reward over 100 episodes: 1884.0
        elif choose_env == "Zaxxon":
            ppo = load_train("Zaxxon-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
        elif choose_env=="Jamesbond":
            ppo = load_train("Jamesbond-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
        elif choose_env=="Qbert":
            ppo = load_train("Qbert-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
        elif choose_env=="Pong":
            ppo = load_train("Pong-ram-v4", discrete=True, test_mark=test_mark, train_steps=4000000)
        elif choose_env=="BankHeist":
            ppo = load_train("BankHeist-ram-v4", discrete=True, test_mark=test_mark, train_steps=4000000)
        elif choose_env=="Boxing":
            ppo = load_train("Boxing-ram-v4", discrete=True, test_mark=test_mark, train_steps=4000000)
        elif choose_env=="HumanoidFlagrunBulletEnv":
            ppo = pybullet_envs_load_train("HumanoidFlagrunBulletEnv-v0", train_steps=10000000, test_mark=test_mark)
        elif choose_env=="HumanoidFlagrunHarderBulletEnv":
            ppo = pybullet_envs_load_train("HumanoidFlagrunHarderBulletEnv-v0", train_steps=10000000, test_mark=test_mark)
        else:
            raise NotImplementedError

        if test_mark == True:
            ppo.test(num_episodes=100, render=False)
# openai，怎么atari和mujoco还用不同的两套超参数