import math
import time
from collections import deque

import numpy as np
import torch
import gym  # gym==0.19.0对应atari-py==0.2.6
import copy

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.distributions import MultivariateNormal, Categorical
#import roboschool

#torch.cuda.is_available = lambda: False

class SlidingWindowDataGenerator:
    def __init__(self, window_size=128, step_size=1, discrete=False):
        self.window_size = window_size  # 滑动窗口大小
        self.step_size = step_size  # 每次滑动步长
        self.discrete=discrete

    def generate_training_samples(self, memory):
        states = np.array(memory["states"])
        done_flags = np.array(memory["dones"])

        training_samples = []
        masks = []

        start_idx = 0
        while start_idx < len(states):
            end_idx = start_idx
            while end_idx < len(states) and not done_flags[end_idx]:
                end_idx += 1
            end_idx += 1

            episode_states = states[start_idx:end_idx]
            episode_length = len(episode_states)
            pad_length = self.window_size - 1
            pad_states = np.full((pad_length,) + episode_states.shape[1:], fill_value=0)
            episode_states = np.concatenate((pad_states, episode_states), axis=0)
            episode_length += pad_length

            # 生成掩码，pad部分为True，实际数据为False
            episode_mask = np.ones((pad_length,), dtype=bool)
            episode_mask = np.concatenate((episode_mask, np.zeros((len(episode_states) - pad_length,), dtype=bool)), axis=0)

            for start in range(0, episode_length - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window_states = episode_states[start:end]
                window_mask = episode_mask[start:end]
                window_states_tensor = torch.FloatTensor(window_states)
                window_mask_tensor = torch.BoolTensor(window_mask)
                training_samples.append(window_states_tensor)
                masks.append(window_mask_tensor)

            start_idx = end_idx

        return torch.stack(training_samples, dim=0), torch.stack(masks, dim=0)

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
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


def calculate_embed_dim(state_dim, num_heads):
    if state_dim % num_heads == 0:
        return state_dim
    else:
        return ((state_dim // num_heads) + 1) * num_heads

def pos_encoding(seq_len, embed_dim, device):
    encoding = torch.zeros(seq_len, embed_dim, device=device)
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-math.log(10000.0) / embed_dim))
    encoding[:, 0::2] = torch.sin(pos * div_term)
    encoding[:, 1::2] = torch.cos(pos * div_term)
    return encoding


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=512, num_layers=2, num_heads=4, discrete=False):
        super(ActorNetwork, self).__init__()
        self.discrete = discrete
        self.embed_dim = calculate_embed_dim(state_dim, num_heads)

        # 状态嵌入层
        self.state_embedding = nn.Linear(state_dim, self.embed_dim)

        # Transformer编码器层
        self.transformer_encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads,
                                                                     dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(self.transformer_encoder_layer1, num_layers=num_layers)

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        orthogonal_init(self.mlp[0], gain=1.0)
        orthogonal_init(self.mlp[2], gain=1.0)

        # 输出层
        if self.discrete:
            self.logits_layer = nn.Linear(hidden_size, action_dim)
            orthogonal_init(self.logits_layer, gain=1.0)
        else:
            self.mu_layer = nn.Linear(hidden_size, action_dim)
            self.sigma_layer = nn.Linear(hidden_size, action_dim)
            # TODO: humnanoidgym对标准差的设计也是用parameter，可能确实要改
            #  self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            orthogonal_init(self.mu_layer, gain=0.01)
            orthogonal_init(self.sigma_layer, gain=0.01)

    def forward(self, state_seq, mask_seq):
        batch_size, seq_len, _ = state_seq.size()
        embedded_state_seq = self.state_embedding(state_seq)
        pos_embed = pos_encoding(seq_len, self.embed_dim, embedded_state_seq.device)
        embedded_state_seq = embedded_state_seq + pos_embed

        # 使用传入的掩码
        mask = mask_seq

        # 将掩码传递给Transformer编码器
        encoded_seq = self.transformer_encoder1(embedded_state_seq, src_key_padding_mask=mask)

        # 掩码均值池化
        mask_expanded = mask.unsqueeze(-1).expand(encoded_seq.size())
        encoded_seq_masked = encoded_seq.masked_fill(mask_expanded, 0.0)
        pooled = encoded_seq_masked.sum(dim=1) / ((~mask).sum(dim=1).unsqueeze(-1) + 1e-6)

        mlp_output = self.mlp(pooled)

        if self.discrete:
            logits = self.logits_layer(mlp_output)
            return logits
        else:
            mu = self.mu_layer(mlp_output)
            sigma = torch.nn.functional.softplus(self.sigma_layer(mlp_output)) + 1e-6
            return mu, sigma

# 定义 Critic 网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=512):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        for layer in self.critic:
            if isinstance(layer, nn.Linear):  # 只对线性层应用初始化
                orthogonal_init(layer)

    def forward(self, state):
        return self.critic(state)


import torch
import torch.nn as nn

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
                 eps_clip=0.2, k_epochs=8, device="cuda", discrete=False, window_size=32):#window_size=32

        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(self.device)
        self.env_name = env.spec.id

        self.actor = ActorNetwork(state_dim, action_dim, discrete=discrete)
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

        self.action_low = action_low
        self.action_high = action_high
        self.discrete = discrete

        self.window_size = window_size
        self.state_sequence = deque(maxlen=window_size)
        self.state_dim = state_dim
        self.data_generator = SlidingWindowDataGenerator(window_size=window_size,discrete=discrete)

    def select_action(self, state, test_mode=False):
        state_tensor = torch.FloatTensor(state)
        self.state_sequence.append(state_tensor)
        if len(self.state_sequence) < self.window_size:
            padding = torch.zeros(self.window_size - len(self.state_sequence), self.state_dim)
            seq = torch.cat((padding, torch.stack(list(self.state_sequence), dim=0)), dim=0)
            mask = torch.cat((torch.ones(self.window_size - len(self.state_sequence), dtype=torch.bool),
                              torch.zeros(len(self.state_sequence), dtype=torch.bool)), dim=0)
        else:
            seq = torch.stack(list(self.state_sequence)[-self.window_size:], dim=0)
            mask = torch.zeros(self.window_size, dtype=torch.bool)
        seq_tensor = seq.unsqueeze(0)  # 形状变为 (1, window_size, state_dim)
        mask_tensor = mask.unsqueeze(0)

        if self.discrete:
            with torch.no_grad():
                action_logits = self.actor(seq_tensor, mask_tensor)
            dist = Categorical(logits=action_logits)
            if test_mode:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            with torch.no_grad():
                action_mean, sigma = self.actor(seq_tensor, mask_tensor)
            cov_matrix = torch.diag_embed(sigma ** 2)
            dist = MultivariateNormal(action_mean, cov_matrix)
            if test_mode:
                action = action_mean
            else:
                action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.cpu().numpy(), action_logprob.cpu().numpy()

    def update(self, memory):
        states_seq, masks = self.data_generator.generate_training_samples(memory) # 此处处理得到的states_seq是包含本时刻状态的（本时刻状态为最新状态）状态序列
        states_seq = states_seq.to(self.device)
        masks = masks.to(self.device)
        # 这里会有一次actor前向计算，在GPU运行
        states = torch.FloatTensor(np.array(memory["states"])).to(self.device)
        next_states = torch.FloatTensor(np.array(memory["next_states"])).to(self.device)
        actions = torch.FloatTensor(np.array(memory["actions"])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(memory["logprobs"])).to(self.device)
        rewards = torch.FloatTensor(np.array(memory["rewards"])).to(self.device)
        dones = torch.FloatTensor(np.array(memory["dones"])).to(self.device)
        dead_end = torch.FloatTensor(np.array(memory["dead_end"])).to(self.device)

        # GAE（Generalized Advantage Estimation，广义优势估计）
        v_targets, advantages = self.compute_vtarget_advantages(states, next_states, rewards, dones, dead_end)

        # 标准化优势函数，提升算法稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新，但是裁剪更新幅度
        for counter in range(self.k_epochs):
            new_logprobs = self.get_logprobs(states_seq, actions, masks)  # Q(s,a)的对数概率分布
            # 这里会有一次actor前向计算，在GPU运行
            if self.discrete:
                dist = Categorical(logits=self.actor(states_seq, masks))
            else:
                mu, sigma = self.actor(states_seq, masks)
                # 这里会有一次actor前向计算；在此处windows_size==128时若使用CPU内存会超过16G；在GPU运行
                dist = torch.distributions.Normal(mu, sigma)
            entropy = dist.entropy().mean()  # 计算熵，并取均值
            state_values = self.critic(states).squeeze()  # critic估计的Q值

            # 新旧概率分布比值，用于实现重要性采样，也即重要性采样比率
            ratios = torch.exp(new_logprobs - old_logprobs)

            # 替代损失（surrogate loss），其中的surr2即为PPO算法的核心，也即clip裁剪更新幅度
            surr1 = ratios * advantages  # 无裁剪
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 裁剪
            critic_loss = self.mse_loss(state_values, v_targets)
            # 两个替代损失中更小的那个，更小的更新幅度，牺牲训练迭代次数换取稳定性
            loss = -torch.min(surr1, surr2).mean() + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer_actor.zero_grad()
            loss.backward(retain_graph=True)  # 只需要一次反向传播，retain_graph=True
            if counter%self.k_epochs==0:
                debug_print_gradients(self.actor)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_critic.step()


    def compute_vtarget_advantages(self, states, next_states, rewards, dones, dead_end):
        # 初始化GAE计算
        v_targets = []
        advantages = []
        G = 0  # 初始化累积奖励
        adv = 0  # 初始化优势
        #  全部反转避免重复计算相同的因子（因为如果不反转，下一条乘式就是上一条乘式的前缀）
        for reward, done, dead_end, value, next_value in zip(rewards.flip(0), dones.flip(0), dead_end.flip(0),
                                                             self.critic(states).flip(0),
                                                             self.critic(next_states).flip(0)):
            # 计算TD误差
            delta = reward + self.gamma * next_value * (
                    1 - dead_end) - value  #delta = reward + self.gamma * next_value * (1 - done) - value

            # GAE优势计算
            adv = delta + self.gamma * self.GAE_lambda * (1 - done) * adv
            advantages.insert(0, adv)

            G = value + adv
            v_targets.insert(0, G)

        # 返回计算出的回报和优势
        return torch.FloatTensor(v_targets).to(self.device), torch.FloatTensor(advantages).to(self.device)

    def get_logprobs(self, states_seq, actions, masks):
        if self.discrete:
            # 离散空间，使用Categorical分布
            logits = self.actor(states_seq, masks)  # 获取logits
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(actions)  # 计算log概率
            return logprobs
        else:
            # 连续空间，使用MultivariateNormal分布
            mu, sigma = self.actor(states_seq, masks)  # 获取均值和标准差
            cov_matrix = torch.diag_embed(sigma ** 2)
            dist = MultivariateNormal(mu, cov_matrix)
            logprobs = dist.log_prob(actions)  # 计算log概率
            return logprobs

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
        memory = {"states": [], "next_states": [], "actions": [], "logprobs": [], "rewards": [], "dones": [],
                  "dead_end": []}
        timestep = 0
        episode_rewards = []
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
            next_state, reward, done, _ = self.env.step(action)
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

            # Store transition in memory
            memory["states"].append(state)
            memory["next_states"].append(next_state)
            memory["actions"].append(action)
            memory["logprobs"].append(action_logprob)
            memory["rewards"].append(reward)
            memory["dones"].append(done)
            memory["dead_end"].append(dead_end)

            state = next_state

            if done and (timestep - last_update_timestep) >= 2048:
                print(
                    f"Updating policy at timestep {timestep}/{max_timesteps}, last episode rewards: {debug_episode_reward:.2f}, episode dones: {debug_episode_counter}")
                self.actor.to(self.device)  #网络训练前加载到GPU
                self.critic.to(self.device)
                self.update(memory)
                self.actor.to("cpu")  # CPU采样；不用管self.critic，它只有在update才有用
                last_update_timestep = timestep  # 更新最后一次更新的时间戳
                memory = {"states": [], "next_states": [], "actions": [], "logprobs": [], "rewards": [], "dones": [],
                          "dead_end": []}
                self.learning_rate_decay(timestep, max_timesteps)

            if timestep % 10000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Progress: {timestep}/{max_timesteps}, Time elapsed: {elapsed_time:.2f}s")
                self.save_model(timestep)  # 定期保存模型

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

        # 平滑 reward（使用滑动平均）
        smoothed_rewards = self.smooth_rewards(episode_rewards, smooth_window)

        # 绘制折线图
        self.plot_rewards(episode_rewards, smoothed_rewards)
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
        """平滑 reward，通过滑动平均"""
        return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    def plot_rewards(self, raw_rewards, smoothed_rewards):
        """绘制 reward 曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(raw_rewards, label="Raw Rewards", alpha=0.7)
        plt.plot(np.arange(len(smoothed_rewards)) + len(raw_rewards) - len(smoothed_rewards), smoothed_rewards,
                 label="Smoothed Rewards", alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Reward Curve')
        plt.show()


def quaternion_to_rotation_matrix(w, x, y, z):
    """
    将四元数转换为旋转矩阵
    """
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R


def reward_for_stability(state):
    """
    根据四元数的z轴分量来计算奖励
    """
    w = state[1]
    x = state[2]
    y = state[3]
    z = state[4]

    # 获取四元数的旋转矩阵
    R = quaternion_to_rotation_matrix(w, x, y, z)

    # 我们关注机器人躯干z轴的分量，理想情况下它应接近[0, 0, 1]
    stability_reward = abs(R[2, 2])  # R[2,2]是z轴分量，理想值为1

    # 根据稳定度调整奖励
    if stability_reward > 0.9:
        reward = 1  # 高奖励，表示机器人接近竖直
    else:
        reward = -1  # 低奖励，表示机器人有倾斜

    return reward


def set_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU 上的随机种子
    torch.cuda.manual_seed(seed)  # CUDA 随机种子
    torch.cuda.manual_seed_all(seed)  # 所有设备的CUDA种子
    env.seed(seed)


def load_train(env_name, train_steps=1000000, discrete=False, test_mark=False, window_size=32):
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
    ppo = PPO(env, state_dim, action_dim, action_low, action_high, device="cuda", discrete=discrete, window_size=window_size)
    #隐藏层64太逆天了不如cpu；隐藏层512时CPU==CPU采样+CUDA训练；隐藏层1024时CPU采样+CUDA训练性能优势明显
    if test_mark:
        ppo.load_model(env.spec.id + "_ppo_model_"+str(train_steps)+".pth")
    else:
        ppo.train(max_timesteps=train_steps)
    return ppo


if __name__ == "__main__":
    choose_env = "InvertedDoublePendulum"
    test_mark = True
    test_mark = False

    if choose_env == "InvertedDoublePendulum":
        ppo = load_train("InvertedDoublePendulum-v2", train_steps=100000, test_mark=test_mark, window_size=4)
    elif choose_env == "Humanoid":
        ppo = load_train("Humanoid-v2", train_steps=5000000, test_mark=test_mark)
    elif choose_env == "InvertedPendulum":
        ppo = load_train('InvertedPendulum-v2', train_steps=50000, test_mark=test_mark)
    elif choose_env == "Walker":
        ppo = load_train("Walker2d-v2", test_mark=test_mark)
    elif choose_env == "HalfCheetah":
        ppo = load_train("HalfCheetah-v2", test_mark=test_mark)
    elif choose_env == "Hopper":  #可能这个环境改用tanh会更好
        ppo = load_train("Hopper-v2", test_mark=test_mark)
    elif choose_env == "Swimmer":
        ppo = load_train("Swimmer-v2", test_mark=test_mark)
    elif choose_env == "Alien":
        ppo = load_train("Alien-ram-v0", discrete=True, test_mark=test_mark, train_steps=5000000)
        # 不进行状态标准化的五百万步Average reward over 100 episodes: 1105.2
        # 状态标准化的五百万步Average reward over 100 episodes: 1884.0
    elif choose_env == "Zaxxon":
        ppo = load_train("Zaxxon-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
    elif choose_env=="Jamesbond":
        ppo = load_train("Jamesbond-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
    elif choose_env=="Qbert":
        ppo = load_train("Qbert-ram-v0", discrete=True, test_mark=test_mark, train_steps=4000000)
    elif choose_env=="Pong":
        ppo = load_train("Pong-ram-v4", discrete=True, test_mark=test_mark, train_steps=4000000)#由于pong长期表现不佳，所以改为512网络
    elif choose_env=="BankHeist":
        ppo = load_train("BankHeist-ram-v4", discrete=True, test_mark=test_mark, train_steps=4000000)
    elif choose_env=="Boxing":
        ppo = load_train("Boxing-ram-v4", discrete=True, test_mark=test_mark, train_steps=4000000)
    else:
        raise NotImplementedError

    if test_mark == True:
        ppo.test(num_episodes=100, render=True)
# openai，怎么atari和mujoco还用不同的两套超参数