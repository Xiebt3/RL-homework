import torch
import torch.nn.functional as F

class Memory:
    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.dead_end = []
        self.episode_counter = []
        self.value = []
        self.next_value = []

    def add(self, state, next_state, action, logprob, reward, done, dead_end, episode_counter, value):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.dead_end.append(dead_end)
        self.episode_counter.append(episode_counter)
        self.value.append(value)

    def add_newest_value(self,  newest_value):
        # 找到最新的 episode 编号
        latest_episode = max(self.episode_counter)
        # 提取最新 episode 对应的 value 值
        latest_values = [v for ep, v in zip(self.episode_counter, self.value) if ep == latest_episode]

        # 向前滚动：去掉第一个值，并拼接 last_value
        next_values = latest_values[1:] + [newest_value]

        self.next_value = self.next_value + next_values#列表拼接


    def get(self):
        return {
            "states": self.states,
            "next_states": self.next_states,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "dones": self.dones,
            "dead_end": self.dead_end,
            "episode_counter": self.episode_counter,
            "value": self.value,
            "next_value": self.next_value
        }

    def clear(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.dead_end = []
        self.episode_counter = []
        self.value = []
        self.next_value = []


class MemoryManager:
    """
    MemoryManager负责管理TransformerXL的记忆机制，包括memories、memories_mask和memories_mask_idx的存储和更新。
    """

    def __init__(self, WINDOW_MEM, num_layers, EMBED_SIZE, num_heads):
        self.WINDOW_MEM = WINDOW_MEM
        self.num_layers = num_layers
        self.EMBED_SIZE = EMBED_SIZE
        self.num_heads = num_heads

        # 这两个将用于transformer的输入（无论是采样还是更新）；但只在采样时，才会更新这两个
        self.memories = torch.zeros((WINDOW_MEM, num_layers, EMBED_SIZE))
        self.memories_mask = torch.zeros((num_heads, 1, WINDOW_MEM + 1), dtype=torch.bool)

        self.memories_mask_idx = WINDOW_MEM  # 标量
        self.episode_snapshots = []  # 用于存储每个episode结束时的记忆和掩码快照
        self.episode_counter=0

        mask_idx_ohot = F.one_hot(torch.tensor(self.memories_mask_idx), self.WINDOW_MEM + 1).unsqueeze(0).unsqueeze(1)
        self.memories_mask = torch.logical_or(self.memories_mask, mask_idx_ohot)
        self.memories_mask_idx -= 1

    def update(self, state_embed, done):
        # Save current memory state in transition data if needed
        if done:
            # Save the final memory state of the episode
            self.episode_snapshots.append((self.memories.clone(), self.memories_mask.clone()))
            self.episode_counter += 1
            # Reset memory for the new episode
            self.reset()
        # Update memory with new state_embed
        self.memories = torch.roll(self.memories, -1, dims=0)
        self.memories[-1] = state_embed
        # Update memory mask
        mask_idx_ohot = F.one_hot(torch.tensor(self.memories_mask_idx), self.WINDOW_MEM + 1).unsqueeze(0).unsqueeze(1)
        self.memories_mask = torch.logical_or(self.memories_mask, mask_idx_ohot)
        if not done:
            self.memories_mask_idx -= 1

    def get(self):
        return self.memories, self.memories_mask, self.memories_mask_idx

    def reset(self):
        self.memories.zero_()
        self.memories_mask.zero_()
        self.memories_mask_idx = self.WINDOW_MEM

    def clear_snapshots(self):
        self.episode_snapshots = []
        self.episode_counter = 0


