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

    def add(self, state, next_state, action, logprob, reward, done, dead_end):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.dead_end.append(dead_end)


    def get(self):
        return {
            "states": self.states,
            "next_states": self.next_states,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "dones": self.dones,
            "dead_end": self.dead_end,
        }

    def clear(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.dead_end = []
