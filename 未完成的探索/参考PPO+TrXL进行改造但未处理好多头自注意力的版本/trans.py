from transformers import TransfoXLModel
import torch.nn as nn
import torch

class Gating(nn.Module):
    def __init__(self, d_model, bias):
        super(Gating, self).__init__()
        self.fc_x = nn.Linear(d_model, d_model)
        self.fc_y = nn.Linear(d_model, d_model)
        self.fc_h = nn.Linear(d_model, d_model)
        self.bias = nn.Parameter(torch.full((d_model,), bias))

    def forward(self, x, y):
        combined = self.fc_x(x) + self.fc_y(y)
        r = torch.sigmoid(combined)
        z = torch.sigmoid(combined - self.bias)
        h = torch.tanh(self.fc_h(r * x + y))
        return (1 - z) * x + z * h


# 自定义 Transformer-XL 的 Layer，加入门控
class GatedTransfoXLLayer(nn.Module):
    def __init__(self, config):
        super(GatedTransfoXLLayer, self).__init__()
        # 原始的 Attention 和 Feed-Forward 层
        self.attention = nn.MultiheadAttention(config.d_model, config.n_head)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        # 加入门控机制
        self.gate1 = Gating(config.d_model, bias=0.0)  # 在 Attention 后应用
        self.gate2 = Gating(config.d_model, bias=0.0)  # 在 Feed-Forward 后应用

    def forward(self, hidden, attn_mask=None, mems=None):
        # Attention 层
        attention_output, _ = self.attention(hidden, hidden, hidden, attn_mask=attn_mask)
        # 在 Attention 后应用门控
        gated_attention_output = self.gate1(hidden, attention_output)
        # Feed-Forward 层
        ff_output = self.ff(gated_attention_output)
        # 在 Feed-Forward 后应用门控
        gated_ff_output = self.gate2(gated_attention_output, ff_output)
        return gated_ff_output

# 自定义 Transformer-XL 模型，替换所有 Layer 为带门控的 Layer
class GatedTransformerXL(TransfoXLModel):
    def __init__(self, config):
        super(GatedTransformerXL, self).__init__(config)
        # 替换所有 Layer 为带门控的 Layer
        self.layers = nn.ModuleList([GatedTransfoXLLayer(config) for _ in range(config.n_layer)])

    def forward(self, input_ids, attention_mask=None, mems=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # 调用父类的 forward 方法，获取隐藏状态
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs.last_hidden_state

        # 逐层应用带门控的 Layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, mems)

        # 更新 outputs 中的隐藏状态
        outputs.last_hidden_state = hidden_states
        return outputs