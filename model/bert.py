import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embed_dim % 2 == 0

        pe = torch.zeros(config.max_len, config.embed_dim)
        position = torch.arange(0, config.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embed_dim, 2).float() * (-math.log(10000.0) / config.embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embed_dim % config.n_head == 0

        self.c_attn = nn.Linear(config.embed_dim, config.embed_dim*3)
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.embed_dim = config.embed_dim
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            attn = attn.masked_fill(attn_mask.logical_not(), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)

            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.embed_dim, config.embed_dim*4)
        self.w2 = nn.Linear(config.embed_dim, config.embed_dim*4)
        self.w3 = nn.Linear(config.embed_dim*4, config.embed_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = SwiGLU(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.max_len = config.max_len
        self.n_head = config.n_head

        self.bert = nn.ModuleDict(dict(
            pe = PositionalEncoding(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))

    def forward(self, x, len):
        B, T, C = x.size()
        assert T <= self.max_len

        x = self.bert.pe(x)
        x = self.bert.drop(x)

        attn_mask = (torch.arange(T, dtype=torch.int64, device=x.device).unsqueeze(0) < len.unsqueeze(1))
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        for block in self.bert.h:
            x = block(x, attn_mask)

        x = self.bert.ln_f(x)

        return x
