import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 2048
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    d_ff: int = 1536
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)

        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AudioTokenTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [B, T], received {tuple(input_ids.shape)}")

        B, T = input_ids.shape

        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence too long: T={T}, max_seq_len={self.config.max_seq_len}"
            )

        tok = self.token_emb(input_ids)

        pos = self.pos_emb[:, :T, :]

        x = tok + pos
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)

        return logits

    def compute_loss(self, input_ids, target_ids):
        logits = self.forward(input_ids)

        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, V),
            target_ids.view(B * T),
        )

        return logits, loss

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, temperature, top_k=50):
        self.eval()
        tokens = start_tokens

        for _ in range(max_new_tokens):
            if tokens.size(1) > self.config.max_seq_len:
                input_cond = tokens[:, -self.config.max_seq_len :]
            else:
                input_cond = tokens

            logits = self.forward(input_cond)  # [B, T, V]
            next_logits = logits[:, -1, :]     # [B, V]
            next_logits = next_logits / temperature

            if top_k is not None:
                values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                min_values = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_values,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )
            probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
