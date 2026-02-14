import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    # Model
    vocab_size: int = 50257 # using gpt tokenizer thats whyy
    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_layers: int = 6
    seq_len: int = 128

    # MoE
    n_real_experts: int = 8
    shared_expert_hidden: int = 768
    expert_hidden: int = 384
    top_k: int = 4
    rho: float = 0.5  # target data-sparsity

    # Loss weights
    balance_loss_weight: float = 2e-2
    z_loss_weight: float = 1e-3

    # Training
    batch_size: int = 32
    max_steps: int = 3000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Eval / logging
    eval_interval: int = 200
    log_interval: int = 50
    save_interval: int = 1000
    gen_tokens: int = 150 # how many tokens to generate mid training

    @property
    def n_null_copies(self) -> int:
        """M = N * (1-rho)/rho  (Eq. 4 in paper)"""
        return int(self.n_real_experts * (1 - self.rho) / self.rho)

    @property
    def n_total_slots(self) -> int:
        return self.n_real_experts + self.n_null_copies


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# Rotary Position Embeddings


def precompute_rope(dim:int,max_len:int,theta:float = 10000.0) -> tuple[torch.Tensor,torch.Tensor]:
    freqs = 1.0/ (theta**(torch.arange(0,dim,2).float()/dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t,freqs) # (max_len, dim//2)
    return angles.cos(),angles.sin()

def apply_rope(x:torch.Tensor,cos:torch.Tensor,sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim)"""
    D = x.size(-1)
    assert D%2 == 0 
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0) # (1, 1, T, head_dim//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[...,::2]
    x2 = x[...,1::2]
    # stack + flatten is just a compact way of interleaving even/odd back into the original embedding layout.
    return torch.stack([x1*cos - x2*sin , x1*sin + x2*cos],dim=-1).flatten(-2)



# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.seq_len,cfg.seq_len)).view(1,1,cfg.seq_len,cfg.seq_len)
        self.register_buffer("mask",mask)
    
    def forward(self,x:torch.Tensor,rope_cos:torch.Tensor,rope_sin:torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads,self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q,rope_cos,rope_sin)
        k = apply_rope(k,rope_cos,rope_sin)

        attn = (q @ k.transpose(-2,-1)) * (self.head_dim**-0.5)
        attn = attn.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        attn = F.softmax(attn,dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

# Expert FFN


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.gelu = nn.GELU(approximate='tanh')
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x 
    

class MoELayer(nn.Module):
    """
    Token-choice MoE with null experts (Section 4.1).

    Router produces N+1 logits (N real + 1 null).  The single null logit is
    duplicated M times to form N+M total slots.  Top-K is applied over all
    N+M slots.  Selected slots pointing to null produce output=0 (no compute).
    Gate weights are renormalised over selected *real* experts only.
    """

    def __init__(self,cfg:Config):
        super().__init__()
        self.cfg = cfg
        N = cfg.n_real_experts
        M = cfg.n_null_copies

        # shared expert 
        self.shared_expert = ExpertFFN(cfg.d_model, cfg.shared_expert_hidden)

        # Routed experts
        self.experts = nn.ModuleList([ExpertFFN(cfg.d_model, cfg.expert_hidden) for _ in range(N)])

        # Router: produces N+1 logits (N real + 1 null)
        self.router = nn.Linear(cfg.d_model, N + 1, bias=False)
        self.N = N
        self.M = M
        self.top_k = cfg.top_k

        # Telemetry accumulators (filled during forward, read externally)
        self.last_expert_counts: torch.Tensor | None = None
        self.last_null_ratio: float = 0.0
        self.last_gate_weights: torch.Tensor | None = None
        self.last_zero_compute_ratio: float = 0.0
        self.last_balance_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        num_tokens = x_flat.size(0)

         # --- Router logits ---
        logits_raw = self.router(x_flat)  # (tokens, N+1)
        real_logits = logits_raw[:, :self.N]         # (tokens, N)
        null_logit = logits_raw[:, self.N:]           # (tokens, 1)

        # Duplicate null logit M times → total N+M slots
        expanded_logits = torch.cat([real_logits, null_logit.expand(-1, self.M)], dim=-1)  # (tokens, N+M)

        # --- Z-loss: log-sum-exp penalty ---
        lse = torch.logsumexp(expanded_logits, dim=-1)  # (tokens,)
        z_loss = (lse ** 2).mean()
        # --- Top-K selection ---
        topk_vals, topk_idxs = torch.topk(expanded_logits, self.top_k, dim=-1)  # (tokens, top_k)
        topk_gates = F.softmax(topk_vals, dim=-1)  # softmax over selected slots

        # build mask for which selection are null and whiich ar real
        is_real = topk_idxs < self.N  # (tokens, top_k) 

        real_gate_sum = (topk_gates*is_real.float()).sum(dim=-1,keepdim=True).clamp(min=1e-9)
        # If a token has no real experts selected, all gates become 0 (zero-compute token)
        has_real = is_real.any(dim=-1, keepdim=True).float()
        renorm_gates = topk_gates * is_real.float() / real_gate_sum * has_real  # (tokens, top_k)

        """
        so first we create a nask for null experts then we multiply the mask with topk gates which nulls out 
        the gates of the null expert then when we sum across experts those null gates doesnt contribute anything 
        and then we create another mask for nullifying any rows which has no rela experts then we take product of that mask
        with the sum 
        """

        # --- Load-balancing loss (Eq. 6) ---
        # f_i = fraction of tokens routed to slot i
        NM = self.N + self.M
        
        """
        topk format is not convenient for counting usage per slot.
        Thats where slot_mask comes in.
        For every token
        for every top-k choice
        keep a one-hot vector over slots.

        slot_mask[token][k][slot] = 0
        
        slot_mask is a one-hot encoding of routing decisions.
        It answers the question:

        “For each token and each of its top-k choices, which slot was selected?”
        

        Assume we are in an MoE router with
        `N = 3` real experts, `M = 1` null expert, so


        NM = N + M = 4   # total slots


        Assume:


        num_tokens = 4
        top_k = 2


        So for each token, the router picks 2 slots out of 4.

        ---

        First, imagine the router already ran and produced `topk_idxs`, which tells us **which slots were selected for each token**.

        Letss say:

        topk_idxs =
        tensor([
        [0, 2],
        [1, 3],
        [0, 1],
        [2, 3]
        ])


        Shape is (num_tokens=4, top_k=2)

        This means:

        * token 0 routed to slots 0 and 2
        * token 1 → slots 1 and 3
        * token 2 → slots 0 and 1
        * token 3 → slots 2 and 3

        ---

        Now this line runs:

        slot_mask = torch.zeros(num_tokens, self.top_k, NM)


        So shape is:

        (4, 2, 4)

        Initially it’s all zeros:

        slot_mask[token, k, slot] = 0

        ---

        Then comes the critical operation:

        slot_mask.scatter_(2, topk_idxs.unsqueeze(-1), 1.0)

        Let’s decode that.

        `topk_idxs.unsqueeze(-1)` changes shape from `(4, 2)` to `(4, 2, 1)` so it can index into dimension 2 (the slot dimension).

        `scatter_(2, index, value)` means:

        > along dimension 2, put `value` at the positions specified by `index`

        So after scatter, `slot_mask` becomes a **one-hot encoding of slot selection**.

        For token 0, which had `[0, 2]`, its slice looks like:

        [
        [1, 0, 0, 0],   # k=0 chose slot 0
        [0, 0, 1, 0]    # k=1 chose slot 2
        ]

        For token 1 `[1, 3]`:

        [
        [0, 1, 0, 0],
        [0, 0, 0, 1]
        ]

        For token 2 `[0, 1]`:

        [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
        ]

        For token 3 `[2, 3]`:

        [
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]

        So `slot_mask` now explicitly marks **every routing decision**.

        ---

        Now comes this line:

        f = slot_mask.sum(dim=1).sum(dim=0) / (num_tokens * self.top_k)

        Let’s do it step by step mentally.

        First:

        slot_mask.sum(dim=1)

        This collapses the `top_k` dimension, giving shape `(4, 4)`.

        For each token, it counts how many times each slot was selected:

        Token 0:

        [1, 0, 1, 0]

        Token 1:

        [0, 1, 0, 1]

        Token 2:

        [1, 1, 0, 0]

        Token 3:

        [0, 0, 1, 1]

        Now:

        .sum(dim=0)

        This sums across tokens, giving total usage per slot:

        Slot 0 → used 2 times
        Slot 1 → used 2 times
        Slot 2 → used 2 times
        Slot 3 → used 2 times

        So:

        tensor([2, 2, 2, 2])

        Now divide by total routing decisions:

        num_tokens * top_k = 4 * 2 = 8

        So:

        f = [2/8, 2/8, 2/8, 2/8]
        = [0.25, 0.25, 0.25, 0.25]

        This `f` is exactly what the comment says:

        > **fᵢ = fraction of tokens routed to slot i**

        It’s the empirical routing frequency.

        ---

        Now we compute the *expected* routing distribution from logits:


        probs_all = F.softmax(expanded_logits, dim=-1)


        Assume `expanded_logits` has shape `(4, 4)` and after softmax we get:


        probs_all =
        [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.4, 0.3, 0.2],
        [0.25,0.25,0.25,0.25],
        [0.3, 0.2, 0.3, 0.2]
        ]


        Each row sums to 1. This is the router’s *probabilistic preference* before top-k selection.

        Now:


        P = probs_all.mean(dim=0)


        So average over tokens:

        Slot 0 → (0.4 + 0.1 + 0.25 + 0.3) / 4 = 0.2625
        Slot 1 → 0.2875
        Slot 2 → 0.2625
        Slot 3 → 0.1875

        So:


        P = [0.2625, 0.2875, 0.2625, 0.1875]


        This matches the comment:

        > **Pᵢ = average routing probability for slot i**

        This is what the router *wants* to do on average.

        ---

        Finally comes the load-balancing loss:


        balance_loss = NM * (f * P).sum()


        Compute elementwise product:


        f * P =
        [
        0.25 * 0.2625,
        0.25 * 0.2875,
        0.25 * 0.2625,
        0.25 * 0.1875
        ]


        Sum them:


        ≈ 0.25


        Multiply by `NM = 4`:


        balance_loss ≈ 1.0


        ---

        ### What this loss is actually doing

        If routing is **perfectly balanced**, `f` will match `P`, and this expression is minimized.

        If some slots get overused while others are ignored, `f` and `P` become misaligned, increasing the loss.

        So this loss softly pushes the router toward:

        * using all slots
        * respecting its own probability distribution
        * avoiding expert collapse

        This loss is small when probability mass and actual usage are spread evenly, and it becomes large when both probability and usage concentrate on the same few slots.

        Thatss why it prevents expert collapse.

        """

        # one-hot encode selected slots
        slot_mask = torch.zeros(num_tokens, self.top_k, NM, device=x.device)
        slot_mask.scatter_(2,topk_idxs.unsqueeze(-1),1.0)
        f = slot_mask.sum(dim=1).sum(dim=0)/ (num_tokens * self.top_k)  # (NM,)
        
         # P_i = average routing probability for slot i
        probs_all = F.softmax(expanded_logits,dim=-1)  # (tokens, NM)
        P = probs_all.mean(dim=0) # (NM,)

        balance_loss = NM * (f * P).sum()

        # --- Dispatch to real experts ---
        # Accumulate expert outputs

        combined_output = torch.zeros_like(x_flat)  # (tokens, D)

        # Per-expert token counts for telemetry
        expert_counts = torch.zeros(self.N,device=x.device)
        gate_weight_sums = torch.zeros(self.N, device=x.device)
        gate_weight_counts = torch.zeros(self.N, device=x.device)


        for k_idx in range(self.top_k):
            slot_ids = topk_idxs[:, k_idx]   #(tokens,)
            gates = renorm_gates[:, k_idx] 

            for e in range(self.N):
                mask = (slot_ids == e)
                if not mask.any():
                    continue
                token_subset = x_flat[mask]  # (n_e, D)
                gate_subset = gates[mask].unsqueeze(-1)  # (n_e, 1)
                expert_out = self.experts[e](token_subset)
                combined_output[mask] += gate_subset * expert_out
                expert_counts[e] += mask.sum().float()
                gate_weight_sums[e] += gates[mask].sum()
                gate_weight_counts[e] += mask.sum().float()
        
        # Shared expert (always active)
        shared_out = self.shared_expert(x_flat)
        output = shared_out + combined_output

        # --- Telemetry ---
        with torch.no_grad():
            null_selections = (~is_real).sum().float()
            total_selections = torch.tensor(num_tokens * self.top_k, dtype=torch.float32, device=x.device)
            self.last_null_ratio = (null_selections / total_selections).item()

            self.last_expert_counts = expert_counts.detach()

            avg_gates = torch.where(gate_weight_counts > 0,
                                    gate_weight_sums / gate_weight_counts,
                                    torch.zeros_like(gate_weight_sums))
            self.last_gate_weights = avg_gates.detach()

            # Zero-compute tokens: all top-K went to null
            zero_compute = (~is_real).all(dim=-1).sum().float()
            self.last_zero_compute_ratio = (zero_compute / num_tokens).item()

        self.last_balance_loss = balance_loss
        self.last_z_loss = z_loss

        return output.view(B, T, D)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.moe_norm = RMSNorm(cfg.d_model)
        self.moe = MoELayer(cfg)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.moe(self.moe_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class MoENullModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE
        rope_cos, rope_sin = precompute_rope(cfg.head_dim, cfg.seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) → logits: (B, T, vocab_size)"""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        return self.lm_head(x)

    def get_aux_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        balance_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        z_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if block.moe.last_balance_loss is not None:
                balance_loss = balance_loss + block.moe.last_balance_loss
            if block.moe.last_z_loss is not None:
                z_loss = z_loss + block.moe.last_z_loss
        balance_loss = balance_loss / self.cfg.n_layers
        z_loss = z_loss / self.cfg.n_layers
        return balance_loss, z_loss

    def get_telemetry(self) -> dict:
        """Gather telemetry from all MoE layers."""
        expert_counts = []
        null_ratios = []
        gate_weights = []
        zero_compute_ratios = []
        for block in self.blocks:
            moe = block.moe
            if moe.last_expert_counts is not None:
                expert_counts.append(moe.last_expert_counts.cpu().tolist())
            null_ratios.append(moe.last_null_ratio)
            if moe.last_gate_weights is not None:
                gate_weights.append(moe.last_gate_weights.cpu().tolist())
            zero_compute_ratios.append(moe.last_zero_compute_ratio)
        return {
            "per_layer_expert_counts": expert_counts,
            "avg_expert_counts": [sum(c) / len(c) for c in zip(*expert_counts)] if expert_counts else [],
            "null_ratio": sum(null_ratios) / len(null_ratios) if null_ratios else 0,
            "avg_gate_weights": [sum(g) / len(g) for g in zip(*gate_weights)] if gate_weights else [],
            "zero_compute_ratio": sum(zero_compute_ratios) / len(zero_compute_ratios) if zero_compute_ratios else 0,
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.cfg.seq_len:]
            logits = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, tokens: list[int], seq_len: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.lr * max(coeff, 0.1)


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(model: MoENullModel, tokens: list[int], cfg: Config, device: torch.device) -> float:
    model.eval()
    n_eval = min(len(tokens) - cfg.seq_len - 1, cfg.batch_size * 20)
    if n_eval <= 0:
        model.train()
        return float("inf")
    total_loss = 0.0
    count = 0
    start = len(tokens) - n_eval - cfg.seq_len - 1
    for i in range(0, n_eval, cfg.seq_len):
        offset = start + i
        if offset + cfg.seq_len + 1 > len(tokens):
            break
        chunk = torch.tensor(tokens[offset : offset + cfg.seq_len + 1], dtype=torch.long, device=device)
        x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    if count == 0:
        return float("inf")
    return math.exp(total_loss / count)


# ---------------------------------------------------------------------------
# ASCII bar chart for console
# ---------------------------------------------------------------------------

def ascii_bar(values: list[float], width: int = 30, labels: list[str] | None = None) -> str:
    if not values:
        return ""
    max_val = max(values) if max(values) > 0 else 1.0
    lines = []
    for i, v in enumerate(values):
        bar_len = int(v / max_val * width)
        label = labels[i] if labels else f"E{i}"
        lines.append(f"  {label:>4s} |{'█' * bar_len}{' ' * (width - bar_len)}| {v:.0f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MoE with Null Experts")
    parser.add_argument("--dataset", type=str, default="input.txt",
                        help="Path to text file or 'tiny' for tiny-shakespeare")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    cfg = Config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Tokenizer ---
    enc = tiktoken.get_encoding("gpt2")

    # --- Data ---
    data_path = args.dataset
    if data_path == "tiny":
        # Download tiny-shakespeare
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path = os.path.join(args.output_dir, "input.txt")
        if not os.path.exists(data_path):
            print("Downloading tiny-shakespeare...")
            urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = enc.encode(text)
    print(f"Dataset: {len(tokens):,} tokens")

    dataset = TextDataset(tokens, cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            drop_last=True, num_workers=0)

    # --- Model ---
    model = MoENullModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Subtract tied weights (counted once in tok_emb, once in lm_head)
    n_params_unique = n_params - model.tok_emb.weight.numel()
    print(f"Model parameters: {n_params_unique:,} (unique, with weight tying)")
    print(f"  Null copies M = {cfg.n_null_copies}, total routing slots = {cfg.n_total_slots}")
    assert n_params_unique < 42_000_000, f"Model too large: {n_params_unique:,} params"

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    # --- Telemetry log ---
    telemetry_path = os.path.join(args.output_dir, "telemetry.json")
    telemetry_log: list[dict] = []

    # --- Training ---
    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(1, cfg.max_steps + 1):
        # Get batch
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_batch, y_batch = next(data_iter)

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # LR schedule
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        logits = model(x_batch)
        lm_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y_batch.view(-1))

        # Auxiliary losses
        balance_loss, z_loss = model.get_aux_losses()
        total_loss = lm_loss + cfg.balance_loss_weight * balance_loss + cfg.z_loss_weight * z_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # --- Telemetry ---
        telem = model.get_telemetry()
        step_data = {
            "step": step,
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "balance_loss": balance_loss.item(),
            "z_loss": z_loss.item(),
            "lr": lr,
            "expert_counts": telem["avg_expert_counts"],
            "null_ratio": telem["null_ratio"],
            "gate_weights": telem["avg_gate_weights"],
            "zero_compute_ratio": telem["zero_compute_ratio"],
        }

        # Periodic perplexity + generation
        if step % cfg.eval_interval == 0 or step == 1:
            ppl = compute_perplexity(model, tokens, cfg, device)
            step_data["perplexity"] = ppl

            # Generate sample
            prompt_tokens = tokens[:10]
            prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            gen_ids = model.generate(prompt, max_new_tokens=cfg.gen_tokens)
            gen_text = enc.decode(gen_ids[0].cpu().tolist())
            step_data["generated_text"] = gen_text

        telemetry_log.append(step_data)

        # --- Console output ---
        if step % cfg.log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            tokens_per_sec = step * cfg.batch_size * cfg.seq_len / elapsed
            print(f"\n{'='*60}")
            print(f"Step {step}/{cfg.max_steps} | LR: {lr:.2e} | {tokens_per_sec:,.0f} tok/s")
            print(f"  Loss: {total_loss.item():.4f} (LM: {lm_loss.item():.4f}, "
                  f"Bal: {balance_loss.item():.4f}, Z: {z_loss.item():.4f})")
            if "perplexity" in step_data:
                print(f"  Perplexity: {step_data['perplexity']:.2f}")
            print(f"  Null routing: {telem['null_ratio']*100:.1f}% | "
                  f"Zero-compute tokens: {telem['zero_compute_ratio']*100:.1f}%")

            if telem["avg_expert_counts"]:
                print(f"  Expert utilization:")
                print(ascii_bar(telem["avg_expert_counts"],
                                labels=[f"E{i}" for i in range(cfg.n_real_experts)]))

            if "generated_text" in step_data:
                print(f"  Generated: {step_data['generated_text'][:200]}")

        # --- Save checkpoint ---
        if step % cfg.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # --- Save telemetry periodically ---
        if step % cfg.log_interval == 0:
            with open(telemetry_path, "w") as f:
                json.dump(telemetry_log, f)

    # --- Final save ---
    with open(telemetry_path, "w") as f:
        json.dump(telemetry_log, f)
    print(f"\nTelemetry saved to {telemetry_path}")

    final_ckpt = os.path.join(args.output_dir, "checkpoint_final.pt")
    torch.save({
        "step": cfg.max_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }, final_ckpt)
    print(f"Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
