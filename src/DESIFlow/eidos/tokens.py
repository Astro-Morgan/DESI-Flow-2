"""
Token self-attention for the grid Eidos: pre-norm transformer blocks over the CNN's tokens (one token per 16 log-lambda pixels = 370 km/s).

    pos="cbias" (default) a learned relative-position table PLUS a CONTENT-CONDITIONED offset preference. The attention logit of asking token i and target token j is
                  q_i . k_j / sqrt(dh) + table[h, i - j] + g(x_i) . R[h, :, i - j]
                  table: one learned number per head and signed token offset (2 L - 1 entries), the same for every token (a content-blind prior);
                  g: a small MLP of the asking token's normalised content giving `rank` (8) numbers per head, R: `rank` learned offset profiles per head, so every token chooses WHICH
                  signed offsets it cares about (token 12 may ignore token 23 while token 23 attends to token 12; a [OIII] token and an [NII] token can prefer different line gaps in the
                  same head). The last layer of g is zero-initialised, so training starts from the plain table. Equivalently the attention WEIGHT is multiplied by exp(u_i(offset)) (a softmax
                  over offsets would only add a per-query constant that cancels). Measured in the benchmark (G0, B=32, 40k steps): random-span hidden-line medians +0.08..+0.10 over the
                  plain table (std 0.52 vs 0.44, dr2 0.44 vs 0.34, hi 0.58 vs 0.48), equal complex-hidden / visible lines, 1.55x the training time of the encoder step, +0.37M parameters.
    pos="bias"    the table alone (the benchmark baseline). Tied to the token spacing and to L (integer offsets).
    pos="rope"    rotary position embedding on the PHYSICAL token positions (km/s from the first token): every head dim is rotated, periods geometric in [min_period, max_period]. No
                  position parameters; works for any sequence length or spacing; min_period must be >= 2 x the token spacing (Nyquist: 370 km/s tokens -> >= 741 km/s). Measured WORSE
                  than the table on random-span medians (0.29 vs 0.44).
    All are redshift-translation equivariant on the uniform log-lambda grid (a redshift is a shift of every token position): the offset between two lines does not change with z.

forward(x (B, L, d)) -> (B, L, d). Parameter names (n1, n2, qkv, o, mlp, rb, g, R) are the ones of the benchmark blocks (scripts/narrow_bench/phys/physbench.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBlock(nn.Module):
    def __init__(self, d, n_heads, n_tokens, mlp_ratio=4, pos="cbias", token_v=None, min_period=800., max_period=1e6, rank=8, g_hidden=64):
        super().__init__()
        if pos not in ("cbias", "bias", "rope"):
            raise ValueError(f"pos must be 'cbias', 'bias' or 'rope', got {pos!r}")
        if d % n_heads or (d // n_heads) % 2:
            raise ValueError(f"d={d} must split into {n_heads} heads of even dimension")
        self.H, self.pos, self.rank = n_heads, pos, rank
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        if pos in ("bias", "cbias"):
            self.rb = nn.Parameter(torch.zeros(n_heads, 2 * n_tokens - 1))              # logit units
            self.register_buffer("idx", (torch.arange(n_tokens)[:, None] - torch.arange(n_tokens)[None, :]) + n_tokens - 1, persistent=False)
            if pos == "cbias":
                self.g = nn.Sequential(nn.Linear(d, g_hidden), nn.GELU(), nn.Linear(g_hidden, n_heads * rank))
                nn.init.zeros_(self.g[2].weight); nn.init.zeros_(self.g[2].bias)
                self.R = nn.Parameter(torch.randn(n_heads, rank, 2 * n_tokens - 1) * 0.1)  # offset profiles; the preference of token i is g(x_i) . R
        else:
            if token_v is None or token_v.numel() != n_tokens:
                raise ValueError("pos='rope' needs token_v (n_tokens,) in km/s")
            m = (d // n_heads) // 2
            omega = 2 * math.pi / torch.logspace(math.log10(min_period), math.log10(max_period), m, dtype=torch.float64)
            ang = token_v.detach().cpu().double().view(n_tokens, 1) * omega.view(1, m)     # float64 then cast: angles reach ~1e3 rad
            self.register_buffer("cos", torch.cos(ang).float(), persistent=False)
            self.register_buffer("sin", torch.sin(ang).float(), persistent=False)

    def _rotate(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        c, s = self.cos.to(x.dtype), self.sin.to(x.dtype)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    def content_bias(self, h):
        """h (B, L, d) normalised block input -> (B, H, L, L) content-conditioned offset preference u[b, h, i, j] = g(h_i)_h . R[h, :, i - j + L - 1] as a strided VIEW of a
        (B, H, L, 2L-1) matrix product (no gather, no atomics in the backward). The table convention is offset = i - j, so R is flipped to index by j - i."""
        B, L, _ = h.shape
        g = self.g(h).view(B, L, self.H, self.rank).permute(0, 2, 1, 3)                  # (B, H, L, r)
        U = torch.matmul(g, self.R.flip(-1))                                             # (B, H, L, 2L-1), U[b, h, i, k] with k = j - i + L - 1
        K = 2 * L - 1
        return U.as_strided((B, self.H, L, L), (self.H * L * K, L * K, K - 1, 1), L - 1)

    def forward(self, x):
        B, L, d = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        if self.pos == "rope":
            a = F.scaled_dot_product_attention(self._rotate(q), self._rotate(k), v)
        else:
            bias = self.rb[:, self.idx].unsqueeze(0).to(q.dtype)
            if self.pos == "cbias":
                bias = bias + self.content_bias(h).to(q.dtype)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.n2(x))
