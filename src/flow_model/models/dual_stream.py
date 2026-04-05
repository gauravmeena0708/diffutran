import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(x, cos, sin):
    # x: [B, N, n_heads, d_head]
    # cos, sin: [1, N, 1, d_head]
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    rx = torch.cat([-x2, x1], dim=-1)
    return x * cos + rx * sin

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
        return self._cos_cached[:, :seq_len, ...], self._sin_cached[:, :seq_len, ...]

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.register_buffer("_device_tracker", torch.zeros(1))

    @property
    def device(self):
        return self._device_tracker.device

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=self.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FlashAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, rope_cos=None, rope_sin=None):
        B, N_q, _ = q.shape
        _, N_k, _ = k.shape
        
        qx = self.q_proj(q).view(B, N_q, self.n_heads, self.head_dim)
        kx = self.k_proj(k).view(B, N_k, self.n_heads, self.head_dim)
        vx = self.v_proj(v).view(B, N_k, self.n_heads, self.head_dim)
        
        if rope_cos is not None and rope_sin is not None:
            qx = apply_rope(qx, rope_cos[:, :N_q, :, :], rope_sin[:, :N_q, :, :])
            if N_q == N_k:
                kx = apply_rope(kx, rope_cos[:, :N_k, :, :], rope_sin[:, :N_k, :, :])

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)

        # F.scaled_dot_product_attention requires PyTorch >= 2.0.
        # Fall back to manual scaled dot-product attention for older envs.
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(qx, kx, vx, dropout_p=self.dropout if self.training else 0.0)
        else:
            scale = self.head_dim ** -0.5
            attn = (qx @ kx.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            if self.training and self.dropout > 0.0:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ vx
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        
        return self.out_proj(out)

class DualStreamBlock(nn.Module):
    """
    A single block of the Dual-Stream Transformer.
    Maintains Local Tokens (Columns) and a Global Token (Row Summary).
    Uses FlashAttention-2 and RoPE.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_global = nn.LayerNorm(d_model)
        self.norm_local = nn.LayerNorm(d_model)
        self.norm_cond = nn.LayerNorm(d_model)
        
        self.attn_global = FlashAttention(d_model, n_heads, dropout)
        self.attn_local_cross = FlashAttention(d_model, n_heads, dropout)
        self.attn_local_self = FlashAttention(d_model, n_heads, dropout)
        self.attn_cond = FlashAttention(d_model, n_heads, dropout)
        
        self.ffn_global = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_local = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_global_2 = nn.LayerNorm(d_model)
        self.norm_local_2 = nn.LayerNorm(d_model)
        self.norm_local_3 = nn.LayerNorm(d_model)

    def forward(self, x_local, x_global, rope_cos, rope_sin, y_emb=None):
        """
        x_local: [B, N, D]
        x_global: [B, 1, D]
        y_emb: Optional [B, 1, D] condition
        """
        out = self.attn_global(x_global, x_local, x_local)
        x_global = self.norm_global(x_global + out)
        
        # Dedicated Cross-Attention Conditioning
        if y_emb is not None:
             out = self.attn_cond(x_global, y_emb, y_emb)
             x_global = self.norm_cond(x_global + out)
        
        out = self.attn_local_cross(x_local, x_global, x_global)
        x_local = self.norm_local(x_local + out)
        
        out = self.attn_local_self(x_local, x_local, x_local, rope_cos, rope_sin)
        x_local = self.norm_local_2(x_local + out)
        
        x_global = self.norm_global_2(x_global + self.ffn_global(x_global))
        x_local = self.norm_local_3(x_local + self.ffn_local(x_local))
        
        return x_local, x_global

class DualStreamTransformer(nn.Module):
    """
    Novel Backbone for Tabular Flow Matching.
    """
    def __init__(self, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1, num_classes=None):
        super().__init__()
        self.d_model = d_model
        
        self.t_embedder = TimestepEmbedder(d_model)
        self.rope = RotaryPositionalEmbedding(d_model // n_heads)
        
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embedder = nn.Embedding(num_classes, d_model)
        
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            DualStreamBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x, t, y=None):
        B, N, D = x.shape
        t_emb = self.t_embedder(t).unsqueeze(1)
        x = x + t_emb
        
        global_tok = self.global_token.expand(B, -1, -1) + t_emb
        y_emb = self.class_embedder(y).unsqueeze(1) if y is not None and self.num_classes is not None else None
        
        rope_cos, rope_sin = self.rope(seq_len=N)
        
        for layer in self.layers:
            x, global_tok = layer(x, global_tok, rope_cos, rope_sin, y_emb)
            
        x = self.output_norm(x)
        return self.head(x)

if __name__ == "__main__":
    t_model = DualStreamTransformer(d_model=64, n_layers=2, n_heads=4, d_ff=128, num_classes=5)
    _x = torch.randn(2, 10, 64)
    _t = torch.rand(2)
    _y = torch.tensor([0, 4])
    _out = t_model(_x, _t, _y)
    assert _out.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {_out.shape}"
    print("DualStreamTransformer verification passed!")
