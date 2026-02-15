import torch
import torch.nn as nn
import math

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

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DualStreamBlock(nn.Module):
    """
    A single block of the Dual-Stream Transformer.
    Maintains Local Tokens (Columns) and a Global Token (Row Summary).
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_global = nn.LayerNorm(d_model)
        self.norm_local = nn.LayerNorm(d_model)
        
        # 1. Global Updates (Cross-Attend to Local)
        # Query: Global, Key/Value: Local
        self.attn_global = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 2. Local Updates (Cross-Attend to Global)
        # Query: Local, Key/Value: Global
        self.attn_local_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 3. Local Self-Attention
        self.attn_local_self = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # FFNs
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

    def forward(self, x_local, x_global):
        """
        x_local: [B, N, D]
        x_global: [B, 1, D]
        """
        # 1. Global Update: Global attends to Local
        # Residual connection on Global
        # Q=Global, K=Local, V=Local
        tgt = x_global
        src = x_local
        
        # Attn
        out, _ = self.attn_global(tgt, src, src)
        x_global = self.norm_global(x_global + out)
        
        # 2. Local Update: Local attends to (Updated) Global
        # Q=Local, K=Global, V=Global
        tgt = x_local
        src = x_global # [B, 1, D] broadcasted by attention mechanism
        
        out, _ = self.attn_local_cross(tgt, src, src)
        x_local = self.norm_local(x_local + out)
        
        # 3. Local Self-Attention
        # Q=Local, K=Local, V=Local
        out, _ = self.attn_local_self(x_local, x_local, x_local)
        x_local = self.norm_local_2(x_local + out)
        
        # 4. FFN
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
        
        # Timestep Embedding
        self.t_embedder = TimestepEmbedder(d_model)
        
        # Condition Embedding (Class Conditional)
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embedder = nn.Embedding(num_classes, d_model)
        
        # Learnable Global Token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Blocks
        self.layers = nn.ModuleList([
            DualStreamBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final Output Projection
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model) # Projects to velocity

    def forward(self, x, t, y=None):
        """
        x: [B, N, D] (Noisy Tokens)
        t: [B] (Timesteps)
        y: [B] (Class Labels) - Optional
        """
        B, N, D = x.shape
        
        # Embed Time
        t_emb = self.t_embedder(t) # [B, D]
        t_emb = t_emb.unsqueeze(1) # [B, 1, D]
        
        # Add Time to Input
        x = x + t_emb
        
        # Prepare Global Token
        global_tok = self.global_token.expand(B, -1, -1) # [B, 1, D]
        global_tok = global_tok + t_emb
        
        # Add Condition to Global Token
        if y is not None and self.num_classes is not None:
             y_emb = self.class_embedder(y).unsqueeze(1) # [B, 1, D]
             global_tok = global_tok + y_emb
        
        # Pass through layers
        for layer in self.layers:
            x, global_tok = layer(x, global_tok)
            
        # Final Head on Local Tokens
        x = self.output_norm(x)
        out = self.head(x) # [B, N, D] velocity
        
        return out

