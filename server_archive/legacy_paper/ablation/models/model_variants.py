"""
Ablation Study: Model Variants Definition
==========================================
Four variants for the Diffusion + Multi-View Transformer IDS framework:
  1. Full Model:      Diffusion Augmentation + Multi-View Transformer
  2. w/o Diffusion:   Multi-View Transformer only (no diffusion augmentation)
  3. w/o Multi-View:  Diffusion + Standard Transformer (single-view)
  4. w/o Both:        Standard Transformer only (no diffusion, no multi-view)

Each variant shares the same classifier head and evaluation protocol.
Only the data augmentation strategy and/or the attention mechanism differs.

Author: [Your Name]
Date:   2026-06-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 1. Core Building Blocks
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout(ffn_out)
        return x


class StandardTransformerEncoder(nn.Module):
    """Standard (single-view) Transformer encoder."""
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1,
                 max_seq_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)
        x = self.norm(x)
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)
        return x


# ============================================================================
# 2. Multi-View Transformer Encoder
# ============================================================================

class MultiViewTransformerEncoder(nn.Module):
    """
    Multi-View Transformer encoder.
    Splits input features into K views, processes each view independently,
    then fuses them via cross-attention or concatenation.

    Args:
        input_dim:        Total input feature dimension
        d_model:          Hidden dimension per view
        nhead:            Number of attention heads per view
        num_layers:       Number of transformer layers per view
        dim_feedforward:  FFN hidden dimension
        dropout:          Dropout rate
        n_views:          Number of views
        view_splits:      List of feature indices for each view (if None, splits evenly)
        fusion_method:    'concat' or 'cross_attn'
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1,
                 n_views: int = 3, view_splits: Optional[List[List[int]]] = None,
                 fusion_method: str = 'concat'):
        super().__init__()
        self.n_views = n_views
        self.fusion_method = fusion_method
        self.d_model = d_model

        if view_splits is not None:
            self.view_splits = view_splits
            self.view_dims = [len(vs) for vs in view_splits]
        else:
            # Default: split evenly
            base_dim = input_dim // n_views
            remainder = input_dim % n_views
            self.view_dims = [base_dim + (1 if i < remainder else 0) for i in range(n_views)]
            splits = []
            start = 0
            for vd in self.view_dims:
                splits.append(list(range(start, start + vd)))
                start += vd
            self.view_splits = splits

        # Per-view projection + Transformer
        self.view_projectors = nn.ModuleList([
            nn.Linear(self.view_dims[i], d_model) for i in range(n_views)
        ])
        self.view_transformers = nn.ModuleList([
            StandardTransformerEncoder(d_model, d_model, nhead, num_layers,
                                       dim_feedforward, dropout, max_seq_len=1)
            for _ in range(n_views)
        ])

        # Fusion
        if fusion_method == 'concat':
            fusion_dim = d_model * n_views
        elif fusion_method == 'cross_attn':
            fusion_dim = d_model
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion_proj = nn.Linear(fusion_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) - flat feature vector
        Returns: (batch, d_model) - fused representation
        """
        batch_size = x.size(0)
        view_reps = []

        for i in range(self.n_views):
            # Extract view features
            view_x = x[:, self.view_splits[i]]  # (batch, view_dim_i)
            view_x = self.view_projectors[i](view_x)  # (batch, d_model)
            # Treat as sequence of length 1
            view_x = view_x.unsqueeze(1)  # (batch, 1, d_model)
            view_rep = self.view_transformers[i](view_x)  # (batch, d_model)
            view_reps.append(view_rep)

        if self.fusion_method == 'concat':
            fused = torch.cat(view_reps, dim=-1)  # (batch, d_model * n_views)
        elif self.fusion_method == 'cross_attn':
            # Stack views as sequence for cross-attention
            stacked = torch.stack(view_reps, dim=1)  # (batch, n_views, d_model)
            fused, _ = self.cross_attn(stacked, stacked, stacked)  # (batch, n_views, d_model)
            fused = fused.mean(dim=1)  # (batch, d_model)

        out = self.fusion_proj(fused)  # (batch, d_model)
        return out


# ============================================================================
# 3. Diffusion Model (Simplified Score-Based / DDPM)
# ============================================================================

class DiffusionAugmenter(nn.Module):
    """
    Simplified Denoising Diffusion Probabilistic Model for tabular data augmentation.
    Generates synthetic network traffic samples to balance training data.

    Uses a simple MLP-based denoiser (epsilon_prediction network).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 256],
                 num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # Precompute diffusion noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Denoiser network (epsilon_theta)
        layers = []
        prev_dim = input_dim + 1  # +1 for time step embedding
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        self.denoiser = nn.Sequential(*layers)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """Training step: predict noise added to x_0."""
        t = torch.randint(0, self.num_timesteps, (x_0.size(0),), device=x_0.device)
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        t_embed = (t.float() / self.num_timesteps).unsqueeze(-1)
        model_input = torch.cat([x_t, t_embed], dim=-1)
        noise_pred = self.denoiser(model_input)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device, return_all: bool = False) -> torch.Tensor:
        """Generate synthetic samples via reverse diffusion."""
        x_t = torch.randn(n_samples, self.input_dim, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_embed = (t_tensor.float() / self.num_timesteps).unsqueeze(-1)
            model_input = torch.cat([x_t, t_embed], dim=-1)
            noise_pred = self.denoiser(model_input)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            # DDPM reverse step
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise

        return torch.clamp(x_t, -5.0, 5.0)  # Clamp to reasonable range


# ============================================================================
# 4. Classifier Head
# ============================================================================

class ClassifierHead(nn.Module):
    """MLP classifier head shared across all ablation variants."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================================
# 5. Variant 1: Full Model (Diffusion + Multi-View Transformer)
# ============================================================================

class FullModel(nn.Module):
    """Variant 1: Full proposed model with diffusion augmentation + multi-view transformer."""
    def __init__(self, input_dim: int, num_classes: int = 2, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, n_views: int = 3,
                 view_splits: Optional[List[List[int]]] = None,
                 fusion_method: str = 'concat'):
        super().__init__()
        self.diffusion = DiffusionAugmenter(input_dim)
        self.encoder = MultiViewTransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, n_views=n_views,
            view_splits=view_splits, fusion_method=fusion_method
        )
        self.classifier = ClassifierHead(d_model, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: Diffusion augmentation happens during data loading, not in forward pass.
        # The encoder receives (potentially augmented) features directly.
        encoded = self.encoder(x)
        return self.classifier(encoded)

    def get_encoder_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# 6. Variant 2: Without Diffusion (w/o Diffusion)
# ============================================================================

class NoDiffusionModel(nn.Module):
    """Variant 2: Multi-View Transformer without diffusion augmentation."""
    def __init__(self, input_dim: int, num_classes: int = 2, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, n_views: int = 3,
                 view_splits: Optional[List[List[int]]] = None,
                 fusion_method: str = 'concat'):
        super().__init__()
        self.encoder = MultiViewTransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, n_views=n_views,
            view_splits=view_splits, fusion_method=fusion_method
        )
        self.classifier = ClassifierHead(d_model, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.classifier(encoded)


# ============================================================================
# 7. Variant 3: Without Multi-View (w/o Multi-View)
# ============================================================================

class NoMultiViewModel(nn.Module):
    """Variant 3: Diffusion + standard (single-view) Transformer."""
    def __init__(self, input_dim: int, num_classes: int = 2, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.diffusion = DiffusionAugmenter(input_dim)
        self.encoder = StandardTransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers
        )
        self.classifier = ClassifierHead(d_model, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.unsqueeze(1))  # (batch, 1, input_dim) as seq_len=1
        return self.classifier(encoded)


# ============================================================================
# 8. Variant 4: Without Both (w/o Both)
# ============================================================================

class NoBothModel(nn.Module):
    """Variant 4: Standard Transformer only - no diffusion, no multi-view."""
    def __init__(self, input_dim: int, num_classes: int = 2, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.encoder = StandardTransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers
        )
        self.classifier = ClassifierHead(d_model, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.unsqueeze(1))
        return self.classifier(encoded)


# ============================================================================
# 9. Factory Function
# ============================================================================

def create_variant(variant_name: str, input_dim: int, num_classes: int = 2,
                   **kwargs) -> nn.Module:
    """Create an ablation variant by name."""
    variants = {
        'full': FullModel,
        'wo_diffusion': NoDiffusionModel,
        'wo_multiview': NoMultiViewModel,
        'wo_both': NoBothModel,
    }
    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}. "
                         f"Choose from {list(variants.keys())}")

    model_class = variants[variant_name]
    
    # Filter kwargs to only include those accepted by the constructor
    import inspect
    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_class(input_dim=input_dim, num_classes=num_classes, **filtered_kwargs)


# ============================================================================
# 10. Utility: Count Parameters
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 11. Self-Test
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Ablation Model Variants - Self Test")
    print("=" * 60)

    input_dim = 49  # UNSW-NB15 feature dimension
    batch_size = 16
    x = torch.randn(batch_size, input_dim)

    variant_names = ['full', 'wo_diffusion', 'wo_multiview', 'wo_both']
    for name in variant_names:
        model = create_variant(name, input_dim=input_dim, num_classes=9)
        out = model(x)
        n_params = count_parameters(model)
        print(f"[{name:15s}] output shape: {out.shape}, params: {n_params:,}")
        assert out.shape == (batch_size, 9), f"Unexpected output shape: {out.shape}"

    print("\nAll variants passed!")
