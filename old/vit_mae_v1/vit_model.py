# vit_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=768, volume_size=(224, 192, 160)):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.grid_size = tuple(v // p for v, p in zip(volume_size, patch_size))
        self.n_patches = np.prod(self.grid_size)
        self.proj = nn.Conv3d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]
        return x


class ViTEncoder(nn.Module):
    def __init__(self, emb_dim=768, depth=12, n_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.emb_dim = emb_dim

        # No hardcoded pos_embed shape — we init lazily
        self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        if self.pos_embed is None or self.pos_embed.size(1) != x.size(1):
            # Lazy init or reinit based on input size
            self.pos_embed = nn.Parameter(torch.zeros(1, x.size(1), self.emb_dim).to(x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        x = x + self.pos_embed
        x = self.transformer(x)
        return self.norm(x)


class MAEDecoder(nn.Module):
    def __init__(self, emb_dim=768, patch_dim=16**3, hidden_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, patch_dim),
        )

    def forward(self, x):
        return self.decoder(x)  # Predicts masked patches

class ViTMAE(nn.Module):
    def __init__(self, patch_size=16, volume_size=(224, 192, 160), in_channels=1, emb_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, emb_dim, volume_size)
        self.encoder = ViTEncoder(emb_dim=emb_dim)
        self.decoder = MAEDecoder(emb_dim=emb_dim)
        self.patch_dim = np.prod([patch_size]*3 if isinstance(patch_size, int) else patch_size)


    def forward(self, x, mask_ratio=0.75):
        patches = self.patch_embed(x)  # [B, N, D]
        B, N, D = patches.shape
    
        num_mask = int(N * mask_ratio)
        rand_idx = torch.rand(B, N).argsort(dim=1)
        keep_idx = rand_idx[:, :-num_mask]
        mask_idx = rand_idx[:, -num_mask:]
    
        batch_range = torch.arange(B)[:, None].to(x.device)
        visible_tokens = patches[batch_range, keep_idx]
    
        encoded = self.encoder(visible_tokens)
        cls_token, encoded_patches = encoded[:, 0], encoded[:, 1:]
    
        full_encoded = torch.zeros(B, N, D, device=x.device)
        full_encoded[batch_range, keep_idx] = encoded_patches
    
        predicted = self.decoder(full_encoded)  # [B, N, patch_dim]
    
        return predicted, mask_idx, cls_token


    def patchify(self, imgs):
        p = 16
        B, C, H, W, D = imgs.shape
        assert H % p == 0 and W % p == 0 and D % p == 0

        nh, nw, nd = H // p, W // p, D // p
        patches = imgs.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
        patches = patches.contiguous().view(B, C, nh * nw * nd, p, p, p)
        patches = patches.permute(0, 2, 1, 3, 4, 5).contiguous()

        return patches.view(B, -1, p ** 3)

