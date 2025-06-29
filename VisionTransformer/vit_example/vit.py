import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        """
        Patch Embedding Layer for Vision Transformer.

        Args:
            patch_size (int): Size of the patches to be extracted from the input image.
            in_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space to which each patch will be projected.
        """

        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Forward pass of the Patch Embedding Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where
                                - B is batch size
                                - C is number of channels
                                - H is height
                                - W is width

        Returns:
            x (torch.Tensor): Output tensor of shape (B, D, H/P, W/P) where
                                - D is the embedding dimension
                                - H/P and W/P are the height and width of the patches.
        """

        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, H/P * W/P)
        x = x.transpose(1, 2)  # (B, H/P * W/P, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, dropout=0.1, attention_dropout=0.1):
        """
        Multi-Head Self-Attention Layer.

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to add a bias term to the query, key, and value projections.
            dropout (float): Dropout rate applied to the output of the MLP and attention layers.
            attention_dropout (float): Dropout rate applied to the attention weights.
        """

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaled Dot-Product Attention 中的 √d_k

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the Multi-Head Self-Attention Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) where
                                - B is batch size
                                - N is the number of patches (or tokens)
                                - D is the embedding dimension

        Returns:
            out (torch.Tensor): Output tensor of shape (B, N, D) after applying multi-head self-attention.
        """

        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        out = (attn @ v)  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, D)
        out = self.projection(out)
        out = self.projection_dropout(out)
        return out


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        """
        MLP Block for Transformer Encoder.

        Args:
            in_dim (int): Input dimension of the features.
            hidden_dim (int): Hidden dimension of the MLP.
            dropout (float): Dropout rate applied to the output of the MLP.
        """

        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the MLP Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) where
                                - B is batch size
                                - N is the number of patches (or tokens)
                                - D is the embedding dimension

        Returns:
            x (torch.Tensor): Output tensor of the same shape as input, after applying MLP.
        """

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, dropout=0.1, attention_dropout=0.1):
        """
        Transformer Encoder Block.

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of the hidden dimension in the MLP block to the embedding dimension.
            qkv_bias (bool): Whether to add a bias term to the query, key, and value projections.
            dropout (float): Dropout rate applied to the output of the MLP and attention layers.
            attention_dropout (float): Dropout rate applied to the attention weights.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x):
        """
        Forward pass of the Transformer Encoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) where
                                - B is batch size
                                - N is the number of patches (or tokens)
                                - D is the embedding dimension

        Returns:
            x (torch.Tensor): Output tensor of the same shape as input, after applying self-attention and MLP.
        """

        x = x + self.self_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        """
        Vision Transformer (ViT) model.

        Args:
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of the patches to be extracted from the input image.
            in_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            num_classes (int): Number of output classes for classification. If None, the model outputs the class token representation.
            embed_dim (int): Dimension of the embedding space to which each patch will be projected.
            num_layers (int): Number of Transformer encoder blocks.
            num_heads (int): Number of attention heads in the Multi-Head Self-Attention.
            mlp_ratio (float): Ratio of the hidden dimension in the MLP block to the embedding dimension.
            qkv_bias (bool): Whether to add a bias term to the query, key, and value projections.
            dropout (float): Dropout rate applied to the output of the MLP and attention layers.
            attention_dropout (float): Dropout rate applied to the attention weights.
        """

        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        # Learnable Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable Position Embedding: [cls_token, patch_1, ..., patch_N]
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.Sequential(*[
            EncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Identity() if num_classes is None else nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where
                                - B is batch size
                                - C is number of channels
                                - H is height
                                - W is width

        Returns:
            logits (torch.Tensor): Output tensor of shape (B, num_classes) if num_classes is specified,
                                   otherwise the output is the class token representation of shape (B, D).
        """

        B = x.shape[0]

        x = self.patch_embedding(x)  # shape: (B, N, D)

        # Prepend a class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, D)

        x = x + self.positional_encoding  # (B, 1+N, D)
        x = self.pos_drop(x)  # (B, 1+N, D)

        x = self.blocks(x)  # (B, 1+N, D)
        x = self.norm(x)  # (B, 1+N, D)

        cls_output = x[:, 0]  # (B, D)

        logits = self.head(cls_output)  # (B, num_classes) or (B, D) if num_classes is None
        return logits
