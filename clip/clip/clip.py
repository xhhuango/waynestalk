import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor of shape [batch_size, sequence_length, d_model]
        """
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x):
        return self.resblocks(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, context_length, width, layers, heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.transformer = Transformer(width, layers, heads)
        self.ln_final = nn.LayerNorm(width)
        self.context_length = context_length

    def forward(self, text):
        """
        Parameters
        ----------
        text: Tensor of shape [batch_size, context_length] containing token indices.
        """

        x = self.token_embedding(text)  # [batch_size, context_length, width]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # [context_length, batch, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, context_length, width]
        x = self.ln_final(x)
        return x[:, -1, :]  # [batch, width]


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()  # [batch, 768]
        self.proj = nn.Linear(self.vit.hidden_dim, output_dim)

    def forward(self, x):
        features = self.vit(x)  # [batch, 768]
        features = self.proj(features)  # [batch, output_dim]
        return features


class CLIP(nn.Module):
    def __init__(self, image_embed_dim, text_vocab_size, context_length, text_width, text_layers, text_heads):
        super().__init__()
        self.image_encoder = ImageEncoder(image_embed_dim)
        self.text_encoder = TextEncoder(text_vocab_size, context_length, text_width, text_layers, text_heads)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_text(self, text):
        return self.text_encoder(text)

    def forward(self, image, text):
        image_features = F.normalize(self.encode_image(image), dim=-1)
        text_features = F.normalize(self.encode_text(text), dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


if __name__ == "__main__":
    vocab_size = 49408
    context_length = 77
    image_embed_dim = 512
    text_width = 512
    text_layers = 12
    text_heads = 8

    model = CLIP(
        image_embed_dim=image_embed_dim,
        text_vocab_size=vocab_size,
        context_length=context_length,
        text_width=text_width,
        text_layers=text_layers,
        text_heads=text_heads
    )

    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    text = torch.randint(0, vocab_size, (batch_size, context_length))

    logits_per_image, logits_per_text = model(image, text)
    print("logits_per_image shape:", logits_per_image.shape)
    print("logits_per_text shape:", logits_per_text.shape)
