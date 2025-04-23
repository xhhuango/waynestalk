from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import GPT2TokenizerFast


# @dataclass
# class GPT2Config:
#     vocab_size: int = 50257
#     n_positions: int = 1024
#     d_model: int = 1600
#     n_layer: int = 48
#     n_head: int = 25
#     d_ff: int = 6400
#     residual_dropout_prob: float = 0.1
#     embedding_dropout_prob: float = 0.1
#     attention_dropout_prob: float = 0.1
#     layer_norm_epsilon: float = 1e-5
#     initializer_range: float = 0.02

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    d_ff: int = 3072
    residual_dropout_prob: float = 0.1
    embedding_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        self.residual_dropout = nn.Dropout(config.residual_dropout_prob)

        attention_mask = (torch.tril(torch.ones(config.n_positions, config.n_positions))
                          .view(1, 1, config.n_positions, config.n_positions))
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def forward(self, x):
        """
        Causal self-attention forward pass.

        Args
        ----
            x: (batch_size, seq_len, d_model)

        Returns
        -------
            output: (batch_size, seq_len, d_model)
        """

        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, d_model) for Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (batch_size, n_head, seq_len, d_head) for Q, K, V
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # Causal mask: we only allow attention to current and previous positions
        mask = self.attention_mask[:, :, :seq_len, :seq_len]
        attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        attention = attention.transpose(1, 2).contiguous()  # (batch_size, seq_len, n_head * d_head)
        attention = attention.view(batch_size, seq_len, -1)  # (batch_size, seq_len, d_model)

        # Linear projection
        output = self.W_o(attention)
        output = self.residual_dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask):
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # (batch_size, n_head, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)  # (batch_size, n_head, seq_len, seq_len)
        weights = self.attention_dropout(weights)
        attention = weights @ V  # (batch_size, n_head, seq_len, d_head)
        return attention, weights


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.residual_dropout_prob)

    def forward(self, x):
        """
        Feed-forward network forward pass.

        Args
        ----
            x: (batch_size, seq_len, d_model)

        Returns
        -------
            x: (batch_size, seq_len, d_model)
        """

        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ffn = FeedForward(config)

    def forward(self, x):
        """
        Transformer block forward pass.

        Args
        ----
            x: (batch_size, seq_len, d_model)

        Returns
        -------
            x: (batch_size, seq_len, d_model)
        """

        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.n_positions, config.d_model)
        self.dropout = nn.Dropout(config.embedding_dropout_prob)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(self, input_ids, position_ids=None):
        """
        Model forward pass.

        Args
        ----
            input_ids: (batch_size, seq_len)
            position_ids: (batch_size, seq_len)

        Returns
        -------
            hidden_states: (batch_size, seq_len, d_model)
        """

        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.dropout(self.wte(input_ids) + self.wpe(position_ids))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.layer_norm(hidden_states)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

    def forward(self, input_ids, labels=None):
        """
        Language model forward pass.

        Args
        ----
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len)

        Returns
        -------
            logits: (batch_size, seq_len, vocab_size)
            loss: (optional) cross-entropy loss
        """

        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits, loss


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT‑2 has no pad token; reuse <EOS>
    return tokenizer


class PretrainingDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, seq_len: int = 1024):
        super().__init__()
        self.seq_len = seq_len
        all_ids: List[int] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_ids.extend(tokenizer.encode(line))

        # drop the tail so we have an exact multiple of seq_len
        total = len(all_ids) - (len(all_ids) % seq_len)
        self.data = torch.tensor(all_ids[:total], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x


def train_loop(model, dataloader, epochs: int, lr: float, save_dir: Path):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=len(dataloader) * epochs
    )
    global_step = 0
    best_loss = float("inf")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            sys.stdout.write(f"\rEpoch {epoch} Batch {batch_idx} / {len(dataloader)}")
            sys.stdout.flush()

            opt.zero_grad()
            _, loss = model(input_ids, labels=labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / len(dataloader)
        print(f",\ttrain_loss={avg_loss:.4f}")

        # simple checkpointing
        torch.save(model.state_dict(), save_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / "best.pt")


def pretrain():
    tokenizer = get_tokenizer()
    dataset = PretrainingDataset(tokenizer, "wikitext2_train.txt")
    dataloader = DataLoader(dataset, batch_size=8, sampler=RandomSampler(dataset))
    model = GPT2LMHeadModel(GPT2Config())
    train_loop(model, dataloader, epochs=5, lr=2.5e-4, save_dir=Path("./model"))


fine_tuning_examples = [
    {"prompt": "Q: Who wrote Frankenstein?", "response": "Mary Shelley."},
    {"prompt": "Translate to Spanish: Hello!", "response": "Hola！"},
    {"prompt": "Summarize: GPT‑2 is", "response": "GPT‑2 is a language model released by OpenAI."}
]


class PromptResponseDataset(Dataset):
    def __init__(self, tokenizer, json: list[dict[str, str]], seq_len: int = 1024, eos_token: str = "\n"):
        self.seq_len = seq_len
        examples: List[torch.Tensor] = []
        for obj in json:
            text = obj["prompt"] + eos_token + obj["response"] + eos_token
            tokens = tokenizer.encode(text)[: seq_len]
            padding = [tokenizer.eos_token_id] * (seq_len - len(tokens))
            examples.append(torch.tensor(tokens + padding, dtype=torch.long))
        self.data = torch.stack(examples)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x


def fine_tune():
    tokenizer = get_tokenizer()
    dataset = PromptResponseDataset(tokenizer, fine_tuning_examples)
    dataloader = DataLoader(dataset, batch_size=4, sampler=RandomSampler(dataset))
    model = GPT2LMHeadModel(GPT2Config())
    model.load_state_dict(torch.load("./model/best.pt", map_location="cpu", weights_only=True))
    train_loop(model, dataloader, epochs=5, lr=2.5e-4, save_dir=Path("./model_fine_tuned"))


def sample(
    model: GPT2LMHeadModel, tokenizer, prompt: str, max_new_tokens: int = 100,
    temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9,
) -> str:
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids[:, -model.transformer.config.n_positions:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.where(logits >= kth_vals, torch.full_like(logits, -float("inf")))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = sorted_probs.cumsum(dim=-1)

                keep_mask = cumprobs <= top_p
                keep_mask[..., 0] = True  # always keep the best token
                remove_idx = sorted_idx[~keep_mask]

                logits[0, remove_idx] = -float("inf")

            probs = torch.softmax(logits, dim=-1)

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate(model_path):
    model = GPT2LMHeadModel(GPT2Config())
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    tokenizer = get_tokenizer()
    prompt = "Q: Who wrote Frankenstein? A:"
    output = sample(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=0, top_p=0.9)
    print(f"Prompt=> {prompt}")
    print(f"Output=> {output}")


if __name__ == "__main__":
    pretrain()
    # generate("./model_fine_tuned/best.pt")
    # fine_tune()
