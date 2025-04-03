import math

import torch
from torch import nn


class SharedEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(SharedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # (vocab_size, d_model)

    def forward(self, x):
        """
        Shared embedding layer.

        Args
            x: (batch_size, seq_len)

        Returns
            x: (batch_size, seq_len, d_model)
        """

        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, start_pos=0):
        """
        Add positional encoding to input tensor.

        Args
            x: (batch_size, seq_len, d_model)
            start_pos: int

        Returns
            x: (batch_size, seq_len, d_model)
        """

        seq_len = x.size(1)
        x = x + self.pe[start_pos:start_pos + seq_len, :].unsqueeze(0)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h_heads == 0, "d_model must be divisible by h_heads"

        self.d_model = d_model
        self.h_heads = h_heads
        self.d_k = d_model // h_heads
        self.d_v = d_model // h_heads

        self.W_q = nn.Linear(d_model, h_heads * self.d_k, bias=False)  # (d_model, h_heads * d_k)
        self.W_k = nn.Linear(d_model, h_heads * self.d_k, bias=False)  # (d_model, h_heads * d_k)
        self.W_v = nn.Linear(d_model, h_heads * self.d_v, bias=False)  # (d_model, h_heads * d_v)
        self.W_o = nn.Linear(h_heads * self.d_v, d_model, bias=False)  # (h_heads * d_v, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Multi-head attention forward pass.

        Args
            q: (batch_size, seq_len, d_model)
            k: (batch_size, seq_len, d_model)
            v: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (1, seq_len, seq_len)

        Returns
            x: (batch_size, seq_len, d_model)
        """

        batch_size, Q_len, _ = q.size()
        batch_size, K_len, _ = k.size()
        batch_size, V_len, _ = v.size()

        # Linear projections
        Q = self.W_q(q)  # (batch_size, Q_len, h_heads * d_k)
        K = self.W_k(k)  # (batch_size, K_len, h_heads * d_k)
        V = self.W_v(v)  # (batch_size, V_len, h_heads * d_v)

        Q = Q.view(batch_size, Q_len, self.h_heads, self.d_k).transpose(1, 2)  # (batch_size, h_heads, Q_len, d_k)
        K = K.view(batch_size, K_len, self.h_heads, self.d_k).transpose(1, 2)  # (batch_size, h_heads, K_len, d_k)
        V = V.view(batch_size, V_len, self.h_heads, self.d_v).transpose(1, 2)  # (batch_size, h_heads, V_len, d_v)

        # Scaled dot-product attention
        if mask is not None:
            mask = mask.unsqueeze(1)
        attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous()  # (batch_size, Q_len, h_heads, d_v)
        attention = attention.view(batch_size, Q_len, self.d_model)  # (batch_size, Q_len, d_model)

        # Linear projection
        output = self.W_o(attention)  # (batch_size, Q_len, d_model)

        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention.

        Args
            Q: (batch_size, h_heads, Q_len, d_k)
            K: (batch_size, h_heads, K_len, d_k)
            V: (batch_size, h_heads, V_len, d_v)
            mask: (batch_size, 1, Q_len, K_len)

        Returns
            attention: (batch_size, h_heads, Q_len, d_v)
            attention_weights: (batch_size, h_heads, Q_len, K_len)
        """

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch_size, h_heads, Q_len, K_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, h_heads, Q_len, K_len)
        attention = attention_weights @ V  # (batch_size, h_heads, Q_len, d_v)
        return attention, attention_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        """
        Position-wise feed forward pass.

        Args
            x: (batch_size, seq_len, d_model)

        Returns
            x: (batch_size, seq_len, d_model)
        """

        return self.linear2(torch.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, d_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Encoder layer forward pass.

        Args
            x: (batch_size, src_len, d_model)
            mask: (batch_size, 1, src_len)

        Returns
            x: (batch_size, src_len, d_model)
        """

        # Multi-head attention
        attention = self.multi_head_attention(x, x, x, mask=mask)
        x = self.norm1(x + attention)  # Residual connection and layer normalization

        # Position-wise feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual connection and layer normalization

        return x


class Encoder(nn.Module):
    def __init__(self, shared_embedding, d_model, n_layers, h_heads, d_ff, max_len):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = shared_embedding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h_heads, d_ff) for _ in range(n_layers)])

    def forward(self, src, src_mask=None):
        """
        Encoder forward pass.

        Args
            src: (batch_size, src_len)
            src_mask: (batch_size, 1, src_len)

        Returns
            x: (batch_size, src_len, d_model)
        """

        x = self.embedding(src)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, d_heads)
        self.cross_multi_head_attention = MultiHeadAttention(d_model, d_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Decoder layer forward pass.

        Args
            x: (batch_size, tgt_len, d_model)
            encoder_output: (batch_size, src_len, d_model)
            tgt_mask: (1, tgt_len, tgt_len)
            memory_mask: (batch_size, 1, src_len)

        Returns
            x: (batch_size, tgt_len, d_model)
        """

        # Mask multi-head attention
        masked_attention = self.masked_multi_head_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + masked_attention)

        # Cross multi-head attention
        cross_attention = self.cross_multi_head_attention(x, encoder_output, encoder_output, mask=memory_mask)
        x = self.norm2(x + cross_attention)

        # Position-wise feed forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x


class Decoder(nn.Module):
    def __init__(self, shared_embedding, d_model, n_layers, h_heads, d_ff, vocab_size, max_len):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = shared_embedding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, h_heads, d_ff) for _ in range(n_layers)])
        self.output_linear = nn.Linear(d_model, vocab_size, bias=False)
        self.output_linear.weight = self.embedding.embedding.weight

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Decoder forward pass.

        Args
            tgt: (batch_size, tgt_len)
            encoder_output: (batch_size, src_len, d_model)
            tgt_mask: (1, tgt_len, tgt_len)
            memory_mask: (batch_size, 1, src_len)

        Returns
            logits: (batch_size, tgt_len, vocab_size)
        """

        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.output_linear(x)
        return logits


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, h_heads, d_ff, max_len):
        super(Transformer, self).__init__()
        shared_embedding = SharedEmbedding(vocab_size, d_model)
        self.encoder = Encoder(shared_embedding, d_model, n_layers, h_heads, d_ff, max_len)
        self.decoder = Decoder(shared_embedding, d_model, n_layers, h_heads, d_ff, vocab_size, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Transformer forward pass.

        Args
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            src_mask: (batch_size, 1, src_len)
            tgt_mask: (1, tgt_len, tgt_len)
            memory_mask: (batch_size, 1, src_len)
        """

        encoder_output = self.encoder(src, src_mask)
        logits = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)
        return logits


data = [
    ("hello world", "hola mundo"),
    ("i love you", "te amo"),
    ("the cat is black", "el gato es negro"),
    ("good morning", "buenos dias"),
    ("this is a book", "este es un libro"),
    ("what is your name", "como te llamas"),
]

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2


def build_single_vocab(pairs):
    words = set()
    for (src, tgt) in pairs:
        for w in src.lower().split():
            words.add(w)
        for w in tgt.lower().split():
            words.add(w)

    vocab = ["<pad>", "<sos>", "<eos>"] + sorted(list(words))
    tkn2idx = {tkn: idx for idx, tkn in enumerate(vocab)}
    idx2tkn = {idx: tkn for tkn, idx in tkn2idx.items()}
    return vocab, tkn2idx, idx2tkn


def sentence_to_idx(sentence, tkn2idx):
    return [tkn2idx[w] for w in sentence.lower().split()]


def idx_to_sentence(idx, idx2tkn):
    return " ".join(idx2tkn[i] for i in idx)


def encode_pair(src, tgt, tkn2idx, max_len):
    src_idx = sentence_to_idx(src, tkn2idx)
    tgt_idx = sentence_to_idx(tgt, tkn2idx)

    tgt_in_idx = [SOS_INDEX] + tgt_idx
    tgt_out_idx = tgt_idx + [EOS_INDEX]

    src_idx = src_idx[:max_len]
    tgt_in_idx = tgt_in_idx[:max_len]
    tgt_out_idx = tgt_out_idx[:max_len]

    src_idx += [PAD_INDEX] * (max_len - len(src_idx))
    tgt_in_idx += [PAD_INDEX] * (max_len - len(tgt_in_idx))
    tgt_out_idx += [PAD_INDEX] * (max_len - len(tgt_out_idx))

    return src_idx, tgt_in_idx, tgt_out_idx


def create_dataset(pairs, tkn2idx, max_len):
    src_data, tgt_in_data, tgt_out_data = [], [], []
    for (src, tgt) in pairs:
        src_idx, tgt_in_idx, tgt_out_idx = encode_pair(src, tgt, tkn2idx, max_len)
        src_data.append(src_idx)
        tgt_in_data.append(tgt_in_idx)
        tgt_out_data.append(tgt_out_idx)
    return (
        torch.tensor(src_data, dtype=torch.long),
        torch.tensor(tgt_in_data, dtype=torch.long),
        torch.tensor(tgt_out_data, dtype=torch.long),
    )


vocab, tkn2idx, idx2tkn = build_single_vocab(data)
vocab_size = len(vocab)

D_MODEL = 512
N_LAYERS = 6
H_HEADS = 8
D_FF = 2048
MAX_LEN = 20

EPOCHS = 100


def create_padding_mask(seq):
    """
    Args
        seq: (batch_size, seq_len)

    Returns
        mask: (batch_size, 1, seq_len) - 1 for valid token, 0 for padding token
    """
    return (seq != PAD_INDEX).unsqueeze(1).long()


def create_subsequence_mask(size):
    """
    Args
        size: int

    Returns
        mask: (1, size, size) - 1 for valid token, 0 for padding token
    """
    return torch.tril(torch.ones((size, size))).unsqueeze(0)


def train():
    model = Transformer(vocab_size, D_MODEL, N_LAYERS, H_HEADS, D_FF, MAX_LEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    src_data, tgt_in_data, tgt_out_data = create_dataset(data, tkn2idx, MAX_LEN)
    model.train()

    for epoch in range(EPOCHS):
        src_mask = create_padding_mask(src_data)  # (batch_size, 1, MAX_LEN)
        tgt_mask = create_subsequence_mask(tgt_in_data.size(1))  # (1, MAX_LEN, MAX_LEN)
        memory_mask = create_padding_mask(src_data)  # (batch_size, 1, MAX_LEN)

        # (batch_size, MAX_LEN, vocab_size)
        logits = model(src_data, tgt_in_data, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)

        logits = logits.reshape(-1, vocab_size)  # (batch_size * MAX_LEN, vocab_size)
        tgt_out = tgt_out_data.reshape(-1)  # (batch_size * MAX_LEN)

        loss = criterion(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

    return model


def translate(model, sentence):
    model.eval()

    src_idx, _, _ = encode_pair(sentence, "", tkn2idx, MAX_LEN)
    src_tensor = torch.tensor([src_idx], dtype=torch.long)
    src_mask = create_padding_mask(src_tensor)

    encoder_output = model.encoder(src_tensor, src_mask)  # (batch_size, src_len, d_model)

    memory_mask = create_padding_mask(src_tensor)
    ys = torch.tensor([[SOS_INDEX]], dtype=torch.long)  # (batch_size, tgt_len)

    for i in range(MAX_LEN):
        tgt_mask = create_subsequence_mask(ys.size(1))
        # (batch_size, tgt_len, vocab_size)
        logits = model.decoder(ys, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        next_token_idx = torch.argmax(next_token_logits, dim=1, keepdim=True)  # (batch_size, 1)
        ys = torch.cat([ys, next_token_idx], dim=1)

        if next_token_idx.item() == EOS_INDEX:
            break

    decoded_idx = ys.squeeze(0).tolist()
    if decoded_idx[0] == SOS_INDEX:
        decoded_idx = decoded_idx[1:]
    if EOS_INDEX in decoded_idx:
        eos_pos = decoded_idx.index(EOS_INDEX)
        decoded_idx = decoded_idx[:eos_pos]

    return idx_to_sentence(decoded_idx, idx2tkn)


def translate_beam_search(model, sentence, beam_width=3):
    model.eval()

    src_idx, _, _ = encode_pair(sentence, "", tkn2idx, MAX_LEN)
    src_tensor = torch.tensor([src_idx], dtype=torch.long)
    src_mask = create_padding_mask(src_tensor)

    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)  # (batch_size, src_len, d_model)

    memory_mask = create_padding_mask(src_tensor)

    beam = [([SOS_INDEX], 0.0)]
    completed_sentences = []

    for i in range(MAX_LEN):
        new_beam = []
        for tokens, score in beam:
            if tokens[-1] == EOS_INDEX:
                completed_sentences.append((tokens, score))
                new_beam.append((tokens, score))
                continue

            ys = torch.tensor([tokens], dtype=torch.long)
            tgt_mask = create_subsequence_mask(ys.size(1))
            with torch.no_grad():
                # (batch_size, tgt_len, vocab_size)
                logits = model.decoder(ys, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            log_probs = torch.log_softmax(next_token_logits, dim=1).squeeze(0)
            topk = torch.topk(log_probs, beam_width)
            for tkn_idx, tkn_score in zip(topk.indices.tolist(), topk.values.tolist()):
                new_tokens = tokens + [tkn_idx]
                new_score = score + tkn_score
                new_beam.append((new_tokens, new_score))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    for tokens, score in beam:
        if tokens[-1] != EOS_INDEX:
            completed_sentences.append((tokens, score))

    completed_sentences.sort(key=lambda x: x[1], reverse=True)
    best_tokens, best_score = completed_sentences[0]

    if best_tokens[0] == SOS_INDEX:
        best_tokens = best_tokens[1:]
    if EOS_INDEX in best_tokens:
        best_tokens = best_tokens[:best_tokens.index(EOS_INDEX)]

    return " ".join([idx2tkn[idx] for idx in best_tokens])


if __name__ == "__main__":
    test_sentences = [
        "hello world",
        "the cat is black",
        "good morning",
        "what is your name",
        "this is a book",
        "i love you",
        "i love cat",
        "this is a cat",
    ]

    model = train()
    for sentence in test_sentences:
        translation = translate_beam_search(model, sentence)
        print(f"Input: {sentence}, Translation: {translation}")
