import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Embeddings(nn.Module):
    def __init__(self, vocab_size, token_type_size, max_position_embeddings, hidden_dim, dropout_prob):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.token_type_embeddings = nn.Embedding(token_type_size, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        Compute the embeddings for the input tokens.

        Args
            x: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)

        Returns
            embeddings: (batch_size, seq_len, hidden_dim)
        """

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (1, seq_len) -> (batch_size, seq_len)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.all_head_size = hidden_dim

        self.query = nn.Linear(hidden_dim, self.all_head_size, bias=False)
        self.key = nn.Linear(hidden_dim, self.all_head_size, bias=False)
        self.value = nn.Linear(hidden_dim, self.all_head_size, bias=False)

        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, mask=None):
        """
        Multi-head attention forward pass.

        Args
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, 1, 1, seq_len)
                  0 for real tokens, -inf for padding tokens

        Returns
            hidden_states: (batch_size, seq_len, hidden_dim)
        """

        query = self.transpose_for_scores(self.query(hidden_states))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(hidden_states))  # (batch_size, num_heads, seq_len, head_size)
        value = self.transpose_for_scores(self.value(hidden_states))  # (batch_size, num_heads, seq_len, head_size)

        # Scaled dot-product attention
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores + mask
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        attention = attention_weights @ value  # (batch_size, num_heads, seq_len, head_size)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_size)
        new_shape = attention.size()[:-2] + (self.all_head_size,)
        attention = attention.view(*new_shape)  # (batch_size, seq_len, all_head_size)

        # Linear projection
        projection_output = self.output(attention)  # (batch_size, seq_len, hidden_dim)
        projection_output = self.dropout(projection_output)

        hidden_states = self.norm(hidden_states + projection_output)
        return hidden_states

    def transpose_for_scores(self, x):
        """
        Args
            x: (batch_size, seq_len, all_head_size)
        Returns
            (batch_size, num_heads, seq_len, head_size)
        """

        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)  # (batch_size, seq_len, num_heads, head_size)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, hidden_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        """
        Feed-forward network forward pass.

        Args
            hidden_states: (batch_size, seq_len, hidden_dim)

        Returns
            hidden_states: (batch_size, seq_len, hidden_dim)
        """

        hidden_states = self.linear2(self.activation(self.linear1(hidden_states)))
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, d_ff, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, hidden_dim, dropout_prob)
        self.ffn = PositionwiseFeedForward(hidden_dim, d_ff)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, mask=None):
        """
        Encoder layer forward pass.

        Args
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, 1, seq_len)
                  0 for real tokens, -inf for padding tokens

        Returns
            hidden_states: (batch_size, seq_len, hidden_dim)
        """

        # Multi-head attention
        attention_output = self.multi_head_attention(hidden_states, mask=mask)

        # Feed-forward network
        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout(ffn_output)
        hidden_states = self.norm(hidden_states + ffn_output)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, d_ff, dropout_prob):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(num_heads, hidden_dim, d_ff, dropout_prob) for _ in range(num_layers)]
        )

    def forward(self, hidden_states, mask=None):
        """
        Encoder forward pass.

        Args
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, 1, seq_len)
                  0 for real tokens, -inf for padding tokens

        Returns
            hidden_states: (batch_size, seq_len, hidden_dim)
        """

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return hidden_states


class Pooler(nn.Module):
    def __init__(self, hidden_dim):
        super(Pooler, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        """
        Pooler forward pass.

        Args
            hidden_states: (batch_size, seq_len, hidden_dim)

        Returns
            pooled_output: (batch_size, hidden_dim)
        """

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.linear(first_token_tensor)
        pooled_output = F.tanh(pooled_output)
        return pooled_output


class Bert(nn.Module):
    def __init__(
        self, vocab_size, token_type_size, max_position_embeddings, hidden_dim, num_layers, num_heads, d_ff,
        dropout_prob
    ):
        super(Bert, self).__init__()
        self.embeddings = Embeddings(vocab_size, token_type_size, max_position_embeddings, hidden_dim, dropout_prob)
        self.encoder = Encoder(hidden_dim, num_layers, num_heads, d_ff, dropout_prob)
        self.pooler = Pooler(hidden_dim)

    def forward(self, input_ids, token_type_ids=None, mask=None):
        """
        Forward pass for the BERT model.

        Args
            input_ids: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            mask: (batch_size, seq_len)

        Returns
            encoder_output: (batch_size, seq_len, hidden_dim)
            pooled_output: (batch_size, hidden_dim)
        """

        if mask is not None:
            extended_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.to(dtype=torch.float32)
            # Convert 1 -> 0, 0 -> large negative (mask out)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, mask=extended_mask)
        pooled_output = self.pooler(encoder_output)

        return encoder_output, pooled_output


# Pre-training BERT - Masked LM (MLM) and Next Sentence Prediction (NSP)
class PreTrainingHeads(nn.Module):
    def __init__(self, vocab_size, hidden_dim, bert_embedding_weights):
        super(PreTrainingHeads, self).__init__()

        # MLM head
        self.predictions = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.pred_bias = nn.Parameter(torch.zeros(vocab_size))
        # Tie weights to the input embeddings matrix
        self.predictions.weight = bert_embedding_weights

        # NSP head
        self.seq_relationship = nn.Linear(hidden_dim, 2)

    def forward(self, encoder_output, pooled_output):
        """
        Args
            encoder_output: (batch_size, seq_len, hidden_dim)
            pooled_output: (batch_size, hidden_dim)

        Returns
            prediction_scores: (batch_size, seq_len, vocab_size)
            seq_relationship_scores: (batch_size, 2)
        """

        # MLM head
        prediction_scores = self.predictions(encoder_output) + self.pred_bias

        # NSP head
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertForPreTraining(nn.Module):
    def __init__(
        self, vocab_size, token_type_size, max_position_embeddings, hidden_dim, num_layers, num_heads, d_ff,
        dropout_prob
    ):
        super(BertForPreTraining, self).__init__()
        self.bert = Bert(
            vocab_size, token_type_size, max_position_embeddings, hidden_dim, num_layers, num_heads, d_ff, dropout_prob
        )
        # Tying the MLM head's weight to the word embedding
        self.cls = PreTrainingHeads(vocab_size, hidden_dim, self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, token_type_ids=None, mask=None):
        """
        Pre-training BERT

        Args
            input_ids: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            mask: (batch_size, seq_len)

        Returns
            prediction_scores: (batch_size, seq_len, vocab_size)
            seq_relationship_scores: (batch_size, 2)
        """

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, mask=mask)
        prediction_scores, seq_relationship_scores = self.cls(sequence_output, pooled_output)
        return prediction_scores, seq_relationship_scores


class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, num_labels, hidden_dim):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        # A classification head: we typically use the [CLS] pooled output
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, token_type_ids=None, mask=None, labels=None):
        """
        Sequence classification with BERT

        Args
            input_ids: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            mask: (batch_size, seq_len)
            labels: (batch_size)

        Returns
            logits: (batch_size, num_classes)
            loss: (optional) Cross entropy loss
        """

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, mask=mask)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


tkn2idx = {
    "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3,
    "i": 4, "like": 5, "dogs": 6, "cats": 7,
    "they": 8, "are": 9, "playful": 10,
    "[UNK]": 11,
}

idx2tkn = {v: k for k, v in tkn2idx.items()}


def tokenize(text):
    tokens = text.split()
    token_ids = [tkn2idx.get(t, tkn2idx["[UNK]"]) for t in tokens]
    return token_ids


corpus = [
    "i like dogs",
    "they are playful",
    "i like cats",
    "they are cute"
]


def create_example_for_mlm_nsp(sentence_a, sentence_b, is_next, max_seq_len=12, mask_prob=0.15):
    cls_id = tkn2idx["[CLS]"]
    sep_id = tkn2idx["[SEP]"]
    mask_id = tkn2idx["[MASK]"]

    tokens_a = tokenize(sentence_a)
    tokens_b = tokenize(sentence_b)

    input_ids = [cls_id] + tokens_a + [sep_id] + tokens_b + [sep_id]
    token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        token_type_ids = token_type_ids[:max_seq_len]

    # -100 for non-masked positions, and the original token for masked positions
    mlm_labels = [-100] * len(input_ids)

    num_to_mask = max(1, int((len(input_ids) - 3) * mask_prob))  # 3 for [CLS], [SEP], [SEP]
    candidate_mask_positions = [i for i, tid in enumerate(input_ids) if tid not in [cls_id, sep_id]]
    random.shuffle(candidate_mask_positions)
    mask_positions = candidate_mask_positions[:num_to_mask]

    for pos in mask_positions:
        mlm_labels[pos] = input_ids[pos]

        # BERT strategy: 80% replace with [MASK], 10% random, 10% keep
        r = random.random()
        if r < 0.8:
            input_ids[pos] = mask_id
        elif r < 0.9:
            input_ids[pos] = random.randint(4, len(tkn2idx) - 2)  # exclude special tokens
        else:
            pass

    nsp_label = 1 if is_next else 0
    return input_ids, token_type_ids, mlm_labels, nsp_label


def build_pretraining_dataset(corpus, num_examples):
    dataset = []
    n = len(corpus)
    for _ in range(num_examples):
        idx_a = random.randint(0, n - 1)
        sentence_a = corpus[idx_a]

        # 50%: pick a real next sentence; 50%: pick a random sentence
        if random.random() < 0.5:
            idx_b = (idx_a + 1) % n
            sentence_b = corpus[idx_b]
            is_next = True
        else:
            idx_b = random.randint(0, n - 1)
            while idx_b == idx_a:
                idx_b = random.randint(0, n - 1)
            sentence_b = corpus[idx_b]
            is_next = False

        input_ids, token_type_ids, mlm_labels, nsp_label = create_example_for_mlm_nsp(sentence_a, sentence_b, is_next)
        dataset.append((input_ids, token_type_ids, mlm_labels, nsp_label))

    return dataset


def collate_pretraining_batch(examples):
    pad_id = tkn2idx["[PAD]"]
    max_len = max(len(ex[0]) for ex in examples)

    batch_input_ids = []
    batch_token_type_ids = []
    batch_mlm_labels = []
    batch_nsp_labels = []
    batch_mask = []

    for (input_ids, token_type_ids, mlm_labels, nsp_label) in examples:
        seq_len = len(input_ids)
        pad_len = max_len - seq_len
        batch_input_ids.append(input_ids + [pad_id] * pad_len)
        batch_token_type_ids.append(token_type_ids + [0] * pad_len)
        batch_mlm_labels.append(mlm_labels + [-100] * pad_len)
        batch_nsp_labels.append(nsp_label)
        batch_mask.append([1] * seq_len + [0] * pad_len)

    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
    batch_mlm_labels = torch.tensor(batch_mlm_labels, dtype=torch.long)
    batch_nsp_labels = torch.tensor(batch_nsp_labels, dtype=torch.long)
    batch_mask = torch.tensor(batch_mask, dtype=torch.long)
    return batch_input_ids, batch_token_type_ids, batch_mlm_labels, batch_nsp_labels, batch_mask


def pretrain_bert():
    dataset = build_pretraining_dataset(corpus, num_examples=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_pretraining_batch)

    model = BertForPreTraining(
        vocab_size=len(tkn2idx),
        token_type_size=2,
        max_position_embeddings=64,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        dropout_prob=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    EPOCHS = 100

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids, token_type_ids, mlm_labels, nsp_labels, mask = batch
            optimizer.zero_grad()

            prediction_scores, seq_relationship_scores = model(input_ids, token_type_ids, mask)

            mlm_loss = F.cross_entropy(prediction_scores.view(-1, len(tkn2idx)), mlm_labels.view(-1), ignore_index=-100)
            nsp_loss = F.cross_entropy(seq_relationship_scores.view(-1, 2), nsp_labels.view(-1))
            loss = mlm_loss + nsp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    return model


def test_pretrain_bert(model):
    sent_a = "i like [MASK]"
    sent_b = "they are playful"

    input_ids, token_type_ids, mlm_labels, nsp_label = create_example_for_mlm_nsp(sent_a, sent_b, is_next=True)
    test_batch = collate_pretraining_batch([(input_ids, token_type_ids, mlm_labels, nsp_label)])
    input_ids_batch, token_type_ids_batch, mlm_labels_batch, nsp_labels_batch, mask_batch = test_batch

    model.eval()
    with torch.no_grad():
        prediction_scores, seq_relationship_scores = model(input_ids_batch, token_type_ids_batch, mask_batch)

    masked_index = (torch.tensor(input_ids) == tkn2idx["[MASK]"]).nonzero(as_tuple=True)[0]
    if len(masked_index) > 0:
        # We'll just look at the first masked token
        mask_position = masked_index[0].item()
        logits = prediction_scores[0, mask_position]  # shape [vocab_size]
        probs = F.softmax(logits, dim=-1)
        top5 = torch.topk(probs, 5)
        print("Top 5 predictions for [MASK]:")
        for prob, idx in zip(top5.values, top5.indices):
            print(f"  Token='{idx2tkn[idx.item()]}' prob={prob.item():.4f}")

    nsp_prob = F.softmax(seq_relationship_scores[0], dim=-1)
    print("NSP probabilities =", nsp_prob)


# 1: positive, 0: negative
sentiment_data = [
    ("i like dogs", 1),
    ("i like cats", 1),
    ("they are playful", 1),
    ("they are bad", 0),  # 'bad' not in vocab, will become [UNK]
    ("i like [UNK]", 0),  # random negative label
]


def create_example_for_classification(sentence):
    cls_id = tkn2idx["[CLS]"]
    sep_id = tkn2idx["[SEP]"]

    tokens = tokenize(sentence)

    input_ids = [cls_id] + tokens + [sep_id]
    token_type_ids = [0] * (len(tokens) + 2)

    return input_ids, token_type_ids


def build_sentiment_dataset(data):
    examples = []
    for sentence, label in data:
        input_ids, token_type_ids = create_example_for_classification(sentence)
        examples.append((input_ids, token_type_ids, label))
    return examples


def collate_sentiment_batch(examples):
    pad_id = tkn2idx["[PAD]"]
    max_len = max(len(ex[0]) for ex in examples)

    batch_input_ids = []
    batch_token_type_ids = []
    batch_labels = []
    batch_mask = []

    for (input_ids, token_type_ids, label) in examples:
        seq_len = len(input_ids)
        pad_len = max_len - seq_len
        batch_input_ids.append(input_ids + [pad_id] * pad_len)
        batch_token_type_ids.append(token_type_ids + [0] * pad_len)
        batch_labels.append(label)
        batch_mask.append([1] * seq_len + [0] * pad_len)

    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    batch_mask = torch.tensor(batch_mask, dtype=torch.long)
    return batch_input_ids, batch_token_type_ids, batch_labels, batch_mask


def fine_tune_bert(bert):
    dataset = build_sentiment_dataset(sentiment_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_sentiment_batch)

    model = BertForSequenceClassification(bert, num_labels=2, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    EPOCHS = 100

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids, token_type_ids, labels, mask = batch
            optimizer.zero_grad()

            logits, loss = model(input_ids, token_type_ids, mask, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    return model


def test_fine_tune_bert(model):
    text = "i like dogs"

    input_ids, token_type_ids = create_example_for_classification(text)
    mask = [1] * len(input_ids)

    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    token_type_ids_tensor = torch.tensor([token_type_ids], dtype=torch.long)
    mask_tensor = torch.tensor([mask], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits, loss = model(input_ids_tensor, token_type_ids_tensor, mask_tensor)

    probs = F.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()

    print("Probabilities =", probs)
    print("Predicted label =", predicted_label)


if __name__ == "__main__":
    pretrain_model = pretrain_bert()
    test_pretrain_bert(pretrain_model)
    fine_tune_model = fine_tune_bert(pretrain_model.bert)
    test_fine_tune_bert(fine_tune_model)
