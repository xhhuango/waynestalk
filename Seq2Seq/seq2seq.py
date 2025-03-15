import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=PAD_INDEX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, input):
        """
        Args:
            input: (batch_size, seq_len)

        Returns
            output: (batch_size, seq_len, hidden_dim)
            hidden: (num_layers, batch_size, hidden_dim)
            cell: (num_layers, batch_size, hidden_dim)
        """
        embedding = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedding)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=PAD_INDEX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        """
        Args
            input: (batch_size,)
            hidden: (num_layers, batch_size, hidden_dim)
            cell: (num_layers, batch_size, hidden_dim)
        """
        embedding = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: float - probability to use teacher forcing
        """

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)

        _, hidden, cell = self.encoder(src)
        input = tgt[:, 0]
        for t in range(1, tgt_len):
            input = input.unsqueeze(1)
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = tgt[:, t] if teacher_force else output.argmax(1)

        return outputs


SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

SOS_INDEX = 0
EOS_INDEX = 1
PAD_INDEX = 2

english_sentences = [
    "hello world",
    "good morning",
    "i love you",
    "cat",
    "dog",
    "go home",
]

spanish_sentences = [
    "hola mundo",
    "buenos dias",
    "te amo",
    "gato",
    "perro",
    "ve a casa",
]


def build_vocab(sentences):
    vocab = list(set([word for sentence in sentences for word in sentence.split(" ")]))
    vocab = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + vocab
    tkn2idx = {tkn: i for i, tkn in enumerate(vocab)}
    idx2tkn = {i: tkn for tkn, i in tkn2idx.items()}
    return tkn2idx, idx2tkn


def convert_sentences_to_idx(sentences, tkn2idx):
    sentences_idx = [[tkn2idx[tkn] for tkn in sentence.split(" ")] for sentence in sentences]
    for sentence_idx in sentences_idx:
        sentence_idx.insert(0, tkn2idx[SOS_TOKEN])
        sentence_idx.append(tkn2idx[EOS_TOKEN])
    return sentences_idx


src_tkn2idx, src_idx2tkn = build_vocab(english_sentences)
tgt_tkn2idx, tgt_idx2tkn = build_vocab(spanish_sentences)

src_data = convert_sentences_to_idx(english_sentences, src_tkn2idx)
tgt_data = convert_sentences_to_idx(spanish_sentences, tgt_tkn2idx)
max_src_len = max([len(sentence) for sentence in src_data])
max_tgt_len = max([len(sentence) for sentence in tgt_data])
pair = []
for src, tgt in zip(src_data, tgt_data):
    src += [src_tkn2idx[PAD_TOKEN]] * (max_src_len - len(src))
    src_tensor = torch.tensor(src, dtype=torch.long)
    tgt += [tgt_tkn2idx[PAD_TOKEN]] * (max_tgt_len - len(tgt))
    tgt_tensor = torch.tensor(tgt, dtype=torch.long)
    pair.append((src_tensor, tgt_tensor))

EMBEDDING_DIM = 16
HIDDEN_DIM = 32
NUM_LAYERS = 4
LEARNING_RATE = 0.01
EPOCHS = 50

encoder = Encoder(len(src_idx2tkn), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
decoder = Decoder(len(tgt_idx2tkn), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
model = Seq2Seq(encoder, decoder)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)


def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for src_tensor, tgt_tensor in pair:
            src_tensor = src_tensor.unsqueeze(0)
            tgt_tensor = tgt_tensor.unsqueeze(0)

            optimizer.zero_grad()
            output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.5)

            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            tgt = tgt_tensor[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss {total_loss / len(pair)}")


def translate(sentence, max_length=10):
    model.eval()

    src_idx = convert_sentences_to_idx([sentence], src_tkn2idx)[0]
    src_idx += [src_tkn2idx[PAD_TOKEN]] * (max_src_len - len(src_idx))
    src_tensor = torch.tensor(src_idx, dtype=torch.long)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor)

    outputs = []
    input = torch.tensor([tgt_tkn2idx[SOS_TOKEN]], dtype=torch.long)
    for _ in range(max_length):
        with torch.no_grad():
            prediction, hidden, cell = model.decoder(input, hidden, cell)

        next_tkn_idx = prediction.argmax(1).item()
        if next_tkn_idx == EOS_INDEX:
            break

        outputs.append(next_tkn_idx)
        input = torch.tensor([next_tkn_idx], dtype=torch.long)

    return " ".join([tgt_idx2tkn[idx] for idx in outputs])


def translate_beam_search(sentence, beam_width=3, max_length=10):
    model.eval()

    src_idx = convert_sentences_to_idx([sentence], src_tkn2idx)[0]
    src_idx += [src_tkn2idx[PAD_TOKEN]] * (max_src_len - len(src_idx))
    src_tensor = torch.tensor(src_idx, dtype=torch.long)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor)

    beam = [([SOS_INDEX], hidden, cell, 0.0)]
    completed_sentences = []

    for _ in range(max_length):
        new_beam = []
        for tokens, hidden, cell, score in beam:
            if tokens[-1] == EOS_INDEX:
                completed_sentences.append((tokens, score))
                new_beam.append((tokens, hidden, cell, score))
                continue

            input_index = torch.tensor([tokens[-1]], dtype=torch.long)
            with torch.no_grad():
                output, hidden, cell = model.decoder(input_index, hidden, cell)
                log_probs = torch.log_softmax(output, dim=1).squeeze(0)

            topk = torch.topk(log_probs, beam_width)
            for tkn_idx, tkn_score in zip(topk.indices.tolist(), topk.values.tolist()):
                new_tokens = tokens + [tkn_idx]
                new_score = score + tkn_score
                new_beam.append((new_tokens, hidden, cell, new_score))

        new_beam.sort(key=lambda x: x[3], reverse=True)
        beam = new_beam[:beam_width]

    for tokens, hidden, cell, score in beam:
        if tokens[-1] != EOS_INDEX:
            completed_sentences.append((tokens, score))

    completed_sentences.sort(key=lambda x: x[1], reverse=True)
    best_tokens, best_score = completed_sentences[0]

    if best_tokens[0] == SOS_INDEX:
        best_tokens = best_tokens[1:]
    if EOS_INDEX in best_tokens:
        best_tokens = best_tokens[:best_tokens.index(EOS_INDEX)]

    return " ".join([tgt_idx2tkn[idx] for idx in best_tokens])


def test():
    test_sentences = [
        "hello world",
        "i love you",
        "cat",
        "go home",
    ]
    for sentence in test_sentences:
        translation = translate_beam_search(sentence)
        print(f"src: {sentence}, tgt: {translation}")


if __name__ == "__main__":
    train()
    test()
