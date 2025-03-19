from bahdanau_attention import *
from luong_attention import *

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

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
    vocab = ["<sos>", "<eos>", "<pad>"] + vocab
    tkn2idx = {tkn: idx for idx, tkn in enumerate(vocab)}
    idx2tkn = {idx: tkn for tkn, idx in tkn2idx.items()}
    return vocab, tkn2idx, idx2tkn


def convert_sentence_to_idx(sentence, tkn2idx):
    sentence_idx = [tkn2idx[tkn] for tkn in sentence.split(" ")]
    sentence_idx.insert(0, SOS_TOKEN)
    sentence_idx.append(EOS_TOKEN)
    return sentence_idx


def pad_sentence(sentence, max_len):
    return sentence + [PAD_TOKEN] * (max_len - len(sentence))


src_vocab, src_tkn2idx, src_idx2tkn = build_vocab(english_sentences)
tgt_vocab, tgt_tkn2idx, tgt_idx2tkn = build_vocab(spanish_sentences)
src_sentences = [convert_sentence_to_idx(sentence, src_tkn2idx) for sentence in english_sentences]
tgt_sentences = [convert_sentence_to_idx(sentence, tgt_tkn2idx) for sentence in spanish_sentences]
src_max_len = max([len(sentence) for sentence in src_sentences])
tgt_max_len = max([len(sentence) for sentence in tgt_sentences])
src_sentences = [pad_sentence(sentence, src_max_len) for sentence in src_sentences]
tgt_sentences = [pad_sentence(sentence, tgt_max_len) for sentence in tgt_sentences]

ENCODER_EMBEDDING_DIM = 16
ENCODER_HIDDEN_DIM = 32

DECODER_EMBEDDING_DIM = 16
DECODER_HIDDEN_DIM = 32

ATTENTION_HIDDEN_DIM = 32

LEARNING_RATE = 0.01
EPOCHS = 50

using = "luong"  # "bahdanau" or "luong"
loung_attention = "concat"  # "general", "concat", or "dot"

if using == "bahdanau":
    encoder = BahdanauAttentionEncoder(len(src_tkn2idx), ENCODER_EMBEDDING_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
    decoder = BahdanauAttentionDecoder(
        len(tgt_tkn2idx), DECODER_EMBEDDING_DIM, DECODER_HIDDEN_DIM, ENCODER_HIDDEN_DIM, ATTENTION_HIDDEN_DIM
    )
    model = BahdanauAttentionModel(encoder, decoder)
else:
    if loung_attention == "general":
        attention = LuongGeneralAttention(DECODER_HIDDEN_DIM)
    elif loung_attention == "concat":
        attention = LuongConcatAttention(DECODER_HIDDEN_DIM)
    else:
        attention = LuongDotAttention()
    encoder = LuongAttentionEncoder(len(src_vocab), ENCODER_EMBEDDING_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
    decoder = LuongAttentionDecoder(attention, len(tgt_vocab), DECODER_EMBEDDING_DIM, DECODER_HIDDEN_DIM)
    model = LuongAttentionModel(encoder, decoder)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)


def train():
    training_pairs = []
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_tensor = torch.tensor(src, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)
        training_pairs.append((src_tensor, tgt_tensor))

    for epoch in range(EPOCHS):
        total_loss = 0
        for src_tensor, tgt_tensor in training_pairs:
            src_tensor = src_tensor.unsqueeze(0)
            tgt_tensor = tgt_tensor.unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(src_tensor, tgt_tensor)

            outputs_dim = outputs.shape[-1]
            outputs = outputs[:, 1:, :].reshape(-1, outputs_dim)
            tgt = tgt_tensor[:, 1:].reshape(-1)
            loss = criterion(outputs, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {total_loss}")


def translate_beam_search(sentence, beam_width=3, max_length=10):
    model.eval()

    src_idx = convert_sentence_to_idx(sentence, src_tkn2idx)
    src_idx = pad_sentence(src_idx, src_max_len)
    src_tensor = torch.tensor(src_idx, dtype=torch.long)

    with torch.no_grad():
        src_tensor = src_tensor.unsqueeze(0)
        if using == "bahdanau":
            encoder_all_hidden, decoder_hidden = model.encoder(src_tensor)
        else:
            encoder_all_hidden, decoder_hidden, decoder_cell = model.encoder(src_tensor)

    if using == "bahdanau":
        beam = [([SOS_TOKEN], decoder_hidden, None, 0.0)]
    else:
        beam = [([SOS_TOKEN], decoder_hidden, decoder_cell, 0.0)]

    completed_sentences = []

    for _ in range(max_length):
        new_beam = []
        for tokens, decoder_hidden, decoder_cell, score in beam:
            if tokens[-1] == EOS_TOKEN:
                completed_sentences.append((tokens, score))
                new_beam.append((tokens, decoder_hidden, decoder_cell, score))
                continue

            input_index = torch.tensor([tokens[-1]], dtype=torch.long)
            with torch.no_grad():
                if using == "bahdanau":
                    prediction, decoder_hidden = model.decoder(input_index, decoder_hidden, encoder_all_hidden)
                else:
                    prediction, decoder_hidden, decoder_cell = model.decoder(
                        input_index, decoder_hidden, decoder_cell, encoder_all_hidden
                    )

                log_probs = torch.log_softmax(prediction, dim=1).squeeze(0)

            topk = torch.topk(log_probs, beam_width)
            for tkn_idx, tkn_score in zip(topk.indices.tolist(), topk.values.tolist()):
                new_tokens = tokens + [tkn_idx]
                new_score = score + tkn_score
                new_beam.append((new_tokens, decoder_hidden, decoder_cell, new_score))

        new_beam.sort(key=lambda x: x[3], reverse=True)
        beam = new_beam[:beam_width]

    for tokens, decoder_hidden, decoder_cell, score in beam:
        if tokens[-1] != EOS_TOKEN:
            completed_sentences.append((tokens, score))

    completed_sentences.sort(key=lambda x: x[1], reverse=True)
    best_tokens, best_score = completed_sentences[0]

    if best_tokens[0] == SOS_TOKEN:
        best_tokens = best_tokens[1:]
    if EOS_TOKEN in best_tokens:
        best_tokens = best_tokens[:best_tokens.index(EOS_TOKEN)]

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


if __name__ == '__main__':
    train()
    test()
