import torch
from torch import nn


class BahdanauAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim):
        super(BahdanauAttentionEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, encoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

    def forward(self, src):
        """
        Args
            src: (batch_size, src_len)

        Returns
            all_hidden: (batch_size, src_len, encoder_hidden_dim * 2)
            decoder_hidden_init: (batch_size, decoder_hidden_dim)
        """

        embedded = self.embedding(src)  # (batch_size, src_len, embed_dim)
        # all_hidden: (batch_size, src_len, hidden_dim * 2)
        # final_hidden_state: (num_layers * 2, batch_size, hidden_dim)
        all_hidden, final_hidden = self.gru(embedded)

        # Map to decoder's initial hidden state
        final_forward_hidden = final_hidden[-2, :, :]
        final_backward_hidden = final_hidden[-1, :, :]
        hidden_cat = torch.cat((final_forward_hidden, final_backward_hidden), dim=1)
        decoder_hidden_init = self.fc(hidden_cat)  # (batch_size, decoder_hidden_dim)

        return all_hidden, decoder_hidden_init


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(decoder_hidden_dim, attention_dim)
        self.U = nn.Linear(encoder_hidden_dim * 2, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_all_hidden):
        """
        Args
            decoder_hidden: (batch_size, decoder_hidden_dim)
            encoder_all_hidden: (batch_size, src_len, encoder_hidden_dim * 2)

        Returns
            alpha: (batch_size, src_len)
        """

        Ws = self.W(decoder_hidden).unsqueeze(1)  # (batch_size, 1, decoder_hidden_dim)
        Uh = self.U(encoder_all_hidden)  # (batch_size, src_len, attention_dim)

        energy = self.v(torch.tanh(Ws + Uh))  # (batch_size, src_len, 1)
        energy = energy.squeeze(2)  # (batch_size, src_len)

        alpha = torch.softmax(energy, dim=1)  # (batch_size, src_len)

        return alpha


class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, decoder_hidden_dim, encoder_hidden_dim, attention_dim):
        super(BahdanauAttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, attention_dim)
        self.gru = nn.GRU(
            embed_dim + encoder_hidden_dim * 2, decoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=False,
        )
        self.fc = nn.Linear(decoder_hidden_dim, output_dim)

    def forward(self, input_token, decoder_hidden, encoder_all_hidden):
        """
        Args
            input_token: (batch_size)
            decoder_hidden: (batch_size, decoder_hidden_dim)
            encoder_all_hidden: (batch_size, src_len, encoder_hidden_dim * 2)

        Returns
            prediction: (batch_size, output_dim)
            final_hidden: (batch_size, decoder_hidden_dim)
        """

        input_token = input_token.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input_token)  # (batch_size, 1, embed_dim)
        alpha = self.attention(decoder_hidden, encoder_all_hidden)  # (batch_size, src_len)
        alpha = alpha.unsqueeze(1)  # (batch_size, 1, src_len)
        context = torch.bmm(alpha, encoder_all_hidden)  # (batch_size, 1, encoder_hidden_dim * 2)

        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_dim + encoder_hidden_dim * 2)

        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch_size, decoder_hidden_dim)
        # all_hidden: (batch_size, 1, decoder_hidden_dim)
        # final_hidden: (1, batch_size, decoder_hidden_dim)
        all_hidden, final_hidden = self.gru(rnn_input, decoder_hidden)

        final_hidden = final_hidden.squeeze(0)  # (batch_size, decoder_hidden_dim)

        prediction = self.fc(all_hidden.squeeze(1))  # (batch_size, output_dim)

        return prediction, final_hidden


class BahdanauAttentionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(BahdanauAttentionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: float - probability to use teacher forcing

        Returns
            outputs: (batch_size, tgt_len, tgt_vocab_size)
        """

        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)

        encoder_all_hidden, decoder_hidden = self.encoder(src)
        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            prediction, decoder_hidden = self.decoder(input_token, decoder_hidden, encoder_all_hidden)
            outputs[:, t] = prediction

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else prediction.argmax(1)

        return outputs
