import torch
from torch import nn


class LuongAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim):
        super(LuongAttentionEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, encoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

    def forward(self, src):
        """
        Args
            src: (batch_size, src_len)

        Returns
            all_hidden: (batch_size, src_len, decoder_hidden_dim)
            decoder_hidden_init: (batch_size, decoder_hidden_dim)
            decoder_cell_init: (batch_size, decoder_hidden_dim)
        """

        embedded = self.embedding(src)  # (batch_size, src_len, embed_dim)
        # all_hidden: (batch_size, src_len, encoder_hidden_dim * 2)
        # final_hidden, final_cell: (num_layers * 2, batch_size, encoder_hidden_dim)
        all_hidden, (final_hidden, final_cell) = self.lstm(embedded)

        # Map to decoder's initial hidden and cell states
        final_forward_hidden = final_hidden[-2, :, :]
        final_backward_hidden = final_hidden[-1, :, :]
        final_hidden = torch.cat((final_forward_hidden, final_backward_hidden), dim=1)
        final_forward_cell = final_cell[-2, :, :]
        final_backward_cell = final_cell[-1, :, :]
        final_cell = torch.cat((final_forward_cell, final_backward_cell), dim=1)
        decoder_hidden_init = self.fc(final_hidden)  # (batch_size, decoder_hidden_dim)
        decoder_cell_init = self.fc(final_cell)  # (batch_size, decoder_hidden_dim)

        b, s, d = all_hidden.shape
        all_hidden_2d = all_hidden.view(b * s, d)
        all_hidden_2d = self.fc(all_hidden_2d)
        all_hidden = all_hidden_2d.view(b, s, self.fc.out_features)  # (batch_size, src_len, decoder_hidden_dim)

        return all_hidden, decoder_hidden_init, decoder_cell_init


class LuongDotAttention(nn.Module):
    def __init__(self):
        super(LuongDotAttention, self).__init__()

    def forward(self, decoder_hidden, encoder_all_hidden):
        """
        Args
            decoder_hidden: (batch_size, decoder_hidden_dim)
            encoder_all_hidden: (batch_size, src_len, decoder_hidden_dim)

        Returns
            alpha: (batch_size, src_len)
        """

        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, decoder_hidden_dim)
        scores = torch.bmm(decoder_hidden, encoder_all_hidden.transpose(1, 2))  # (batch_size, 1, src_len)
        scores = scores.squeeze(1)  # (batch_size, src_len)

        alpha = torch.softmax(scores, dim=1)  # (batch_size, src_len)

        return alpha


class LuongGeneralAttention(nn.Module):
    def __init__(self, decoder_hidden_dim):
        super(LuongGeneralAttention, self).__init__()
        self.W_a = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_all_hidden):
        """
        Args
            decoder_hidden: (batch_size, decoder_hidden_dim)
            encoder_all_hidden: (batch_size, src_len, decoder_hidden_dim)

        Returns
            alpha: (batch_size, src_len)
        """

        encoder_all_hidden_transformed = self.W_a(encoder_all_hidden)  # (batch_size, src_len, decoder_hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, decoder_hidden_dim)
        scores = torch.bmm(decoder_hidden, encoder_all_hidden_transformed.transpose(1, 2))  # (batch_size, 1, src_len)
        scores = scores.squeeze(1)  # (batch_size, src_len)

        alpha = torch.softmax(scores, dim=1)  # (batch_size, src_len)

        return alpha


class LuongConcatAttention(nn.Module):
    def __init__(self, decoder_hidden_dim):
        super(LuongConcatAttention, self).__init__()
        self.W_a = nn.Linear(decoder_hidden_dim * 2, decoder_hidden_dim, bias=False)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_all_hidden):
        """
        Args
            decoder_hidden: (batch_size, decoder_hidden_dim)
            encoder_all_hidden: (batch_size, src_len, decoder_hidden_dim)

        Returns
            alpha: (batch_size, src_len)
        """

        b, s, d = encoder_all_hidden.shape
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, s, -1)  # (batch_size, src_len, hidden_dim)
        concat_input = torch.cat((decoder_hidden_expanded, encoder_all_hidden), dim=2)  # (batch_size, src_len, hidden_dim * 2)
        concat_output = torch.tanh(self.W_a(concat_input))  # (batch_size, src_len, hidden_dim)
        scores = self.v(concat_output)  # (batch_size, src_len, 1)
        scores = scores.squeeze(2)  # (batch_size, src_len)

        alpha = torch.softmax(scores, dim=1)  # (batch_size, src_len)

        return alpha


class LuongAttentionDecoder(nn.Module):
    def __init__(self, attention, output_dim, embed_dim, decoder_hidden_dim):
        super(LuongAttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, decoder_hidden_dim, num_layers=1, batch_first=True)
        self.attention = attention
        self.W_c = nn.Linear(decoder_hidden_dim * 2, decoder_hidden_dim)
        self.W_s = nn.Linear(decoder_hidden_dim, output_dim)

    def forward(self, input_token, decoder_hidden, decoder_cell, encoder_all_hidden):
        """
        Args
            input_token: (batch_size)
            decoder_hidden: (batch_size, decoder_hidden_dim)
            decoder_cell: (batch_size, decoder_hidden_dim)
            encoder_all_hidden: (batch_size, src_len, decoder_hidden_dim)

        Returns
            prediction: (batch_size, output_dim)
            final_hidden: (batch_size, hidden_dim)
            final_cell: (batch_size, hidden_dim)
        """

        input_token = input_token.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input_token)  # (batch_size, 1, embed_dim)

        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch_size, hidden_dim)
        decoder_cell = decoder_cell.unsqueeze(0)  # (1, batch_size, hidden_dim)

        # decoder_all_hidden: (batch_size, 1, hidden_dim)
        # final_hidden, final_cell: (1, batch_size, hidden_dim)
        decoder_all_hidden, (final_hidden, final_cell) = self.lstm(embedded, (decoder_hidden, decoder_cell))

        final_hidden = final_hidden.squeeze(0)  # (batch_size, hidden_dim)
        final_cell = final_cell.squeeze(0)  # (batch_size, hidden_dim)

        alpha = self.attention(final_hidden, encoder_all_hidden)  # (batch_size, src_len)
        alpha = alpha.unsqueeze(1)  # (batch_size, 1, src_len)
        context = torch.bmm(alpha, encoder_all_hidden)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        hidden_cat = torch.cat((context, final_hidden), dim=1)  # (batch_size, hidden_dim * 2)
        luong_hidden = torch.tanh(self.W_c(hidden_cat))  # (batch_size, hidden_dim)

        prediction = self.W_s(luong_hidden)  # (batch_size, output_dim)

        return prediction, final_hidden, final_cell


class LuongAttentionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(LuongAttentionModel, self).__init__()
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
        tgt_vocab_size = self.decoder.W_s.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)

        encoder_all_hidden, hidden, cell = self.encoder(src)
        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_all_hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else output.argmax(1)

        return outputs
