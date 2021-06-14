import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl

import random
from src.constants import const
# DEVICE = const.DEVICE


class Encoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoder(pl.LightningModule):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=const.MAX_LENGTH):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_s = dropout
        self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_s)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_, hidden, encoder_outputs):
        embedded = self.embedding(input_).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# noinspection PyUnboundLocalVariable
class Seq2Seq(pl.LightningModule):
    def __init__(self, n_input, n_hidden, n_output, teacher_forcing_rate: float = 0.4):
        super(Seq2Seq, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_output
        self.TFR = teacher_forcing_rate

        self.encoder = Encoder(self.n_input, self.n_hidden)
        self.decoder = AttnDecoder(self.n_hidden, self.n_out, dropout=0.2)

    def forward(self, nl: torch.Tensor, sql: torch.Tensor):
        in_len = nl.shape[0]
        tar_len = sql.shape[0]
        enc_hid = self.encoder.initHidden()

        enc_outputs = torch.zeros(const.MAX_LENGTH, self.encoder.hidden_size)
        dec_outputs = torch.zeros(tar_len)

        for i in range(in_len):
            enc_out, enc_hid = self.encoder(nl[i], enc_hid)
            enc_outputs[i] = enc_out[0, 0]

        dec_input = torch.tensor([const.BEG_IDX], dtype=torch.long)
        dec_hidden = enc_hid

        use_teacher_force = True if random.random() < self.TFR else False
        if use_teacher_force:
            for di in range(tar_len):
                dec_out, dec_hidden, dec_attn = self.decoder(dec_input, dec_hidden, enc_outputs)
                dec_outputs[di] = dec_out
                dec_input = sql[di]
        else:
            for di in range(tar_len):
                dec_out, dec_hidden, dec_attn = self.decoder(dec_input, dec_hidden, enc_outputs)
                dec_outputs[di] = dec_out
                _, topi = dec_out.topk(1)
                dec_input = topi.squeeze().detach()
                if dec_input.item() == const.END_IDX:
                    break
        return dec_outputs
    
    def beam_search(self, dec_out, k):
        sequences = [[list(), 0.0]]
        for row in data:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - log(row[j])]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            sequences = ordered[:k]
        return sequences

    def training_step(self, train_batch):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x, y)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
