import random

import torch

from seq2seq import Seq2Seq
seq2seq = Seq2Seq(1000, 1000, 1000)
MAX = 20
SOS_token = 0
EOS_token = 1


# noinspection PyUnboundLocalVariable
def forward(nl, sql):
    enc_hidden = seq2seq.encoder.initHidden()
    inlen = nl.size(0)
    outlen = sql.size(0)

    enc_outputs = torch.zeros(MAX, seq2seq.encoder.hidden_size)
    dec_outputs = []

    for ei in range(inlen):
        enc_out, enc_hid = seq2seq.encoder(nl[ei], enc_hidden)
        enc_outputs[ei] = enc_out[0, 0]

    dec_input = torch.tensor([[SOS_token]])
    dec_hidden = enc_hid

    use_tf = True if random.random() < 0.4 else False

    if use_tf:
        for di in range(outlen):
            dec_out, dec_hid, dec_attn = seq2seq.decoder(dec_input, dec_hidden, enc_outputs)
            dec_outputs.append(dec_out)
            dec_input = sql[di]

    else:
        for di in range(outlen):
            dec_out, dec_hid, dec_attn = seq2seq.decoder(dec_input, dec_hidden, enc_outputs)
            _, topi = dec_out.topk(1)
            dec_outputs.append(dec_out)
            dec_input = topi.squeeze().detach()
            if dec_input.item() == EOS_token:
                break
    dec_outputs = torch.tensor(dec_outputs)

    return dec_outputs
