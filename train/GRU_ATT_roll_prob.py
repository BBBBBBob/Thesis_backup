import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GRU_enc(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, num_layers, dropout):
        super(GRU_enc, self).__init__()
        self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        self.fc_in = nn.Linear(input_size, hidden_size)
        # self.ln = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.fc = nn.Linear(hidden_size, hidden_size*2)
        self.fc_out_mean = nn.Linear(hidden_size*2, output_size)
        self.fc_out_var = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, device):
        x = self.fc_in(x)
        x = self.leaky(x)
        h_0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device=device))
        enc_output, h_n = self.encoder(x, h_0)
        x = self.leaky(self.fc(x))
        pred_pos_mean = self.fc_out_mean(x[:, -1:, :])
        pred_pos_var = nn.functional.elu(self.fc_out_var(x[:, -1:, :])) + 1
        return pred_pos_mean, pred_pos_var, enc_output, h_n

# class GRU_dec(nn.Module):
#     def __init__(self, hidden_size, input_size, output_len, num_layers):
#         super(GRU_dec, self).__init__()
#         self.hidden_size = hidden_size
#         self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
#                               batch_first=True)
#         self.fc = nn.Linear(input_size, hidden_size)
#         self.fc_embed = nn.Linear(hidden_size, hidden_size)
#         # self.ln = nn.LayerNorm(hidden_size)
#         self.leaky = nn.LeakyReLU(negative_slope=0.1)
#         self.output_len = output_len

#     def forward(self, x, h):
#         x = self.leaky(self.fc(x[:, -1:, :]))
#         h = h
#         output_list = []
#         for i in range(self.output_len):
#             if i == 0:
#                 x, h = self.decoder(x, h)
#                 output_list.append(x)
#             else:
#                 x = self.leaky(self.fc_embed(x))
#                 x, h = self.decoder(x, h)
#                 output_list.append(x)
        
#         x = torch.cat(output_list, dim=1)

#         return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # repeat decoder hidden state src_len times
        hidden = hidden.permute(1, 0, 2).repeat(1, encoder_outputs.shape[1], 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)

class GRU_dec(nn.Module):
    def __init__(self, hidden_size, output_size, output_len, num_layers, dropout):
        super(GRU_dec, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        self.attention = Attention(hidden_size=hidden_size)
        self.fc_in = nn.Linear(hidden_size, hidden_size)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.output_len = output_len

    def forward(self, encoder_output, decoder_output, h):
        output = torch.concat([encoder_output, decoder_output], dim=1)
        x = self.leaky(self.fc_in(output[:, -1:, :]))
        weight = self.attention(h, output[:, -8:-1, :])
        weight_enc = torch.bmm(weight.unsqueeze(1), output[:, -8:-1, :])
        x = torch.concat([x, weight_enc], dim=2)
        x = self.leaky(x)
        dec_output, h = self.decoder(x, h)

        return dec_output, h

class MDN(nn.Module):
    def __init__(self, input_size, output_size, mode, temp, dropout):
        super(MDN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mode = mode
        self.temp = temp
        self.fc = nn.Linear(input_size, input_size*2)
        self.dropout = nn.Dropout(p=dropout)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.pi_fc = nn.Linear(input_size*2, mode)
        self.mu_fc = nn.Linear(input_size*2, mode*output_size)
        self.var_fc = nn.Linear(input_size*2, mode*output_size)

    def forward(self, x):
        x = self.leaky(self.fc(x))
        x = self.dropout(x)
        # pi = nn.functional.softmax(self.pi_fc(x) / self.temp, dim=-1)
        pi = self.pi_fc(x)
        mu = self.mu_fc(x).view(x.shape[0], x.shape[1], self.mode, -1)
        var = (nn.functional.elu(self.var_fc(x)) + 1).view(x.shape[0], x.shape[1], self.mode, -1)

        return pi, mu, var

class GRU_ATT_roll_prob(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, output_len, num_layers, mode, temp, dropout):
        super(GRU_ATT_roll_prob, self).__init__()
        self.output_len = output_len
        self.encoder = GRU_enc(hidden_size=hidden_size, input_size=input_size, output_size=output_size, num_layers=num_layers, dropout=dropout)
        self.decoder = GRU_dec(hidden_size=hidden_size, output_size=output_size, output_len=output_len,
                               num_layers=num_layers, dropout=dropout)
        self.MDN = MDN(input_size=hidden_size, output_size=output_size, mode=mode, temp=temp, dropout=dropout)

    def forward(self, x, device):
        pred_pos_mean, pred_pos_var, enc_output, h = self.encoder(x, device)
        dec_output_in = torch.tensor([], device=device, dtype=torch.float32)
        for _ in range(self.output_len):
            dec_output, h = self.decoder(enc_output, dec_output_in, h)
            dec_output_in = torch.cat([dec_output_in, dec_output], dim=1)
        pi, mu, var = self.MDN(dec_output_in)

        return pred_pos_mean, pred_pos_var, pi, mu, var
