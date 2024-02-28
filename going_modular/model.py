import random
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.hidden_dim = configs['hidden_dim']
        self.num_layers = configs['num_layers']
        self.embedding = nn.Embedding(configs['input_dim'], configs['embedding_dim'])

        self.rnn = nn.LSTM(input_size=configs['embedding_dim'], 
                           hidden_size=configs['hidden_dim'], 
                           num_layers=configs['num_layers'],
                           dropout=configs['dropout_rate'])
        
        self.dropout = nn.Dropout(configs['dropout_rate'])

    def forward(self, x):
        # x (seq_length, batch_size)

        x = self.dropout(self.embedding(x))
        # x: (seq_length, batch_size, embedd_dim)

        output, (hidden, cell) = self.rnn(x)
        # output : (seq_length, batch_size, hidden_dim)
        # hidden : (num_layer, batch_size, hidden_dim)
        # cell : (num_layer, batch_size, hidden_dim)

        return hidden, cell
    


class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.output_dim = configs['output_dim']
        self.hidden_dim = configs['hidden_dim']
        self.embedding = nn.Embedding(configs['output_dim'], configs['embedding_dim'])

        self.rnn = nn.LSTM(input_size=configs['embedding_dim'],
                           hidden_size=configs['hidden_dim'],
                           num_layers=configs['num_layers'],
                           dropout=configs['dropout_rate'])
        
        self.dropout = nn.Dropout(configs['dropout_rate'])
        self.linear = nn.Linear(in_features=configs['hidden_dim'], out_features=configs['output_dim'])

    def forward(self, x, hidden, cell):
        # x: (batch_size)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell: (num_layers, batch_size, hidden_dim)

        x = x.unsqueeze(0)
        # x: (1, batch_size) --> 1 is our seq_length here

        x = self.dropout(self.embedding(x))
        # x: (1, batch_size, embedd_dim)

        output , (hidden, cell) = self.rnn(x, (hidden, cell))
        # output: (seq_length=1, batch_size, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell: (seq_length=1, batch_size, hidden_dim)

        prediction = self.linear(output.squeeze(0))
        # prediction: (batch_size, output_dim=vocab_size)

        return prediction, hidden, cell


class Seq_To_Seq(nn.Module):
    def __init__(self, configs, device):
        super().__init__()

        self.encoder = Encoder(configs=configs)
        self.decoder = Decoder(configs=configs)
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio, device):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
