import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
    def __init__(self,vocab_dim, embedding_dim, hidden_dim, output_dim,dropout=0,word_vecs=None,pretrain=False):
        super(RNN, self).__init__()

        self.pretrain = pretrain

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_dim,embedding_dim)
        if word_vecs is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(np.asarray(word_vecs)))

        if pretrain is False:
            self.lstm = nn.LSTM(embedding_dim,hidden_dim,2,dropout=dropout, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim,hidden_dim,dropout=dropout)

        self.h2o = nn.Linear(hidden_dim*2, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        if pretrain is True:
            self.softmax = nn.Softmax(2)
        else:
            self.softmax = nn.LogSoftmax(1)
        self.hidden = self.init_hidden()

    def init_hidden(self,seq_len=1):
        if self.pretrain is False:
            h1 = autograd.Variable(torch.zeros(4, seq_len, self.hidden_dim).cuda())
            h2 = autograd.Variable(torch.zeros(4, seq_len, self.hidden_dim).cuda())
        else:
            h1 = autograd.Variable(torch.zeros(2, seq_len, self.hidden_dim))
            h2 = autograd.Variable(torch.zeros(2, seq_len, self.hidden_dim))

        return (h1,h2)

    def forward(self, sentence, training=False):

        embeds = self.word_embeddings(sentence)

        lstm_out, self.hidden = self.lstm(embeds,self.hidden)
        
        if self.pretrain is False:
            lstm_out = lstm_out[-1] # get last step of output

        hidden_out = self.h2o(lstm_out)
        tan_out = self.tanh(hidden_out)
        out = self.softmax(tan_out)
        
        return out
        
