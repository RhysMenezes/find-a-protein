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
            self.lstm = nn.LSTM(embedding_dim,hidden_dim,2,dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim,hidden_dim,dropout=dropout)

        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        if pretrain is True:
            self.softmax = nn.Softmax(2)
        else:
            self.softmax = nn.LogSoftmax(1)
        self.hidden = self.init_hidden()

    def init_hidden(self,seq_len=1):
        if self.pretrain is False:
            h1 = autograd.Variable(torch.zeros(2, seq_len, self.hidden_dim).cuda())
            h2 = autograd.Variable(torch.zeros(2, seq_len, self.hidden_dim).cuda())
        else:
            h1 = autograd.Variable(torch.zeros(2, seq_len, self.hidden_dim))
            h2 = autograd.Variable(torch.zeros(2, seq_len, self.hidden_dim))

        return (h1,h2)

    def forward(self, sentence, training=False):

        #print ("input: " + str(sentence.shape))

        embeds = self.word_embeddings(sentence)

        #print ("embeds: " + str(embeds.shape))
        #print(embeds.view(len(sentence), 1, -1))

        lstm_out, self.hidden = self.lstm(embeds,self.hidden)
	
        #lstm_out = lstm_out[:, :, :self.hidden_dim] + lstm_out[:, :, self.hidden_dim:]

        #print(lstm_out.shape)
        
        if self.pretrain is False:
            lstm_out = lstm_out[-1] # get last step of output


        hidden_out = self.h2o(lstm_out)

        #print ("lstm_out: " + str(lstm_out.shape))
        #hidden_out = self.h2o(lstm_out)
        #print ("hidden_out: " + str(hidden_out.shape))

        relu_out = self.tanh(hidden_out)
        # print ("relu_out: " + str(relu_out.shape))

        out = self.softmax(relu_out)
            # print ("out: " + str(out.shape))
        return out
        