import torch
import numpy as numpy
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid 
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim*2

        self.rnn1 = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.hidden_dim,
            num_layers = 1,
            batch_first = True

        ) 
        self.rnn2 = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.embedding_dim,
            num_layers = 1,
            batch_first = True

        )


    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features)) 

        x, _ = self.rnn1(x)
        x, (hidden_state, _) = self.rnn2(x)  

        return hidden_state.reshape((self.n_features, self.embedding_dim))  
    
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim = 64, n_features = 1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = input_dim*2

        self.rnn1 = nn.LSTM(
            input_size = input_dim,
            hidden_size = input_dim,
            num_layers = 1,
            batch_first = True

        ) 
        self.rnn2 = nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = 1,
            batch_first = True

        )

        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)


    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features) 
        x = x.reshape((self.n_features, self.seq_len, self.input_dim)) 

        x, _ = self.rnn1(x)
        x, (hidden_state, _) = self.rnn2(x) 
        x = x.reshape((self.seq_len, self.hidden_dim)) 

        return self.output_layer(x)
    
class RecurrentVAE(nn.Module):

    def __init__(self, seq_len, input_dim, embedding_dim=64):
        super(RecurrentVAE, self).__init__()

        self.encoder = Encoder(seq_len, input_dim, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, input_dim).to(device)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)


        return x


    
    


