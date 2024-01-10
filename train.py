from ECG.Ecg_data import dataset
from VAE_LSTM import Encoder, Decoder, RecurrentVAE
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
def get_weights_file_path(epoch:str):
    model_folder = 'C:/Users/amade/Documents/GitHub/Time_series_with_lstm_and_vaes/vae_model_lstm_'
    model_basename = 'vaemodel_'
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train, seq_len, n_features, val, test_anomaly, test_normal = dataset()

model = RecurrentVAE(seq_len, n_features, 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.L1Loss(reduction='sum').to(device)


def train_model(model, train_set, val_set, n_epochs):
    history = dict(train = [], val = [])
    best_loss = np.inf
    train_losses = []
    for epoch in range(n_epochs):
        model.train()
        for seq_true in train_set:

            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = loss_fn(seq_pred, seq_true)

            #print(f'loss: {loss.item()}')

        
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_set:

                

                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = loss_fn(seq_pred, seq_true)

                #print(f'test_loss: {loss.item()}')

            
                
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f'Epoch: {epoch}......train_loss: {train_loss}......test_loss: {val_loss}')

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        
        model_filename = get_weights_file_path(f'{epoch}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)


    return model.eval(), history

import warnings
import matplotlib.pyplot as plt
if  __name__ == '__main__':
    warnings.filterwarnings('ignore') 
    print(np.array(train).shape) 
    _, v_l = train_model(model, train, val, 100)
    plt.plot(v_l['train'], 'r')
    plt.plot(v_l['val'], 'b')
    plt.show


