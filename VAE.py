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


device = 'cpu'

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device = 'cpu'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.mean = nn.Linear(latent_dim, 2)
        self.log_var = nn.Linear(latent_dim, 2)

        self.decoder =  nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()

        )
        

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean(x), self.log_var(x)
        return mean, logvar

    def reparametrization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        return mean + var*epsilon
    

    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    



model = VAE(input_dim=784, hidden_dim=400, latent_dim=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)


def loss_fn(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KL_div = - 0.5*torch.sum(1 + logvar - mean**2 - torch.exp(logvar))

    return reproduction_loss + KL_div



def dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    path = '~/datasets'
    train_dataset = MNIST(path, transform=transform, download=True)
    test_dataset = MNIST(path, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


train_loader, test_loader = dataset(100)
print(train_loader)


def visualize(num_samples):
    dataiter = iter(train_loader)
    image = next(dataiter)
    sample_images = [image[0][i, 0] for i in range(num_samples)]


    fig = plt.figure(figsize=(5,5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5,5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        ax.imshow(im, cmap = 'gray')
        ax.axis('off')

    plt.show()

visualize(25)
from pathlib import Path
def get_weights_file_path(epoch:str):
    model_folder = 'C:/Users/amade/Documents/GitHub/VAEs_and_Time_Series/model_vae'
    model_basename = 'vaemodel_'
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)

def train(model, optimizer, loss_fn = loss_fn, num_epochs = 100):
    for epoch in range(num_epochs):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f'Processing Epoch {epoch: 02d}')
        overall_loss = 0
        for batch_idx, (x,_) in enumerate(batch_iterator):
            x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_fn(x, x_hat, mean, log_var)

            overall_loss += loss.item()
            avg_loss = overall_loss/((batch_idx+1)*x.shape[0])
            batch_iterator.set_postfix({f'loss': f'{avg_loss:6.3f}'})

        
            loss.backward()
            optimizer.step()

        
        model_filename = get_weights_file_path(f'{epoch}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

import warnings
if  __name__ == '__main__':
    warnings.filterwarnings('ignore')  
    train(model, optimizer)



