import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from IPython.display import Image
from tqdm import tqdm

from torchview import draw_graph
import numpy as np

import matplotlib.pyplot as plt

import sys

from utils import NpyDataset
from model import CVAE
from model import to_var
from model import loss_fn



if __name__ == "__main__":
    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, "models", "vae_latentdim_10.pth")

    print("cuda available:", torch.cuda.is_available())
    bs = 64
    epochs = 20
    lr = 1e-3
    latent_dim = 10

    data_folder = "nfft4096_hop1024_nframes16_sr44100data_augmented"
    processed_data_path = os.path.join(base_dir, 'data', 'training_processed', data_folder)

    # Load Data
    dataset = NpyDataset(processed_data_path)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    # Check input shape
    fixed_x, label = next(iter(dataloader))
    print("input shape:", fixed_x.size())

    vae = CVAE(h_dim=32768, z_dim=latent_dim)
    if torch.cuda.is_available():
        vae.cuda()

    print(vae)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for idx, (images, _) in enumerate(tqdm(dataloader, desc="Epoch[{}/{}]".format(epoch+1, epochs), unit="batch")):
            #images = flatten(images)
            recon_images, mu, logvar = vae(to_var(images))
            loss = loss_fn(recon_images, to_var(images), mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()/bs
            epoch_loss += batch_loss
        
            #tqdm.set_postfix(loss=loss.item()/bs, accuracy=0.9)
        print("Average Loss: {}".format(epoch_loss/24))

    torch.save(vae.state_dict(), model_dir)