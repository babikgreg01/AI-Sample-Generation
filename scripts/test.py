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

sys.path.append("../scripts")
from utils import NpyDataset
from model import CVAE
from model import to_var
from model import loss_fn


def generate_latent_points(vae, latent_points_folder):
    latent_points = []
    latent_labels = []

    for idx, (images, labels) in enumerate(dataloader):
        #images = flatten(images)
        recon_images, mu, logvar = vae(to_var(images))

        latent_points.append(mu.cpu().detach().numpy())
        latent_labels.append(labels.cpu().detach().numpy())

    #latent_points = np.array(latent_points)
    #latent_labels = np.array(latent_labels)
    latent_points = np.concatenate(latent_points, axis=0)
    latent_labels = np.concatenate(latent_labels, axis=0)

    latent_points_dir = os.path.join(base_dir, "data/saved_latent_points/latentdim_2", "points.npy")
    latent_labels_dir = os.path.join(base_dir, "data/saved_latent_points/latentdim_2", "labels.npy")

    np.save(latent_points_dir, latent_points)
    np.save(latent_labels_dir, latent_labels)


def load_latent_points():
    latent_points_dir = os.path.join(base_dir, "data/saved_latent_points/latentdim_2", "points.npy")
    latent_labels_dir = os.path.join(base_dir, "data/saved_latent_points/latentdim_2", "labels.npy")
    latent_points = np.load(latent_points_dir)
    latent_labels = np.load(latent_labels_dir)
    return latent_points, latent_labels


def load_model():
    latent_dim = 2
    model_dir = os.path.join(base_dir, "models", "vae_latentdim_2.pth")
    #16384
    #32768
    vae = CVAE(h_dim=32768, z_dim=latent_dim)
    vae.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        vae.cuda()

    print("loaded model:", model_dir)
    return vae


def plot_latent_space(latent_points, category_labels, categories):
    fig, ax = plt.subplots()

    x = latent_points[:, 0]  # Array of x coordinates
    y = latent_points[:, 1]  # Array of y coordinates

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']

    # Create a scatter plot
    for category in np.unique(category_labels):
        ix = np.where(category_labels == category)
        ax.scatter(x[ix], y[ix], c=colors[int(category)], label=categories[int(category)], s=0.1)

    # Increase legend marker size
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_sizes([30.0])

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Scatter Plot with Legend')


    # Connect the event handler
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


# Event handler 
def onclick(event):
    # Check if click is within axes
    if event.xdata is None or event.ydata is None:  
        return

    print(f'Clicked at: x={event.xdata}, y={event.ydata}')




def define_categories(processed_data_path):
    sample_packs = sorted(os.listdir(processed_data_path))

    categories = ['Claps', 'Hihats', 'Kicks', 'Percussion', 'Snares', 'Toms']
    category_keywords = {
        "Kicks": ["Bassdrum", "Kicks", "Sub Perc"],
        "Hihats": ["Hihat", "Hats"],
        "Snares": ["Snare"],
        "Toms": ["Toms"],
        "Percussion": ["Perc"],
        "Claps": ["Claps"]
    }

    def determine_category(item_name, category_keywords):
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in item_name.lower():
                    return category
        return "Other"

    category_lookup = []
    for idx, item in enumerate(sample_packs):
        category_lookup.append(categories.index(determine_category(item, category_keywords)))

    return category_lookup, categories



if __name__ == "__main__":
    base_dir = os.getcwd()
    data_folder = "nfft4096_hop1024_nframes16_sr44100data_augmented"
    processed_data_path = os.path.join(base_dir, 'data', 'training_processed', data_folder)

    #sorted(os.listdir(processed_data_path))

    # Load Dataset
    bs = 128
    dataset = NpyDataset(processed_data_path)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)


    vae = load_model()

    latent_points, latent_labels = load_latent_points()

    category_lookup, categories = define_categories(processed_data_path)

    # generate category labels: 
    len_dataset = len(latent_labels)
    category_labels = np.empty(len_dataset)
    for i in range(len_dataset):
        category_labels[i] = category_lookup[latent_labels[i]]
    print(category_labels)

    plot_latent_space(latent_points, category_labels, categories)


