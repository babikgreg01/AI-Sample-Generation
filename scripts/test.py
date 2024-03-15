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
from functools import partial
import sounddevice as sd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import soundfile as sf

import audio_utils as au
from utils import NpyDataset
from model import CVAE
from model import to_var
from model import loss_fn


def generate_latent_points(vae, latent_points_folder, dataloader):
    print("Encoding data into latent space...")
    latent_points = []
    latent_labels = []
    
    for (images, labels) in tqdm(dataloader):
        #images = flatten(images)
        #recon_images, mu, logvar = vae(to_var(images))
        _, mu, _ = vae.encode(to_var(images))

        latent_points.append(mu.cpu().detach().numpy())
        latent_labels.append(labels.cpu().detach().numpy())

    latent_points = np.concatenate(latent_points, axis=0)
    latent_labels = np.concatenate(latent_labels, axis=0)

    latent_points_dir = os.path.join(base_dir, "data/saved_latent_points", latent_points_folder)
    if not os.path.exists(latent_points_dir):
        os.mkdir(latent_points_dir)

    np.save(latent_points_dir + "/points.npy", latent_points)
    np.save(latent_points_dir + "/labels.npy", latent_labels)


def load_latent_points(latent_points_folder):
    latent_points_dir = os.path.join(base_dir, "data/saved_latent_points", latent_points_folder, "points.npy")
    latent_labels_dir = os.path.join(base_dir, "data/saved_latent_points", latent_points_folder, "labels.npy")
    latent_points = np.load(latent_points_dir)
    latent_labels = np.load(latent_labels_dir)
    return latent_points, latent_labels


def load_model(model_name, latent_dim):
    
    model_dir = os.path.join(base_dir, "models", model_name)
    #16384
    #32768
    vae = CVAE(h_dim=32768, z_dim=latent_dim)
    vae.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        vae.cuda()

    print("loaded model:", model_dir)
    return vae


def plot_latent_space(latent_points, category_labels, categories, params):
    
    # Apply PCA
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(latent_points)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_standardized)
    print("principle components shape:", principal_components.shape)

    params["scaler"] = scaler
    params["pca"] = pca

    fig, ax = plt.subplots()
    x = principal_components[:, 0]  # Get 2D representation
    y = principal_components[:, 1]  

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']

    # Create a scatter plot
    for category in np.unique(category_labels):
        ix = np.where(category_labels == category)
        ax.scatter(x[ix], y[ix], c=colors[int(category)], label=categories[int(category)], s=1.0)

    # Increase legend marker size
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_sizes([30.0])

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Latent Space Projection')

    # Connect the event handler
    onclick_with_params = partial(onclick, params=params)
    fig.canvas.mpl_connect('button_press_event', onclick_with_params)

    plt.show()


# Event handler 
def onclick(event, params):
    # Check if click is within axes
    if event.xdata is None or event.ydata is None:  
        return

    print(f'Clicked at: x={event.xdata}, y={event.ydata}')
    
    sample = np.array([event.xdata, event.ydata]).reshape(1, -1)
    sample = params["pca"].inverse_transform(sample)
    sample = params["scaler"].inverse_transform(sample)

    print(np.shape(sample))
    
    # process sample into audio
    sample = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
    spectrogram = vae.decode(sample) # decode sample into spectrogram

    spectrogram = spectrogram.detach().cpu().numpy() # convert to numpy array
    spectrogram = np.squeeze(spectrogram)
    spectrogram = au.re_scale_spectrogram(spectrogram)
    spectrogram = au.re_pad_spectrogram(spectrogram)
    reconstructed_waveform = au.spectrogram_to_waveform(spectrogram, params["n_fft"], params["hop_length"])
    reconstructed_waveform = au.apply_fadeout(reconstructed_waveform, params["sample_rate"], duration=0.01)
    print("shape of waveform:", np.shape(reconstructed_waveform))


    sd.play(reconstructed_waveform, blocking=True) # play generated audio

    target_file_path = os.path.join(params["audio_path"], f"sample_x_{event.xdata}_y_{event.ydata}.wav")
    sf.write(target_file_path, reconstructed_waveform, params["sample_rate"])
    


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
    sd.default.samplerate = 44100 # sample rate for audio playback
    base_dir = os.getcwd()

    # preprocessed spectrograms path
    data_folder = "nfft4096_hop1024_nframes16_sr44100data_augmented" 
    processed_data_path = os.path.join(base_dir, 'data', 'training_processed', data_folder)

    # parameters for conversion from spectrogram to wav file
    params = {
    "n_fft":4096,
    "hop_length":1024,
    "sample_rate":44100,
    "n_frames": 16
    }

    # Load Dataset
    bs = 32
    dataset = NpyDataset(processed_data_path)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    # Load Model
    latent_dim = 10
    model_name = "vae_latentdim_10.pth"
    vae = load_model(model_name, latent_dim)

    # Load or generate latent points
    latent_points_folder = "latentdim_10"
    gen_new_points = False
    if(gen_new_points):
        generate_latent_points(vae, latent_points_folder, dataloader)

    latent_points, latent_labels = load_latent_points(latent_points_folder)
    category_lookup, categories = define_categories(processed_data_path)

    # generate category labels: 
    len_dataset = len(latent_labels)
    category_labels = np.empty(len_dataset)
    for i in range(len_dataset):
        category_labels[i] = category_lookup[latent_labels[i]]
    

    print(np.shape(latent_points))


    saved_audio_path = "test_10d_v1"
    saved_audio_path = os.path.join(base_dir, 'data/sample_reconstructed', saved_audio_path)
    if not os.path.exists(saved_audio_path):
        os.mkdir(saved_audio_path)
    params["audio_path"] = saved_audio_path
    
    plot_latent_space(latent_points, category_labels, categories, params)


