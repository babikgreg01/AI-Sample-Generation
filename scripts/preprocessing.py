import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf


def pad_spectrogram(spectrogram, num_frames, n_fft):
    # pad or trim to reach desired number of frames, also cut freq bins to n_fft/2
    pad_amount = max(0, num_frames - np.shape(spectrogram)[1])
    pad_mat = ((0, 0), (0, pad_amount)) 
    spectrogram = np.pad(spectrogram, pad_mat, mode='constant', constant_values=-80)

    #cutoff_bin = np.shape(spectrogram)[0] - 1000
    cutoff_bin = n_fft//2
    spectrogram = spectrogram[:cutoff_bin, :num_frames]  
    return spectrogram


def scale_spectrogram(spectrogram):
    # scale spectrogram to a range of [0,1]
    return np.divide(spectrogram, 80) + 1


def re_scale_spectrogram(spectrogram):
    # scale spectrogram back to dB scale
    return np.multiply(spectrogram - 1, 80)


def re_pad_spectrogram(spectrogram):
    # pad spectrogram back to original size
    pad_mat = ((0,1), (0,0))
    spectrogram = np.pad(spectrogram, pad_mat, mode='constant', constant_values=-80)
    return spectrogram


def generate_spectrogram(wav_file_path, n_fft, hop_length, plot=False):
    # Load the audio file
    y, sr = librosa.load(wav_file_path, sr=None)

    # Generate a spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, win_length=n_fft, hop_length=hop_length)), ref=np.max)

    # Display the spectrogram
    if(plot):
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of {}'.format(os.path.basename(wav_file_path)))
        plt.show()
        print("dimension: ", np.shape(D)[0], " * ", np.shape(D)[1])

    return D, sr


def spectrogram_to_waveform(spectrogram, n_fft, hop_length):
    # Convert the spectrogram back to a waveform
    y_reconstructed = librosa.griffinlim(librosa.db_to_amplitude(spectrogram), n_fft=n_fft, win_length=n_fft, hop_length=hop_length)
    
    # scale such that audio peaks at 0dB
    max_peak = np.max(np.abs(y_reconstructed))
    gain = 1 / max_peak 
    y_reconstructed = y_reconstructed * gain

    return y_reconstructed


def apply_fadeout(audio, sr, duration=0.1):
    # convert to audio indices (samples)
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length

    # compute fade out curve (linear fade)
    fade_curve = np.linspace(1.0, 0.0, length)

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve

    return audio


def preprocess_data(raw_data_path, processed_data_path, params):

    destination_subfolder_name = "nfft"+str(params["n_fft"])+"_hop"+str(params["hop_length"])+"_nframes"+str(params["n_frames"])
    destination_subfolder = os.path.join(processed_data_path, destination_subfolder_name)

    print("saving to ", destination_subfolder)
    
    if os.path.exists(destination_subfolder):
            print("config exists - overwriting...")
    else:
        os.mkdir(destination_subfolder)

    for folder_name in os.listdir(raw_data_path):
        print(folder_name)
        
        source_file_path = os.path.join(raw_data_path, folder_name)
        destination_file_path = os.path.join(destination_subfolder, folder_name)
        print(destination_file_path)

        if not os.path.exists(destination_file_path):
            os.mkdir(destination_file_path)

        for file_name in os.listdir(source_file_path):

            if not file_name.startswith('.') and file_name.endswith('.wav'):
                wav_path = os.path.join(source_file_path, file_name)
                npy_path = os.path.join(destination_file_path, os.path.splitext(file_name)[0])

                # generate spectrogram, pad and scale to range [0,1]
                spectrogram, _ = generate_spectrogram(wav_path, n_fft=params["n_fft"], hop_length=params["hop_length"])
                spectrogram = pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                spectrogram = scale_spectrogram(spectrogram)

                # save spectrogram
                np.save(npy_path, spectrogram)


if __name__ == "__main__":
    base_dir = os.getcwd()
    print(base_dir)

    raw_data_path = os.path.join(base_dir, 'data', 'training_raw')
    processed_data_path = os.path.join(base_dir, 'data', 'training_processed')

    #params1 = {
    #"n_fft":2048,
    #"hop_length":1024,
    #"sample_rate":44100,
    #"n_frames": 16,
    #}

    params = {
    "n_fft":4096,
    "hop_length":1024,
    "sample_rate":44100,
    "n_frames": 16,
    }

    preprocess_data(raw_data_path, processed_data_path, params)