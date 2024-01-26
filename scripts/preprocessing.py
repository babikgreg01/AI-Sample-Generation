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


def generate_spectrogram(audio, n_fft, hop_length, plot=False):
    # Load the audio file
    
    # Generate a spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=hop_length)), ref=np.max)

    # Display the spectrogram
    if(plot):
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        #plt.title('Spectrogram of {}'.format(os.path.basename(wav_file_path)))
        plt.show()
        print("dimension: ", np.shape(D)[0], " * ", np.shape(D)[1])

    return D


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


def pitch_shift(audio, semitones, sample_rate):
    # shift the pitch through resampling
    new_sample_rate = int(sample_rate * (2.0 ** (semitones/12)))
    repitched_audio = librosa.resample(audio, orig_sr=new_sample_rate, target_sr=sample_rate)

    return repitched_audio


def preprocess_data(raw_data_path, processed_data_path, params):

    destination_subfolder_name = "nfft"+str(params["n_fft"])+"_hop"+str(params["hop_length"])+"_nframes"+str(params["n_frames"]) + "_sr"+str(params["sample_rate"])
    if(params["data_augmentation"]):
        destination_subfolder_name += "data_augmented"
        
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

                # load audio file
                audio, sr = librosa.load(wav_path, sr=params["sample_rate"])

                # generate spectrogram, pad and scale to range [0,1]
                spectrogram = generate_spectrogram(audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
                spectrogram = pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                spectrogram = scale_spectrogram(spectrogram)

                # save spectrogram
                np.save(npy_path, spectrogram)


                if(params["data_augmentation"]):
                    pitched_down_audio = pitch_shift(audio, -2, sr)
                    spectrogram = generate_spectrogram(pitched_down_audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
                    spectrogram = pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                    spectrogram = scale_spectrogram(spectrogram)
                    np.save(npy_path+"_down2", spectrogram)

                    pitched_up_audio = pitch_shift(audio, 2, sr)
                    spectrogram = generate_spectrogram(pitched_up_audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
                    spectrogram = pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                    spectrogram = scale_spectrogram(spectrogram)
                    np.save(npy_path+"_up2", spectrogram)

                


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
    "data_augmentation": True
    }

    preprocess_data(raw_data_path, processed_data_path, params)