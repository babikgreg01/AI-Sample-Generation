import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import audio_utils as au


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
                spectrogram = au.generate_spectrogram(audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
                spectrogram = au.pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                spectrogram = au.scale_spectrogram(spectrogram)

                # save spectrogram
                np.save(npy_path, spectrogram)

                if(params["data_augmentation"]):
                    # data augmented samples pitched +/-2 
                    pitched_down_audio = au.pitch_shift(audio, -2, sr)
                    spectrogram = au.generate_spectrogram(pitched_down_audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
                    spectrogram = au.pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                    spectrogram = au.scale_spectrogram(spectrogram)
                    np.save(npy_path+"_down2", spectrogram)

                    pitched_up_audio = au.pitch_shift(audio, 2, sr)
                    spectrogram = au.generate_spectrogram(pitched_up_audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
                    spectrogram = au.pad_spectrogram(spectrogram, num_frames=params["n_frames"], n_fft=params["n_fft"])
                    spectrogram = au.scale_spectrogram(spectrogram)
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