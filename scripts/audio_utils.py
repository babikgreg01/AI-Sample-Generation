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