"""Get a histogram of average frequencies to find out which to filter"""

import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.signal import stft
from scipy.io.wavfile import write
from os import makedirs


def update_last_freqs(last_freqs, freq_id, magnitude, averaged_number_of_freqs):
    last_freqs[freq_id].append(magnitude)
    if len(last_freqs[freq_id]) > averaged_number_of_freqs:
        last_freqs[freq_id].pop(0)


def plot_frequencies(title, freqs, amplitudes, path=None):
    plt.title(title)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude')
    plt.plot(freqs, amplitudes)
    if path:
        plt.savefig(path)
    plt.show()
    plt.clf()


def eliminate_frequencies(amplitudes, freqs, threshold):
    amplitudes[(freqs > threshold)] = 0
    signal_post_fft = np.fft.irfft(amplitudes)
    return signal_post_fft


def plot_audio_signal(title, signal, path=None):
    plt.title(title)
    plt.plot(signal[:10000])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    if path:
        plt.savefig(path)
    plt.clf()


def merge_windows(windows):
    return np.concatenate(windows)



def main():
    filenames = [
        '../audio_files/sine_experiment1.wav']  # ['audio_files/ellie-with-noise.wav', 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

    # Ellie recordings
    ellie_recordings = [wave.open(filename, 'r') for filename in filenames]

    # Iterate over recordings
    for recording, filename in zip(ellie_recordings, filenames):

        # Open file
        file = wave.open(filename, 'r')

        # Get audio data
        total_samples = wave.Wave_read.getnframes(file)
        sample_rate = wave.Wave_read.getframerate(file)

        # Get the signal from file
        filename = filename.rsplit(".", 1)[0]
        signal = np.frombuffer(recording.readframes(-1), dtype=np.int)

        # ##### Parameters for windowed fft ########

        freqs = np.fft.rfftfreq(total_samples, 1 / sample_rate)

        freqs_after_fft = np.abs(np.fft.rfft(sine_y))

        plot_audio_signal("signal", sine_y)

        plot_frequencies("freqs", freqs, freqs_after_fft)

        # Perform fft on small window
        last_freqs = [[] for _ in range(len(freqs))]
        i = 0
        iteration_count = 0
        post_fft_frequencies = []



if __name__ == '__main__':
    main()
