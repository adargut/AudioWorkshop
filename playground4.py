"""Filter out noise from recordings of Ellie"""

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.io.wavfile import write
from os import makedirs


def eliminate_frequencies(signal, threshold=3500):
    n = len(signal)
    samplerate = 48000
    freqs = np.fft.rfftfreq(n, 1 / samplerate)
    signal_after_fft = (np.fft.rfft(signal))
    signal_after_fft[(freqs > threshold)] = 0
    signal_post_fft = np.fft.irfft(signal_after_fft)
    return signal_post_fft


def plot_audio_signal(title, signal):
    plt.title(title)
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    plt.clf()


def merge_windows(windows):
    return np.concatenate(windows)


def main():
    filenames = [
        'audio_files/ellie-with-noise.wav']  # ['audio_files/ellie-with-noise.wav', 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

    # Store plotted audio waves in plots dir
    try:
        makedirs('plots')
    except OSError as err:
        pass

    # Ellie recordings
    ellie_recordings = [wave.open(filename, 'r') for filename in filenames]

    # Iterate over recordings
    for recording, filename in zip(ellie_recordings, filenames):

        # Open file
        file = wave.open(filename, 'r')

        # Find number of samples of audio file
        total_samples = wave.Wave_read.getnframes(file)

        # Find samplerate of audio file
        sample_rate = wave.Wave_read.getframerate(file)

        # Get the signal from file
        filename = filename.rsplit(".", 1)[0]
        signal = np.frombuffer(recording.readframes(-1), dtype=np.int)

        # Plot the audio file
        plt.plot(signal)
        title = filename + ' before changing frequencies'
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        try:
            makedirs('plots/before')
        except OSError as err:
            pass

        plt.savefig('plots/before/' + filename.split('/')[1])
        plt.clf()

        # Parameters for windowed fft
        window_size = 1 / 5
        window = int(sample_rate * window_size)
        step_size = 1 / 100
        step = int(sample_rate * step_size)

        # Perform fft on small window
        all_windows_post_fft = []
        for i in tqdm(range(0, int(total_samples - window), window)):
            # Take a small window on audio file
            windowed_signal = signal[i:i + window]

            # Eliminate high frequencies from window
            filtered_windowed_signal = eliminate_frequencies(signal=windowed_signal, threshold=3500)
            all_windows_post_fft.append(filtered_windowed_signal)

        merged_audio_file = merge_windows(all_windows_post_fft)
        saved_name = 'audio_files/after/' + filename.split('/')[1] + '.wav'
        print('saving:', saved_name)
        write(saved_name, sample_rate, merged_audio_file)


if __name__ == '__main__':
    main()
