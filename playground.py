"""Cancel noise from a couple of recordings by Ellie"""

from scipy import fft, ifft, fftpack
from scipy.signal import stft
import numpy as np
from matplotlib import pyplot as plt
import wave
import librosa
from scipy.io.wavfile import write
from os import makedirs


def plot_signal():
    np.random.seed(1234)

    time_step = 0.02
    period = 5.

    time_vec = np.arange(0, 20, time_step)
    sig = (np.sin(2 * np.pi / period * time_vec)
           + 0.5 * np.random.randn(time_vec.size))

    plt.figure(figsize=(6, 5))
    plt.plot(time_vec, sig, label='Original signal')
    plt.show()

    return sig


def filter_frequency(audio_file, frequencies_to_transform, frequency_values):
    transformed_audio = fft(audio_file)

    for frequency_to_transform, frequency_value in zip(frequencies_to_transform, frequency_values):
        transformed_audio[frequencies_to_transform] = frequency_value

    return ifft(transformed_audio)


def main():
    filenames = ['audio_files/ellie-with-noise.wav' , 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

    # Store plotted audio waves in plots dir
    try:
        makedirs('plots')
    except OSError as err:
        pass

    # Ellie recordings
    ellie_recordings = [wave.open(filename, 'r') for filename in filenames]

    # Iterate over recordings
    for recording, filename in zip(ellie_recordings, filenames):

        # Find duration of audio file
        duration = librosa.get_duration(filename=filename)

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

        # Sample rate of original file
        samplerate = 48000

        # Number of samples
        n = int(samplerate * duration)

        window_size = 1 / 5
        increment_size = 1 / 100
        running_avg = 0

        # Perform fft on small window
        st = []
        i = 0
        while i < n - 1 / 5:
            window = int(samplerate * window_size)
            windowed_signal = signal[i:i + int(samplerate * window_size)]
            fft_windowed_signal = np.abs(fftpack.rfft(windowed_signal))

            # x axis
            freqs = fftpack.rfftfreq(window, 1 / samplerate)

            plt.plot(freqs, fft_windowed_signal)
            title = 'Frequency Histogram from ' + str(i) + ' to ' + str(i + int(samplerate * window_size))
            plt.title(title)

            # plt.plot(windowed_signal)
            # plt.title('Windowed Signal')
            plt.show()
            plt.clf()
            i += int(increment_size * samplerate)

            j += 1

            if j > 1000: break

        # Output of Fourier transform
        fft_signal = fftpack.rfft(signal)

        # x axis
        freqs = fftpack.rfftfreq(n, 1 / samplerate)

        # Filter all frequencies above threshold
        threshold = 300
        fft_signal[(freqs > threshold)] = 0

        # Plot the fft output
        plt.plot(fft_signal)
        plt.show()

        # Revert back to original
        audio_after_frequency_transform = fftpack.irfft(fft_signal)

        # Plot the audio file after transformation
        plt.plot(audio_after_frequency_transform)
        title = filename + ' after changing frequencies'
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        # Create dir to save plots after transform
        try:
            makedirs('plots/after')
        except OSError as err:
            pass

        plt.savefig('plots/after/' + filename.split('/')[1])
        plt.clf()

        # Save the audio file after transformation
        try:
            makedirs('audio_files/after')
        except OSError as err:
            pass

        saved_name = 'audio_files/after/' + filename.split('/')[1] + '.wav'
        write(saved_name, samplerate, audio_after_frequency_transform)


if __name__ == '__main__':
    main()
