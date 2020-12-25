"""Cancel noise from a couple of recordings by Ellie"""

from scipy import fft, ifft, fftpack
from scipy.signal import stft
import numpy as np
from matplotlib import pyplot as plt
import wave
import librosa
from scipy.io.wavfile import write
from os import makedirs


def update_average_freqs(last_100_freqs, magnitude, freq):
    last_100_freqs[freq].append(magnitude)
    if len(last_100_freqs[freq]) > 100:
        last_100_freqs[freq].pop(0)


def plot_average_freqs(last_100_freqs):
    width = 5
    plt.title('Histogram of frequencies in last 100 0.01 seconds')
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude')
    averages = {}
    for key in last_100_freqs.keys():
        if len(last_100_freqs[key]) > 0:
            averages[key] = np.average(last_100_freqs[key])
    plt.bar(averages.keys(), averages.values(), width, color='b')
    print('yo')
    plt.show()


def main():
    filenames = ['audio_files/ellie-with-noise.wav']  # , 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

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

        # Parameters for windowed fft
        window_size = 1 / 5
        increment_size = 1 / 100
        running_avg = 0
        window = int(samplerate * window_size)
        freqs = fftpack.rfftfreq(window, 1 / samplerate)

        # Perform fft on small window
        last_100_freqs = {freq: [] for freq in freqs}
        i = 0
        j = 0  # todo remove this, debug parameter
        while i < n - 1 / 5:
            # Take a small window on audio file
            windowed_signal = signal[i:i + int(samplerate * window_size)]

            title = 'Signal with window of size ' + str(window_size) + ' second'
            plt.title(title)
            plt.plot(windowed_signal)
            # plt.show()
            plt.clf()

            # Use fft
            fft_windowed_signal = np.abs(fftpack.rfft(windowed_signal))

            # Average out the last 100 1/100 seconds
            k = 0
            while freqs[k] < len(fft_windowed_signal):
                freq = freqs[k]
                magnitude = fft_windowed_signal[int(freq)]
                update_average_freqs(last_100_freqs=last_100_freqs, magnitude=magnitude, freq=freq)
                k += 1

            plt.plot(freqs, fft_windowed_signal)
            title = 'Frequency Histogram from ' + str(i) + ' to ' + str(i + int(samplerate * window_size))
            plt.title(title)
            # plt.show()
            plt.clf()
            i += int(increment_size * samplerate)

            j += 1
            if j > 100:
                plot_average_freqs(last_100_freqs)
                break

        exit()  # todo remove this, for debug
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
