"""Get a histogram of average frequencies to find out which to filter"""

import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.io.wavfile import write
from os import makedirs


def update_last_freqs(last_freqs, freq, magnitude, averaged_number_of_freqs):
    last_freqs[freq].append(magnitude)
    if len(last_freqs[freq]) > averaged_number_of_freqs:
        last_freqs[freq].pop(0)


def plot_average_freqs(last_freqs, title):
    width = 5
    plt.title(title)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude')
    averages = {}
    for key in last_freqs.keys():
      #  if len(last_freqs[key]) > 0:
            averages[key] = np.average(last_freqs[key])
    plt.bar(averages.keys(), averages.values(), width, color='b')
    plt.show()
    plt.clf()


def eliminate_frequencies(amplitudes, freqs, threshold):
    amplitudes[(freqs > threshold)] = 0
    signal_post_fft = np.fft.irfft(amplitudes)
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
    filenames = ['audio_files/ellie-with-noise.wav']  # ['audio_files/ellie-with-noise.wav', 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

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

        # Find sample rate of audio file
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
        secs_in_window = 1 / 5
        secs_in_increment = 1 / 100
        samples_in_window = int(sample_rate * secs_in_window)
        samples_in_increment = int(sample_rate * secs_in_increment)
        averaged_number_of_freqs = 10

        freqs = np.fft.rfftfreq(samples_in_window, 1 / sample_rate)

        # Perform fft on small window
        last_freqs = {freq: [] for freq in freqs}
        i = 0
        all_windows_post_fft = []

        while i < sample_rate/2: #total_samples - secs_in_window:
            # Take a small window on audio file
            windowed_signal = signal[i:i + samples_in_window]

            # Plot before fft
            title = 'Pre-FFT signal in time ' + str(i/sample_rate) + ' - ' + str((i+samples_in_window)/sample_rate)
            plot_audio_signal(title=title, signal=windowed_signal)

            # Use fft
            fft_windowed_signal = np.abs(np.fft.rfft(windowed_signal))

            # Average out the last 100 1/100 seconds
            for k in range(len(freqs)):
                freq = freqs[k]
                magnitude = fft_windowed_signal[k]
                update_last_freqs(last_freqs=last_freqs, freq=freq, magnitude=magnitude,
                                  averaged_number_of_freqs=averaged_number_of_freqs)

            # plt.plot(freqs, fft_windowed_signal)
            # title = 'Frequency Histogram from ' + str(i) + ' to ' + str(i + int(samplerate * window_size))
            # plt.title(title)
            # plt.show()
            # plt.clf()
            i += int(secs_in_increment * sample_rate)

            title = 'Average frequencies in time ' + str(i/sample_rate) + ' - ' + str((i+samples_in_window)/sample_rate)
            plot_average_freqs(last_freqs=last_freqs, title=title)

        # exit()  # todo remove this, for debug
        # Output of Fourier transform
        fft_signal = np.ftt.rfft(signal)

        # x axis
        freqs = np.ftt.rfftfreq(total_samples, 1 / sample_rate)

        # Filter all frequencies above threshold
        threshold = 300
        fft_signal[(freqs > threshold)] = 0

        # Plot the fft output
        plt.plot(fft_signal)
        # plt.show()

        # Revert back to original
        audio_after_frequency_transform = np.ftt.irfft(fft_signal)

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
        write(saved_name, sample_rate, audio_after_frequency_transform)


if __name__ == '__main__':
    main()
