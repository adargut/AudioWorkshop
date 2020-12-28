"""Get a histogram of average frequencies to find out which to filter"""

import numpy as np
from matplotlib import pyplot as plt
import wave
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
    plt.clf()


def eliminate_frequencies(amplitudes, freqs, threshold):
    amplitudes[(freqs > threshold)] = 0
    signal_post_fft = np.fft.irfft(amplitudes)
    return signal_post_fft


def plot_audio_signal(title, signal, path=None):
    plt.title(title)
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    if path:
        plt.savefig(path)
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
        window = int(sample_rate * secs_in_window)
        increment = int(sample_rate * secs_in_increment)
        averaged_number_of_freqs = 20
        bool_plot_audio_signal_pre_fft = True
        bool_plot_frequencies = True
        bool_plot_average_freqs = True

        freqs = np.fft.rfftfreq(window, 1 / sample_rate)

        # Perform fft on small window
        last_freqs = [[] for _ in range(len(freqs))]
        i = 0
        iteration_count = 0
        post_fft_frequencies = []

        while i < total_samples - window:
            # Take a small window on audio file
            windowed_signal = signal[i:i + window]

            # Plot before fft
            if bool_plot_audio_signal_pre_fft:
                title = 'Pre-FFT signal in time ' + str(i / sample_rate) + ' - ' + str((i + window) / sample_rate)
                plot_audio_signal(title, windowed_signal,
                                  'plots/audio_signal_pre_fft/' + str(iteration_count))

            # Use fft
            fft_windowed_signal = np.abs(np.fft.rfft(windowed_signal))

            # Update averages
            for freq_id in range(len(freqs)):
                magnitude = fft_windowed_signal[freq_id]
                update_last_freqs(last_freqs=last_freqs, freq_id=freq_id, magnitude=magnitude,
                                  averaged_number_of_freqs=averaged_number_of_freqs)
            if bool_plot_frequencies:
                title = 'Frequency Histogram in time ' + str(i / sample_rate) + ' - ' + str((i + window) / sample_rate)
                plot_frequencies(title, freqs, fft_windowed_signal,
                                 'plots/frequencies/' + str(iteration_count))

            if True: #iteration_count % averaged_number_of_freqs == 0:
                averages = []
                for j in range(len(freqs)):
                    averages.append(np.average(last_freqs[j]))
                post_fft_frequencies.append(averages)
                if bool_plot_average_freqs:
                    title = 'Average Frequency Histogram in time ' + str(i / sample_rate) + ' - ' + str(
                        (i + window) / sample_rate)
                    plot_frequencies(title, freqs, averages,
                                     'plots/average_frequencies/' + str(iteration_count))

            i += increment
            iteration_count += 1

        exit()  # todo remove this, for debug
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
