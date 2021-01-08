import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.signal import stft
from scipy.io import wavfile
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
    plt.show()
    if path:
        plt.savefig(path)
    plt.clf()


def merge_windows(windows):
    return np.concatenate(windows)


def main():
    # Open the sine wave
    filename = '../audio_files/sine_experiment.wav'
    sine_wave = wavfile.read(filename)[1]

    # Audio parameters
    sample_rate = 48000
    duration = 10
    total_samples = sample_rate * duration

    # Get the signal from file
    np.frombuffer(sine_wave, dtype=np.int)
    plot_audio_signal("Original sine audio signal", sine_wave)

    # # ##### Parameters for windowed fft ########
    secs_in_window = 1 / 5
    secs_in_increment = 1 / 100
    averaged_number_of_freqs = 10
    bool_plot_audio_signal_pre_fft = False
    bool_plot_frequencies = False
    bool_plot_average_freqs = False
    window = int(sample_rate * secs_in_window)
    increment = int(sample_rate * secs_in_increment)
    freqs = np.fft.rfftfreq(window, 1 / sample_rate)

    # Perform fft on small window
    last_freqs = [[] for _ in range(len(freqs))]
    i = 0
    iteration_count = 0
    post_fft_frequencies = []

    while i < (total_samples - window):
        # Take a small window on audio file
        windowed_signal = sine_wave[i:i + window]

        # Plot before fft
        if bool_plot_audio_signal_pre_fft:
            title = 'Pre-FFT signal in time ' + str(i / sample_rate) + ' - ' + str((i + window) / sample_rate)
            plot_audio_signal(title, windowed_signal,
                              'plots/audio_signal_pre_fft/' + str(iteration_count))

        # Use fft
        fft_windowed_signal = np.fft.rfft(windowed_signal)

        # Update averages
        for freq_id in range(len(freqs)):
            magnitude = fft_windowed_signal[freq_id]
            update_last_freqs(last_freqs=last_freqs, freq_id=freq_id, magnitude=magnitude,
                              averaged_number_of_freqs=averaged_number_of_freqs)
        if bool_plot_frequencies:
            title = 'Frequency Histogram in time ' + str(i / sample_rate) + ' - ' + str((i + window) / sample_rate)
            plot_frequencies(title, freqs, fft_windowed_signal,
                             'plots/frequencies/' + str(iteration_count))

        averages = []
        for j in range(len(freqs)):
            averages.append(np.average(last_freqs[j]))
        post_fft_frequencies.append(averages)
        if bool_plot_average_freqs:
            title = 'Average Frequency Histogram in time ' + str((i + window) / sample_rate)
            plot_frequencies(title, freqs, averages,
                             'plots/average_frequencies/' + str(iteration_count))

        i += increment
        iteration_count += 1

    audio_after_fft = []
    for block in post_fft_frequencies:
        new_block = []
        for i in range(0, int(len(block) / int(window / increment)) - 1):
            start = i * int(window / increment)
            end = (i + 1) * int(window / increment)
            new_block.append(np.average(block[start:end]))
            # new_block.append(block[i*int(window/increment)])
        processed_signal = np.fft.irfft(new_block)
        audio_after_fft.append(processed_signal)
    audio_after_fft = np.concatenate(audio_after_fft)
    audio_after_fft = audio_after_fft.astype(np.int32)
    plot_audio_signal("Post-fft with windows", audio_after_fft)
    saved_name = 'audio_files/after/' + filename.split('/')[1] + '.wav'

    wavfile.write(saved_name, sample_rate, audio_after_fft)

    # Filter all frequencies above threshold
    # threshold = 300
    # fft_signal[(freqs > threshold)] = 0


if __name__ == '__main__':
    main()
