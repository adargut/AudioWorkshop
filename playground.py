"""Get a histogram of average frequencies to find out which to filter"""

import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.io.wavfile import write
import signal_converter
import struct

def read_whole(filename):
    wav_r = wave.open(filename, 'r')
    ret = []
    while wav_r.tell() < wav_r.getnframes():
        decoded = struct.unpack("<h", wav_r.readframes(1))
        ret.append(decoded)
    return ret

def update_last_freqs(last_freqs, freq_id, magnitude, averaged_number_of_freqs):
    last_freqs[freq_id].append(magnitude)
    if len(last_freqs[freq_id]) > averaged_number_of_freqs:
        last_freqs[freq_id].pop(0)


def plot_frequencies(title, freqs, amplitudes, path=None):
    plt.title(title)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude')
    plt.plot(freqs, amplitudes)
    plt.show()
    # if path:
    #     plt.savefig(path)
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


def plot_graph(freqs, post_fft_frequencies):
    plt.title("Scatter Plot")
    thresh = 50000
    x = []
    y = []
    z = []

    for freq_id in range(len(freqs)):
        for time_id in range(len(post_fft_frequencies)):
            block = post_fft_frequencies[time_id]
            amp = np.abs(block[freq_id])
            if amp > thresh:
                x.append(time_id)
                y.append(np.log(freqs[freq_id]))
                z.append(np.log(amp))
            # elif len(x) > 0:
                # plt.scatter(x, y, c=z, cmap="plasma", s=0.1)
                # x = []
                # y = []
                # z = []
        # if len(x) > 0:
            # plt.scatter(x, y, c=z, cmap="plasma", s=0.1)

    plt.scatter(x, y, c=z, cmap='inferno', s=1)
    plt.colorbar()

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    plt.clf()

def main():
    filenames = [
        'audio_files/guitar1.wav']  # ['audio_files/ellie-with-noise.wav', 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

    #  recordings
    recordings = [wave.open(filename, 'r') for filename in filenames]

    # Iterate over recordings
    for recording, filename in zip(recordings, filenames):

        # Open file
        file = wave.open(filename, 'r')

        # Get audio data
        def generate_sine_wave(freq, sample_rate, duration, amplitude):
            x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
            frequencies = x * freq
            # 2pi because np.sin takes radians, y_i = f(x_i) where f(x)=sin(2pi*x)
            y = amplitude * np.sin((2 * np.pi) * frequencies)
            return x, y

        # Parameters for sine wave
        # freq = 800
        # sample_rate = 48000
        # amplitude = 20
        # duration = 1
        # total_samples = sample_rate * duration
        #
        # # Generate the wave
        # sine_x, sine_y = generate_sine_wave(freq, sample_rate, duration, amplitude)

        # Get the signal from file
        filename = filename.rsplit(".", 1)[0]
        # signal = sine_y
        samples = file.getnframes()
        audio = file.readframes(samples)
        signal = np.frombuffer(audio, dtype=np.int16)
        total_samples = wave.Wave_read.getnframes(file)
        sample_rate = wave.Wave_read.getframerate(file)
        plot_audio_signal("Original audio signal", signal)

        ####### Parameters for windowed fft ########
        secs_in_window = 1 / 50
        secs_in_increment = 1 / 100
        averaged_number_of_freqs = 2

        bool_plot_audio_signal_pre_fft = False
        bool_plot_frequencies = False
        bool_plot_average_freqs = False
        # ##########################################

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
            windowed_signal = signal[i:i + window]

            # Plot before fft
            if bool_plot_audio_signal_pre_fft:
                title = 'Pre-FFT signal in time ' + str(i / sample_rate) + ' - ' + str((i + window) / sample_rate)
                plot_audio_signal(title, windowed_signal,
                                  'plots/audio_signal_pre_fft/' + str(iteration_count))

            # Use fft
            fft_windowed_signal = np.fft.rfft(windowed_signal)
            # for j in range(len(freqs)):
            #     if freqs[j] > 1000 and np.abs(fft_windowed_signal[j]) > 100000000:
            #         fft_windowed_signal[j] = 0
            # for i in range(len(fft_windowed_signal)):
            #     if np.abs(fft_windowed_signal[i]) < thresh:
            #         fft_windowed_signal[i] = 0

            # fft_windowed_signal2 = np.abs(np.fft.rfft(windowed_signal2))

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

        # plot_graph(freqs, post_fft_frequencies)

        audio_after_fft = []
        # Combine blocks into a signal by irffting each block.
        for block in post_fft_frequencies:
            new_block = block
            processed_signal = np.fft.irfft(new_block)
            audio_after_fft.append(processed_signal)
        freqs_log, post_fft_frequencies_log = signal_converter.convert_to_logscale(freqs, post_fft_frequencies)
        new_signal = signal_converter.convert_to_signal(freqs_log, post_fft_frequencies_log, sample_rate, window)
        plot_audio_signal("Post-fft self implemented (log-scale)", new_signal)
        plot_graph(freqs_log, post_fft_frequencies_log)
        audio_after_fft = np.concatenate(audio_after_fft)
        audio_after_fft = audio_after_fft.astype(np.int32)
        plot_audio_signal("Post-fft from IRFTT (linear)", audio_after_fft)
        # audio_after_fft_and_log = audio_after_fft_and_log.astype(np.int32)

        saved_name = 'audio_files/after/' + filename.split('/')[1] + '.wav'
        write(saved_name, sample_rate, audio_after_fft)
        # saved_name = 'audio_files/after/' + filename.split('/')[1] + '_log.wav'
        # write(saved_name, sample_rate, new_signal)


        # Filter all frequencies above threshold
        # threshold = 300
        # fft_signal[(freqs > 300)] = 0

if __name__ == '__main__':
    main()
