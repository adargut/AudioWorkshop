from scipy import fft, ifft
import numpy as np
from matplotlib import pyplot as plt
import wave


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


def filter_frequency(audio_file, frequency):
    my_fft = fft(audio_file)
    my_fft[frequency] = -25

    return ifft(my_fft)


def main():
    # Open audio file
    audio_file = wave.open('audio_files/London_Trafalgar_Square.wav', 'r')

    # Convert the audio file into signal
    audio_signal = audio_file.readframes(-1)
    audio_signal = np.fromstring(audio_signal, "Int16")

    # Plot the signal
    plt.plot(audio_signal)
    plt.show()

    # Perform fft, then change frequency and return the original file after inverse fft
    frequencies_to_filter = range(100)
    filtered_audio = filter_frequency(audio_signal, frequencies_to_filter)
    plt.plot(filtered_audio)
    plt.show()


if __name__ == '__main__':
    main()
