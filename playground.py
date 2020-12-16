from scipy import fft, ifft, fftpack
import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.io.wavfile import write


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
    # # Open audio file
    # audio_file = wave.open('audio_files/London_Trafalgar_Square.wav', 'r')
    #
    # # Convert the audio file into signal
    # audio_signal = audio_file.readframes(-1)
    # audio_signal = np.fromstring(audio_signal, "Int16")

    filenames = ['audio_files/ellie-with-noise.wav', 'audio_files/background-noise.wav', 'audio_files/ellie.wav']

    # Ellie recordings
    ellie_recordings = [wave.open(filename, 'r') for filename in filenames]

    # Iterate over recordings
    for recording, filename in zip(ellie_recordings, filenames):

        # Get the signal from file
        filename = filename.rsplit(".", 1)[0]
        signal = np.frombuffer(recording.readframes(-1), dtype=np.int)

        # Plot the audio file
        plt.plot(signal)
        title = filename + ' before changing frequencies'
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig('plots/before/' + filename.split('/')[1])
        plt.clf()

        # Output of Fourier transform
        fft_signal = fftpack.rfft(signal)
        fft_signal[10:2000] = 0
        audio_after_frequency_transform = fftpack.irfft(fft_signal)

        # Sample rate of original file
        samplerate = 48000

        # Plot the audio file after transformation
        plt.plot(audio_after_frequency_transform)
        title = filename + ' after changing frequencies'
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig('plots/after/' + filename.split('/')[1])
        plt.clf()

        # Save the audio file after transformation
        saved_name = 'audio_files/after/' + filename.split('/')[1] + '.wav'
        write(saved_name, samplerate, audio_after_frequency_transform)

    # Plot the signal
    # plt.plot(audio_signal)
    # plt.show()
    #
    # # Perform fft, then change frequency and return the original file after inverse fft
    # frequencies = range(100)
    # values = [-55] * len(frequencies)
    # filtered_audio = filter_frequency(audio_signal, frequencies, values)
    # plt.plot(filtered_audio)
    # plt.show()


if __name__ == '__main__':
    main()
