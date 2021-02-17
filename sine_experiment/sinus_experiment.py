import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.io import wavfile
from playground import plot_audio_signal, update_last_freqs


def generate_sine_wave(freq, sample_rate, duration, amplitude):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians, y_i = f(x_i) where f(x)=sin(2pi*x)
    y = amplitude * np.sin((2 * np.pi) * frequencies)
    return x, y


# Parameters for sine wave
freq = 800
sample_rate = 48000
amplitude = 20
duration = 10
total_samples = sample_rate * duration

# Generate the wave
sine_x, sine_y = generate_sine_wave(freq, sample_rate, duration, amplitude)

# Save sine to .wav file

wavfile.write('../audio_files/sine_experiment2.wav', sample_rate, sine_y)
