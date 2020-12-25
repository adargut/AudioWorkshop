"""Generate audio signal with noise and cancel noise using fft"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq

SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians, y_i = f(x_i) where f(x)=sin(2pi*x)
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y)
plt.show()

_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(3727, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone

normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

plt.plot(normalized_tone[:1000])
plt.show()

# Number of samples in normalized_tone
N = SAMPLE_RATE * DURATION

yf = rfft(normalized_tone)
xf = rfftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
# plt.show()

# Filter frequencies above threshold
threshold = 1000
yf[(xf > threshold)] = 0

plt.plot(xf, np.abs(yf))
plt.show()

filtered_yf = irfft(yf)
plt.plot(filtered_yf[:1000])
plt.show()
