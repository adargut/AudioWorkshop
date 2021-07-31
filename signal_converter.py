import numpy as np


def generate_sine_wave(freq, sample_rate, duration, amplitude):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians, y_i = f(x_i) where f(x)=sin(2pi*x)
    y = amplitude * np.sin((2 * np.pi) * frequencies)
    return x, y


def compute_sum_of_sines(freqs, block, time):
    sum = 0
    for i in range(len(block)):
        freq = freqs[i]
        amp = block[i]
        sum += amp*np.sin(freq*time)
    return sum


def convert_to_signal(freqs, amplitudes_over_time, sample_rate, window_size):
    signal = []
    time = 0
    for block in amplitudes_over_time:
        for j in range(window_size):
            sum = 0
            for i in range(len(freqs)):
                freq = freqs[i]
                amp = block[i]
                sum += amp * np.sin(20*freq * time)
            signal.append(sum)
            time += 1/sample_rate
        # print(block)
    return signal

def convert_to_logscale(freqs, amplitudes_over_time):
    i = 0
    new_freqs = []
    while freqs[i] < 200:
        new_freqs.append(freqs[i])
        i += 1
    max_freq = max(freqs)
    curr_freq = freqs[i-1]
    while curr_freq < max_freq:
        new_freqs.append(curr_freq)
        curr_freq *= 2**(1/12)

    new_amplitudes_over_time = []
    for block in amplitudes_over_time:
        new_block = []
        j = i
        for new_freq in new_freqs:
            amp_sum = 0
            while freqs[j] <= new_freq:
                amp_sum += block[j]
                j += 1
            new_block.append(amp_sum)
        new_amplitudes_over_time.append(new_block)

    return new_freqs, new_amplitudes_over_time

x, y = generate_sine_wave(100, 48000, 1, 1000)
