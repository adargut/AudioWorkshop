from librosa import display
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import IPython

def get_sine_wave(samples, freq, amp):
    x = np.arange(samples)
    y = amp * np.sin(2 * np.pi * freq * ( x / samples ))
    return x, y

def plot_signal(x, y):
    plt.figure()
    plt.stem(x, y, 'r', )
    plt.plot(x, y)
    plt.xlabel('Time --->')
    plt.ylabel('<--- Amplitude -->')
    plt.show()

def plot_signal_with_sr(samples, sample_rate):
    librosa.display.waveplot(y = samples, sr = sample_rate)
    plt.show()


def fft(audio, sampling_rate, freqs_to_dest):
    n = len(audio)
    T = 1/sampling_rate
    y = scipy.fft(audio)
    x = np.linspace(0.0, 1.0/(2.0 * T), n/2)

    if True:
        for i, f in enumerate(y):
            if i > 10000:
                y[i] = 0
    
    return x, y

    
def fft_plot(audio, sampling_rate, x, y):
    n = len(audio)
    T = 1/sampling_rate
    fig, ax = plt.subplots()
    ax.plot(x, 2.0/n * np.abs(y[:n//2]))
    plt.grid()
    plt.xlabel('Frequency -->')
    plt.ylabel('Magnitude')
    return plt.show()

x1, y1 = get_sine_wave(samples=100, freq=11, amp=2)
x2, y2 = get_sine_wave(samples=100, freq=3, amp=1)

# y3 = y1 + y2
# plot_signal(x1, y1 + y2)

path = 'ellie-with-noise.wav'
samples, sample_rate = librosa.load(path)
IPython.display.Audio(path)

# # plot_signal_with_sr(samples, sample_rate)
# x_post_fft, y_post_fft = fft(samples, sampling_rate=sample_rate, freqs_to_dest=1000)
# fft_plot(samples, sample_rate, x_post_fft, y_post_fft)

# # Ellie after fft
# IPython.display.Audio(scipy.ifft(y_post_fft), rate=sample_rate)

def windowed_fft(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = scipy.fft(windows * weighting, axis=0)
#     fft = np.absolute(fft)
#     fft = fft**2

#     return fft    
#     scale = np.sum(weighting**2) * sample_rate
#     fft[1:-1, :] *= (2.0 / scale)
#     fft[(0, -1), :] /= scale
    return fft
    
    # # Prepare fft frequency list
    # freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # # Compute spectrogram feature
    # ind = np.where(freqs <= max_freq)[0][-1] + 1
    # specgram = np.log(fft[:ind, :] + eps)
    # return specgram

fft = windowed_fft(samples=samples, sample_rate=sample_rate)
reconstruced_signal = scipy.ifft(fft)
# TODO make windowed signal sound OK post reconstruction
IPython.display.Audio(reconstruced_signal, rate=sample_rate)
print('done')
