import numpy as np
from matplotlib import pyplot as plt
import wave
from scipy.io.wavfile import write
import signal_converter
from midiutil import MIDIFile


def get_info_from_audio_file(path):
    file = wave.open(path, 'r')
    samples = file.getnframes()
    audio = file.readframes(samples)
    signal = np.frombuffer(audio, dtype=np.int16)
    sample_rate = wave.Wave_read.getframerate(file)

    return signal, sample_rate


def get_post_fft_amplitudes(signal, sample_rate, window_ms=20, increment_ms=10, averages=-1):
    run_average = averages > 0

    # Measured in number of samples
    window = int(0.001 * window_ms *sample_rate)
    if run_average:
        increment = int(0.001 * increment_ms * sample_rate)
    else:
        # If we don't average there's no point in making smaller increments
        increment = window


    freqs = np.fft.rfftfreq(window, 1 / sample_rate)

    curr_averages = []
    if run_average:
        curr_averages = [[] for _ in range(len(freqs))]
    post_fft_amps = []

    for i in range(0, len(signal) - window, increment):
        windowed_signal = signal[i: i + window]
        fft_windowed_signal = np.fft.rfft(windowed_signal)
        if run_average:
            update_averages(fft_windowed_signal, curr_averages, averages)
            if i % window == 0:
                post_fft_amps.append([np.average(curr_averages[j]) for j in range(len(freqs))])
        else:
            post_fft_amps.append(fft_windowed_signal)

    return post_fft_amps, freqs


def update_averages(fft_windowed_signal, curr_averages, averages):
    for freq_id in range(len(fft_windowed_signal)):
        curr_averages[freq_id].append(fft_windowed_signal[freq_id])
        if len(curr_averages[freq_id]) > averages:
            curr_averages[freq_id].pop(0)


def get_signal_from_blocks_irfft(post_fft_amps):
    signal = []
    for block in post_fft_amps:
        block_signal = np.fft.irfft(block)
        signal.extend(block_signal)
    return np.array(signal, dtype=np.int16)


def convert_to_logscale(freqs, amplitudes_over_time):
    i = 0
    new_freqs = []
    while freqs[i] < 7000:
        new_freqs.append(freqs[i])
        i += 1
    max_freq = max(freqs)
    curr_freq = freqs[i - 1]
    while curr_freq < max_freq:
        new_freqs.append(curr_freq)
        curr_freq *= 2 ** (1 / 24)

    new_amplitudes_over_time = []
    for block in amplitudes_over_time:
        new_block = [0] * len(block)
        j = 0
        for new_freq in new_freqs:
            amp_sum = 0
            while j < len(block) and freqs[j] <= new_freq:
                amp_sum += block[j]
                j += 1
            new_block[j - 1] = amp_sum
        new_amplitudes_over_time.append(new_block)

    return new_freqs, new_amplitudes_over_time


def delete_noise(amps, freqs):
    for block in amps:
        for i in range(len(block)):
            if np.abs(block[i]) < 1e4:
                block[i] = 0

def plot_audio_signal(signal, title, sample_rate):
    plt.plot(np.arange(0, len(signal)/sample_rate, 1/sample_rate), signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    plt.clf()


def plot_graph1(freqs, post_fft_frequencies):
    plt.title("Scatter Plot")
    x = []
    y = []
    z = []
    for freq_id in range(len(freqs)//4):
        for time_id in range(len(post_fft_frequencies)):
            block = post_fft_frequencies[time_id]
            amp = np.abs(block[freq_id])
            if amp < 1e6:
                continue
            x.append(time_id)
            y.append(freqs[freq_id])
            z.append(amp)
    plt.scatter(x, y, c=z, s=3)
    # plt.clim(0, 7e5)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    plt.clf()


def get_max_freq(amps, freqs):
    return freqs[np.argmax(amps)]

def get_note_name(idx):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = notes[idx % 12]
    number = str(idx // 12)
    return note + number


def get_closest_note(freq):
    idx = np.searchsorted(notes, freq)
    if idx > 107:
        return 107
    if idx > 0:
        idx = idx-1 if freq-notes[idx-1] < notes[idx]-freq else idx
    return idx


# def get_note_from_block(block, freqs):
#         idx = get_closest_note(freqs[np.argmax(block)])
#         return get_note_name(idx)


def get_note_idx_from_block(block, freqs):
    notes_average_amplitude = [[0, 0] for j in range(108)]
    for i in range(len(freqs)):
        note_idx = get_closest_note(freqs[i])
        notes_average_amplitude[note_idx][0] += block[i]
        notes_average_amplitude[note_idx][1] += 1

    idx = 0
    value = notes_average_amplitude[0][0] / notes_average_amplitude[0][1]
    for j in range(1, len(notes_average_amplitude)):
        if notes_average_amplitude[j][1] == 0:
            continue
        tmp = notes_average_amplitude[j][0] / notes_average_amplitude[j][1]
        if tmp > value:
            value = tmp
            idx = j
    return idx



notes = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.5, 25.96, 27.5, 29.14, 30.87, 32.7, 34.65, 36.71, 38.89,
         41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98, 103.83,
         110, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94,
         261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392, 415.3, 440, 466.16, 493.88, 523.25, 554.37,
         587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880, 932.33, 987.77, 1046.5, 1108.73, 1174.66,
         1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760, 1864.66, 1975.53, 2093, 2217.46, 2349.32, 2489.02,
         2637, 2793, 2959, 3135, 3322, 3520, 3729, 3951, 4186, 4434, 4698, 4978, 5274, 5587, 5919, 6271, 6644, 7040,
         7458, 7902]



sg, sr = get_info_from_audio_file("audio_files/C-Guitar.wav")
plot_audio_signal(sg, "1", sr)
amps1, freqs1 = get_post_fft_amplitudes(sg, sr, window_ms=300, increment_ms=50, averages=-1)
notes_per_block = []
degrees = []

for block in amps1:
    notes_per_block.append(get_note_name(get_note_idx_from_block(block, freqs1)))
    degrees.append(get_note_idx_from_block(block, freqs1) + 12)
print(notes_per_block)
print(degrees)
# processed_signal = get_signal_from_blocks_irfft(amps1)
plot_graph1(freqs1, amps1)
# plot_audio_signal(processed_signal, "2", sr)
# write("audio_files/after/C-Nofar.wav", sr, processed_signal)

# delete_noise(amps1, freqs1)

# freqs_log, amps_log = convert_to_logscale(freqs1, amps1)
# new_signal = get_signal_from_blocks_irfft(amps_log)
# plot_audio_signal(new_signal, "3", sr)
# plot_graph(freqs1, amps_log)
# write("audio_files/after/guitar1_after_log.wav", sr, new_signal)


track    = 0
channel  = 0
time     = 0    # In beats
duration = 0.3    # In beats
tempo    = 60   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, time, tempo)

for i, pitch in enumerate(degrees):
    MyMIDI.addNote(track, channel, pitch, time + i*0.3, duration, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)