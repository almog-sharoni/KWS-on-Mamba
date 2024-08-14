import numpy as np


FREQUENCY = 16000
DURATION = 16000
def add_time_shift_noise_and_align(audio, max_shift_in_ms=100):
    # randomly shift the audio by at most max_shift_in_ms
    max_shift = (max_shift_in_ms * FREQUENCY) // 1000
    time_shift = np.random.randint(0, max_shift)
    future = np.random.randint(0, 2)

    if future == 0:
        audio = np.pad(audio[time_shift:], (0, time_shift), 'constant')
    else:
        audio = np.pad(audio[:-time_shift], (time_shift, 0), 'constant')

    # Ensure the audio tensor has the correct length
    if len(audio) < DURATION:
        audio = np.pad(audio, (DURATION - len(audio), 0), 'constant')

    return audio[:DURATION]

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise