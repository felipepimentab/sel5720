import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the sine wave function
def sine_wave(freq, t):
    return np.sin(2 * np.pi * freq * t)

# Parameters
freq = 5  # Frequency of the sine wave (Hz)
sampling_rate = 11  # Sampling rate (Hz)
duration = 1  # Duration of the signal (seconds)

# Time arrays
t_continuous = np.linspace(0, duration, 1000)
t_sampled = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

# Generate the sine waves
continuous_wave = sine_wave(freq, t_continuous)
sampled_wave = sine_wave(freq, t_sampled)

# Interpolation for reconstruction
reconstructed_func = interp1d(t_sampled, sampled_wave, kind='linear', fill_value="extrapolate")
t_reconstructed = np.linspace(0, duration, 1000)
reconstructed_wave = reconstructed_func(t_reconstructed)

# Plotting
plt.figure(figsize=(14, 8))

# Plot the continuous sine wave
plt.plot(t_continuous, continuous_wave, label='Original Continuous Sine Wave')

# Plot the sampled points
plt.scatter(t_sampled, sampled_wave, color='red', label='Sampled Points')

# Plot the reconstructed sine wave
plt.plot(t_reconstructed, reconstructed_wave, '--', label='Reconstructed Sine Wave', color='orange')

# Labels and legend
plt.title(f'Sine Wave with Frequency {freq}Hz and Sampling Rate {sampling_rate}Hz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig("plot.png")
