import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Create directory to save images
if not os.path.exists("imgs"):
  os.makedirs("imgs")

# Define the sine wave function
def sine_wave(freq, t):
    return np.sin(2 * np.pi * freq * t + np.pi/2)

# Parameters
freq = 20  # Frequency of the sine wave (Hz)
sampling_rate = [4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 100]  # Sampling rate (Hz)
duration = 1  # Duration of the signal (seconds)

for i, rate in enumerate(sampling_rate):
    # Time arrays
    t_continuous = np.linspace(0, duration, 1000)
    t_sampled = np.linspace(0, duration, duration * rate, endpoint=False)

    # Generate the sine waves
    continuous_wave = sine_wave(freq, t_continuous)
    sampled_wave = sine_wave(freq, t_sampled)

    # Interpolation for reconstruction using UnivariateSpline
    spline = UnivariateSpline(t_sampled, sampled_wave, k=3, s=0)
    t_reconstructed = np.linspace(0, duration, 1000)
    reconstructed_wave = spline(t_reconstructed)

    # Apply thresholding to the reconstructed wave
    reconstructed_wave[reconstructed_wave > 1] = 1
    reconstructed_wave[reconstructed_wave < -1] = -1

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot the continuous sine wave
    plt.plot(t_continuous, continuous_wave, label='Original Continuous Sine Wave')

    # Plot the sampled points
    plt.scatter(t_sampled, sampled_wave, color='red', label='Sampled Points')

    # Plot the reconstructed sine wave
    plt.plot(t_reconstructed, reconstructed_wave, '--', label='Reconstructed Sine Wave', color='orange')

    # Labels and legend
    plt.title(f'Sine Wave with Frequency {freq}Hz and Sampling Rate {rate}Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'imgs/plot_{rate}.png')