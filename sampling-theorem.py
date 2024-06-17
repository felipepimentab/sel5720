import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

# Define the sine wave function with phase
def sine_wave(freq, t, phase=0):
    return np.sin(2 * np.pi * freq * t + phase)

# Ensure the output directory exists
output_dir = 'imgs'
os.makedirs(output_dir, exist_ok=True)

# Parameters
freq = 20  # Frequency of the sine wave (Hz)
sampling_rates = [8, 10, 12, 14, 16, 20, 40, 42, 53, 61, 74, 80, 100]  # Array of sampling rates (Hz)
duration = 1.5  # Duration of the signal (seconds)
time_start = -0.25  # Start time for the signal
time_end = 1.25  # End time for the signal
crop_start = 0.0  # Start time for cropping
crop_end = 1.0  # End time for cropping
phase = np.pi / 4  # Phase shift of the sine wave

# Time arrays
t_continuous = np.linspace(time_start, time_end, 1500)

# Generate the continuous sine wave
continuous_wave = sine_wave(freq, t_continuous, phase)

for sampling_rate in sampling_rates:
    # Time array for sampled points
    t_sampled = np.linspace(time_start, time_end, int((time_end - time_start) * sampling_rate), endpoint=False)

    # Generate the sampled sine wave
    sampled_wave = sine_wave(freq, t_sampled, phase)

    # Interpolation for reconstruction using UnivariateSpline
    spline = UnivariateSpline(t_sampled, sampled_wave, k=3, s=0)
    t_reconstructed = np.linspace(time_start, time_end, 1500)
    reconstructed_wave = spline(t_reconstructed)

    # Apply thresholding to the reconstructed wave
    reconstructed_wave[reconstructed_wave > 1] = 1
    reconstructed_wave[reconstructed_wave < -1] = -1

    # Extract the central part of the waves for FFT
    crop_start_idx = np.searchsorted(t_continuous, crop_start)
    crop_end_idx = np.searchsorted(t_continuous, crop_end)
    central_continuous_wave = continuous_wave[crop_start_idx:crop_end_idx]
    central_reconstructed_wave = reconstructed_wave[crop_start_idx:crop_end_idx]

    # FFT of the central parts of the waves
    fft_continuous = np.fft.fft(central_continuous_wave)
    fft_reconstructed = np.fft.fft(central_reconstructed_wave)

    # Frequency axis for the FFT
    central_t_continuous = t_continuous[crop_start_idx:crop_end_idx]
    freq_axis = np.fft.fftfreq(len(central_t_continuous), d=(central_t_continuous[1] - central_t_continuous[0]))

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))

    # Plot the continuous sine wave and reconstructed sine wave
    axs[0].plot(t_continuous, continuous_wave, label='Original Continuous Sine Wave')
    axs[0].scatter(t_sampled, sampled_wave, color='red', label='Sampled Points')
    axs[0].plot(t_reconstructed, reconstructed_wave, '--', label='Reconstructed Sine Wave', color='orange')
    axs[0].set_title(f'Sine Wave with Frequency {freq}Hz, Phase {phase} and Sampling Rate {sampling_rate}Hz')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim([crop_start, crop_end])  # Crop the wave plot

    # Plot the FFTs of both waves
    axs[1].plot(freq_axis, np.abs(fft_continuous), label='FFT of Original Sine Wave')
    axs[1].plot(freq_axis, np.abs(fft_reconstructed), label='FFT of Reconstructed Sine Wave', color='orange')
    axs[1].set_xlim([0, 100])  # Crop the FFT plot to -100 Hz to 100 Hz
    axs[1].set_title('FFT of Original and Reconstructed Sine Waves')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_{sampling_rate}.png')
    plt.close(fig)

# Update README.md
readme_path = 'README.md'
readme_content = []

with open(readme_path, 'r') as f:
    readme_content = f.readlines()

# Find the position of the first '---'
insert_idx = readme_content.index('---\n') + 1

# Remove everything after the first '---'
readme_content = readme_content[:insert_idx]

# Add new lines for each sampling rate
for sampling_rate in sampling_rates:
    new_lines = [
        f'\n\n#### Amostragem em {sampling_rate}Hz',
        f'\n\n![Amostragem em {sampling_rate}Hz](imgs/plot_{sampling_rate}.png)',
        '\n\n---'
    ]
    readme_content.extend(new_lines)

# Write the updated content back to README.md
with open(readme_path, 'w') as f:
    f.writelines(readme_content)