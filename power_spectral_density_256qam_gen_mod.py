import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Constants
A = 1  # Amplitude
f_c = 1000  # Carrier frequency
fs = 10000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector

# Generate bit sequence
np.random.seed(0)  # For reproducibility
num_bits = 1000
bits = np.random.randint(0, 2, num_bits)

# Function to generate 256-QAM signal
def generate_256qam_signal(bits, f_c, A, t):
    # Map bits to symbols (each symbol consists of 8 bits for 256-QAM)
    symbols = [int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8)]
    
    # Create a complex signal array
    signal = np.zeros(len(t), dtype=complex)

    # Construct the signal based on the symbols
    for symbol in symbols:
        if symbol < 256:
            phase = (symbol * 2 * np.pi) / 256  # Map symbol to phase
            signal += A * np.exp(1j * phase) * np.cos(2 * np.pi * f_c * t)  # QAM modulation

    return signal.real  # Return only the real part

# Function to generate validated modulation 1
def generate_valid_modulation_1(t, A, f_c, phi):
    return A * np.cos(2 * np.pi * f_c * t + phi)  # Simplified example

# Function to generate validated modulation 2
def generate_valid_modulation_2(t, A, f_c, phi):
    return A * np.cos(2 * np.pi * f_c * t) + (A * np.sin(2 * np.pi * f_c * t))

# Function to generate validated modulation 3
def generate_valid_modulation_3(t, A, f_c, phi):
    return A * np.sin(2 * np.pi * f_c * t + phi)

# Generate signals
qam_signal = generate_256qam_signal(bits, f_c, A, t)
valid_modulation_1 = generate_valid_modulation_1(t, A, f_c, 0)
valid_modulation_2 = generate_valid_modulation_2(t, A, f_c, 0)
valid_modulation_3 = generate_valid_modulation_3(t, A, f_c, 0)

# Calculate PSD using Welch's method
frequencies, psd_qam = welch(qam_signal, fs, nperseg=256)
_, psd_valid_mod_1 = welch(valid_modulation_1, fs, nperseg=256)
_, psd_valid_mod_2 = welch(valid_modulation_2, fs, nperseg=256)
_, psd_valid_mod_3 = welch(valid_modulation_3, fs, nperseg=256)

# Print results
print("Power Spectral Density Results:")
print("Frequencies (Hz):", frequencies)
print("256-QAM PSD:", psd_qam)
print("Validated Modulation 1 PSD:", psd_valid_mod_1)
print("Validated Modulation 2 PSD:", psd_valid_mod_2)
print("Validated Modulation 3 PSD:", psd_valid_mod_3)

# Plotting with larger font sizes
plt.figure(figsize=(12, 8))
plt.semilogy(frequencies, psd_qam, label='256-QAM')
plt.semilogy(frequencies, psd_valid_mod_1, label='Validated Modulation 1')
plt.semilogy(frequencies, psd_valid_mod_2, label='Validated Modulation 2')
plt.semilogy(frequencies, psd_valid_mod_3, label='Validated Modulation 3')

# Set larger font sizes
plt.title('Power Spectral Density Comparison', fontsize=20)
plt.xlabel('Frequency (Hz)', fontsize=18)
plt.ylabel('PSD (V^2/Hz)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)

# Display the plot
plt.show()
