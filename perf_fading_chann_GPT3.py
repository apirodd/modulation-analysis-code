import numpy as np
import matplotlib.pyplot as plt

# Define parameters
f_c = 1e3  # Carrier frequency in Hz
A = 1      # Amplitude
fs = 1e4   # Sampling frequency
T = 1      # Duration of the signal
num_symbols = 1000  # Number of symbols
t = np.linspace(0, T, int(fs * T))  # Time vector

# Define phi, m, d, and other necessary variables
phi = np.pi / 4  # Example phase shift
m = np.random.rand(len(t))  # Random modulation index for example, same length as t
d = np.random.rand(len(t))  # Random d for the second modulation, same length as t

# Function to generate QPSK signal
def generate_qpsk_signal(bits, f_c, A, t):
    symbols = [int(bits[i] + bits[i + 1], 2) for i in range(0, len(bits) - 1, 2)]  # Combine pairs of bits
    signal = np.zeros(len(t))
    symbol_duration = len(t) // len(symbols)
    for i, symbol in enumerate(symbols):
        if symbol == 0:  # 00
            signal[i * symbol_duration:(i + 1) * symbol_duration] = A * np.cos(2 * np.pi * f_c * t[i * symbol_duration:(i + 1) * symbol_duration])
        elif symbol == 1:  # 01
            signal[i * symbol_duration:(i + 1) * symbol_duration] = A * np.cos(2 * np.pi * f_c * t[i * symbol_duration:(i + 1) * symbol_duration] + np.pi / 2)
        elif symbol == 2:  # 10
            signal[i * symbol_duration:(i + 1) * symbol_duration] = A * np.cos(2 * np.pi * f_c * t[i * symbol_duration:(i + 1) * symbol_duration] + np.pi)
        elif symbol == 3:  # 11
            signal[i * symbol_duration:(i + 1) * symbol_duration] = A * np.cos(2 * np.pi * f_c * t[i * symbol_duration:(i + 1) * symbol_duration] + 3 * np.pi / 2)
    return signal

# Function to generate the validated modulations
def generate_valid_modulation_1(t, A, f_c, phi, m):
    return A * np.cos(2 * np.pi * f_c * t + phi) + m  # Use m with the same length as t

def generate_valid_modulation_2(t, A, f_c, phi, d):
    return A * np.cos(2 * np.pi * f_c * t + phi) + d  # Use d with the same length as t

def generate_valid_modulation_3(t, A, f_c, phi):
    return A * np.cos(2 * np.pi * f_c * t + phi)  # No additional parameters needed

# Function to simulate multipath and fading
def apply_fading(signal):
    multipath_delays = [0, 0.5e-3, 1e-3]  # Example delays in seconds
    multipath_gains = [1, 0.5, 0.25]      # Gains for different paths
    received_signal = np.zeros_like(signal)
    
    for delay, gain in zip(multipath_delays, multipath_gains):
        delayed_signal = np.roll(signal, int(delay * fs))
        received_signal += gain * delayed_signal
        
    fading_factor = np.random.normal(1, 0.1, len(received_signal))
    return received_signal * fading_factor

# Function to calculate BER
def calculate_ber(original_bits, received_bits):
    bit_errors = np.sum(original_bits != received_bits)
    return bit_errors / len(original_bits)

# Function to calculate SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    return 10 * np.log10(signal_power / noise_power)

# Main simulation
# Generate random bits for QPSK (2 bits per symbol)
bits = np.random.randint(0, 2, size=num_symbols * 2).astype(str)  # Generate enough bits for QPSK

# Generate QPSK signal
qpsk_signal = generate_qpsk_signal(bits, f_c, A, t)

# Apply fading to the QPSK signal
qpsk_signal_faded = apply_fading(qpsk_signal)

# Generate validated modulations
modulation_1_signal = generate_valid_modulation_1(t, A, f_c, phi, m)
modulation_2_signal = generate_valid_modulation_2(t, A, f_c, phi, d)
modulation_3_signal = generate_valid_modulation_3(t, A, f_c, phi)

# Apply fading to the validated modulations
modulation_1_faded = apply_fading(modulation_1_signal)
modulation_2_faded = apply_fading(modulation_2_signal)
modulation_3_faded = apply_fading(modulation_3_signal)

# Simulate received bits for QPSK
received_bits_qpsk = bits  # In a real scenario, this would be received bits

# Calculate BER and SNR for QPSK
ber_qpsk = calculate_ber(bits, received_bits_qpsk)  # Simulated received bits
snr_qpsk = calculate_snr(qpsk_signal_faded, np.random.normal(0, 0.1, len(qpsk_signal_faded)))

# Calculate BER and SNR for validated modulations
# Here we need to define how we simulate received bits for the validated modulations.
# This is a placeholder for the actual logic.
received_bits_modulation_1 = np.random.randint(0, 2, size=num_symbols * 2)
received_bits_modulation_2 = np.random.randint(0, 2, size=num_symbols * 2)
received_bits_modulation_3 = np.random.randint(0, 2, size=num_symbols * 2)

ber_modulation_1 = calculate_ber(bits, received_bits_modulation_1)
ber_modulation_2 = calculate_ber(bits, received_bits_modulation_2)
ber_modulation_3 = calculate_ber(bits, received_bits_modulation_3)

snr_modulation_1 = calculate_snr(modulation_1_faded, np.random.normal(0, 0.1, len(modulation_1_faded)))
snr_modulation_2 = calculate_snr(modulation_2_faded, np.random.normal(0, 0.1, len(modulation_2_faded)))
snr_modulation_3 = calculate_snr(modulation_3_faded, np.random.normal(0, 0.1, len(modulation_3_faded)))

# Display results
print("QPSK - BER: {:.4f}, SNR: {:.2f} dB".format(ber_qpsk, snr_qpsk))
print("Validated Modulation 1 - BER: {:.4f}, SNR: {:.2f} dB".format(ber_modulation_1, snr_modulation_1))
print("Validated Modulation 2 - BER: {:.4f}, SNR: {:.2f} dB".format(ber_modulation_2, snr_modulation_2))
print("Validated Modulation 3 - BER: {:.4f}, SNR: {:.2f} dB".format(ber_modulation_3, snr_modulation_3))
