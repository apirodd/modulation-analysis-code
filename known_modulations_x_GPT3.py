import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to add AWGN noise to a signal
def add_awgn_noise(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

# Function to calculate the SNR of the signal
def calculate_snr(signal, noisy_signal):
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noisy_signal - signal))
    snr_linear = signal_power / noise_power
    return 10 * np.log10(snr_linear)

# Function to calculate BER by comparing original and received signals
def calculate_ber(original_signal, received_signal):
    original_bits = (original_signal > 0).astype(int)
    received_bits = (received_signal > 0).astype(int)
    bit_errors = np.sum(original_bits != received_bits)
    return bit_errors / len(original_bits)

# Function to normalize the power of the signal
def normalize_signal(signal):
    power = np.mean(np.abs(signal)**2)
    return signal / np.sqrt(power)

# Modulation schemes
def generate_chirp_signal(t, A, f_0, f_1):
    return A * np.sin(2 * np.pi * (f_0 + (f_1 - f_0) * t / max(t)) * t)

def generate_gmsk_signal(bits, f_c, A, t, BT=0.3):
    phase = np.cumsum(bits) * 2 * np.pi / BT
    return A * np.cos(2 * np.pi * f_c * t + phase)

def generate_msk_signal(bits, f_c, A, t):
    I = 2 * bits[::2] - 1
    Q = 2 * bits[1::2] - 1
    phase = np.cumsum(I - Q) * 2 * np.pi
    return A * np.cos(2 * np.pi * f_c * t + phase)

def generate_qam_signal(bits, f_c, A, t, order):
    levels = int(np.sqrt(order))  # Number of levels based on order (16, 64, etc.)
    I = 2 * np.random.randint(0, levels, len(bits)//2) - (levels - 1)
    Q = 2 * np.random.randint(0, levels, len(bits)//2) - (levels - 1)
    return A * (I * np.cos(2 * np.pi * f_c * t[:len(I)]) - Q * np.sin(2 * np.pi * f_c * t[:len(Q)]))

def generate_bpsk_signal(bits, f_c, A, t):
    modulated_bits = 2 * bits - 1  # Convert bits to +1/-1
    return A * modulated_bits * np.cos(2 * np.pi * f_c * t)

def generate_qpsk_signal(bits, f_c, A, t):
    I = 2 * bits[::2] - 1  # In-phase bits
    Q = 2 * bits[1::2] - 1  # Quadrature bits
    return A * (I * np.cos(2 * np.pi * f_c * t[:len(I)]) - Q * np.sin(2 * np.pi * f_c * t[:len(Q)]))

def generate_ook_signal(bits, f_c, A, t):
    return A * bits * np.cos(2 * np.pi * f_c * t)

def generate_bfsk_signal(bits, A, t, f_c1, f_c2):
    return A * (np.cos(2 * np.pi * f_c1 * t) * (bits == 0) + np.cos(2 * np.pi * f_c2 * t) * (bits == 1))

# Evaluate each modulation scheme
def evaluate_modulation(bits, f_c, A, t, modulation_type, snr_db):
    if modulation_type == 'Chirp':
        signal = generate_chirp_signal(t, A, f_0=500, f_1=1000)
    elif modulation_type == 'GMSK':
        signal = generate_gmsk_signal(bits, f_c, A, t)
    elif modulation_type == 'MSK':
        signal = generate_msk_signal(bits, f_c, A, t)
    elif modulation_type == 'QAM':
        signal = generate_qam_signal(bits, f_c, A, t, order=16)  # Example with 16-QAM
    elif modulation_type == '16-QAM':
        signal = generate_qam_signal(bits, f_c, A, t, order=16)
    elif modulation_type == '64-QAM':
        signal = generate_qam_signal(bits, f_c, A, t, order=64)
    elif modulation_type == '128-QAM':
        signal = generate_qam_signal(bits, f_c, A, t, order=128)
    elif modulation_type == '256-QAM':
        signal = generate_qam_signal(bits, f_c, A, t, order=256)
    elif modulation_type == 'BPSK':
        signal = generate_bpsk_signal(bits, f_c, A, t)
    elif modulation_type == 'QPSK':
        signal = generate_qpsk_signal(bits, f_c, A, t)
    elif modulation_type == 'OOK':
        signal = generate_ook_signal(bits, f_c, A, t)
    elif modulation_type == 'BFSK':
        signal = generate_bfsk_signal(bits, A, t, f_c1=500, f_c2=1000)
    else:
        raise ValueError(f"Unsupported modulation type: {modulation_type}")
    
    # Normalize the signal for fair comparison
    signal = normalize_signal(signal)
    
    noisy_signal = add_awgn_noise(signal, snr_db)
    snr_calculated = calculate_snr(signal, noisy_signal)
    ber = calculate_ber(signal, noisy_signal)

    return snr_calculated, ber

# Main function to loop through known modulations and save results
def evaluate_known_modulations():
    # Parameters
    f_c = 1000  # Carrier frequency in Hz
    A = 1       # Signal amplitude
    t = np.linspace(0, 1, 1000)  # Time vector
    snr_db = 20  # Target SNR in dB
    
    modulations = ['Chirp', 'GMSK', 'MSK', '16-QAM', '64-QAM', '128-QAM', '256-QAM', 'BPSK', 'QPSK', 'OOK', 'BFSK']
    
    # Generate random bits
    bits = np.random.randint(0, 2, 1000)  # Random bits for modulations

    results = []

    for modulation_type in modulations:
        print(f"Evaluating {modulation_type} modulation:")
        try:
            snr_calculated, ber = evaluate_modulation(bits, f_c, A, t, modulation_type, snr_db)
            print(f"SNR (Calculated): {snr_calculated:.2f} dB")
            print(f"BER: {ber:.6f}")
            results.append([modulation_type, snr_calculated, ber])
        except Exception as e:
            print(f"Error evaluating {modulation_type}: {e}")
        print("-" * 40)

    # Convert results to a DataFrame and save to CSV
    df_results = pd.DataFrame(results, columns=["Modulation", "SNR (dB)", "BER"])
    df_results.to_csv("known_modulation_performance.csv", index=False)
    print("Performance results saved to 'known_modulation_performance.csv'")

# Run the evaluation for known modulations
evaluate_known_modulations()
