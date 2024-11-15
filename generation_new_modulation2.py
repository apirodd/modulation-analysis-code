import random
import numpy as np
import pandas as pd

# Definiamo le formule di modulazione come stringhe che possiamo utilizzare
modulation_formulas = {
    "AM": "A_c * (1 + m * np.cos(2 * np.pi * f_m * t + phi_m)) * np.cos(2 * np.pi * f_c * t + phi_c)",
    "FM": "A_c * np.cos(2 * np.pi * f_c * t + k_f * np.cumsum(m) + phi_c)",
    "PM": "A_c * np.cos(2 * np.pi * f_c * t + k_p * m + phi_c)",
    "QAM": "I(t) * np.cos(2 * np.pi * f_c * t) - Q(t) * np.sin(2 * np.pi * f_c * t)",
    "BPSK": "A_c * np.cos(2 * np.pi * f_c * t + np.pi * d(t))",
    "QPSK": "A_c * np.cos(2 * np.pi * f_c * t + (np.pi / 2) * d(t))",
    "DPSK": "A_c * np.cos(2 * np.pi * f_c * t + phi_prev + delta_phi)",
    "FSK": "A_c * np.cos(2 * np.pi * f(t) * t + phi)",
    "Chirp": "A * np.cos(2 * np.pi * (f_0 * t + 0.5 * K * t**2) + phi)",
    "GMSK": "A * np.cos(2 * np.pi * f_c * t + np.pi * h * np.cumsum(m) + phi)",
    "OFDM": "np.sum([A_k * np.cos(2 * np.pi * f_k * t + phi_k) for k in range(N)])",
    "ASK": "A * np.cos(2 * np.pi * f_c * t + phi) if d(t) == 1 else 0",
    "CPM": "A * np.cos(2 * np.pi * f_c * t + theta(t))",
    "MSK": "A * np.cos(2 * np.pi * f_c * t + np.pi * np.cumsum(m) + phi)",
    "OOK": "A * d(t) * np.cos(2 * np.pi * f_c * t + phi)",
    "16QAM": "I(t) * np.cos(2 * np.pi * f_c * t) - Q(t) * np.sin(2 * np.pi * f_c * t)",
    "64QAM": "I(t) * np.cos(2 * np.pi * f_c * t) - Q(t) * np.sin(2 * np.pi * f_c * t)",
    "128QAM": "I(t) * np.cos(2 * np.pi * f_c * t) - Q(t) * np.sin(2 * np.pi * f_c * t)",
    "256QAM": "I(t) * np.cos(2 * np.pi * f_c * t) - Q(t) * np.sin(2 * np.pi * f_c * t)",
    "BFSK": "A * np.cos(2 * np.pi * f_d * t + phi)",
    "PAM": "np.sum([A_n * p(t - n * T) for n in range(N)])",
    "DPSK-BPSK": "A * np.cos(2 * np.pi * f_c * t + phi_prev + delta_phi)",
    "DPSK-QPSK": "A * np.cos(2 * np.pi * f_c * t + phi_prev + delta_phi)",
    "Hamming": "Codifica usata in altre modulazioni",
    "Pulse Position Modulation (PWM)": "A * square(2 * np.pi * f_c * t, duty=m(t))",
    "Delta Modulation": "Delta * np.sum([d_n * p(t - n * T) for n in range(N)])",
    "Spectral Amplitude Coding": "np.sum([A_k * np.cos(2 * np.pi * f_k * t + phi_k) for k in range(N)])",
    "Code-Division Multiple Access": "np.sum([d_k * np.cos(2 * np.pi * f_c * t + phi_k) * c_k(t) for k in range(N)])",
    "Frequency Hopping Spread Spectrum": "A * np.cos(2 * np.pi * f_h(t) * t + phi)",
    "Multi-Carrier CDMA": "np.sum([d_k * np.cos(2 * np.pi * f_ck * t + phi_k) * c_k(t) for k in range(N)])",
    "Adaptive Modulation and Coding": "A * np.cos(2 * np.pi * f_c * t + phi + (2 * np.pi * d(t) / M))"
}

# Funzione per generare una combinazione casuale di modulazioni
def generate_random_modulation():
    modulation_types = list(modulation_formulas.keys())
    selected_modulations = random.sample(modulation_types, 2)
    combined_modulation = f"({modulation_formulas[selected_modulations[0]]}) + ({modulation_formulas[selected_modulations[1]]})"
    return combined_modulation, f"Combined_{selected_modulations[0]}_{selected_modulations[1]}"

# Creare un dataset di combinazioni casuali
def create_modulation_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        formula, label = generate_random_modulation()
        dataset.append({"Label": label, "Formula": formula})
    return pd.DataFrame(dataset)

# Creare un dataset delle modulazioni standard
def create_standard_modulation_dataset():
    dataset = [{"Label": label, "Formula": formula} for label, formula in modulation_formulas.items()]
    return pd.DataFrame(dataset)

# Creare dataset delle modulazioni standard e casuali
standard_modulation_dataset = create_standard_modulation_dataset()
random_modulation_dataset = create_modulation_dataset(50)

# Combinare i dataset
combined_dataset = pd.concat([standard_modulation_dataset, random_modulation_dataset], ignore_index=True)

# Salvare il DataFrame combinato in un file CSV
file_path = "random_modulations.csv"
combined_dataset.to_csv(file_path, index=False)

print(f"File '{file_path}' creato con le modulazioni standard e casuali.")
