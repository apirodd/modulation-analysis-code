import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Parametri generali
fs = 1e3  # Frequenza di campionamento
t = np.arange(0, 1, 1/fs)  # Asse temporale
fc = 50  # Frequenza portante
A = 1  # Ampiezza del segnale
phi = np.pi / 4  # Fase iniziale
n = 5  # Numero di componenti nel sommatorio
m = np.random.randn(len(t))  # Segnale modulante casuale
I = np.cos(2 * np.pi * t)  # Componente I
Q = np.sin(2 * np.pi * t)  # Componente Q
d = np.sign(np.sin(2 * np.pi * 3 * t))  # Segnale dati

# Funzione per calcolare componenti In-Phase e Quadrature
def calculate_iq(signal):
    I = np.cos(2 * np.pi * fc * t) * signal
    Q = -np.sin(2 * np.pi * fc * t) * signal
    return I, Q

# Modulation Schemes
# Modulation 1
mod1 = I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t) + \
       (A * np.cos(2 * np.pi * fc * t + phi)) + \
       (A * np.cos(2 * np.pi * fc * t + phi)) + \
       (A * np.sum(m) / len(m))  # Normalizzato

# Modulation 2
mod2 = I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t) + \
       (A * np.cos(2 * np.pi * fc * t)) + \
       (A * np.pi * d * np.sin(2 * np.pi * fc * t))

# Modulation 3
mod3 = I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t) + \
       phi + (A * np.sin(2 * np.pi * fc * t)) / (Q + 1e-3)  # Epsilon per evitare divisione per zero

# Traditional Modulations
# QPSK
symbols_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j])  # Punti di costellazione
data_qpsk = np.random.choice(symbols_qpsk, len(t) // 100)
I_qpsk = np.repeat(data_qpsk.real, 100)
Q_qpsk = np.repeat(data_qpsk.imag, 100)
qpsk = I_qpsk * np.cos(2 * np.pi * fc * t[:len(I_qpsk)]) - Q_qpsk * np.sin(2 * np.pi * fc * t[:len(Q_qpsk)])

# 16-QAM
qam_levels = np.array([-3, -1, 1, 3])  # Livelli di ampiezza
I_qam16 = np.random.choice(qam_levels, size=len(t) // 100)
Q_qam16 = np.random.choice(qam_levels, size=len(t) // 100)
I_upsampled = np.repeat(I_qam16, 100)  # Simboli up-sampled
Q_upsampled = np.repeat(Q_qam16, 100)
qam16 = I_upsampled * np.cos(2 * np.pi * fc * t[:len(I_upsampled)]) + \
        Q_upsampled * np.sin(2 * np.pi * fc * t[:len(Q_upsampled)])

# Funzione per plottare la mappa tempo-frequenza
def plot_spectrogram(signal, title):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=256)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title(f"Spectrogram - {title}")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power [dB]')
    plt.show()

# Funzione per plottare il diagramma di costellazione
def plot_constellation(I, Q, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(I, Q, color='blue', s=10)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title(f"Constellation Diagram - {title}")
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True)
    plt.show()

# Genera visualizzazioni
signals = {
    "Generated Modulation 1": mod1,
    "Generated Modulation 2": mod2,
    "Generated Modulation 3": mod3,
    "QPSK": qpsk,
    "16-QAM": qam16
}

for name, sig in signals.items():
    print(f"Plotting spectrogram for {name}...")
    plot_spectrogram(sig, name)

    # Calcola e plotta la costellazione
    I_mod, Q_mod = calculate_iq(sig)
    plot_constellation(I_mod, Q_mod, name)
