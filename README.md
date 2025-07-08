# Generating Modulation Schemes for Wireless Communications Using Transformer Models

This repository contains the source code, sample data, and visual outputs associated with the paper:

> **"Generating Modulation Schemes for Wireless Communications Using Transformer Models"**  
> Andrea Melis, Andrea Piroddi, Roberto Girau  
> Accepted for oral presentation at the **V. International Conference on Electrical, Computer and Energy Technologies (ICECET 2025)** – Paris, France.

---

## 🧠 Overview

This work explores the use of GPT-2, a Transformer-based generative model, to create novel wireless modulation schemes for Cognitive Radio systems.  
The generated formulas are syntactically validated and evaluated through numerical simulations to compare their performance with traditional schemes like QPSK and 16-QAM.

---

## 📂 Repository Structure

├── data/ # Dataset of known modulation formulas
├── models/ # Fine-tuned GPT-2 model and tokenizer
├── generation/ # Code for sampling and validating generated modulations
├── simulation/ # Performance evaluation (BER, SNR, spectrograms, etc.)
├── results/ # Figures: spectrograms, constellation diagrams, tables
├── notebooks/ # Example Jupyter notebooks
└── README.md


---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- Transformers (`transformers==4.x`)
- NumPy, SciPy, Matplotlib
- PyTorch or TensorFlow (for model inference)
- Jupyter (for notebooks)

### Installation

```bash
git clone https://github.com/your-username/modulation-transformer.git
cd modulation-transformer
pip install -r requirements.txt

@inproceedings{melis2025modulation,
  title={Generating Modulation Schemes for Wireless Communications Using Transformer Models},
  author={Melis, Andrea and Piroddi, Andrea and Girau, Roberto},
  booktitle={Proceedings of the V. International Conference on Electrical, Computer and Energy Technologies (ICECET)},
  year={2025}
}

---

🔧 Vuoi che ti generi anche il file `requirements.txt`? Posso aiutarti a impacchettare il repo completo. Fammi sapere se usi Hugging Face, PyTorch o un altro stack!
