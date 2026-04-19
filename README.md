# NLP Neural Pipeline: Urdu Language Processing
**Student ID:** i23-2631  
**Course:** Natural Language Processing (CS-4063)

This repository contains a comprehensive neural pipeline for Urdu NLP, spanning from custom word embedding generation to advanced deep learning architectures for sequence labeling and text classification.

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- `pytorch-crf`

### Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### Data Requirements
To successfully run the notebook, ensure the following three files are present in the root directory:

1. `raw.txt`: Raw article text.
2. `metadata.json`: Article metadata (titles and dates).
3. `cleaned.txt`: Pre-processed text (stopwords and punctuation removed).

---

## Implementation Details

### Part 1: Word Embeddings
- **Methodology:** Implementation of TF-IDF, PPMI, and Skip-gram (Word2Vec).
- **Analytics:** Includes t-SNE dimensionality reduction to visualize semantic clusters and nearest-neighbor analysis for Urdu vocabulary.

### Part 2: BiLSTM Sequence Labeling
- **Task:** Part-of-Speech (POS) tagging and Named Entity Recognition (NER).
- **Model:** 2-layer Bidirectional LSTM with a CRF output layer.
- **Handling:** Implemented variable-length sequence padding and masking to ensure accurate loss calculation.

### Part 3: Transformer Encoder
- **Architecture:** 4-head Multi-Head Attention, 4 Encoder blocks, and Sinusoidal Positional Encodings.
- **Training:** Optimized using the AdamW optimizer and a Cosine Annealing learning rate scheduler.
- **Interpretability:** Includes self-attention heatmaps to visualize token-level focus during classification.

---

## True Performance Results
The metrics below are derived from the actual code execution outputs in the provided notebook.

| Model Type | Task | Accuracy | Status |
|---|---|---|---|
| Transformer Encoder | News Classification | ~53.3% | Overfitting (85% Train vs 53% Val) |
| BiLSTM | POS Tagging | ~19.0% | Minority Class Struggle |

### Key Findings:
- **Transformer Efficiency:** While the Transformer converges significantly faster due to parallelization, it exhibits high variance on small datasets (overfitting).
- **BiLSTM Challenges:** The POS tagger's lower accuracy reflects the high complexity of Urdu syntax and the limitations of training on small-scale annotated sets.

---

## Repository Structure
- `i23-2631_Assignment2_DS_B.ipynb`: Core implementation and training logs.
- `embeddings/`: Directory containing saved `.npy` and `.json` embedding files.
- `models/`: Saved state dictionaries for the trained neural layers.
- `Detailed_NLP_Insights_Report.pdf`: Formal analysis and error report.

---

## 📝 Reproduction Instructions

1. Prepare the environment using the `requirements.txt`.
2. Ensure the Urdu dataset files (`raw.txt`, `metadata.json`, `cleaned.txt`) are in the root folder.
3. Open the Jupyter Notebook and select "Run All".
