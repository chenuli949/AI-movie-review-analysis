🎬 Sentiment Analysis of IMDb Movie Reviews — Deep Learning Models Comparison

This project predicts whether a movie review expresses a **positive** or **negative** sentiment using deep learning.
The dataset contains 50,000 English-language movie reviews from IMDb, released as the *Large Movie Review Dataset* (Maas et al., 2011) and made available on Kaggle.

The study compares the performance of four neural network architectures — **MLP, 1D-CNN, LSTM, and Bidirectional LSTM** — to identify the most effective model for binary sentiment classification of long-form review text.

## 📘 Project Overview

Automatically judging the sentiment of a review is a core natural language processing task with applications in recommendation, reputation monitoring and market research.
This project uses the review text to predict the binary variable **sentiment** (`positive` / `negative`), so that large volumes of user-written reviews can be summarized without manual reading.

---

## 📂 Repository Structure

```tree
AI-movie-review-analysis/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Python package dependencies
├── data/                              # Saved partition indices & dataset instructions
│   ├── train_indices.npy              # Standardized training split indices (N=40,000)
│   ├── validation_indices.npy         # Standardized validation split indices (N=5,000)
│   ├── test_indices.npy               # Standardized test split indices (N=5,000)               
├── notebooks/                         # Jupyter Notebooks for pipeline steps
│   ├── 01_EDA_Preprocessing.ipynb    # EDA, cleaning, and dataset splitting
│   ├── 02_MLP Model.ipynb            # TF-IDF + MLP with Embedding Layer
│   ├── 03_1D_CNN.ipynb              # 1D-CNN Text Classifier
│   ├── 04_LSTM.ipynb                 #LSTM Classifier for Sequential Context
│   ├── 05_BiLSTM.ipynb               # Bidirectional LSTM with Dropout
│   
└── 
```

---

## 🧠 Models Implemented

| Model | Description | Key Strength |
|---|---|---|
| **MLP (Multi-Layer Perceptron)** | Baseline feedforward network over TF-IDF bag-of-words features. | Fast, strong on lexical (word-presence) signal. |
| **1D-CNN (Convolutional Neural Network)** | Learns local n-gram patterns from word embeddings via sliding convolution filters. | Detects key phrases regardless of position; efficient to train. |
| **LSTM (Long Short-Term Memory)** | Sequence-based recurrent model that reads the review word by word. | Captures word order, negation and long-range context. |
| **Bidirectional LSTM** | Two LSTMs reading the review forwards and backwards, concatenated. | Uses context from both directions of the sentence. |

Each model was trained independently using the **same preprocessed dataset and the same train/validation/test split**, and compared on Accuracy, Precision, Recall, F1-score and ROC-AUC.

## 📊 Dataset Information

- **Dataset:** [IMDB Dataset of 50K Movie Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Original source:** Maas, A.L., Daly, R.E., Pham, P.T., Huang, D., Ng, A.Y. and Potts, C. (2011), *Large Movie Review Dataset v1.0*, Stanford AI Lab
- **Records:** 50,000 reviews
- **Features:** 1 free-text column (`review`)
- **Target variable:** `sentiment` (`positive` / `negative`)
- **Class balance:** Perfectly balanced — 25,000 positive / 25,000 negative → **no resampling required**
- **Review length:** Highly right-skewed; median ≈ 171 words, 75th percentile ≈ 277 words, max ≈ 2,459 words

## 🧹 Data Preprocessing and Feature Engineering

The preprocessing pipeline (`notebooks/01_EDA_Preprocessing.ipynb`) included:

- **Missing-value check:** none found in either column.
- **HTML tag removal:** stripped `<br />` line-break artifacts from review text.
- **Whitespace normalization:** collapsed repeated whitespace.
- **Label encoding:** `negative → 0`, `positive → 1`.
- **Duplicate check:** 418 exact duplicate reviews identified in the raw data (flagged, see Known Issues in `CONTRIBUTING.md`).
- **Stratified splitting:** 80% train / 10% validation / 10% test, `random_state=42`, saved as shared index files so every model uses identical partitions.
- **Model-specific vectorization** (fitted on the training split only, to prevent leakage):
  - **TF-IDF** (10,000 terms) for the MLP.
  - **Integer token sequences** (20,000-word vocabulary, padded/truncated to 300 tokens) with a learned 128-dim embedding for the 1D-CNN, LSTM and Bidirectional LSTM.
- **No stop-word removal or stemming:** words like "not" and "but" are essential for negation and were kept intact.

## ⚙️ Model Training Workflow

1. Load the cleaned dataset and shared split indices from `data/`.
2. Fit each model's vectorizer on the training split only.
3. Train each of the 4 models with early stopping on validation loss (`patience=2`, best weights restored).
4. Evaluate once on the held-out test set using Accuracy, Precision, Recall, F1-score, ROC-AUC and the confusion matrix.
5. Compare results and visualize performance across models.

## 🧩 Evaluation Metrics

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- ROC-AUC
- Confusion Matrix
- Matthews Correlation Coefficient (supplementary)

These metrics give a fair, rounded comparison across models — accuracy alone can hide asymmetric error patterns (e.g. a model that misses positive reviews more than negative ones).

## 🖼️ Visualizations

- Sentiment class distribution
- Review length distribution
- Training vs. validation accuracy/loss curves (per model)
- Confusion matrix (per model)
- ROC curve (per model)
- Model performance comparison bar chart and ROC overlay 


## 🚀 How to Run

**1️⃣ Clone the repository**
```bash
git clone https://github.com/chenuli949/AI-movie-review-analysis.git
cd AI-movie-review-analysis
```

**2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Run the notebooks in order**
```
notebooks/01_EDA_Preprocessing.ipynb   # cleaning, EDA, train/val/test split
notebooks/02_MLP_Model.ipynb           # TF-IDF + MLP
notebooks/03_1D_CNN.ipynb              # Embedding + 1D-CNN
notebooks/04_LSTM.ipynb                # Embedding + LSTM
notebooks/05_BiLSTM.ipynb              # Embedding + Bidirectional LSTM
```
Each notebook can also be opened directly in Google Colab (badge at the top of the notebook) — no local setup needed.


## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Key Observations |
|---|---|---|---|---|---|---|
| MLP | 89.88% | 90.40% | 89.24% | 89.81% | 0.9574 | Best overall result; full-review TF-IDF coverage, balanced precision/recall. |
| 1D-CNN | 86.88% | 89.84% | 83.16% | 86.37% | — | Fast, competitive; slightly favors precision over recall. |
| LSTM | 87.38% | 93.20% | 80.64% | 86.47% | 0.9465 | High precision but conservative — misses more positive reviews (false negatives) than it wrongly flags negative ones. |
| Bidirectional LSTM | 85.74% | 93.23% | 77.08% | 84.39% | 0.9512 | Highest AUC among sequence models but lowest recall; bidirectional context did not translate into higher accuracy here. |


## 👥 Members

|IT Number      | Name                
|---|---|
IT23225442      Abeysekara W.C.S.M.   
IT23291546	    Munidasa T.G.D.L      
IT23268258	    Fernando B.P.L        
IT23282872	    Alawaththa A.K.R.T 
