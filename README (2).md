<div align="center">

# 🏥 Healthcare Disease Prediction & Clinical Guidance System

### *Bridging Natural Language and Clinical Diagnosis with AI*

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A state-of-the-art NLP-powered system that transforms natural language health queries into accurate disease predictions and personalized clinical recommendations.**

[Overview](#-overview) • [Architecture](#-system-architecture) • [Model](#-model-architecture) • [Results](#-results--performance) • [Setup](#️-installation--setup) • [Contributors](#-contributors)

---

</div>

## 🩺 Overview

Traditional health systems require precise medical keywords — impractical for the average person. This project eliminates that barrier by enabling users to describe their symptoms naturally, just as they would to a doctor.

The system uses advanced **NLP and deep learning** to:
- 🔍 **Extract and normalize symptoms** from free-form text
- 🧠 **Predict diseases** using a fine-tuned Enhanced PubMedBERT model
- 💊 **Generate personalized recommendations** for medication, diet, exercise, and precautions
- 📊 **Explain predictions** using SHAP and LIME for clinical transparency

> **Problem Statement:** Build an AI assistant that predicts diseases from user-provided **natural language health queries** by accurately extracting symptoms and mapping them to probable medical conditions.

---

## 🏗️ System Architecture

The system follows a multi-stage pipeline — from raw symptom input to clinical guidance output.

<div align="center">

![System Architecture](1771869761697_image.png)

*Figure 1: End-to-end pipeline from user symptom query to top-3 disease predictions with confidence scores*

</div>

### Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| **Input** | Symptom Query | Free-text natural language input from user |
| **Preprocessing** | Text Cleaning + MCN | Normalize medical concepts via Symptom Dictionary |
| **NLP Processing** | BERT Embeddings | Tokenization, attention mechanism, contextual feature extraction |
| **Classification** | Multi-Branch Model | Main Disease Classifier + Symptom Classifier with combined predictions |
| **Output** | Top-3 Predictions | Disease predictions ranked by confidence score |

---

## 🧠 Model Architecture

The core model is an **Enhanced PubMedBERT** with a dual-classifier architecture specifically designed for biomedical NLP tasks.

<div align="center">

![Model Architecture](1771869772097_image.png)

*Figure 2: EnhancedSymptomClassifier — PubMedBERT backbone with attention pooling and dual-branch classification head*

</div>

### Architecture Components

**1. PubMedBERT Backbone**
- Domain-specific BERT pre-trained on biomedical literature
- Input: `(batch × seq_len)` tokenized symptom text
- Output: Hidden states `(batch × seq_len × 768)`

**2. Attention Mechanism**
- Custom attention layer: `768 → 128 → 1`
- Learns to weight the most clinically relevant tokens
- Produces weighted context vector for classification

**3. Dual Classification Head**
- **[CLS] Path:** `Linear(512) → LeakyReLU + BatchNorm → Linear(256) → LeakyReLU + BatchNorm → Linear(128) → classes`
- **Auxiliary Branch:** `Linear(128) → LeakyReLU → Linear(64) → classes`
- **Combined Output:** Weighted ensemble `(0.7 × main + 0.3 × auxiliary)`

---

## 📊 Results & Performance

### Training Configuration

<div align="center">

![Training Metrics](1771869785073_image.png)

*Figure 3: Full training configuration and final performance metrics*

</div>

| Metric | Value |
|--------|-------|
| **Model** | Enhanced PubMedBERT (attention + dual classifier) |
| **Best Validation Accuracy** | **98.07%** |
| **Best Validation F1-Score** | **0.9808** |
| **Best Validation Loss** | **0.1658** |
| **Training Loss (Final Epoch)** | 0.4232 |
| Epochs | 10 |
| Batch Size | 16 |
| Learning Rate | 2e-5 (BERT layers: 2e-6) |
| Optimizer | AdamW with LR scheduling & weight decay |
| Loss Function | Weighted Cross-Entropy |

### Training History

The model demonstrated consistent improvement across all 10 epochs with no signs of overfitting:

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 |
|-------|-----------|----------|-------------|--------|
| 1 | 3.358 | 2.435 | 89.43% | 0.8911 |
| 3 | 2.115 | 1.312 | 93.39% | 0.9273 |
| 5 | 1.462 | 0.811 | 96.65% | 0.9665 |
| 7 | 0.957 | 0.443 | 97.46% | 0.9751 |
| **10** | **0.423** | **0.166** | **98.07%** | **0.9808** |

---

## 💡 Live Demo — Prediction Output

<div align="center">

![Prediction Result](1771869794883_image.png)

*Figure 4: Sample prediction UI showing Top-3 disease predictions with confidence scores and full clinical guidance for Heart Attack*

</div>

The system outputs:
- **Top 3 predicted diseases** with confidence scores (e.g., Heart Attack: 51.35%)
- **Extracted symptoms** identified from the natural language query
- **Disease description** in plain language
- **Specific precautions** (e.g., call ambulance, chew aspirin, keep calm)
- **Treatment recommendations** — mild (Aspirin, nitroglycerin) and intense (angioplasty, bypass surgery)

---

## ⚙️ Core Methodology

### 1. Natural Language Processing

**Medical Concept Normalization (MCN)**
A custom Symptom Dictionary maps everyday expressions to standardized medical terms:
- `"stomach ache"` → `abdominal_pain`
- `"tummy hurts"` → `abdominal_pain`
- `"can't breathe"` → `breathlessness`

**Symptom Extraction Pipeline**
- Spell correction using SymSpell
- Fuzzy matching with RapidFuzz
- Sentence embeddings via SentenceTransformers
- spaCy NLP for entity recognition

### 2. Disease Prediction Models

| Model | Type | Notes |
|-------|------|-------|
| **Enhanced PubMedBERT** | Deep Learning | Best performer — 98.07% accuracy |
| MCN-BERT | Deep Learning | Baseline transformer model |
| BiLSTM | Deep Learning | Sequential symptom modeling |
| SVC | Machine Learning | High accuracy baseline |
| Random Forest | Machine Learning | Robust ensemble baseline |

### 3. Recommendation & Explainability

- **Medication, diet, and exercise** recommendations mapped to each disease
- **Critical precautions** triggered by high-risk symptom combinations
- **SHAP** (SHapley Additive exPlanations) for global feature importance
- **LIME** (Local Interpretable Model-agnostic Explanations) for per-prediction transparency

---

## 📊 Datasets

| Dataset | Size | Coverage | Use |
|---------|------|----------|-----|
| **Medical Dataset-1** | 4,920 instances | 132 symptoms, 41 diseases | Primary training & recommendation |
| **Symptom2Disease** | 1,200 instances | 24 disease classes | LLM fine-tuning (LLaMA 3, Mistral-7B) |

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/sayyedsabirali/Minor-Project.git
cd Minor-Project
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Run the application**
```bash
# Launch Streamlit web interface
streamlit run main_app.py

# Or run the Jupyter notebook
jupyter notebook main.ipynb
```

### Dependencies

```
spacy · nltk · numpy · sentence_transformers · scikit-learn
pandas · torch · transformers · matplotlib · seaborn
symspellpy · rapidfuzz · streamlit · ipykernel
```

---

## 🚀 Future Scope

- **Expanded Coverage** — Scale beyond 132 symptoms and 41 diseases for broader clinical applicability
- **Symptom Dictionary** — Continuous refinement with clinical ontologies (SNOMED CT, ICD-10)
- **LLM Benchmarking** — Fine-tuning and evaluation of LLaMA 3, Mistral-7B, and newer models
- **Multilingual Support** — Extend NLP pipeline to handle non-English symptom descriptions
- **EHR Integration** — Connect with Electronic Health Records for richer patient context
- **Mobile Deployment** — Lightweight model variants for edge/mobile deployment

---

## 👨‍💻 Contributors

<div align="center">

| Name | Faculty No. | Role |
|------|-------------|------|
| **Sayyed Sabir Ali** | 22AIB320 | Lead Developer & NLP Engineer |
| **Mohd Fazil** | 22AIB261 | ML Engineer & System Architect |

**Supervisor:** Ms. Ayesha Khan  
**Institution:** ZHCET AMU, Aligarh

</div>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Made with ❤️ for accessible healthcare — ZHCET AMU, 2024*

**⭐ Star this repo if you found it helpful!**

</div>
