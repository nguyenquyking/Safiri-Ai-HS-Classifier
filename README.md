# 🌍 Safiri AI - HS Code Classification System

An advanced, dual-architecture AI classification system designed to predict Harmonized System (HS) codes from unstructured product descriptions. Developed as a high-fidelity solution for the **Safiri AI Take-Home Challenge**.

---

## 🏗️ Project Architecture

The project is organized into a modular, professional structure to ensure code quality, scalability, and clear separation of concerns:

```text
/safiri-ai/
├── apps/               # Front-end Applications (Gradio)
│   ├── main_rag/       # Main Generative RAG + LLM App (Port 7861)
│   └── baseline_ml/    # Traditional ML Baseline App (Port 7860)
├── src/                # Core Logic (The Engines)
│   ├── rag/            # Semantic Vector Search Engine (SBERT)
│   └── ml/             # ML Training & Models (TF-IDF + Logistic Reg)
├── data/               # Data Warehouse
│   ├── raw/            # Artificial Intelligence Base Data (WCO)
│   ├── processed/      # Cleaned Train/Test Datasets
│   └── generate_data.py# Synthetic Data Generation Tool
├── evaluation/         # Testing & Metrics
│   ├── scripts/        # Unified evaluation framework
│   └── reports/        # Accuracy & Performance reports
├── config/             # Environment configurations
├── README.md           # Documentation
├── requirements.txt    # Unified dependencies
└── .env                # API Keys (Git-ignored)
```

---

## 🚀 Getting Started

### 1. Prerequisites
Clone the repository and install all dependencies using the unified requirement file:
```bash
git clone https://github.com/nguyenquyking/Safiri-Ai-HS-Classifier.git
cd Safiri-Ai-HS-Classifier
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory (or use the existing one) and add your Google Gemini API Key:
```text
GEMINI_API_KEY=your_api_key_here
```

### 3. Running the Applications

#### **Option A: Main Generative RAG App (The Final Solution)**
This is the full pipeline featuring Vector Search re-ranked by Google Gemini 2.5-Flash.
```bash
python apps/main_rag/app.py
```
> Access at: `http://127.0.0.1:7861`

#### **Option B: Traditional ML Baseline (The Comparator)**
Experience the raw speed of TF-IDF + Logistic Regression with LIME explainability.
```bash
python apps/baseline_ml/app.py
```
> Access at: `http://127.0.0.1:7860`

---

## 📊 Dataset & AI Data Generation

To ensure the system handles real-world complexity, we utilized an **AI-Driven Synthetic Data Engine** (`data/generate_data.py`) to create a high-fidelity dataset based on official **WCO (World Customs Organization)** taxonomies.

### Data Composition
The dataset consists of **252 unique samples** across 6 HS classes, with 4 distinct difficulty levels:

| Sample Type | Description | N |
| :--- | :--- | :---: |
| **Standard** | Clear, unambiguous product descriptions with primary keywords. | 120 |
| **Ambiguous** | Descriptions where primary identifying keywords are intentionally omitted. | 60 |
| **Overlapping** | Products that genuinely straddle the boundary between two similar HS codes. | 36 |
| **Edge Case** | Adversarial descriptions where modifiers act as "traps" for keyword-based models. | 36 |

---

## 📈 Detailed Performance Analysis

Our evaluation framework (`evaluation/scripts/evaluate_all.py`) provides deep insights into how the hybrid RAG architecture outperforms traditional baselines.

### Accuracy by Sample Type (Test Set)
| Type | N | ML Baseline | Vector RAG | **LLM Re-Ranking** |
| :--- | :---: | :---: | :---: | :---: |
| **Standard** | 24 | 95.8% | 91.7% | **100.0%** |
| **Ambiguous** | 11 | 63.6% | 81.8% | **100.0%** |
| **Overlapping** | 8 | 75.0% | 100.0% | 87.5% |
| **Edge Case** | 8 | 50.0% | 62.5% | **100.0%** |

### Agentic Pipeline Insights
- **Override Rate (15.7%)**: The LLM autonomously decided to change the initial Vector Search prediction in ~16% of cases.
- **Correction Rate (87.5%)**: When the LLM chose to override, it was correct **87.5%** of the time, effectively "saving" the system from vector retrieval errors.
- **Mean Reciprocal Rank (MRR)**: The Vector Engine achieved an MRR of **0.92**, indicating the correct code is almost always in the Top 2 results.

---

## 📊 Evaluation & Verification

To verify the system performance across all architectures (ML vs RAG vs LLM), run the unified evaluation script:
```bash
python evaluation/scripts/evaluate_all.py
```
The results will be generated in `evaluation/reports/evaluation_report.txt`.

---

## 🛠️ System Components

- **Method 1 (RAG + LLM)**: Uses `SentenceTransformers` for embedding and `Google Gemini` for grammatical reasoning. It excels at breaking "keyword rigidity" and correctly identifying the subject noun in complex descriptions.
- **Method 2 (ML Baseline)**: Uses `scikit-learn` pipeline. It serves as a performance baseline and demonstrates the limitations of purely statistical text matching.
- **Data Engine**: Includes a custom synthetic data generator that intentionally injects **Ambiguous**, **Overlapping**, and **Edge Case** samples to stress-test the classification logic.
