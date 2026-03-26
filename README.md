# 🌍 Safiri AI - HS Code Classification System

An advanced, dual-architecture AI classification system designed to predict Harmonized System (HS) codes from unstructured product descriptions. Developed as a complete solution for the **Safiri AI Take-Home Challenge**.

---

## 🏗️ Architecture Overview

This repository demonstrates two distinct approaches to solve the HS Code classification problem, highlighting a deep understanding of both classical Machine Learning architectures and modern LLM-based Generative AI engineering.

### 1. Method 2: Traditional Machine Learning (Baseline)
- **Path**: `/Method2_Traditional_ML`
- **Algorithm**: TF-IDF Text Vectorization + Logistic Regression.
- **Explainability**: Integrated with **LIME (Local Interpretable Model-agnostic Explanations)** to showcase exactly which keywords heavily influence the model's prediction (Support vs. Oppose weights).
- **Architecture Strengths**: Extremely fast inference, low compute footprint, mathematics-backed transparency.
- **Flaws Demonstrated**: Exhibits "Keyword Rigidity/Bias". It may fail on edge cases where modifier words visually clash with the grammatical subject (e.g., failing to classify a *"Molded plastic case for Apple iPhone 14"* as Plastic `3926` due to the overwhelming TF-IDF weight of the word "iPhone").

### 2. Method 1: Generative RAG Re-Ranking (Advanced Solution)
- **Path**: `/Method1_RAG`
- **Algorithm**: `SentenceTransformers` (Vector Semantic Search) + `Google Gemini API` (LLM Re-Ranking).
- **Workflow**: 
  1. **Retriever**: Embeds the user's query into an n-dimensional vector space using `all-MiniLM-L6-v2` and retrieves the Top 3 closest historical matches using Cosine Similarity.
  2. **Reader/Generator**: Passes the 3 semantic contexts to Gemini, forcing it to evaluate the grammar, discard deceptive noise keywords, and autonomously **Re-rank** the Vector prediction.
- **Architecture Strengths**: Defeats Keyword Rigidity natively. It isolates the grammatical subject beautifully and acts as an interpretable "Subject Matter Expert".

---

## 📊 The "Real-World Complexity" Dataset

To prove structural awareness, the system is backed by a custom, perfectly balanced synthetic dataset containing **180 rows** (`synthetic_customs_data.csv`). The dataset intentionally injects 3 "real-world" complexities as requested by the assignment:

1. **Semantic Overlap:** Separates overlapping items like "Knitted T-Shirts" (`6109`) and "Woven Shirts" (`6205`) using nearly identical noise keywords to test the model's precision.
2. **Ambiguous Descriptions:** Includes vague queries maliciously stripped of key identifiers (e.g., *"heavy cotton relaxed fit graphic tee black"*).
3. **Edge Cases:** Includes cross-domain items designed to trick the baseline ML, such as *"Plastic outdoor garden storage box"* (Conflicting Furniture `9403` vs Plastics `3926`).

---

## 🚀 Setup & Execution instructions

### 1. Prerequisites
Clone the repository and install the dependencies:
```bash
git clone https://github.com/nguyenquyking/Safiri-Ai-HS-Classifier.git
cd Safiri-Ai-HS-Classifier
pip install -r Method2_Traditional_ML/requirements.txt
pip install sentence-transformers google-generativeai python-dotenv
```

### 2. Running Method 2 (Traditional ML)
To run the standard Machine Learning Grado UI with LIME explanations:
```bash
cd Method2_Traditional_ML
python app.py
```
> Open `http://127.0.0.1:7860` in your browser.

### 3. Running Method 1 (Generative RAG Re-Ranking)
To run the advanced LLM architecture:
1. Navigate to `/Method1_RAG`.
2. Open the `.env` file and insert your API Key: `GEMINI_API_KEY=your_key_here`. (If you leave it blank, the system will elegantly fallback to pure Vector Retrieval without crashing).
3. Start the server:
```bash
cd Method1_RAG
python app_rag.py
```
> Open `http://127.0.0.1:7861` and test complex edge cases like *"A simple molded plastic carrying case designed to hold and protect the Apple iPhone 14 Pro Max Smartphone."* to watch the Generative Re-ranker in action.
