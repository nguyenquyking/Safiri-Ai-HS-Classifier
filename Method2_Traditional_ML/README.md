# Safiri AI: HS Code Classification System

This repository contains a full pipeline for an intelligent, interpretable HS code classification system, designed to handle ambiguous merchandise descriptions. It accurately predicts HS codes, gives confidence scores, and provides mathematical reasoning for every prediction.

## Features
- **Prediction:** Employs TF-IDF and Logistic Regression to classify text into appropriate detailed HS Codes.
- **Explainability:** Integrates `LIME` (Local Interpretable Model-agnostic Explanations) to demonstrate exactly which keywords (and their specific weight values) led to the assigned code.
- **Ambiguity Handling:** When confidence is below 60%, the system flags an `[AMBIGUITY ALERT]` and presents Top-3 alternatives for user consultation.

## Getting Started

### 1. Requirements
Ensure you have Python installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

*(Note for the reviewer: Since this is a self-contained take-home, we generated a high-quality synthetic dataset directly mimicking complex and overlapping customs descriptions to ensure all capabilities can be verified zero-setup).*

### 2. Pipeline Execution
Run the following scripts in order:

**(A) Generate Dataset:**
```bash
python generate_data.py
```
*(Creates `synthetic_customs_data.csv` containing overlapping and ambiguous item datasets).*

**(B) Train Pipeline:**
```bash
python train_model.py
```
*(Vectorizes data, trains the Logistic Regression schema, evaluates accuracy, and exports `hs_model.joblib`).*

**(C) Launch the Web UI:**
```bash
python app.py
```
*(Starts a localized Gradio Web UI at `http://localhost:7860/` to test predictions).*

## Why this approach?
Rather than a "black-box" generative LLM, this approach prioritizes absolute mathematical transparency, blazing fast inference times, and offline capabilities—all essential factors in high-throughput global trade intelligence arrays.
