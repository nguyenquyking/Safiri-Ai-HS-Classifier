import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import lime
import lime.lime_text

# Define base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_and_evaluate():
    try:
        df = pd.DataFrame()
        try:
            raw_data_path = os.path.join(BASE_DIR, "../../data/raw/synthetic_customs_data.csv")
            df = pd.read_csv(raw_data_path)
            print(f"Loaded raw data from: {raw_data_path}")
        except:
            print(f"Data not found at {raw_data_path}. Did you run generate_data.py?")
            return
            
        X = df["description"]
        y = df["hs_code"].astype(str)
        
        # Split Data (80% Train, 20% Evaluate)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save chunks for RAG Engine audit
        train_df = df.loc[X_train.index]
        test_df = df.loc[X_test.index]
        train_out = os.path.join(BASE_DIR, "../../data/processed/train_dataset.csv")
        test_out = os.path.join(BASE_DIR, "../../data/processed/test_dataset.csv")
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        print(f"Datasets saved to {os.path.dirname(train_out)}")
        
        print("Training TF-IDF + Logistic Regression model on 80% Train Data...")
        # Create pipeline
        model = make_pipeline(
            TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2)),
            LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced', random_state=42)
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model (Testing on unseen 20% Test Data)
        preds = model.predict(X_test)
        print(f"\n--- Classification Report ({len(X_test)} Unseen Edge Cases) ---")
        print(classification_report(y_test, preds))
        
        probs = model.predict_proba(X_test)
        top3_preds = [model.classes_[np.argsort(prob)[-3:]] for prob in probs]
        top3_correct = sum([1 if true_y in preds else 0 for true_y, preds in zip(y_test, top3_preds)])
        top3_acc = top3_correct / len(y_test)
        
        print(f"Top-1 Accuracy: {model.score(X_test, y_test):.2%}")
        print(f"Top-3 Accuracy: {top3_acc:.2%}")
        
        # Save model
        model_out = os.path.join(BASE_DIR, "hs_model.joblib")
        joblib.dump(model, model_out)
        print(f"Model saved to {model_out}")
        
        return model
        
    except Exception as e:
        print(f"Error during training: {e}")

def predict_query(model, query):
    print(f"\n[QUERY]: {query}")
    
    # Probabilities
    probs = model.predict_proba([query])[0]
    classes = model.classes_
    
    # Sort top 3
    top3_indices = probs.argsort()[-3:][::-1]
    
    top1_class = classes[top3_indices[0]]
    top1_prob = probs[top3_indices[0]]
    
    print("\n--- Top 3 Predictions ---")
    for idx in top3_indices:
        print(f"HS Code {classes[idx]}: {probs[idx]:.2%}")
        
    # Ambiguity Check
    if top1_prob < 0.6:
        print("\n⚠️ [AMBIGUITY ALERT]: The confidence is below 60%. Please consider the top 3 options or consult an expert.")
    
    # Explainability using LIME
    print("\n--- Explanation ---")
    explainer = lime.lime_text.LimeTextExplainer(class_names=classes)
    
    # We create a prediction function for LIME that outputs probabilities
    def predict_proba_lime(texts):
        return model.predict_proba(texts)
        
    # Explain the instance for the top predicted class
    top1_idx_in_classes = list(classes).index(top1_class)
    exp = explainer.explain_instance(query, predict_proba_lime, num_features=5, top_labels=1)
    
    # Extract the features that contributed to the top class
    try:
        contributions = exp.as_list(label=top1_idx_in_classes)
        print(f"Why was {top1_class} chosen?")
        for word, weight in contributions:
            direction = "Supports" if weight > 0 else "Opposes"
            print(f"- '{word.strip()}': {direction} (weight: {weight:.4f})")
    except KeyError:
        # Fallback if label is string vs int issue
        print(f"LIME extracted terms for context: {exp.as_list()}")

if __name__ == "__main__":
    model = train_and_evaluate()
    if model:
        print("\nTesting Inference...")
        test_queries = [
            "Apple iPhone 14 Pro Max 256GB Black Smartphone, new in box",
            "Wooden dining table set with 4 chairs oak finish",
            "Men's casual collar shirt short sleeve cotton blend", # Ambiguous case (Woven/Knitted overlap)
        ]
        
        for q in test_queries:
            predict_query(model, q)
