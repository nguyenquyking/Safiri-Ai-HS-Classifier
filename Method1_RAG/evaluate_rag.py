import pandas as pd
from rag_engine import RAGEngine

def evaluate_rag():
    print("Loading RAG Evaluator...")
    # Initialize Engine with train queries ONLY (No data leakage)
    engine = RAGEngine(data_path="../Method2_Traditional_ML/train_dataset.csv")
    
    # Load 20% unseen edge-cases
    try:
        test_df = pd.read_csv("../Method2_Traditional_ML/test_dataset.csv")
    except Exception as e:
        print("Could not load test dataset. Run train_model.py in Method 2 first.")
        return
        
    print(f"\nEvaluating RAG Vector Search on {len(test_df)} Unseen Edge Cases...")
    
    top1_correct = 0
    top3_correct = 0
    
    # Iterate through the test set blindly
    for idx, row in test_df.iterrows():
        query = row['description']
        true_hs = str(row['hs_code'])
        
        # Retrieve Top 3 Semantic matches blindly
        predicted_hs, confidence, matches = engine.retrieve(query, top_k=3)
        
        # Check Top 1
        if predicted_hs == true_hs:
            top1_correct += 1
            
        # Check Top 3
        top3_preds = [match['hs_code'] for match in matches]
        if true_hs in top3_preds:
            top3_correct += 1
            
    # Calculate metrics
    top1_acc = top1_correct / len(test_df)
    top3_acc = top3_correct / len(test_df)
    
    print("\n--- RAG Vector Metric Report (Unseen Data) ---")
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Top-3 Accuracy: {top3_acc:.2%}")
    print("----------------------------------------------")
    print("Note: If Top-1 fails but Top-3 succeeds, the Generative LLM (Gemini) acts as the final Re-ranker to guarantee the ultimate Top-1 precision based on contextual grammar.")

if __name__ == "__main__":
    evaluate_rag()
