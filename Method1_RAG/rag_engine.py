import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGEngine:
    def __init__(self, data_path="synthetic_customs_data.csv"):
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        # L6-v2 is ultra fast and great for semantic sentences
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            self.df = pd.read_csv(data_path)
            print(f"Loaded {len(self.df)} records from Vector Database.")
            
            # Embed all the descriptions to form our "Vector Database"
            print("Encoding Vector Database...")
            self.db_embeddings = self.model.encode(self.df['description'].tolist())
            print("Vectorization complete!")
            
        except Exception as e:
            print(f"Error loading data: {e}. Are you sure synthetic_customs_data.csv is present?")
            self.df = None
            self.db_embeddings = None

    def retrieve(self, query, top_k=3):
        if self.df is None or self.db_embeddings is None:
            return None, "Error: Database not loaded"
            
        # 1. Embed the incoming query into the same n-dimensional space
        query_embedding = self.model.encode([query])
        
        # 2. Compute Cosine Similarity between query and ALL items in the DB
        # Returns an array of shape (1, num_items)
        similarities = cosine_similarity(query_embedding, self.db_embeddings)[0]
        
        # 3. Find Top K matches
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = similarities[idx]
            row = self.df.iloc[idx]
            results.append({
                "hs_code": str(row['hs_code']),
                "matched_description": row['description'],
                "similarity_score": score
            })
            
        # The Top-1 HS code is our logical predicted class
        top_1_prediction = results[0]['hs_code']
        # The Top-1 Similarity Score serves as our "Confidence Score"
        max_confidence = results[0]['similarity_score']
        
        return top_1_prediction, max_confidence, results

if __name__ == "__main__":
    # Test the Engine directly
    rag = RAGEngine()
    
    test_query = "A simple molded plastic carrying case designed to hold and protect the Apple iPhone 14 Pro Max Smartphone."
    print(f"\n[QUERY]: {test_query}")
    
    predicted_hs, confidence, matches = rag.retrieve(test_query)
    print(f"\n=> Predicted HS Code: {predicted_hs} (Confidence: {confidence:.1%})")
    
    print("\n--- Semantic Retrievals ---")
    for i, match in enumerate(matches):
        print(f"#{i+1}: HS Code {match['hs_code']} | Score {match['similarity_score']:.1%} | Text: '{match['matched_description']}'")
