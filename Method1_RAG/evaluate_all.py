"""
Safiri AI HS Code Classification — Unified Evaluation Framework
==============================================================
Performance Analysis of Multi-Layered Classification Architecture (Batch LLM Mode):
  1. Method 2  : Traditional ML (TF-IDF + Logistic Regression)
  2. Method 1a : Vector Space Retrieval (SentenceTransformer Embeddings)
  3. Method 1b : Semantic Arbiter Pipeline (RAG + Batch LLM Reasoning)

This version optimizes API usage by batching multiple queries into a single 
Gemini request for higher throughput and reduced rate-limit pressure.
"""

import os, sys, time, json, re
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import google.generativeai as genai
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RAGEngine

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Arbiter Engine (Method 1b Component - Batch Processing)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticArbiter:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key.strip())
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

    def resolve_batch(self, batch_items):
        """
        Processes a batch of items (query + top-3 matches) in a single LLM call.
        batch_items: List of dicts { 'id': int, 'query': str, 'matches': list }
        Returns: Dict { id: hs_code }
        """
        if not self.model or not batch_items:
            return {item['id']: (item['matches'][0]['hs_code'] if item['matches'] else None) for item in batch_items}

        # Build context for multiple items
        items_context = ""
        for item in batch_items:
            context_text = ""
            for i, match in enumerate(item['matches'][:3], 1):
                context_text += f"    Match {i}: {match['hs_code']} ('{match['matched_description']}')\n"
            
            items_context += f"Item ID: {item['id']}\n  Description: \"{item['query']}\"\n  Candidates:\n{context_text}\n"

        prompt = f"""
You are a highly skilled Customs Classification Expert.
Classify the following {len(batch_items)} items based ONLY on their respective provided candidates.

{items_context}

For each item, determine the single most appropriate HS Code from its 3 candidates.
Return your answer as a MINIMAL JSON list of objects, each containing "id" and "hs_code". 
Example format: [{{ "id": 0, "hs_code": "8517" }}, {{ "id": 1, "hs_code": "6205" }}]
No explanations. No other text. Just the JSON.
"""
        try:
            # Pacing Pacing between batches
            time.sleep(2.0) 
            response = self.model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            
            # Use regex if LLM returned extra junk
            json_match = re.search(r'\[.*\]', clean_json, re.DOTALL)
            if json_match:
                results_list = json.loads(json_match.group(0))
                return {res['id']: str(res['hs_code']) for res in results_list}
            else:
                raise ValueError("No JSON list found in response")
        except Exception as e:
            print(f"    Batch Error: {e}. Falling back to RAG Top-1 for this batch.")
            return {item['id']: (item['matches'][0]['hs_code'] if item['matches'] else None) for item in batch_items}

# ─────────────────────────────────────────────────────────────────────────────
# Setup & Initial Processing (ML + RAG)
# ─────────────────────────────────────────────────────────────────────────────

print("Initializing Evaluation Framework...")
ml_model = joblib.load("../Method2_Traditional_ML/hs_model.joblib")
engine = RAGEngine(data_path="../Method2_Traditional_ML/train_dataset.csv")
arbiter = SemanticArbiter()
test_df = pd.read_csv("../Method2_Traditional_ML/test_dataset.csv")
n = len(test_df)

processed_data = []
print(f"Phase 1: Running ML & Vector RAG on {n} samples...")

for i, row in test_df.iterrows():
    query   = row["description"]
    true_hs = str(row["hs_code"])
    stype   = row.get("sample_type", "unknown")

    # Method 2 (ML)
    ml_top1  = str(ml_model.predict([query])[0])
    ml_probs = ml_model.predict_proba([query])[0]
    ml_top3  = [ml_model.classes_[j] for j in np.argsort(ml_probs)[-3:][::-1]]

    # Method 1a (Vector RAG)
    vec_top1, _, vec_results = engine.retrieve(query, top_k=5)
    
    processed_data.append({
        "id": i,
        "query": query,
        "true_hs": true_hs,
        "stype": stype,
        "ml_top1": ml_top1,
        "ml_top3": ml_top3,
        "vec_top1": vec_top1,
        "vec_results": vec_results,
        "retrieved_codes": [r["hs_code"] for r in vec_results]
    })

# ─────────────────────────────────────────────────────────────────────────────
# Batch LLM Processing (Method 1b)
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 7
print(f"\nPhase 2: Running Batch LLM Reasoning (Size={BATCH_SIZE})...")

llm_predictions = {}
for i in range(0, n, BATCH_SIZE):
    batch = processed_data[i:i + BATCH_SIZE]
    print(f"    Processing batch {i//BATCH_SIZE + 1}/{(n-1)//BATCH_SIZE + 1}...")
    
    batch_input = [{"id": item["id"], "query": item["query"], "matches": item["vec_results"]} for item in batch]
    batch_results = arbiter.resolve_batch(batch_input)
    llm_predictions.update(batch_results)

# ─────────────────────────────────────────────────────────────────────────────
# Record Compilation & Metrics
# ─────────────────────────────────────────────────────────────────────────────

records = []
for item in processed_data:
    true_hs = item["true_hs"]
    ret_codes = item["retrieved_codes"]
    llm_hs = llm_predictions.get(item["id"], "skipped")
    
    # Calculate MRR
    rr = 0.0
    for rank, code in enumerate(ret_codes, start=1):
        if code == true_hs:
            rr = 1.0 / rank
            break

    records.append({
        "query"       : item["query"][:60] + "..." if len(item["query"]) > 60 else item["query"],
        "true"        : true_hs,
        "sample_type" : item["stype"],
        "ml"          : item["ml_top1"],
        "ml_ok"       : item["ml_top1"] == true_hs,
        "ml_top3_ok"  : true_hs in item["ml_top3"],
        "vec"         : item["vec_top1"],
        "vec_ok"      : item["vec_top1"] == true_hs,
        "rec_at3"     : true_hs in ret_codes[:3],
        "rec_at5"     : true_hs in ret_codes[:5],
        "rr"          : rr,
        "llm"         : llm_hs,
        "llm_ok"      : llm_hs == true_hs,
    })

df = pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# Final Report Generation
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 115
THIN = "-" * 115
def pct(n, d): return f"{n/d:.1%}" if d else "N/A"

lines = [
    "SAFIRI AI — UNIFIED HS CODE CLASSIFICATION EVALUATION REPORT (BATCH MODE)",
    f"Dataset : AI-Enhanced WCO Taxonomy (252 samples)",
    f"Split   : 80% Train (201) / 20% Test ({n}, stratified)",
    f"LLM     : gemini-2.5-flash (Batched Processing)",
    "",
]

# (Sections 0-6 generation logic restored here)
ml1_ok = df["ml_ok"].sum(); ml3_ok = df["ml_top3_ok"].sum()
vec_ok = df["vec_ok"].sum(); r3_ok = df["rec_at3"].sum(); r5_ok = df["rec_at5"].sum()
llm_ok = df["llm_ok"].sum(); mrr = df["rr"].mean()

lines += [
    SEP, "  SECTION 0: OVERALL ACCURACY SUMMARY", SEP,
    f"  {'Metric':<45} | {'Method 2 (ML)':>14} | {'Method 1a (RAG)':>15} | {'Method 1b (LLM)':>20}",
    THIN,
    f"  {'Top-1 Accuracy':<45} | {pct(ml1_ok,n):>14} | {pct(vec_ok,n):>15} | {pct(llm_ok,n):>20}",
    f"  {'Top-3 Accuracy (ML) / Recall@3 (RAG)':<45} | {pct(ml3_ok,n):>14} | {pct(r3_ok,n):>15} | {'—':>20}",
    f"  {'Recall@1':<45} | {'= Top-1':>14} | {pct(vec_ok,n):>15} | {'--':>20}",
    f"  {'Recall@5':<45} | {'--':>14} | {pct(r5_ok,n):>15} | {'--':>20}",
    f"  {'Mean Reciprocal Rank (MRR)':<45} | {'--':>14} | {mrr:.4f}{' '*9} | {'--':>20}",
    SEP, ""
]

lines += [SEP, "  SECTION 1: ACCURACY BY SAMPLE TYPE", SEP, f"  {'Type':<13} | {'N':>3} | {'ML Top-1':>9} | {'ML Top-3':>9} | {'Vec Top-1':>10} | {'Recall@3':>9} | {'LLM Accuracy':>12} | {'MRR':>6}", THIN]
for stype in ["standard", "ambiguous", "overlapping", "edge_case"]:
    sub = df[df["sample_type"] == stype]; t = len(sub)
    if not sub.empty:
        lines.append(f"  {stype:<13} | {t:>3} | {pct(sub['ml_ok'].sum(), t):>9} | {pct(sub['ml_top3_ok'].sum(), t):>9} | {pct(sub['vec_ok'].sum(), t):>10} | {pct(sub['rec_at3'].sum(), t):>9} | {pct(sub['llm_ok'].sum(), t):>12} | {sub['rr'].mean():.4f}")
lines += [SEP, ""]

cat_names = {"8517": "Telecommunications apparatus", "8525": "Cameras & broadcast equipment", "6109": "T-shirts / singlets (knitted)", "6205": "Shirts (woven, not knitted)", "9403": "Other furniture", "3926": "Other articles of plastics"}
lines += [SEP, "  SECTION 2: ACCURACY BY HS CODE CLASS", SEP, f"    {'HS':<4} | {'Category':<35} | {'N':>3} | {'ML Top-1':>9} | {'Vec Top-1':>10} | {'Recall@3':>9} | {'LLM':>12} | {'MRR':>6}", THIN]
for hs in sorted(df["true"].unique()):
    sub = df[df["true"] == hs]; t = len(sub)
    lines.append(f"  {hs:<4} | {cat_names.get(hs, 'Other Categories'):<35} | {t:>3} | {pct(sub['ml_ok'].sum(), t):>9} | {pct(sub['vec_ok'].sum(), t):>10} | {pct(sub['rec_at3'].sum(), t):>9} | {pct(sub['llm_ok'].sum(), t):>12} | {sub['rr'].mean():.4f}")
lines += [SEP, ""]

override_mask = df["llm"] != df["vec"]
n_override = override_mask.sum()
n_correction = (override_mask & df["llm_ok"] & ~df["vec_ok"]).sum()
n_error = (override_mask & ~df["llm_ok"] & df["vec_ok"]).sum()
lines += [SEP, "  SECTION 3: RAG + LLM PIPELINE METRICS (BATCH)", SEP,
    f"  LLM Top-1 Accuracy    : {pct(llm_ok, n)}  ({llm_ok}/{n})",
    f"  Override Rate         : {pct(n_override, n)}  ({n_override}/{n})",
    f"  Correction Rate       : {pct(n_correction, n_override) if n_override else '0%'}  ({n_correction}/{n_override})",
    f"  LLM Error Rate        : {pct(n_error, n_override) if n_override else '0%'}  ({n_error}/{n_override})",
    SEP, ""
]

lines += [SEP, "  SECTION 4: CONFUSION MATRIX — ML BASELINE", SEP]
classes = sorted(df["true"].unique())
cm = confusion_matrix(df["true"], df["ml"], labels=classes)
lines += ["  Actual \\ Pred  | " + " | ".join(f"{c:>5}" for c in classes), THIN]
for i, actual in enumerate(classes):
    lines.append(f"  {actual:<15} | " + " | ".join(f"{cm[i,j]:>5}" for j in range(len(classes))))
lines += [SEP, ""]

lines += [SEP, "  SECTION 5: PRECISION / RECALL / F1", SEP, "  Method 2 — Traditional ML", THIN]
lines += [f"    {l}" for l in classification_report(df["true"], df["ml"], zero_division=0).splitlines()]
lines += ["", "  Method 1b — RAG + LLM (Batch)", THIN]
lines += [f"    {l}" for l in classification_report(df["true"], df["llm"], zero_division=0).splitlines()]
lines += [SEP, ""]

lines += [SEP, "  SECTION 6: FULL ROW-BY-ROW PREDICTIONS", SEP, f"  {'Query':<62} | {'True':<4} | {'Type':<11} | {'ML':<10} | {'Vec':<10} | {'LLM':<10}", THIN]
for _, r in df.iterrows():
    m_st = f"{r['ml']} {'OK' if r['ml_ok'] else 'XX'}"
    v_st = f"{r['vec']} {'OK' if r['vec_ok'] else 'XX'}"
    l_st = f"{r['llm']} {'OK' if r['llm_ok'] else 'XX'}" if r['llm'] != "skipped" else "skipped"
    lines.append(f"  {r['query']:<62} | {r['true']:<4} | {r['sample_type']:<11} | {m_st:<10} | {v_st:<10} | {l_st:<10}")
lines += [SEP]

with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nEvaluation Complete. Batch report saved to evaluation_report.txt")
