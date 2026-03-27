import sys
import os
import json
import gradio as gr
from dotenv import load_dotenv

# Setup path to allow imports from the root src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.rag.rag_engine import RAGEngine

import google.generativeai as genai

# Load environment variables from the root .env file
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path=env_path)

try:
    engine = RAGEngine()
except Exception as e:
    print("Warning: Engine failed to initialize:", e)
    engine = None

def process_query(query):
    if engine is None:
        return "ERROR", "0%", "ERROR", "0%", "RAG Engine offline. Please check logs."
        
    api_key = os.getenv("GEMINI_API_KEY")
        
    # Phase 1: Retrieval (Vector Search)
    predicted_hs, confidence, matches = engine.retrieve(query, top_k=3)
    
    # Format the retrieved context
    context_text = ""
    for i, match in enumerate(matches, 1):
        context_text += f"[Context {i}] Description: '{match['matched_description']}' -> HS Code: {match['hs_code']}\n"
    
    # Phase 2: Generation (LLM Synthesis & Re-ranking)
    explanation = f"### Retrieval Context (RAG Vector Search)\n"
    for i, match in enumerate(matches, 1):
        color = "🟢" if i == 1 else "🟡"
        explanation += f"**{color} Semantic Match #{i}:** `{match['hs_code']}` - {match['matched_description']} (Sim: {match['similarity_score']:.1%})\n"
        
    explanation += "\n---\n### 🤖 Expert Analysis (Google Gemini LLM Re-Ranking)\n"
    
    if not api_key or not api_key.strip():
        explanation += "\n*(❗ You have not configured the Gemini API Key in the `.env` file. Operating in pure Semantic Retrieval Mode. Generative reasoning is disabled.)* \n\n"
        return predicted_hs, f"{confidence:.1%}", "N/A (No API Key)", "N/A", explanation
        
    try:
        # Build Generative RAG Prompt for JSON Schema Re-ranking
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
You are a highly skilled Customs Classification Expert.
The user wants to classify the following newly imported goods:
"{query}"

I have queried our internal customs historical database and retrieved the top 3 items with the closest semantic vector structure:
{context_text}

Based EXCLUSIVELY on the 3 historical contexts provided above, you must:
1. Determine the single most appropriate HS Code for the user's item among the 3 contexts. Do not hallucinate external HS codes.
2. Formulate a self-assessed Confidence Score (e.g. 95%) reflecting your certainty in choosing this context over the others.
3. Write 1-2 concise paragraphs providing STRONG GRAMMATICAL REASONING (e.g. identifying the core subject vs. modifier words) to justify why you chose this HS code context and discarded the others.

Return your response STRICTLY as a raw JSON object with NO markdown formatting around it (do not use ```json). The JSON must EXACTLY match this structure:
{{
  "final_hs_code": "The 4-digit HS code you selected",
  "llm_confidence_score": "Your confidence percentage (e.g. 95%)",
  "explanation": "Your detailed reasoning formatted nicely in Markdown"
}}
"""
        
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Robust parsing cleanup just in case LLM adds markdown wrappers
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        
        llm_hs = data.get('final_hs_code', "N/A")
        llm_conf = data.get('llm_confidence_score', "N/A")
        explanation += data.get('explanation', "No explanation provided.")
        
    except json.JSONDecodeError as json_err:
        explanation += f"**JSON Parsing Error:** {json_err}\n\n**Raw Output:**\n{response.text}"
        llm_hs = "Parse Error"
        llm_conf = "Parse Error"
    except Exception as e:
        explanation += f"**Gemini API Error:** {str(e)} \n Please verify your `.env` file."
        llm_hs = "API Error"
        llm_conf = "API Error"
        
    return predicted_hs, f"{confidence:.1%}", str(llm_hs), str(llm_conf), explanation

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Method 1 (Generative RAG Re-ranking) - Vector Engine vs LLM Final Decision")
    gr.Markdown("Demonstrating how Gemini natively re-evaluates the Top 3 Vector distances to formulate its own Final Prediction and Confidence Score.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Product Description", 
                lines=3
            )
            gr.Examples(
                examples=[
                    "A simple molded plastic carrying case designed to hold and protect the Apple iPhone 14 Pro Max Smartphone.",
                    "Smartphone with built-in advanced 4K digital camera and lens attachment",
                    "Wooden dining table set with 4 chairs oak finish"
                ],
                inputs=input_text
            )
            submit_btn = gr.Button("Find HS Code via RAG", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 1. Vector Retriever Result (Mathematical Distance)")
            with gr.Row():
                out_hs_code = gr.Textbox(label="Vector Match (HS Code)")
                out_confidence = gr.Textbox(label="Vector Confidence (Cosine %)")
                
            gr.Markdown("### 2. Gemini Generator Result (Syntactic Reasoning + Re-Ranking)")
            with gr.Row():
                out_llm_code = gr.Textbox(label="Gemini Final Decision (HS Code)")
                out_llm_conf = gr.Textbox(label="Gemini Confidence Level")
                
            out_logic = gr.Markdown(label="Explainability Logic (RAG Context)")
            
    submit_btn.click(
        fn=process_query,
        inputs=[input_text],
        outputs=[out_hs_code, out_confidence, out_llm_code, out_llm_conf, out_logic]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, theme=gr.themes.Default())
