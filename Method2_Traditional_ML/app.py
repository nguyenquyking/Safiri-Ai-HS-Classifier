import gradio as gr
import joblib
import lime
import lime.lime_text

# Load the trained model
try:
    model = joblib.load("hs_model.joblib")
except Exception as e:
    model = None
    print("Warning: Model not found. Please run train_model.py first.")

def predict_hs_code(query):
    if not model:
        return "Error: Model not trained", "0%", "Please run train_model.py to generate hs_model.joblib", ""
        
    # Get probabilities
    probs = model.predict_proba([query])[0]
    classes = model.classes_
    
    # Sort top 3 indices
    top3_indices = probs.argsort()[-3:][::-1]
    
    top1_class = classes[top3_indices[0]]
    top1_prob = probs[top3_indices[0]]
    
    # LIME Explanation
    explainer = lime.lime_text.LimeTextExplainer(class_names=classes)
    def predict_proba_lime(texts):
        return model.predict_proba(texts)
        
    top1_idx_in_classes = list(classes).index(top1_class)
    exp = explainer.explain_instance(query, predict_proba_lime, num_features=5, top_labels=1)
    
    # Format explanation
    explanation_text = f"**Why was HS Code {top1_class} chosen?**\nThe following keywords explicitly influenced the prediction:\n\n"
    try:
        contributions = exp.as_list(label=top1_idx_in_classes)
        for word, weight in contributions:
            direction = "Supports" if weight > 0 else "Opposes"
            color = "🟢" if weight > 0 else "🔴"
            explanation_text += f"- {color} **'{word.strip()}'**: {direction} (weight: {weight:.4f})\n"
    except:
        explanation_text += "Explanation generation failed for this text."
        
    # Format Top 3 Alternatives
    alt_text = "### Alternative Predictions\n"
    for idx in top3_indices[1:]:
        alt_text += f"- **HS {classes[idx]}**: {probs[idx]:.2%}\n"
        
    # Ambiguity Warning
    if top1_prob < 0.6:
        alt_text = "⚠️ **[AMBIGUITY ALERT]** Confidence is below 60%. Please carefully consider the following alternatives:\n\n" + alt_text

    confidence_str = f"{top1_prob:.2%}"
    
    return top1_class, confidence_str, explanation_text, alt_text

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Safiri AI - HS Code Classification System")
    gr.Markdown("Intelligent HS code prediction with reasoning and confidence scores.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Product Description", 
                placeholder="e.g. Apple iPhone 14 Pro Max 256GB Black Smartphone",
                lines=3
            )
            submit_btn = gr.Button("Predict HS Code", variant="primary")
            
            gr.Examples(
                examples=[
                    "Apple iPhone 14 Pro Max 256GB Black Smartphone, new in box",
                    "Wooden dining table set with 4 chairs oak finish",
                    "Men's casual collar shirt short sleeve cotton blend",
                    "GoPro HERO11 Black Waterproof Action Camera 5.3K Video"
                ],
                inputs=input_text
            )
            
        with gr.Column(scale=1):
            with gr.Row():
                out_hs_code = gr.Textbox(label="Predicted HS Code (Top 1)", lines=1)
                out_confidence = gr.Textbox(label="Confidence Score", lines=1)
            
            out_explanation = gr.Markdown(label="Reasoning / Explainability")
            out_alternatives = gr.Markdown(label="Alternatives (Handling Ambiguity)")
            
    submit_btn.click(
        fn=predict_hs_code,
        inputs=[input_text],
        outputs=[out_hs_code, out_confidence, out_explanation, out_alternatives]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
