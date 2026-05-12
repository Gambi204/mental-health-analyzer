import gradio as gr
import torch
import numpy as np
from transformers import pipeline
import pickle
from lime.lime_text import LimeTextExplainer
import time

# Load Model 1: Disorder Classifier
disorder_path = "./FinalMentalHealthModel_DistilBERT"
disorder_pipe = pipeline("text-classification", model=disorder_path, tokenizer=disorder_path, top_k=None)

# Load Model 2: Root Cause Classifier
root_path = "./FinalMentalHealthModel_RootCause"
root_pipe = pipeline("text-classification", model=root_path, tokenizer=root_path, top_k=None)

# Load label encoder for root cause
with open("./label_encoder_root.pkl", "rb") as f:
    label_encoder_root = pickle.load(f)

# LIME Explainer
explainer = LimeTextExplainer(class_names=label_encoder_root.classes_.tolist())

# Predict Function
def analyze_post(text):
    try:
        start = time.time()

        # Predict Disorder
        disorder_preds = disorder_pipe(text)[0]
        disorder_preds.sort(key=lambda x: x['score'], reverse=True)
        disorder_label = disorder_preds[0]['label']
        disorder_conf = disorder_preds[0]['score']

        # Predict Root Cause
        root_preds = root_pipe(text)[0]
        root_preds.sort(key=lambda x: x['score'], reverse=True)
        root_label_raw = root_preds[0]['label']
        root_label_idx = int(root_label_raw.split("_")[-1]) if "LABEL_" in root_label_raw else int(root_label_raw)
        root_label = label_encoder_root.inverse_transform([root_label_idx])[0]
        root_conf = root_preds[0]['score']

        # LIME Explanation
        def root_predict(texts):
            preds = root_pipe(texts)
            all_scores = []
            for pred in preds:
                try:
                    pred_sorted = sorted(pred, key=lambda x: int(x["label"].replace("LABEL_", "")))
                except:
                    pred_sorted = pred
                scores = [x["score"] for x in pred_sorted]
                all_scores.append(scores)
            return np.array(all_scores)

        explanation = explainer.explain_instance(text, root_predict, num_features=5)
        exp_html = explanation.as_html()

        # Save explanation to file
        lime_path = "lime_explanation.html"
        with open(lime_path, "w", encoding="utf-8") as f:
            f.write(exp_html)

        end = time.time()
        print(f"🕒 Analysis took {end - start:.2f} seconds")

        return (
            f"**Predicted Disorder:** {disorder_label} (Confidence: {disorder_conf:.2f})\n"
            f"**Predicted Root Cause:** {root_label} (Confidence: {root_conf:.2f})",
            exp_html,
            lime_path
        )
    
    except Exception as e:
        print("❌ Error in analyze_post:", e)
        return (
            "**An error occurred during prediction. Check logs.**",
            "<p style='color:red;'>LIME Explanation failed.</p>",
            None  # No file
        )


# Gradio UI
with gr.Blocks(title="Mental Health Analyzer") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>🧠 Mental Health Detection Tool</h1>"
        "<p style='text-align: center;'>Analyze Reddit-style posts for likely mental health disorders and their root causes using explainable AI (LIME).</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="✍️ Reddit Post (title + selftext)",
                placeholder="E.g. I’ve been feeling very anxious lately and struggling to sleep...",
                lines=7
            )
            btn = gr.Button("🔍 Analyze")
            status = gr.Markdown("")

        with gr.Column(scale=2):
            output_pred = gr.Markdown(label="📊 Prediction Results")
            output_explainer = gr.HTML(label="🧠 LIME Explanation")
            lime_file = gr.File(label="📁 Download LIME Explanation")

    btn.click(fn=analyze_post, inputs=input_text, outputs=[output_pred, output_explainer, lime_file], show_progress=True)

demo.launch()
