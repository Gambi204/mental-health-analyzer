# 🧠 Mental Health Detection Using DistilBERT & Explainable AI (LIME)

This project is a robust mental health analysis tool that uses two fine-tuned DistilBERT models to detect **mental health disorders** and their possible **root causes** from Reddit-style posts. It includes **explainable AI** via LIME to provide users with transparency and interpretability into model decisions.

## 📌 Project Overview

The tool helps users analyze free-form text and receive:
- Predicted **mental health disorder** (e.g., depression, anxiety)
- Probable **root cause** of the issue (e.g., trauma, personality, addiction)
- **LIME explanations** for model decisions (with HTML visualization + download option)

This project is built using:
- `Hugging Face Transformers` for NLP pipelines
- `Scikit-learn` for label encoding
- `LIME` for explainability
- `Gradio` for a beautiful, interactive UI

---
