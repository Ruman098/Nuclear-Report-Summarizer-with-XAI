# 🔬 Nuclear Report Summarizer with XAI

An interactive **AI-powered nuclear report summarization system** with **Explainable AI (XAI)** support.  
This project uses **Transformer-based NLP models** to summarize nuclear technical reports (PDF/TXT) and applies **SHAP** to explain *which parts of the report influenced the summary*.

Built with **Streamlit**, **Hugging Face Transformers**, and **SHAP** for transparency and interpretability.

🚀 **Live Demo (Streamlit Cloud):**  
👉 **https://nuclear-report-summarizer-with-xai.streamlit.app/**

---

## ✨ Features

- 📄 Upload **PDF or TXT nuclear reports**
- 🧠 Automatic **AI-generated summaries**
- 🔍 **Explainable AI (XAI)** using SHAP
- 🎨 Visual token-level importance highlighting
- ⚡ Lightweight and easy to run locally
- 🧪 Suitable for technical & safety documentation analysis

---

## 🗂 Project Structure
```bash
Nuclear-Report-Summarizer-with-XAI/
├── app.py
├── summarizer.py
├── requirements.txt
├── README.md
├── Nuclear Reactor Pressure Test Report.txt
└── pycache/
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ruman098/Nuclear-Report-Summarizer-with-XAI.git
cd Nuclear-Report-Summarizer-with-XAI
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

### Summarization
- Uses a Transformer-based model (**t5-small**) to generate concise summaries.

### Explainability
- **SHAP** analyzes token contributions to the generated summary.
- 🟥 **Red tokens** → Positive influence on the summary  
- 🟦 **Blue tokens** → Negative influence on the summary

### Visualization
- Interactive **SHAP text plots** embedded directly in Streamlit.

---

## 📦 Key Libraries Used

- **Streamlit** – UI framework  
- **Transformers (Hugging Face)** – NLP models  
- **PyTorch** – Model backend  
- **SHAP** – Explainable AI  
- **PyPDF2** – PDF text extraction  
- **Matplotlib** – Visualization support  

---

## 📌 Example Use Cases

- Nuclear safety report analysis  
- Technical documentation summarization  
- Regulatory compliance reviews  
- Research and academic studies  
- Explainable AI demonstrations  

---

## ⚠️ Notes

- SHAP explanations may take longer for large documents.
- GPU acceleration is automatically used if available.
- Designed for educational and research purposes.

---
