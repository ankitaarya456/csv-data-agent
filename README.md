# CSV Insight Studio 🧠📊
**An AI-powered Streamlit application for intelligent CSV data exploration, visualization, machine learning, clustering, and natural language dataset understanding.**

CSV Insight Studio acts like a **personal data lab**, allowing users to upload CSV files and instantly gain insights using interactive analytics and AI-driven explanations — all from a single unified interface.

🔗 **Live App:**  
https://csv-data-agent-ankita-arya.streamlit.app/

---

## 🎯 Why This Project?
This project demonstrates **end-to-end applied Data Science & AI skills**, combining:
- Exploratory Data Analysis (EDA)
- Classical Machine Learning workflows
- Unsupervised learning & dimensionality reduction
- LLM-powered natural language dataset understanding
- Deployment-ready Streamlit application

It is designed to reflect **real-world data analysis tools** used by analysts, ML engineers, and data scientists.

---

## ✨ Key Capabilities

### 📂 CSV Exploration
- Upload one or multiple CSV files
- Automatic detection of numeric & categorical columns
- Dataset snapshot (head, shape, column info)
- Missing value analysis & column profiling

---

### 📊 Interactive Visualization
- Histograms for feature distributions
- Scatter plots with optional categorical coloring
- Correlation heatmaps for numeric features
- Dynamic Plotly-based visualizations

---

### 🤖 Machine Learning Studio
- Automatic task detection:
  - Classification
  - Regression
- Supported Models:
  - Logistic Regression
  - Random Forest (Classifier & Regressor)
  - Linear Regression
- Evaluation Metrics:
  - Accuracy, Precision, Recall, F1-score
  - MAE, RMSE, R²
  - Confusion Matrix

---

### 🧩 Clustering
- K-Means clustering on selected numeric features
- Adjustable number of clusters
- PCA-based 2D cluster visualization for interpretability

---

### 🧠 AI Dataset Mentor
- Ask questions about the dataset in **plain English**
- Handles:
  - Column names & data types
  - Dataset size & structure
  - Missing values
  - Summary statistics & correlations
- Powered by **Hugging Face FLAN-T5**

---

## 🧠 Skills Demonstrated
- Exploratory Data Analysis (EDA)
- Feature profiling & correlation analysis
- Supervised Machine Learning
- Unsupervised Learning (K-Means, PCA)
- Model evaluation & interpretation
- LLM integration for data understanding
- Streamlit app development
- Cloud deployment

---

## 🛠 Tech Stack

- **Frontend:** Streamlit (custom UI & layout)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Machine Learning:** Scikit-learn
- **AI / NLP:** Hugging Face Transformers (FLAN-T5)
- **Deployment:** Streamlit Community Cloud

---

## ▶️ Run Locally

```bash
git clone https://github.com/ankitaarya456/csv-data-agent.git
cd csv-data-agent
pip install -r requirements.txt
streamlit run app.py

---
## License

This project is licensed under the Apache License 2.0.
