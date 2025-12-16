# CSV Insight Studio 🧠📊

**CSV Insight Studio** is an AI-powered Streamlit application that enables interactive exploration, visualization, machine learning, clustering, and intelligent questioning of CSV datasets — all in one unified interface.

🔗 **Live App:**  
https://csv-data-agent-ankita-arya.streamlit.app/

---

## ✨ Key Capabilities

### 📂 CSV Exploration
- Upload one or multiple CSV files
- Automatic detection of numeric & categorical columns
- Missing value and column profiling

### 📊 Interactive Visualization
- Histograms for feature distributions  
- Scatter plots with optional categorical coloring  
- Correlation heatmaps for numeric features  

### 🤖 Machine Learning Studio
- Automatic task detection (classification / regression)
- Models supported:
  - Logistic Regression
  - Random Forest (Classifier & Regressor)
  - Linear Regression
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - MAE, RMSE, R²
  - Confusion Matrix

### 🧩 Clustering
- K-Means clustering on selected numeric features
- PCA-based 2D cluster visualization
- Adjustable number of clusters

### 🧠 AI Dataset Mentor
- Ask questions about the dataset in plain English
- Handles:
  - Column names & data types
  - Dataset size & structure
  - Missing values
  - Summary statistics & correlations
- Powered by **Hugging Face FLAN-T5**

---

## 🛠 Tech Stack

- **Frontend:** Streamlit (custom UI & CSS)
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
