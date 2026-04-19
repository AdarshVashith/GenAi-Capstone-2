# 🌾 Crop Yield Prediction & Intelligent Farm Advisory System (Milestone 2)

## 📌 Project Overview

This project focuses on building an **AI-powered Crop Yield Prediction and Farm Advisory System** that combines **machine learning, retrieval-augmented generation (RAG), and agentic AI** to support data-driven agricultural decisions.

The system not only predicts crop yield but also acts as an **intelligent advisory assistant**, helping farmers and stakeholders make better decisions based on environmental conditions, historical data, and agronomic knowledge.

---

## 🎯 Problem Statement

Agriculture faces multiple uncertainties such as:

* Weather variability
* Soil conditions
* Crop selection challenges
* Lack of real-time expert guidance

Traditional methods of yield estimation and farm advisory are:

* Manual and time-consuming
* Inconsistent
* Dependent on limited expertise

👉 This project addresses these challenges by building an **automated AI-driven system** that:

* Predicts crop yield
* Provides contextual farm recommendations
* Uses domain knowledge through RAG pipelines

---

## 🚀 Key Features

### 🌱 Crop Yield Prediction

* Predicts crop yield using machine learning models
* Handles structured agricultural datasets
* Supports preprocessing and feature engineering

### 🧠 Agronomy Knowledge Base (RAG)

* Local knowledge base stored in `rag_docs/`
* Uses **Retrieval-Augmented Generation** for contextual responses
* Provides domain-aware recommendations

### 🗂️ Vector Database Integration

* ChromaDB vectorstore for semantic search
* Hugging Face embeddings for document representation
* Fast retrieval of relevant agricultural insights

### 🤖 Agentic AI Farm Advisor

* Built using **LangGraph**
* Multi-step reasoning for advisory generation
* Context-aware decision-making system

### 📊 Structured Report Generation

* Powered by **Groq LLM API**
* Generates:

  * Crop insights
  * Risk analysis
  * Recommendations

### 🌐 Interactive UI

* Built using **Streamlit**
* User-friendly interface for:

  * Input data
  * Viewing predictions
  * Receiving advisory reports

---

## 🧩 Project Architecture

```text
User Input → Streamlit UI → ML Model Prediction
                     ↓
              LangGraph Agent
                     ↓
        RAG (ChromaDB + Embeddings)
                     ↓
          Groq LLM Report Generation
                     ↓
              Final Advisory Output
```

---

## 🛠️ Tech Stack

| Component       | Technology   |
| --------------- | ------------ |
| Programming     | Python       |
| ML Models       | Scikit-learn |
| Vector DB       | ChromaDB     |
| Embeddings      | Hugging Face |
| LLM             | Groq API     |
| Agent Framework | LangGraph    |
| UI              | Streamlit    |

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone <your-repo-link>
cd <your-repo-name>
```

---

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 LLM API Setup (Groq)

The project uses **Groq API** for report generation.

### Option 1: Environment Variable

```bash
export GROQ_API_KEY="your_api_key_here"
```

---

### Option 2: .env File

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

---

## 📦 Build the Vector Store

Run the following command to create embeddings and store them in ChromaDB:

```bash
python build_vectorstore.py
```

---

## ▶️ Run the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

👉 The application will open automatically in your browser.

---

## 📊 Machine Learning Workflow

### Data Processing

* Handling missing values
* Feature selection
* Data normalization

### Models Used

* Regression-based models for yield prediction
* (Extendable to advanced models like Random Forest / XGBoost)

---

## 📈 Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 🤖 Agent Capabilities

The LangGraph-based agent can:

* Interpret user inputs
* Retrieve relevant agronomy knowledge
* Generate actionable insights
* Provide structured farm recommendations

---

## 📂 Project Structure

```bash
.
├── app.py
├── build_vectorstore.py
├── rag_docs/
├── vectorstore/
├── models/
├── utils/
├── requirements.txt
└── README.md
```

---

## 👥 Team Contribution

| Member       | Contribution                                                      |
| ------------ | ----------------------------------------------------------------- |
| Adarsh | Model Development, RAG Integration, Streamlit UI |
| Satyam | Model Developement, Testing, Documentation |
|Himank | Model Developement , Documentation |
|Daksh | Report , Documentation , Agent Testing |
---

## 📌 Future Enhancements

* Integration with real-time weather APIs
* Advanced deep learning models
* Multi-crop prediction support
* Mobile-friendly interface
* Cloud deployment

---

## ✅ Conclusion

The **Crop Yield Prediction & Advisory System** demonstrates how **AI + RAG + Agentic workflows** can transform agriculture.

It provides:

* Accurate yield predictions
* Intelligent advisory support
* Scalable AI-driven decision-making

👉 This project is a step toward **smart farming and precision agriculture** using modern AI technologies.

---

