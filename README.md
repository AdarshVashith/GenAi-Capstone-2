# Crop Yield Prediction - Milestone 2

This project now includes:

- A local agronomy knowledge base in `rag_docs/`
- A ChromaDB vectorstore builder using Hugging Face embeddings
- A LangGraph farm advisory agent
- Groq-backed structured report generation
- A simple Streamlit UI in `app.py`

## Install

```bash
pip install -r requirements.txt
```

## LLM API Setup

The assistant uses **Groq** for report generation.

Set your Groq API key before running the advisory app:

```bash
export GROQ_API_KEY="your_api_key_here"
```

You can also create a local `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
GROQ_API_KEY=your_api_key_here
```

## Build the Vectorstore

```bash
python build_vectorstore.py
```

## Run the Streamlit App

```bash
streamlit run app.py
```

# GenAi-Capstone-2
