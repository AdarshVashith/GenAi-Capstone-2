"""LangGraph farm advisory agent with prediction, retrieval, and report generation."""

from __future__ import annotations

import os
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

import logging

from agent.prompts import REPORT_PROMPT_TEMPLATE
from agent.retriever import get_vectorstore
from config import LLM_MODEL_NAME
from src.predict import predict_yield

logger = logging.getLogger(__name__)


class FarmAgentState(TypedDict):
    farm_data: dict[str, Any]
    yield_prediction: dict[str, Any]
    retrieved_docs: list[dict[str, Any]]
    final_report: str


def predict_node(state: FarmAgentState) -> FarmAgentState:
    """Predict yield from structured farm data."""
    prediction = predict_yield(state["farm_data"])
    return {**state, "yield_prediction": prediction}


def assess_risk_node(state: FarmAgentState) -> FarmAgentState:
    """Assign a simple risk band from the predicted yield value."""
    predicted_value = float(state["yield_prediction"]["predicted_yield"])
    if predicted_value >= 4.0:
        risk_level = "Low"
    elif predicted_value >= 2.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    updated_prediction = {**state["yield_prediction"], "risk_level": risk_level}
    return {**state, "yield_prediction": updated_prediction}


def retrieve_node(state: FarmAgentState) -> FarmAgentState:
    """Retrieve agronomic references for the crop and risk level."""
    vectorstore = get_vectorstore()
    query = f"{state['farm_data']['Item']} crop guidance risk level {state['yield_prediction']['risk_level']}"
    docs = vectorstore.similarity_search(query, k=4)
    retrieved_docs = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
        }
        for doc in docs
    ]
    return {**state, "retrieved_docs": retrieved_docs}


def generate_report_node(state: FarmAgentState) -> FarmAgentState:
    """Generate a structured farm advisory report from retrieved evidence."""
    retrieved_context = "\n\n".join(
        f"Source: {doc['source']}\n{doc['content']}" for doc in state["retrieved_docs"]
    )
    prompt = REPORT_PROMPT_TEMPLATE.format(
        farm_inputs=state["farm_data"],
        predicted_yield=state["yield_prediction"]["predicted_yield"],
        risk_level=state["yield_prediction"]["risk_level"],
        retrieved_context=retrieved_context or "No agronomic references retrieved.",
    )
    if os.getenv("GROQ_API_KEY"):
        llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        report = llm.invoke(prompt).content
    else:
        references = (
            "\n".join(f"- {doc['source']}" for doc in state["retrieved_docs"])
            or "- No references retrieved."
        )
        actions = (
            "\n".join(
                f"- {doc['content'].splitlines()[1] if len(doc['content'].splitlines()) > 1 else doc['content'][:120]}"
                for doc in state["retrieved_docs"][:3]
            )
            or "- Review irrigation, nutrient, and pest management guidance."
        )
        report = (
            "## Crop & Field Summary\n"
            f"- Area: {state['farm_data']['Area']}\n"
            f"- Crop: {state['farm_data']['Item']}\n"
            f"- Avg Temperature: {state['farm_data']['avg_temp']}\n"
            f"- Rainfall: {state['farm_data']['average_rain_fall_mm_per_year']}\n"
            f"- Pesticide Usage: {state['farm_data']['pesticides_tonnes']}\n\n"
            "## Yield Prediction\n"
            f"- Predicted Yield: {state['yield_prediction']['predicted_yield']:.2f}\n\n"
            "## Yield Risk Status\n"
            f"- Risk Level: {state['yield_prediction']['risk_level']}\n\n"
            "## Recommended Farming Actions\n"
            f"{actions}\n\n"
            "## Agronomic References (from retrieved docs only)\n"
            f"{references}\n\n"
            "## Disclaimer\n"
            "- This fallback report was generated without a live LLM because `GROQ_API_KEY` is not set."
        )
    return {**state, "final_report": report}


def build_farm_agent():
    """Build the sequential LangGraph workflow."""
    graph = StateGraph(FarmAgentState)
    graph.add_node("predict", predict_node)
    graph.add_node("assess_risk", assess_risk_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "predict")
    graph.add_edge("predict", "assess_risk")
    graph.add_edge("assess_risk", "retrieve")
    graph.add_edge("retrieve", "generate_report")
    graph.add_edge("generate_report", END)
    logger.info("LangGraph workflow compiled successfully.")
    return graph.compile()


def run_farm_agent(farm_data: dict[str, Any]) -> FarmAgentState:
    """Run the farm advisory workflow and return the final state."""
    agent = build_farm_agent()
    initial_state: FarmAgentState = {
        "farm_data": farm_data,
        "yield_prediction": {},
        "retrieved_docs": [],
        "final_report": "",
    }
    logger.info("Invoking farm agent workflow...")
    return agent.invoke(initial_state)


def answer_follow_up(report_context: str, question: str) -> str:
    """Answer a follow-up question based on the provided report context."""
    if not os.getenv("GROQ_API_KEY"):
        return "Cannot answer follow-up questions without a `GROQ_API_KEY`."

    prompt = (
        "You are an expert agronomy consultant. A user has received the following farm advisory "
        "report and has a follow-up question. Use the report content to provide a helpful, "
        "accurate, and professional answer.\n\n"
        f"--- ADVISORY REPORT ---\n{report_context}\n\n"
        f"--- USER QUESTION ---\n{question}\n\n"
        "Expert Answer:"
    )
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0.7)
    return llm.invoke(prompt).content


if __name__ == "__main__":
    try:
        example_state = run_farm_agent(
            {
                "Area": "India",
                "Item": "Wheat",
                "avg_temp": 25.0,
                "average_rain_fall_mm_per_year": 800.0,
                "pesticides_tonnes": 50.0,
            }
        )
        logger.info("Generated Prediction: %s", example_state["yield_prediction"])
        logger.info(
            "Generated Report Snippet: %s...", example_state["final_report"][:200]
        )
    except BaseException as exc:
        logger.exception("Unable to run farm agent: %s", exc)
