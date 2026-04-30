"""
ModexAI Agent Demo
==================
A LangChain agent that autonomously discovers, evaluates, and purchases
a fine-tuned model from the ModexAI Exchange API.

Task: "Optimize finance analysis" → agent self-trades a LoRA model.

Usage
-----
    # Start the API first:
    #   docker compose up
    #
    # Then run this script:
    python demo_agent.py

Environment variables (optional, set in ../.env or shell):
    MODEXAI_API_URL  – defaults to http://localhost:8000
    OPENAI_API_KEY   – required only for GPT-backed agent;
                       set to "ollama" to use the local Ollama LLM instead.
"""

from __future__ import annotations

import json
import os

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEXAI_API_URL = os.getenv("MODEXAI_API_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------


class SearchModelsInput(BaseModel):
    niche: str = Field(..., description="The model niche to search for, e.g. 'finance', 'claims', 'devops'")


class EvalModelsInput(BaseModel):
    niche: str = Field(..., description="The niche to evaluate models for")
    samples: list[str] = Field(..., description="A list of task prompt samples to evaluate models against")


class BuyModelInput(BaseModel):
    model_id: str = Field(..., description="The ID of the model to purchase")
    token: str = Field(default="mock", description="Payment token; use 'mock' for the local demo")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def search_models(niche: str) -> str:
    """Search the ModexAI marketplace for models matching a niche."""
    try:
        resp = requests.get(f"{MODEXAI_API_URL}/models", params={"niche": niche}, timeout=10)
        resp.raise_for_status()
        models = resp.json()
        if not models:
            return f"No models found for niche '{niche}'."
        lines = [f"Found {len(models)} model(s) for niche '{niche}':"]
        for m in models:
            lines.append(
                f"  • {m['id']} — {m['name']} | ${m['price_usd']} | "
                f"accuracy={m['benchmarks'].get('accuracy', 'n/a')}"
            )
        return "\n".join(lines)
    except requests.RequestException as exc:
        return f"Error contacting ModexAI API: {exc}"


def evaluate_models(niche: str, samples: list[str]) -> str:
    """Evaluate top-3 models for a niche using provided sample prompts."""
    try:
        payload = {"niche": niche, "samples": samples}
        resp = requests.post(f"{MODEXAI_API_URL}/eval", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No evaluation results returned."
        lines = [f"Evaluation results for niche '{niche}':"]
        for r in results:
            lines.append(
                f"  • {r['model_id']} — score={r['score']} | latency={r['avg_latency_ms']}ms"
            )
        best = results[0]["model_id"]
        lines.append(f"\nRecommended model: {best}")
        return "\n".join(lines)
    except requests.RequestException as exc:
        return f"Error contacting ModexAI API: {exc}"


def buy_model(model_id: str, token: str = "mock") -> str:
    """Purchase a model from the ModexAI marketplace."""
    try:
        resp = requests.get(
            f"{MODEXAI_API_URL}/buy/{model_id}",
            params={"token": token},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            f"Purchase successful!\n"
            f"  Model:          {data['model_id']}\n"
            f"  Status:         {data['status']}\n"
            f"  Download token: {data['download_token']}\n"
            f"  Model path:     {data['model_path']}"
        )
    except requests.RequestException as exc:
        return f"Error contacting ModexAI API: {exc}"


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

modexai_tools = [
    StructuredTool.from_function(
        func=search_models,
        name="search_models",
        description=(
            "Search the ModexAI marketplace for fine-tuned models by niche. "
            "Returns a list of available models with IDs, names, prices, and benchmark scores."
        ),
        args_schema=SearchModelsInput,
    ),
    StructuredTool.from_function(
        func=evaluate_models,
        name="evaluate_models",
        description=(
            "Evaluate the top-3 models for a given niche by running sample prompts through them. "
            "Returns ranked results with scores and latency. Use this before buying."
        ),
        args_schema=EvalModelsInput,
    ),
    StructuredTool.from_function(
        func=buy_model,
        name="buy_model",
        description=(
            "Purchase a model from ModexAI. Pass the model_id of the model you want to buy. "
            "Use token='mock' for the local demo. Returns a download token and model path."
        ),
        args_schema=BuyModelInput,
    ),
]

# ---------------------------------------------------------------------------
# LLM selection
# ---------------------------------------------------------------------------


def _build_llm():
    if OPENAI_API_KEY and OPENAI_API_KEY not in ("", "ollama"):
        print("Using OpenAI GPT-4o-mini as the agent LLM.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    print(f"Using local Ollama (phi3:mini) at {ollama_host} as the agent LLM.")
    return ChatOllama(model="phi3:mini", base_url=ollama_host)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous AI agent operating on the ModexAI model exchange.
Your goal is to optimize AI model usage by discovering, evaluating, and purchasing the best
fine-tuned model for a given task — just like a trader on a stock exchange.

Follow this process:
1. Search for models matching the required niche.
2. Evaluate the top candidates using representative task samples.
3. Purchase the highest-scoring model.

Always complete the full loop: search → evaluate → buy.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def run_agent(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"ModexAI Agent Demo")
    print(f"Task: {task}")
    print(f"API:  {MODEXAI_API_URL}")
    print(f"{'='*60}\n")

    llm = _build_llm()
    agent = create_tool_calling_agent(llm, modexai_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=modexai_tools, verbose=True, max_iterations=10)

    result = executor.invoke({"input": task})
    print(f"\n{'='*60}")
    print("Final output:")
    print(result["output"])
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_agent(
        "Optimize finance analysis: find and deploy the best finance model available on ModexAI."
    )
