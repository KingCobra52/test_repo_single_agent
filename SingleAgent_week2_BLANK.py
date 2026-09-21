"""Week 2 breakout starter for the Single-Agent team in the Nittany AI ELP.

Run locally:
    python -m pip install -r requirements.txt
    python -m streamlit run SingleAgent_week2_BLANK.py

Create a .env file beside this script containing:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key

Complete TODO 1, TODO 2, and TODO 3.
"""

import json
import os
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tavily import TavilyClient

st.set_page_config(page_title="AI Research Assistant", page_icon="🔍")


# --- Configuration: provided ---
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside SingleAgent_week2_BLANK.py."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


# =============================================================================
# TODO 1 — WRITE THE SYSTEM PROMPT (about 8-10 minutes)
# =============================================================================
# Include all four parts:
#   Role: Who is the model?
#   Task: What must it produce?
#   Constraints: What makes each task useful?
#   Format: What exact JSON structure must it return?
#
# Requirements:
#   - 4-6 tasks
#   - each task is a specific, searchable, non-empty string
#   - tasks collectively cover the user's question
#   - tasks do not substantially overlap
#   - tasks can be answered using public sources
#   - output shape stays {"tasks": ["...", "..."]}
# =============================================================================
SYSTEM_PROMPT = """You are a query decomposition assistant. Your role is to break a complex user question into a small set of independent, searchable sub-questions that, taken together, fully answer the original question.

TASK
Given a user's question, produce 4-6 tasks. Each task is a specific, self-contained question or search query that can be answered using public sources (web search, documentation, articles, etc.). The tasks should collectively cover every important aspect of the user's question.

CONSTRAINTS
- Produce between 4 and 6 tasks inclusive.
- Each task must be a specific, searchable, non-empty string (not a vague topic like "background" or "details").
- Tasks must not substantially overlap — each should target a distinct facet of the question.
- Every task must be answerable using publicly available information.
- Together, the tasks must cover the user's question completely, so that answering all of them yields a full answer.
- If the user's question is simple, still decompose it into its distinct verifiable sub-parts rather than returning a single task.
- Do not include explanations, preamble, or commentary outside the JSON.

FORMAT
Return ONLY valid JSON in exactly this shape:
{"tasks": ["<task 1>", "<task 2>", "<task 3>", "<task 4>"]}

Rules for the format:
- The top-level object has exactly one key: "tasks".
- "tasks" is a JSON array of strings.
- The array contains 4-6 strings.
- No trailing commas, no extra keys, no markdown fences around the JSON.
""".strip()


# --- Graph state: provided ---
class ResearchState(TypedDict):
    question: str
    research_plan: list
    search_results: list[dict]
    response: str


# --- Shared model helper: provided ---
def call_model(system_prompt: str, question: str) -> dict:
    """Call DeepSeek and parse a JSON object from the response."""
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# TODO 2 — COMPLETE THE PLANNER NODE (about 3-4 minutes)
# =============================================================================
# 1. Call call_model() using SYSTEM_PROMPT and state["question"].
# 2. Read "tasks" from the returned dictionary.
# 3. Keep the provided validation.
# 4. Return the tasks under the research_plan state key.
# =============================================================================
def plan_node(state: ResearchState) -> dict:
    output = call_model(SYSTEM_PROMPT, state["question"])
    tasks = output.get("tasks")

    # Keep this validation after defining tasks.
    if (
        not isinstance(tasks, list)
        or not 4 <= len(tasks) <= 6
        or any(not isinstance(task, str) or not task.strip() for task in tasks)
    ):
        raise ValueError(
            "The model must return 4-6 non-empty research-task strings "
            "inside a 'tasks' list."
        )

    return {"research_plan": tasks}


def task_to_query(task) -> str:
    """Convert a Team 1 task into a Tavily search query."""
    return str(task).strip()


# --- Node 2: Search with Tavily: provided ---
def search_node(state: ResearchState) -> dict:
    # Week 2 intentionally searches only the first generated task.
    query = task_to_query(state["research_plan"][0])

    if not query:
        raise ValueError("The first research task is empty.")

    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=5,
    )
    results = response.get("results", [])

    if not results:
        raise ValueError("Tavily did not return any search results.")

    return {"search_results": results}


# --- Node 3: Answer the first task: provided ---
def response_node(state: ResearchState) -> dict:
    first_task = task_to_query(state["research_plan"][0])

    source_text = "\n\n".join(
        f"Title: {result.get('title', 'Untitled')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Content: {result.get('content', '')}"
        for result in state["search_results"]
    )

    prompt = f"""You are a research assistant. This is one research task from a
larger question. Answer only this task, using only the search results below.
Keep the response concise and cite source URLs next to the claims they support.

Overall question:
{state['question']}

Research task to answer:
{first_task}

Search results:
{source_text}
"""

    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"response": response.choices[0].message.content}


# =============================================================================
# TODO 3 — CONNECT THE LANGGRAPH WORKFLOW (about 2-3 minutes)
# =============================================================================
# Add four edges in this order:
#   START -> plan -> search -> respond -> END
# =============================================================================
def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("respond", response_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# --- Streamlit UI: provided ---
st.title("🔍 Research Assistant - Team 1: Planner")
st.caption("Planner → Tavily Search → DeepSeek Response")

question = st.text_input(
    "Enter a business research question",
    value="Compare OpenAI and Anthropic for a company choosing an enterprise AI platform.",
)

if st.button("Research", type="primary") and question.strip():
    initial_state: ResearchState = {
        "question": question.strip(),
        "research_plan": [],
        "search_results": [],
        "response": "",
    }

    try:
        # Compiling here allows the page to open before TODO 3 is finished.
        research_graph = build_graph()
        with st.spinner("Planning, searching, and preparing the response..."):
            result = research_graph.invoke(initial_state)
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
    else:
        st.subheader("Research plan")
        st.json(result["research_plan"])

        st.subheader("Answer (first task only this week)")
        st.caption(f"Task: {task_to_query(result['research_plan'][0])}")
        st.write(result["response"])

        with st.expander("Raw search results"):
            st.json(result["search_results"])
