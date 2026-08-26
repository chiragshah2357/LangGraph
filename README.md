# LangGraph — Complete Notes

A single reference merged from the topic notes. Flow: **what it is → the state schema → reducers → notebook walkthroughs → memory & tool calls → routers, tools & agents**.

## Contents

1. [Getting Started with LangGraph](#1-getting-started-with-langgraph)
2. [What Actually Is the State Schema?](#2-what-actually-is-the-state-schema)
3. [Reducers — How State Updates Merge](#3-reducers--how-state-updates-merge)
4. [LangGraph Basics — Notebook Walkthroughs](#4-langgraph-basics--notebook-walkthroughs)
5. [Memory Across Conversations & How Tool Calls Really Work](#5-memory-across-conversations--how-tool-calls-really-work)
6. [Router — the LLM as a Router (a Basic Agent)](#6-router--the-llm-as-a-router-a-basic-agent)
7. [Tools — the Bridge to External Systems](#7-tools--the-bridge-to-external-systems)
8. [Chatbot with Multiple Tools](#8-chatbot-with-multiple-tools)
9. [Agents — the ReAct Architecture](#9-agents--the-react-architecture)
10. [Practical Gotchas — Fixes from Building the Multi-Tool Chatbot](#10-practical-gotchas--fixes-from-building-the-multi-tool-chatbot)
11. [Types of RAG — Agentic & Adaptive RAG](#11-types-of-rag--agentic--adaptive-rag)
12. [Autonomous RAG — the Self-Managing RAG System](#12-autonomous-rag--the-self-managing-rag-system)

---

# 1. Getting Started with LangGraph

## What is LangGraph?

**LangGraph** is a library for building **stateful, multi-actor applications with LLMs** — used to create **agent and multi-agent workflows**.

- Inspired by **Pregel** and **Apache Beam**; its public interface draws from **NetworkX**.
- Built by **LangChain Inc** (creators of LangChain), but **can be used without LangChain**.
- Powers **production-grade agents**, trusted by LinkedIn, Uber, Klarna, GitLab, and many more.
- Provides **fine-grained control over both the flow and state** of your agent applications via a central **persistence layer**.

### Key features enabled by persistence

- **Memory** — persists arbitrary aspects of application state, supporting memory of conversations and other updates within and across user interactions.
- **Human-in-the-loop** — because state is **checkpointed**, execution can be **interrupted and resumed**, allowing decisions, validation, and corrections at key stages via human input.

## The LangChain Ecosystem

| Layer | What it holds | License |
|-------|---------------|---------|
| **Deployment** | LangGraph Platform | Commercial |
| **Components** | Integrations, Vector Databases, Tools | OSS |
| **Architecture** | LangChain + LangGraph | OSS |
| **LangSmith** (credits) | Debugging, Playground, Prompt Management, Annotation, Testing, Monitoring | Commercial |

## AI Agents / Agentic AI

An **agentic** system is more than a single LLM call:

```
Input ──▶ LLM ──▶ Output
             │
             └──▶ 3rd-party APIs / Tools  (can loop back)
```

Instead of answering in one shot, the LLM can call **tools** and **3rd-party APIs**, then continue reasoning.

## LangGraph Internals

**LangGraph = Lang + Graph.** It models a workflow as a **DAG (Directed Acyclic Graph)** of **stateful AI agents** that control the **flow of information**.

### Two core components

1. **Nodes** — units of work (e.g. an LLM call, `llm(input)`).
2. **Edges** — connections that decide what runs next.

Every graph runs **START → nodes → END**.

### Simple / Sequential workflow

```
START ──▶ Chatbot (calls LLM) ──▶ END
```

### Complex / Conditional workflow

```
        START
          │
          ▼
       Chatbot ──(edges)──┐
        │                 │
        ▼                 ▼
     Weather          Temperature
        │                 │
        ▼                 ▼
       END               END
```

The **Chatbot** node branches via **conditional edges** to a **Weather** or **Temperature** node — each ending at **END**.

## In One Line

LangGraph is a **graph-based framework** where you wire LLM-powered **nodes** together with **edges** to build **stateful, controllable, multi-agent AI workflows** — from a single chatbot to branching, tool-using agents.

---

# 2. What Actually Is the State Schema?

## The plain-English version

The **state** is just **one dictionary that travels through your graph** — the single shared "notepad" every node reads from and writes to.

The **state schema** is just you **declaring, up front, what fields that dictionary is allowed to have and what type each one is.** It's a *blueprint*, not a container — it describes the shape of the data that will flow, it doesn't hold the data itself.

## Why it needs declaring

In LangGraph, nodes **don't call each other** — they can't hand data to the next node directly. Instead there's one shared object:

- a node **reads** what it needs from the state,
- **returns a small update**,
- LangGraph **merges** that update in and passes the state to the next node.

For that hand-off to work, LangGraph must know what the state looks like. That declaration *is* the schema.

## The diagram

```
                 STATE SCHEMA  (the blueprint / form)
             ┌──────────────────────────────────────┐
             │  class State(TypedDict):             │
             │      name: str                        │
             │      game: Literal["cricket", ...]    │
             └──────────────────────────────────────┘
                              │ StateGraph(State)
                              ▼
      the shared "notepad" (one dict per run) flows through:

   START
     │   state = {"name": "Krish"}
     ▼
 ┌───────────┐   reads state['name']
 │ playgame  │   returns {"name": "... want to play"}   ── update ──┐
 └───────────┘                                                       │
     │   state = {"name": "Krish want to play"}   ◄── merged in ─────┘
     ▼
 ┌───────────┐   reads state['name']
 │  cricket  │   returns {"name": "... cricket", "game": "cricket"}
 └───────────┘
     │   state = {"name": "Krish want to play cricket", "game": "cricket"}
     ▼
   END  ──►  final state returned to caller
```

**Mental model:** `state` = the shared memory of one graph run; `schema` = the form/template that memory must fill in. Nodes never pass data directly — they all read and write the *same* notepad.

## The part that trips people up

The **schema is only the shape definition.** Four separate things sit on top of it — and Notebooks 1–4 were really about these differences:

| Thing | What it controls | In the notebooks |
|-------|------------------|------------------|
| **How you define it** | Syntax of the blueprint | `TypedDict` / `@dataclass` / Pydantic `BaseModel` |
| **How you read it** | Dict vs attribute access | `state['name']` vs `state.name` |
| **Whether types are enforced** | Is bad input rejected? | TypedDict & dataclass = **no**; Pydantic = **yes** (runtime `ValidationError`) |
| **How updates merge** | Overwrite vs accumulate | default = **overwrite**; a **reducer** (`add_messages`) = **append** |

## One sentence

The **state schema is your declaration of what the shared data dictionary flowing through the graph looks like** — its field names and types — and every node is guaranteed to receive that dictionary, read from it, and return updates to it.

---

# 3. Reducers — How State Updates Merge

## The plain-English version

A **reducer** is a **merge rule for one state field**. It tells LangGraph *how* to combine the field's **old value** with the **update a node returns** — instead of just replacing it.

```
new_value = reducer(old_value, node_returned_value)
```

This is the fourth row of the state-schema table: *how updates merge* — overwrite vs accumulate.

## The problem it solves

Nodes never call each other. Each node just **returns a small dict** of the keys it wants to update, and LangGraph merges that into the shared state. By **default that merge is an overwrite** — the new value clobbers the old one.

For a field like `messages` (a growing conversation), overwrite is a disaster:

```
Start:      messages = [A]
Node 1 returns {"messages": [B]}  →  messages = [B]        ✗ lost A
Node 2 returns {"messages": [C]}  →  messages = [C]        ✗ lost A and B
```

Every turn wipes the history. You wanted it to **accumulate**.

## The two behaviours

| | Rule | Result |
|---|------|--------|
| **No reducer (default)** | `reducer(old, new) → new` | overwrite — old value thrown away |
| **`add_messages` reducer** | `reducer(old, new) → old + new` | append — history grows |

Same idea as Python's `functools.reduce` or a Redux reducer: a function that folds a new update into existing state.

## How you declare it — `Annotated`

You attach the reducer to a field with `Annotated`. The reducer rides along as the field's metadata:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    #                    ↑type  ↑reducer (the merge rule)
```

`Annotated[list, add_messages]` reads as: *messages is a list, and whenever a node updates it, merge with `add_messages` instead of overwriting.* Now the same run accumulates:

```
Start:      messages = [A]
Node 1 returns {"messages": [B]}  →  messages = [A, B]      ✓
Node 2 returns {"messages": [C]}  →  messages = [A, B, C]   ✓
```

Any field **without** an `Annotated[..., reducer]` keeps the default overwrite behaviour — reducers are per-field.

## Why `add_messages` and not just `+`

`add_messages` is smarter than a plain list append. It also:

- **Assigns an ID** to any message that doesn't have one.
- **Updates in place** instead of duplicating when the incoming message shares an ID with an existing one — useful for streaming and editing a message mid-run.

That's why LangGraph ships it as a prebuilt reducer rather than making you write `lambda old, new: old + new`.

## One sentence

A **reducer is the per-field merge rule**: without one a node's update **overwrites** the field, and annotating a field like `messages` with `add_messages` switches the rule to **append**, so conversation history grows across nodes instead of being replaced.

---

# 4. LangGraph Basics — Notebook Walkthroughs

Notes on the notebooks in `1-LangGraphBasics/`.

## Notebook 1 — `1-simplelangchain.ipynb`: a graph with conditional branching

A toy **"which sport to play"** graph that teaches the core LangGraph mechanics.

- **State schema** — `State(TypedDict)` with one key `graph_info: str`. This is the shared data passed between nodes.
- **Nodes are just functions** — `start_play`, `cricket`, `badminton`. Each takes `state`, appends text to `graph_info`, and returns the updated value. By default the returned value **overwrites** the state key.
- **Conditional edge** — `random_play(state)` returns `"cricket"` or `"badminton"` based on `random.random() > 0.5`. Its return value is the **name of the next node** to route to.
- **Graph wiring:**

  ```
  START → start_play → (random_play decides) → cricket   → END
                                             ↘ badminton → END
  ```

  Built with `StateGraph(State)`, `add_node`, `add_edge`, `add_conditional_edges`, then `.compile()`.
- **Invocation** — `.invoke({"graph_info": "Hey My name is Krish"})` runs START → node → END, and the text accumulates along the path taken, e.g. `"Hey My name is Krish I am planning to play Cricket"`.

**Takeaway:** how to define state, add nodes, and use a **conditional edge** to branch the flow.

### Key snippet

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    graph_info: str

def start_play(state: State):
    return {"graph_info": state['graph_info'] + " I am planning to play"}

def random_play(state: State) -> Literal['cricket', 'badminton']:
    return "cricket" if random.random() > 0.5 else "badminton"

graph = StateGraph(State)
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)
graph_builder = graph.compile()
```

## Notebook 2 — `2-chatbot.ipynb`: a minimal LLM chatbot graph

The simplest possible chatbot — teaches **reducers + real LLM nodes**.

- **State with a reducer** — `messages: Annotated[list, add_messages]`. Unlike notebook 1 (overwrite), `add_messages` **appends** each new message to the list instead of replacing it, so conversation history accumulates.
- **The LLM** — `ChatGroq(model="openai/gpt-oss-120b")`, with `GROQ_API_KEY` loaded from `.env`.
- **One node** — `superbot(state)` calls `llm.invoke(state['messages'])` and returns `{"messages": [response]}`; the reducer merges that `AIMessage` into the history.
- **Graph wiring:**

  ```
  START → SuperBot → END
  ```

- **Invocation** — `.invoke({'messages': "Hi, My name is Chirag..."})` returns the state containing both the `HumanMessage` and the LLM's `AIMessage`.

**Takeaway:** the same graph pattern, but now with (a) a **reducer** to accumulate messages and (b) a **node that calls an actual LLM** — the foundation of every LangGraph chatbot.

### Key snippet

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="openai/gpt-oss-120b")

def superbot(state: State):
    return {"messages": [llm.invoke(state['messages'])]}

graph = StateGraph(State)
graph.add_node("SuperBot", superbot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)
graph_builder = graph.compile()
```

## Notebook 3 — `3-DatclassStateScheme.ipynb`: different ways to define the state schema

**Point of the notebook:** the state schema doesn't have to be a `TypedDict` — LangGraph accepts other Python types too. It builds the **same sport graph two ways** to compare how you define and access state.

### Part A — TypedDict (recap)

- State = `TypedDictState` with `name: str` and `game: Literal["cricket", "badminton"]`.
- Nodes read state with **dictionary syntax**: `state['name']`.
- Same flow as Notebook 1: `START → playgame → (decide_play 50/50) → cricket / badminton → END`.
- Important note: TypedDict types are just **hints** — checked by tools like mypy / the IDE, **not enforced at runtime**. That's why `graph.invoke({"name": "123"})` runs fine even though `"123"` isn't a real name.

### Part B — Dataclass

- Same graph, but state = a `@dataclass` (`DataClassState`) instead of a dict.
- Only real difference: nodes access fields with **attribute syntax** → `state.name` instead of `state['name']`.
- Everything else (nodes, conditional edge, wiring, invoke) is identical.

### Key snippet

```python
from typing_extensions import TypedDict
from typing import Literal
from dataclasses import dataclass

# Option A: dict-style access -> state['name']
class TypedDictState(TypedDict):
    name: str
    game: Literal["cricket", "badminton"]

# Option B: attribute-style access -> state.name
@dataclass
class DataClassState:
    name: str
    game: Literal["badminton", "cricket"]
```

**Takeaway:** LangGraph is **flexible about how you define the state schema**. `TypedDict` (accessed like a dict, `state['name']`) and `@dataclass` (accessed like an object, `state.name`) are interchangeable for the same graph. (A later step usually adds **Pydantic**, which — unlike these two — *does* enforce types at runtime.)

## Chains with LangGraph — the 4 building blocks

(From the "Chains with LangGraph" notes — the roadmap for turning the basic graph into a tool-using agent.)

The base shape is the familiar chain: `START → Node 1 → Node 2 → … → Node n → END`, using **nodes**, **normal edges**, and **conditional edges** (a conditional edge can loop back). On top of this skeleton, four new ideas are added:

1. **Chat Messages** — the graph state now holds **chat messages** instead of plain strings: `HumanMessage` = input, `AIMessage` (from the LLM) = output. The state carries a conversation.
2. **Chat Models** — use actual **LLM chat models inside a graph node** (the node calls the model to do its work).
3. **Binding Tools** — give the LLM **tools** so it can reach **external sources** — 3rd-party APIs, a vector database, etc. (e.g. fetching *recent AI news*). Input → LLM decides to use a tool → output. This is "**bind tools to our chat model.**"
4. **Execute Tool Calls** — actually **run the tool calls** the LLM requests, from within the graph nodes: the LLM says "call this tool," the node executes it and feeds the result back.

**Takeaway:** this upgrades the basic graph into a **chain that passes chat messages through LLM nodes, binds tools to the model for external data, and executes the tool calls the LLM makes** — the foundation of a tool-using agent.

## In one line

Notebook 1 = graph structure + **conditional routing** (overwrite state); Notebook 2 = same skeleton but with a **reducer** (`add_messages`) and a **real LLM node** = a working chatbot; Notebook 3 = the same graph showing the state schema defined as a **TypedDict vs a dataclass** (dict access vs attribute access). **Chains** = adding chat messages, chat-model nodes, tool binding, and tool-call execution to build a tool-using agent.

---

# 5. Memory Across Conversations & How Tool Calls Really Work

Two common points of confusion, cleared up.

## 1. Does the graph "remember" previous conversations?

**Within one conversation/run: yes. Across separate conversations: not automatically — you need persistence.**

- The `add_messages` reducer accumulates messages **inside a single thread/run**. As long as the growing `messages` list is passed along, the model sees the whole history — that's memory *within* the conversation.
- But two separate `graph.invoke(...)` calls are independent — the second starts **fresh**. In-memory state is gone once a run ends.
- To remember **across** runs (or app restarts), add a **checkpointer** + a **`thread_id`**:

```python
from langgraph.checkpoint.memory import MemorySaver   # or SqliteSaver / Postgres for real persistence

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [...]}, config)   # later calls with the same thread_id resume the same history
```

**Rule of thumb:**
- `add_messages` → accumulate history **within** a run.
- checkpointer + `thread_id` → remember **across** runs.

This is the "Memory" + persistence-layer feature from the getting-started notes.

## 2. Can the LLM put chunks directly into a vector DB by calling tools?

**Mostly yes — but the LLM never touches the database itself. It only *requests* the tool; your code runs it.**

```
LLM does NOT write to the DB.
LLM outputs a tool call  →  your graph node executes the tool  →  the tool writes to the vector DB
```

- The LLM's output is just a **structured request**, e.g. *"call `store_document(text=...)`."*
- The tool's actual Python code (embed the text → insert into Pinecone / FAISS / etc.) does the work.
- The LLM decides *whether* and *with what arguments*; it has **no direct DB access**.

This is exactly the two Chains concepts:
- **Binding Tools** = giving the model the capability.
- **Execute Tool Calls** = your node actually running the requested tool.

### Practical note

You *can* wire it this way, but **bulk ingestion of chunks into a vector DB is usually a plain data pipeline, not an LLM job.** Loading + chunking + embedding thousands of documents is deterministic bulk work — you don't want to pay an LLM to decide each insert. Letting the LLM call a write-tool makes sense for **selective memory** ("remember this fact"), not bulk ingestion.

| Direction | Common? | Who does it |
|-----------|---------|-------------|
| **Read** from vector DB via tool (retrieval / RAG) | Very common | LLM calls a `search` tool; tool queries the DB |
| **Write** to vector DB via tool | Possible, selective | LLM requests it; the **tool** writes — usually for saving specific memories, not bulk ingest |

## One sentence

Message history only persists across separate conversations if you add a **checkpointer + thread_id**; and an LLM never touches a vector DB directly — it **emits a tool call** and your **graph node executes the tool** that does the reading or writing.

---

# 6. Router — the LLM as a Router (a Basic Agent)

## Recap: what we already have

A graph that (1) uses **messages as state** and (2) a **chat model with bound tools** (`bind_tools([add])`). When that model runs, it returns **one of two things**:

1. **A tool call** — "please run `add(a, b)`"
2. **A natural-language response** — a direct answer, no tool

## The key idea: the LLM *is* the router

Compare the two graph shapes:

```
Workflow (rigid):          Router / Agent (branches):
  start                       start
    │                           │
    ▼                           ▼
 llm_tool                    llm_tool ──┐  (routes)
    │                         │         │
    ▼                         ▼         ▼
   end                      tools      end   (direct response)
                              │
                              ▼
                             end
```

> *"We can think of this as a **router**, where the chat model routes between a direct response or a tool call based upon the user input."*

The branching decision is made by **`tools_condition`**: it inspects the LLM's last message — if it contains a tool call → route to the **`tools`** node; if not → route to **`END`**.

## Why this is a "Basic Agent"

> *"This is a simple example of an **agent**, where the LLM is directing the control flow either by calling a tool or just responding directly."*

```
                    (tools bound = its "menu")
START ──▶ LLM (the BRAIN) ──┬──▶ node ──▶ End     (Step 2: respond directly)
                            └──▶ node ──▶ End     (Step 3: call a tool)
```

An **agent = an LLM that decides what the program does next.** The router is the smallest possible version of that: one decision, two outcomes.

## One sentence

A **router** is a graph where the LLM itself chooses the next step — call a tool or answer directly — via `tools_condition`; that single act of the model *directing control flow* is what makes it the simplest kind of agent.

---

# 7. Tools — the Bridge to External Systems

## The problem tools solve

An LLM is just a **brain** — text in, text out. Ask *"what's the current temperature in New York?"* and the raw model can't answer; that data isn't in its weights. It needs to reach the outside world.

```
 I/P ──▶ ┌─────────┐ ──▶ O/P ("current temp of New York")  ──▶ Human
         │   LLM   │
         │ (Brain) │ ◀──▶  Tool Call  ──▶  3rd-party API
         └─────────┘        Schema {JSON}    Weather API
              ▲                               Database / external source
           Binding
```

A **tool** is that bridge — a documented function the LLM can invoke to reach 3rd-party APIs, a weather API, a database, or any external source.

## How the LLM uses a tool

- **Schema `{JSON}`** — every tool exposes a JSON schema (name, purpose, argument types). That's how the model knows *what the tool does* and *what inputs to pass*.
- **Binding** — `llm.bind_tools([...])` hands the model the **menu** of available tools. Binding does **not** let the model *run* them; it lets the model *request* them. Your graph node runs the actual function.

## Example — the `add` tool

```
 I/P "What is 2 plus 2?"  (natural language)
        │
        ▼
      START ──▶ Chatbot (LLM) ──▶ LLM → BRAIN → Tool_call
                   │  ┌──────────────┐
                   ├─▶│ Tool: add()  │──▶ END
                   │  │ """docstring"""
                   └─▶ END            │  return a + b
                                      └──────────────┘
```

- The model reads the natural-language input and decides to call `add()`.
- **The docstring is mandatory** — LangChain uses it as the tool's description in the JSON schema. A tool function with no docstring (and no explicit `description`) raises `ValueError: Function must have a docstring if description not provided.` Decorate with `@tool` and give it a one-line docstring.

## One sentence

A **tool** is a schema-described, docstring-documented function you **bind** to the LLM so it can *request* external actions; the model chooses the tool and its arguments, and your **graph node executes it**.

---

# 8. Chatbot with Multiple Tools

## Same router, bigger toolbox

The graph shape is **identical** to the single-tool router — only the `tools` node now wraps a whole **collection**, and `tool_calling_llm` has all of them bound:

```
__start__ ──▶ tool_calling_llm ──▶ tools ──▶ __end__
                     │                          ▲
                     └──────────────────────────┘  (conditional: no tool → straight to end)
```

## The toolbox

| Tool | What it fetches | Kind |
|------|-----------------|------|
| **Arxiv** | research papers | prebuilt integration |
| **Wikipedia** | encyclopedia content | prebuilt integration |
| **Internet Search (Tavily)** | live web results (needs a **Tavily API key**) | prebuilt integration |
| **add()** | custom arithmetic | your own function |
| **multiply()** | custom arithmetic | your own function |

## The insight: you don't route per tool

You **bind all of them** and let the **LLM pick the right one(s)** per query:

- *"recent papers on RAG?"* → Arxiv
- *"who invented Python?"* → Wikipedia
- *"news today?"* → Tavily
- *"12 × 7?"* → multiply

`tools_condition` still answers just **one** yes/no question — *did the model request any tool?* — and `ToolNode` dispatches to whichever specific tool the model named. Mixing **prebuilt integrations** (Arxiv, Wikipedia, Tavily) with **your own functions** (add, multiply) in a single `tools=[...]` list is the whole point.

## One sentence

Scaling from one tool to many changes **nothing** about the graph — you bind a mixed list of prebuilt and custom tools, and the LLM + `tools_condition` + `ToolNode` handle selection and dispatch automatically.

---

# 9. Agents — the ReAct Architecture

## Recap: the four building blocks

Everything so far has been climbing one ladder:

```
Chains  →  Routers  →  Tools  →  Basic Agent
```

Supporting pieces: **`ToolNode`** + **`tools_condition`** (the conditional-edge router), and **LangSmith** for **tracking & monitoring** a running agent.

## Basic Agent = the router you already have

```
Natural-language i/p
        │
        ▼
      START ──▶ LLM (BRAIN) ──┬──▶ END        (direct answer)
                              └──▶ Tools ──▶ End
```

This is the multi-tool chatbot. The LLM decides: answer directly, or call a tool. But notice — the `Tools` node goes **straight to End**. It calls a tool *once*.

## The new idea: ReAct — a *general* agent architecture

A real agent loops through three steps until the task is done:

1. **Act** — the model calls a specific tool.
2. **Observe** — the tool's output is fed **back** to the model.
3. **Reason** — using that output + the original input, the model decides the **next step**: call another tool, or finish.

## Why the loop matters

Prompt: **"Please add 5 plus 5 and then multiply by 3."**

- A **basic router** calls one tool and stops — it *can't* complete this two-step task.
- A **ReAct agent** loops:

```
Act      → add(5, 5)
Observe  → 10
Reason   → "now multiply that by 3"
Act      → multiply(10, 3)
Observe  → 30
Reason   → "done"  → END
```

## The one structural change

```
Router (calls a tool once):        ReAct agent (loops):
  llm ──▶ tools ──▶ END               llm ──▶ tools
                                       ▲         │
                                       └─────────┘   (tools loops BACK to llm)
```

Turning a router into a ReAct agent is a **single edge change**: make the `tools` node route **back to the LLM** (`tools → llm`) instead of to `END`. That back-edge is what lets the model observe each tool result and chain multiple tool calls. Everything else — state, `add_messages`, `tools_condition`, `ToolNode` — stays exactly the same.

## One sentence

**ReAct = Act → Observe → Reason, on a loop**; the only graph change from the basic router is that the `tools` node feeds its output *back to the LLM* instead of ending, which is what enables multi-step tool use.

---

# 10. Practical Gotchas — Fixes from Building the Multi-Tool Chatbot

Real errors hit while building the notebooks, and what actually fixed them. These are environment/version issues, not concept issues — but they'll stop you cold.

## A tool function needs a docstring

`ToolNode([add])` (or `@tool`) auto-converts a function into a tool and uses its **docstring as the description**. A function with no docstring raises:

```
ValueError: Function must have a docstring if description not provided.
```

**Fix:** decorate with `@tool` and give it a one-line docstring.
```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b
```

## Name your nodes and edges consistently

`builder.add_node("llm_tools", ...)` but then `builder.add_edge(START, "llm_tool")` (missing `s`) fails at `compile()` with an "unknown node" error. The node name in `add_node` must match every `add_edge` / `add_conditional_edges` reference **exactly**. (`tools_condition` routes to a node literally named `"tools"` by default.)

## Display/invoke the graph variable you actually compiled

If your tool graph compiles into `graph_builder` but your display cell draws `graph` (an earlier, simpler compile), you'll render the **wrong** graph and never see your `tools` node. Keep one consistent variable name — compile the final graph into `graph` and both `display(...)` and `graph.invoke(...)` use it.

## arxiv — the version trap

`langchain_community`'s `ArxivQueryRun` calls `arxiv.Search(...).results()`, which **modern `arxiv` (>=2) removed** → `AttributeError: 'Search' object has no attribute 'results'`. But pinning `arxiv<2` hits the opposite wall: 1.4.x uses `http://` and Arxiv now 301-redirects to https → `HTTPError: HTTP 301`. No single version satisfies the old wrapper.

**Fix:** skip the broken wrapper — wrap the **modern arxiv client** as your own `@tool`.
```python
import arxiv as arxiv_lib          # uv add "arxiv>=2"
from langchain_core.tools import tool

_client = arxiv_lib.Client()       # modern client uses https

@tool
def arxiv(query: str) -> str:
    """Search arxiv.org for papers and return titles, authors, dates and summaries."""
    papers = _client.results(arxiv_lib.Search(query=query, max_results=2))
    return "\n\n".join(
        f"Title: {p.title}\nAuthors: {', '.join(a.name for a in p.authors)}\nSummary: {p.summary[:500]}"
        for p in papers
    ) or "No results found."
```

## wikipedia — HTTP 429 → JSONDecodeError

The old `wikipedia` package sends a **generic User-Agent**, which Wikimedia now rate-limits (HTTP 429). The 429 body is plain text, and the library blindly calls `.json()` on it →

```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Fix:** set a **descriptive User-Agent** (and the https endpoint) *before* creating the wrapper.
```python
import wikipedia
wikipedia.wikipedia.USER_AGENT = "MyApp/1.0 (https://github.com/you/your-repo)"
wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
```

## Restart the kernel after changing packages or tool objects

Jupyter caches imported modules and variables. After a `uv add`/downgrade, or after editing a cell that (re)defines a tool, **re-running one cell isn't enough** — the kernel still holds the old module/object in `sys.modules` and the namespace. Use **Restart → Run All** so the change actually takes effect. (This is why a fix that's correct on disk can still show the identical old error.)

## One sentence

Most "it doesn't work" moments here are **version/environment traps** — docstring-less tools, node-name typos, stale kernels, and langchain-community's aging arxiv/wikipedia wrappers — fixed by adding docstrings, keeping names consistent, wrapping modern clients yourself, setting a real User-Agent, and restarting the kernel.

---

# 11. Types of RAG — Agentic & Adaptive RAG

Two advanced RAG patterns, both built as **LangGraph state machines** (nodes + conditional edges that can loop). They're the natural payoff of everything above: an agent that decides *how* to retrieve.

## Baseline: Traditional RAG (a fixed pipeline)

```
Question ──▶ retrieve from DB ──▶ stuff docs into LLM ──▶ Answer
```

It **always** retrieves, from **one** source, the **same** way — no matter the question. No decisions, no recovery if the docs are irrelevant.

## 11a. Agentic RAG — let the agent decide how/whether to retrieve

**Agentic RAG puts an LLM agent in charge of retrieval.** You give the LLM a **retriever tool** and it makes the decisions:

- **Whether to retrieve at all** (a simple question may not need it).
- **Which source** to query (company policy DB, legal docs DB, …).
- **Whether the retrieved docs are relevant** — a **"check relevance" conditional edge** grades them.
- **Rewrite the query and retry** if they're not relevant.
- **Fail gracefully** — say *"I don't know"* instead of hallucinating.

Graph shape (LangGraph):

```
        ┌───────── rewrite ◀──┐ (docs not relevant)
        ▼                     │
START ─▶ agent ──(should retrieve?)──▶ retrieve (tool) ──(check relevance)──┐
          │  no                                                             │ yes
          ▼                                                                 ▼
         END                                                            generate ──▶ Answer
```

> "To implement a retrieval agent, we simply need to give an LLM access to a **retriever tool**." The agent node decides to call a `function_call` (retrieve) or end; the conditional edge after retrieval routes to **generate** (relevant) or **rewrite** (not).

**Traditional vs Agentic:** traditional RAG's retrieval is hardwired; agentic RAG's retrieval is a *decision the LLM makes at runtime* — including which of several DBs to hit, and whether to give up.

## 11b. Adaptive RAG — route by complexity, then self-correct

**Adaptive RAG dynamically adjusts its strategy based on the query's complexity** — dig deep for hard questions, answer simple ones directly. It unites **two** ideas:

**(1) Query analysis (routing).** A classifier inspects the question and picks a route:
- *related to the index* → **retrieve** from the vector store
- *unrelated to the index* → **web search**
- *simple / factual* → answer directly (no retrieval)

This is the jump from **single-step** (always one retrieve) and **multi-step** (always many) to **adaptive** (as many steps as the query actually needs).

**(2) Active / self-corrective RAG (a.k.a. self-reflective RAG).** After retrieving, the graph *grades its own work* in a loop:
- **Grade documents** — relevant? If not → **rewrite the question** and retrieve again.
- **Generate**, then **check for hallucinations** — is the answer grounded in the docs? If not → regenerate.
- **Check answer relevance** — does it actually address the question? If not → rewrite/retry.

```
Question ─▶ Query Analysis ─┬─(related)────▶ Retrieve ─▶ Grade ─(relevant?)─┬─yes─▶ Generate ─(grounded? answers Q?)─▶ Answer
                            │                              ▲                 └─no──▶ Re-write question ─┘
                            ├─(unrelated)─▶ Web search ─▶ Generate ─▶ Answer
                            └─(simple)────▶ Answer directly
```

Because it grades, rewrites, and re-checks, it's a **RAG state machine** (loops via conditional edges), not a linear **RAG chain**. **CRAG (Corrective RAG)** and **Self-RAG** are the named self-reflection variants.

## One sentence

**Agentic RAG** makes retrieval a decision the LLM controls (retrieve? where? relevant? rewrite? give up?); **Adaptive RAG** routes each query by complexity and then self-corrects — grading docs, checking for hallucinations, and retrying — so both replace the fixed RAG pipeline with a looping LangGraph state machine.

---

# 12. Autonomous RAG — the Self-Managing RAG System

These notes build a ladder from a plain retrieval graph up to a fully autonomous system. Each rung adds one capability.

## 12a. The ladder

**1) Basic Agentic RAG.** A linear graph: `__start__ → retriever → responder → __end__`.
- Document → **embedding** → **vectorstore**; a query hits the **retriever**, which pulls **context** from the store.
- **Responder** = `LLM + context` → output. This is ordinary RAG expressed as a graph.

**2) Agentic RAG with ReAct.** The LLM becomes the **brain**, *bound* to tools and looping via ReAct.
- `__start__ → react_agent → __end__`, where `react_agent` is `LLM` with **Tool 1 = Retriever**, **Tool 2 = Wikipedia**, **Tool 3 = Web Search** bound to it.
- Adds a **self-reflection** loop: `Retriever + LLM → (self-reflection) → LLM judges: Yes → output / No → loop back`.

**3) Query Planning & Decomposition vs Chain-of-Thought.** Two ways to handle a hard question:

| Aspect | Chain-of-Thought (CoT) | Query Planning & Decomposition |
|--------|------------------------|--------------------------------|
| Purpose | Let the LLM reason step-by-step | Break a complex query into structured sub-queries |
| Style | Natural-language reasoning path | Explicit sub-queries / formal question segments |
| Inspiration | Human-like scratchpad thinking | Structured task planning / modular Q&A |
| Agent behavior | Think → Retrieve → Think → Answer | Plan all → Retrieve all → Answer once |

Query-planning flow: `PLAN query → SQ1/SQ2/SQ3 → retrieve each → combine context → output`.

**4) Iterative Retrieval.** A grading loop instead of a single shot:
```
Retrieve → Generate → Reflect → Refine ──(not good enough)──▶ back to Retrieve
                                        └─(good)─────────────▶ END
```

**5) Answer Synthesis from multiple sources.** One query fans out across sources and the LLM **synthesizes** one answer: `LLM → {Retriever, Wikipedia, ArXiv, YouTube} → synthesize → output`.

**6) Autonomous RAG.** The top of the ladder — it *combines all of the above*.

## 12b. What is Autonomous RAG?

**Autonomous RAG is a Retrieval-Augmented Generation system where the LLM (or agent) can reason, plan, act, reflect, and improve — on its own — without manual control over each step.**

It combines:
- **Agentic reasoning** (ReAct / LangGraph agents)
- **Self-reflection & self-correction**
- **Dynamic tool selection**
- **Multi-source retrieval**

Put differently, it stacks: **ReAct + CoT + Query Planning/Decompose + Retrieval strategies + Self-Reflection**.

## 12c. Core components

| Component | Role |
|-----------|------|
| **Planner Agent** | Breaks complex queries into sub-questions |
| **Tool Selector** | Chooses between Wikipedia, ArXiv, vector DBs, APIs, etc. |
| **Retriever** | Executes tool calls to retrieve relevant documents |
| **Synthesizer** | Uses the LLM to generate the final answer |
| **Reflector** | Verifies whether context or answer is good enough |
| **Retry Loop** | Refines and retries if reflection fails |
| **Memory** (optional) | Stores feedback, logs bad queries, improves prompts/tools |

## 12d. Complete flow (workflow with LangGraph)

```
Query
  ▼
Query Planning & Decomposition        ①
  ▼
Chain of Thought                      ②
  ▼
ReAct:  Reason → Act → Observe        (Agentic RAG)
  ▼
Iterative Retrieval Check ◀─────────────────┐  (Multiple Sources)
  ▼                                          │
Answer Synthesis                             │
  ▼                                          │
Self-Reflection ─(No)────────────────────────┘
  │
  └─(Good)─▶ END
```

## 12e. Agentic RAG vs Autonomous RAG

| Concept | Agentic RAG | Autonomous RAG |
|---------|-------------|----------------|
| **Definition** | A RAG system using an **agentic approach** — an LLM reasons, plans, and acts using tools | A RAG system that **operates independently**, with **full self-management** of planning, retrieving, reflection, and improvement |
| **Focus** | Structured reasoning and tool use (ReAct, LangGraph) | Complete autonomy in task execution, retry, and learning |
| **Behavior** | Think → Act → Observe → Answer | Think → Act → **Reflect → Retry → Learn** → Answer |
| **Retry logic** | Optional — usually static agent plans | Built-in retry/refine strategies (context + answer reflection) |
| **Self-reflection** | May include it optionally | **Core feature** — reflects on retrieval & answers before finalizing |
| **Tool use** | Uses tools via agents (Wikipedia, SQL, ArXiv) | Selects and adapts tools **dynamically** based on reasoning |
| **Planner** | Often present (manual or LLM-generated plans) | **Always present** — triggers multi-step workflows adaptively |
| **Learning loop** | Not always present | May log feedback and improve over time |

## One sentence

**Autonomous RAG** is the fully self-managing end of the RAG ladder: it plans and decomposes the query, reasons (CoT + ReAct), retrieves from multiple sources, then **reflects, retries, and learns on its own** — where Agentic RAG stops at *Think → Act → Observe → Answer*, Autonomous RAG closes the loop with *Reflect → Retry → Learn*.
