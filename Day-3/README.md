# Day 3: LangChain Expression Language (LCEL), Automated Evaluation & Agent Architectures

> **Drive Ready – AI Agents | One Person Company (OPC) Initiative**  
> *Department of Technical Training, Aditya University*

---

## Overview

Day 3 focuses on productionizing LLM applications through **LangChain Expression Language (LCEL)**, establishing automated testing and evaluation harnesses, measuring prompt iteration benchmarks, analyzing temperature variance on evaluation stability, and establishing a decision framework for determining when a problem requires an autonomous AI Agent versus traditional code or a single LLM call.

---

## Tasks Covered

### Task 1: Conversational Chains with LCEL
* **Implementation:** Modern declarative pipeline built using LangChain Expression Language:
  ```python
  prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_V1),
      MessagesPlaceholder("history"),
      ("human", "{question}"),
  ])
  chain = prompt | llm | StrOutputParser()
  ```
* **Memory Management:** Stateful `ask(question)` function injecting message history via `MessagesPlaceholder` and updating history with `("human", question)` and `("ai", answer)` tuples.
* **Test Verification:**
  * Turn 1: `"Hi, I'm Karthikeya. Which ancient scripts do you study?"` $\rightarrow$ Lists Brahmi, Grantha, Latin, Sanskrit.
  * Turn 2: `"What's my name?"` $\rightarrow$ Successfully responds: *"Your name is Karthikeya! How can I help you explore KLG Ancient World today?"*.

### Task 2: Automated Testing & Evaluation Framework (`run_tests`)
* **Implementation:** Custom evaluation harness (`run_tests`) that parses structured test cases from CSV format and evaluates substring assertions against live model outputs.
* **Key Design Principle:** Each test invocation passes an isolated, empty conversation history (`"history": []`) to eliminate cross-test conversational contamination.
* **Benchmark Test Suite (10 Test Cases for KLG Ancient World):**
  1. *Which ancient scripts do you study?* $\rightarrow$ `PASS` (contains "Brahmi", "Grantha", etc.)
  2. *Do you cover South Indian inscriptions like Grantha?* $\rightarrow$ `PASS` (contains "Grantha")
  3. *Tell me about the Sacred Landscape Routes.* $\rightarrow$ `PASS` (contains "Alqueva" / "St. Peter")
  4. *Is St. Peter of the Orchards part of your routes?* $\rightarrow$ `PASS` (contains "St. Peter of the Orchards")
  5. *What is MythiLens?* $\rightarrow$ `PASS` (contains "MythiLens")
  6. *Does MythiLens have quests or learning features?* $\rightarrow$ `PASS` (contains "quest" / "learning")
  7. *What kind of traditional stories do you explore?* $\rightarrow$ `PASS` (contains "Puranic" / "Sthala Purana")
  8. *How much does a guided tour to Alqueva cost?* $\rightarrow$ `PASS` (contains "I'll check with the KLG team")
  9. *When is your next live excavation trip?* $\rightarrow$ `PASS` (contains "I'll check with the KLG team")
  10. *Do you sell physical car tyres?* $\rightarrow$ `PASS` (contains "I'll check with the KLG team")
* **Baseline Score (`rate_v1`):** `10/10 passed (100%)`.

### Task 3: Prompt Engineering Iteration & Comparative Evaluation
* **Implementation:** Refined prompt definition from `SYSTEM_V1` to `SYSTEM_V3`:
  * Explicitly codified strict fallback: *"If the answer isn't in FACTS, always reply with: 'I'll check with the KLG team.' Answer directly using only FACTS."*
* **Comparative Results:**
  * Baseline (`v1`): `100%` pass rate
  * Refined (`v3`): `100%` pass rate
  * Result: Pass rate maintained 100% while reducing ambiguity and eliminating hallucinated conversational filler.

### Task 4: Temperature Sensitivity & Stability Testing
* **Implementation:** Benchmarking prompt `SYSTEM_V3` against a high-temperature model instance (`llm_hot = ChatGoogleGenerativeAI(temperature=1.0)`).
* **Test Protocol:** Executed identical test suite across two consecutive evaluation passes (Run 1 and Run 2).
* **Observed Metrics:**
  * **Run 1:** `10/10 (100%)`
  * **Run 2:** `10/10 (100%)`
* **Takeaway:** Higher temperatures introduce significant vocabulary and syntactic variation between runs. While semantic meaning remains aligned, non-deterministic phrasing risks flipping borderline substring assertions, demonstrating why temperature must be pinned low (`0.0 - 0.2`) during automated evaluation and regression testing.

### Task 5: Architectural Decision Framework: "Agent or Not?"
A decision matrix established to prevent over-engineering:
* **Plain Code (Deterministic):**
  * *Example:* Computing offline image cache checksums or looking up route distances from a dictionary.
  * *Why:* Faster, zero token cost, deterministic execution, zero latency.
* **Single LLM Call (Transformative / Creative):**
  * *Example:* Translating a temple Sthala Purana summary from English into Pure Telugu.
  * *Why:* No state, planning, or external tool execution required; standard one-shot prompt completion.
* **AI Agent (Multi-Step / Goal-Oriented):**
  * *Example:* Handling complex user requests such as: *"Check if the Alqueva route is available offline in MythiLens, log my quest progress, and recommend my next epigraphy module."*
  * *Why:* Requires environmental perception, sequential decision-making, conditional branching, state management, and multiple tool/API calls.

---

## Tech Stack & Setup

### Required Libraries
```bash
pip install -q langchain-google-genai langchain-core
```

### Environment Variable Setup
```python
import os
from google.colab import userdata

try:
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
except Exception:
    print("Please set your GEMINI_API_KEY.")
```

---

## How to Run

### Option 1: Run Locally
To run the automated tests locally, ensure a `test.csv` file exists with columns `question` and `must_include`:

```csv
question,must_include
Which ancient scripts do you study?,Brahmi
Do you cover South Indian inscriptions like Grantha?,Grantha
Tell me about the Sacred Landscape Routes.,Alqueva
Is St. Peter of the Orchards part of your routes?,St. Peter of the Orchards
What is MythiLens?,MythiLens
Does MythiLens have quests or learning features?,quest
What kind of traditional stories do you explore?,Puranic
How much does a guided tour to Alqueva cost?,check with the KLG team
When is your next live excavation trip?,check with the KLG team
Do you sell physical car tyres?,check with the KLG team
```

Run the script:
```bash
cd Day-3
python task.py
```

### Option 2: Run in Google Colab / Interactive Python
Upload `task.py` to Google Colab or an interactive Python environment alongside `test.csv`, ensure `GEMINI_API_KEY` is set, and run the test harness.

---

## Key Observations & Learnings

1. **Declarative Composition with LCEL:** The pipe syntax (`prompt | llm | output_parser`) eliminates boilerplate code and standardizes streaming, batching, and async execution.
2. **State Isolation in Automated Testing:** Automated test suites must explicitly clear conversation history (`history: []`) between test cases to prevent semantic spillover and bias.
3. **Deterministic Evaluation Harnesses:** When using keyword/substring assertions, keep model temperature near zero (`0.0 - 0.2`) to ensure reproducible evaluation benchmarks.
4. **Agentic System Selection:** Autonomous agents should only be deployed when a problem requires dynamic planning, multiple external tool invocations, and environmental feedback; deterministic logic belongs in pure code.
