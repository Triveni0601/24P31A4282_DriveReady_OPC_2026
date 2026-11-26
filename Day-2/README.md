# Day 2: Advanced Prompting, Few-Shot Classification, Structured JSON & Memory State

> **Drive Ready – AI Agents | One Person Company (OPC) Initiative**  
> *Department of Technical Training, Aditya University*

---

## Overview

Day 2 delves into advanced prompt engineering patterns, few-shot in-context learning for intent routing, deterministic JSON data extraction, and client-side conversational state architectures. It culminates with building an interactive conversational agent with persistent short-term memory and function logging for the **KLG Ancient World** ([klgancientworld.com](https://klgancientworld.com)) OPC project.

---

## Tasks Covered

### Task 1: Prompt Anatomy & Negative Constraint Enforcement
* **Implementation:** Evaluating system instructions with and without explicit behavioral negative rules.
* **Test Input:** `"Can you deliver in 1 hour?"`
* **Logic & Behavior:**
  * **Without Strict Rules:** The assistant attempts to be helpful and explains the 24-hour notice policy, but leaves ambiguity regarding future delivery timelines.
  * **With Explicit Negative Rules (`never promise delivery times; if you do not know, say so.`):** The assistant strictly declines the 1-hour request while clearly enforcing that specific delivery times are never promised.

### Task 2: Few-Shot In-Context Intent Classification
* **Implementation:** Low-temperature classification using 3-shot exemplar message pairs routing queries into three domain categories: `epigraphy`, `route`, or `bug`.
* **Few-Shot Exemplars:**
  * `"What does this Brahmi inscription say?"` $\rightarrow$ `epigraphy`
  * `"Tell me about the Alqueva sacred route"` $\rightarrow$ `route`
  * `"The offline map failed to load"` $\rightarrow$ `bug`
* **Test Messages & Model Classification Results:**
  1. `"Can you help transliterate a Grantha line?"` $\rightarrow$ `epigraphy`
  2. `"Where is the St. Peter of the Orchards route?"` $\rightarrow$ `route`
  3. `"MythiLens crashed when I opened the quest screen"` $\rightarrow$ `bug`
  4. `"What is the Sthala Purana story for this temple?"` $\rightarrow$ `epigraphy`
  5. `"How do offline GPS quests work?"` $\rightarrow$ `route`
* **Outcome:** Achieved 100% classification precision without requiring fine-tuning or complex classifiers.

### Task 3: Structured JSON Extraction & Sanitization
* **Implementation:** Zero-temperature extraction prompt forcing the model to output valid, parseable JSON with strictly defined keys (`name`, `item`, `qty`, `date`, `eggless`).
* **Input Query:** `"hi ravi here, need 2 plum cakes for 24th dec, eggless pls"`
* **Parsing Logic:**
  ```python
  cleaned = reply.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
  order = json.loads(cleaned)
  ```
* **Output Parsed Dict:**
  ```json
  {
    "name": "Ravi",
    "item": "plum cake",
    "qty": 2,
    "date": "24th dec",
    "eggless": true
  }
  ```

### Task 4: Memory Buffer & Function Logging (Sweet Crumbs Bakery)
* **Implementation:**
  * Created a `@log_call` decorator to track deterministic backend tool/pricing calls (`get_price`).
  * Implemented a sliding window buffer (`MAX_TURNS = 2`) for conversation history.
* **Execution Trace:**
  * `get_price('cupcake', 6)` $\rightarrow$ logged and returned `480`.
* **Critical Context Loss Finding:**
  * User introduces their name: `"Hi, my name is Karthik"`.
  * After 4 subsequent questions about menu and eggless options, the user asks: `"What is my name?"`.
  * **Bot Response:** *"You haven't told me your name yet! What should I call you? 😊"*
  * **Diagnosis:** Because `MAX_TURNS` was constrained to 2 exchanges, the earliest introduction was evicted from the sliding window, causing memory truncation.

### Task 5: Production-Ready Conversational Agent for OPC (KLG Ancient World)
* **Implementation:** Built an end-to-end interactive chat assistant with catalogue pricing lookups (`get_access_fee`), function call logging, and an expanded sliding window buffer (`MAX_TURNS = 8`).
* **Features:**
  * **Catalogue Integration:** Brahmi guide, Grantha guide, Alqueva sacred route, St. Peter of the Orchards route (`Rs 0` access fee).
  * **Bilingual Support:** Responds seamlessly in both English and Telugu (e.g., handling *"St. Peter of the Orchards route gurinchi cheppandi"*).
  * **Out-of-Domain Guardrail:** Rejects out-of-scope inquiries (*"Do you sell physical car tyres?"* $\rightarrow$ *"I'll check with the KLG team"*).
  * **Context Retention Verification:**
    * Query: `"What is my name and what routes did I ask about?"`
    * Response: *"Your name is Karthik, and you asked about the Alqueva sacred route and the St. Peter of the Orchards route!"*
    * Outcome: Full context successfully retained across extended dialogue turns.

---

## Tech Stack & Setup

### Required Libraries
```bash
pip install -q langchain-google-genai openai
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
```bash
# Navigate to Day-2 folder
cd Day-2

# Run the task script (note: Tasks 4 and 5 launch interactive chat sessions; type 'quit' or 'bye' to exit)
python task.py
```

### Option 2: Run in Google Colab / Interactive Python
Upload `task.py` to Google Colab or run within an interactive Python environment, ensure `GEMINI_API_KEY` is configured, and execute the script.

---

## Key Observations & Learnings

1. **Few-Shot Prompting Reliability:** Providing just 3 exemplar input-output pairs at `temperature=0.2` reliably constrained Gemini to output single-label intent classifications without extra conversational filler.
2. **Defensive JSON Cleaning:** Modern LLMs often wrap JSON responses in markdown fences (````json ... ````). Robust parsing requires stripping prefixes and suffixes prior to calling `json.loads`.
3. **Sliding Window Buffer Trade-offs:**
   * A small buffer (`MAX_TURNS = 2`) conserves tokens and reduces latency but causes catastrophic forgetting of initial user context (such as names and early preferences).
   * Expanding the buffer (`MAX_TURNS = 8`) preserves conversational continuity across multi-turn interactions while remaining within safe token bounds.
4. **Tool/Function Logging Decorators:** Wrapping helper functions with `@functools.wraps` provides clean audit trails of agent tool calls without polluting application business logic.
