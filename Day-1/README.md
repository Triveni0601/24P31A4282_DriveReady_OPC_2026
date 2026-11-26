# Day 1: LLM Foundations, Prompt Engineering & API Mechanics

> **Drive Ready – AI Agents | One Person Company (OPC) Initiative**  
> *Department of Technical Training, Aditya University*

---

## Overview

Day 1 introduces the fundamental mechanics of Large Language Models (LLMs), covering how model parameters, system constraints, tokenization, and stateless architectures influence agent design. The codebase transitions from introductory toy examples (Sweet Crumbs Home Bakery) to building an enterprise assistant grounded in a real-world **One Person Company (OPC)** initiative: **KLG Ancient World** ([klgancientworld.com](https://klgancientworld.com)).

---

## Tasks Covered

### Task 1: Temperature Tuning & Output Determinism
* **Implementation:** `ChatGoogleGenerativeAI` configured with `model="gemini-3.5-flash-lite"`.
* **Objective:** Compare temperature settings at `temperature=0.2` (focused/deterministic) versus `temperature=1.2` (creative/divergent) by asking for a bakery tagline three consecutive times.
* **Logic & Behavior:**
  * **Low Temperature (0.2):** Generates focused, highly structured, and consistent taglines sticking close to the provided facts (freshly baked, eggless options, 24 hours notice).
  * **High Temperature (1.2):** Produces broader vocabulary, stylistic variations, and creative phrasing across consecutive invocations.

### Task 2: Grounding & Hallucination Mitigation
* **Implementation:** Strict negative constraint in the system prompt: `"never promise delivery times; if you do not know, say so."`
* **Test Input:** `"Do you deliver to Kakinada?"`
* **Logic & Behavior:** Because the bakery location radius is absent from the system prompt facts, the model adheres to its grounding rules and explicitly states its lack of knowledge (`"I am not entirely sure if we deliver to Kakinada..."`), preventing false promises or geographic hallucination.

### Task 3: Statelessness vs. Conversational Memory
* **Implementation:** OpenAI-compatible client interface (`client.chat.completions.create`) interacting with the Google Generative Language API.
* **Logic & Behavior:**
  * **Call 1:** `"My name is Ravi"` $\rightarrow$ Greets Ravi.
  * **Call 2 (Stateless Isolation):** `"What is my name?"` $\rightarrow$ LLM replies: *"I don't know your name since we haven't been introduced yet."* (Demonstrating that LLMs possess zero inherent memory between calls).
  * **Call 3 (History Injection):** Supplying full context `[system, user ("My name is Ravi"), assistant ("Hi Ravi..."), user ("What is my name?")]` $\rightarrow$ Model correctly responds: *"Your name is Ravi!"*.

### Task 4: Token Usage & Multilingual Efficiency Analysis
* **Implementation:** Comparing API token consumption (`response.usage.prompt_tokens`) across languages with identical semantics.
* **Test Inputs:**
  * English: `"Can I get an eggless cake today evening?"`
  * Telugu: `"ఈ రోజు సాయంత్రం నాకు ఎగ్‌లెస్ కేక్ దొరుకుతుందా?"`
* **Recorded Token Metrics:**
  * **English Query:** `125` prompt tokens
  * **Telugu Query:** `136` prompt tokens
* **Insight:** Non-Latin scripts require multi-byte subword/byte-fallback tokenization, resulting in higher token counts for identical semantic messages.

### Task 5: Domain Adaptation for OPC (KLG Ancient World)
* **Implementation:** Grounding an AI assistant for **KLG Ancient World** ([klgancientworld.com](https://klgancientworld.com)).
* **Facts Injected:**
  * Ancient scripts studied: Brahmi, Grantha, Latin, and Sanskrit.
  * Research narratives: Puranic narratives and Sthala Puranas.
  * Sacred Landscape Routes: Alqueva and St. Peter of the Orchards.
  * Digital products: MythiLens heritage exploration app (quests, learning, offline features).
* **Negative Rule:** If an answer is not in facts, reply: *"I'll check with the KLG team."*
* **Test Inquiries:**
  * Script query: `"Which ancient scripts do you study?"` $\rightarrow$ Correctly lists Brahmi, Grantha, Latin, Sanskrit.
  * Route query: `"Tell me about the Sacred Landscape Routes."` $\rightarrow$ Correctly highlights Alqueva and St. Peter of the Orchards.
  * Out-of-Domain Guardrail: `"Do you sell physical car tyres?"` $\rightarrow$ Correctly responds: *"I'll check with the KLG team."*

---

## Tech Stack & Setup

### Required Libraries
```bash
pip install -q langchain-google-genai openai
```

### Environment Variable Setup
Ensure your Google Gemini API key is configured in your environment or Google Colab Secrets under `GEMINI_API_KEY`:

```python
import os

# Local Python:
# export GEMINI_API_KEY="your-api-key-here"
# os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# Google Colab:
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```
> **Security Note:** Never commit or hardcode actual API keys into version control.

---

## How to Run

### Option 1: Run Locally
```bash
# Navigate to Day-1 folder
cd Day-1

# Run the task script
python task.py
```

### Option 2: Run in Google Colab / Interactive Python
Upload `task.py` to Google Colab or run within an interactive Python session, ensure `GEMINI_API_KEY` is configured in Colab Secrets or environment variables, and execute the script.

---

## Key Observations & Learnings

1. **Temperature Effects:** Lower temperatures (`0.2`) are crucial for enterprise fact-answering and constraint compliance, while higher temperatures (`1.2`) suit brainstorming and taglines.
2. **Stateless API Nature:** Chat models do not maintain server-side sessions; conversation state must be explicitly serialized and re-submitted on every turn.
3. **Multilingual Token Overhead:** Indic scripts like Telugu consume ~9% more prompt tokens for identical semantic phrases, impacting latency and inference costs.
4. **Reliable Fallback Mechanisms:** Clear fallback rules (`"If the answer isn't in FACTS, say 'I'll check with the KLG team.'"`) prevent the model from answering out-of-scope queries.
