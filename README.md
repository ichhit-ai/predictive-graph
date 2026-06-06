# 🧠 Predictive Swarm Engine (Graphify Swarm)

A production-grade, high-fidelity predictive intelligence platform. It transitions standard GraphRAG from simple fact retrieval into a dynamic **causal-chain simulation arena** by combining structured graph databases with a swarm of adversarial specialist agents.

```
                  [ Raw PDF Document ]
                           │
                  Ingestor & Semantic Chunking
                           │
            Adaptive Spaced Ontology Generation
                           │
        Cascaded Graph Extraction (Sliding Window)
                           │
         Strict Entity Resolution & Deduplication
              (Jaccard Stemming & Initials Stripping)
                           │
           ┌───────────────┴───────────────┐
     SQLite Database              Specialist Agent Swarm
     (WAL Enabled)                (10 Domain-Specific Personas)
           │                               │
           └───────────────┬───────────────┘
                           ▼
              Dynamic Semantic Subgraphing
              (Agent-Specific Context)
                           │
          Adversarial Multi-Agent Debate Loop
          (Round 1 + Round 2 Counter-Arguments)
                           │
               Keyless Web & Reddit Grounding
                           │
               Executive Prediction Synthesis
```

---

## 🚀 Key Architectural Upgrades

### 1. Ingestion & Graph Construction
* **Adaptive Spaced Ontology:** Samples text fragments across the **Start, Middle, and End** of the document, ensuring complete visibility and prevention of schema loss in large files.
* **Sliding Context Window:** Tracks key proper nouns from preceding chunks ($N-2$, $N-1$) and injects them into chunk $N$'s extraction prompt to resolve cross-chunk entity references.
* **Hardened Entity Resolution:** Standardizes orthographic variations (e.g., resolving `"Joe R. Reeder"`, `"Joe Reeder"`, and `"J. Reeder"` to a single canonical entity) using **strict Jaccard token/N-gram character similarity** and initials stripping, eliminating loose containment snowballing.
* **Junk Entity Filtering:** Excludes pronouns, fragments, and non-descriptive nouns (e.g., `"body"`, `"face"`, `"victim"`) from polluting the network topology.

### 2. High-Performance Bottleneck Mitigations
* **Batch Embedding Generation:** Bundles chunk and entity vector generations into single parallel requests (up to 100 items per request), accelerating ingestion speed by **10x**.
* **Sliding Character & Token Limiter:** Monitors token and character flow (max 90,000 characters per minute), automatically pausing and resuming requests to prevent hitting Google's TPM limits.
* **Intelligent API Guardrails:** Automatically intercepts Google Gemini safety blocks (common when uploading personal diaries) and Daily Quota Exhaustion limits, stopping requests instantly and reporting clear explanations to the user instead of timing out or retrying.

### 3. Scaled Multi-Agent Debate Arena
* **10-Agent Specialist Swarm:** Dynamically spawns **10 distinct domain specialist personas** customized to the uploaded document's context for a wider variety of forecasting perspectives.
* **Dynamic Semantic Subgraphs:** Cosine similarity maps each agent's specialty directly to entity vector embeddings, injecting only the top 15 most relevant nodes/edges into their prompt context.
* **Adversarial Debate Loop:** 
  * *Round 1:* Agents analyze their subgraphs and make initial forecasts.
  * *Round 2 (Adversarial):* Agents are explicitly instructed to reference, challenge, or counter at least one claim made by another specialist in Round 1.
* **Real-World Scrapers Grounding:** Agents can execute keyless searches via Yahoo Search or scrape Reddit comments using inline `[SEARCH: <query>]` and `[REDDIT: <query>]` tags, grounding simulations in live information.

### 4. Advanced 2D ForceGraph UI
* **Degree-Centrality Scaling:** Node radius scales dynamically based on the connection density of each node (larger hubs stand out immediately).
* **Relationship Directionality:** Adds directional arrowheads on graph edges to map causal propagation paths clearly.
* **Camera Controls:** Includes a dedicated **Recenter Graph** button to dynamically reset the canvas zoom and position.
* **Rich Glassmorphic Design:** Standardised neon-glow shadows, clean styling, and dark mode optimizations.
* **Data Transparency Explorer:** Built-in tabbed layout for viewing structured **Nodes Table** and **Edges Table** with confidence scores, relationship types, and source evidence quotes.

---

## 🛠️ Installation & Setup

### 1. Clone & Prepare Directory
Make sure you are in the workspace folder containing the code:
```bash
cd /home/ichhit/Downloads/microsoft
```

### 2. Install Dependencies
Install the required packages (including the missing scraper dependencies):
```bash
pip install -r requirements.txt
```

Ensure your `requirements.txt` contains:
```text
streamlit>=1.35.0
google-generativeai>=0.5.4
pypdf>=4.2.0
beautifulsoup4>=4.12.0
```

### 3. Environment Variable Setup
Set your Gemini API Key in your shell or add it directly to a local `.env` file:
```bash
export GEMINI_API_KEY="your-api-key"
```

---

## 🎮 How to Run

Launch the Streamlit web application:
```bash
streamlit run app.py
```

### Steps to Simulate:
1. **API Credentials:** Provide your `Gemini API Key` in the sidebar (if not set in the environment).
2. **Ingest Document:** Upload a PDF file (e.g., a report, incident log, or narrative). The system will automatically chunk, extract, resolve, and store your graph.
3. **Inspect the Graph:** Explore the interactive 2D Canvas force-graph in the dashboard, or switch tabs to check raw Nodes/Edges tables.
4. **Trigger Scenario:** Type a what-if query (e.g., *"What if the windows were locked from the inside?"*) and press **Run Simulation**.
5. **Watch the Debate:** Observe the 10 specialists request real-world information and debate.
6. **Executive Summary:** Read the final consolidated predictive synthesis, containing causal steps and confidence metrics.
