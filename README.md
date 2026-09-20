# 🧭 AI Training Assistant for New Employees

An AI-powered onboarding assistant that answers new-employee questions by
**routing** them to the right knowledge source (company info, role guides,
or admin/policy documents) and generating **grounded, cited answers** with
Retrieval-Augmented Generation (RAG). Questions that don't match any
knowledge source fall back to a direct-LLM response with safe redirection
(e.g. "ask HR" for payroll/approval requests).

Built for the capstone: *AI Training Assistant for New Employees*.

---

## 1. Architecture

```
                         ┌───────────────────────┐
                         │   Employee Question    │
                         └───────────┬────────────┘
                                     │
                           ┌─────────▼─────────┐
                           │   Query Router     │  keyword hints +
                           │  (decision node)   │  TF-IDF similarity
                           └─────────┬─────────┘
              ┌──────────────┬───────┴────────┬──────────────┐
              ▼              ▼                ▼              ▼
      general_company  role_specific   admin_policy     direct_llm
     (company/*.md)    (roles/*.md)  (policies/admin/  (no doc match /
                                        faq/*.md)        sensitive topic)
              │              │                │              │
              └──────┬───────┴───────┬────────┘              │
                     ▼               ▼                       │
              TF-IDF Retriever (top-k chunks)                 │
                     │                                        │
                     ▼                                        ▼
              LLM Answer Generation (grounded, cited)   LLM Answer Generation
                     │                                    (redirect / refuse)
                     └───────────────┬────────────────────────┘
                                     ▼
                          Answer + Sources + Route shown in Chat UI
```

**Routing logic** blends two signals per candidate knowledge source:
1. **Keyword hints** — fast, interpretable priors (e.g. "expense", "PTO" → admin_policy).
2. **TF-IDF cosine similarity** — data-driven relevance against each knowledge
   source's document chunks.

If the best combined score is below a configurable confidence threshold, or the
question matches a sensitive-topic override (payroll, approvals, security bypass
attempts), the router sends the question to the **direct_llm** fallback path
instead of forcing an answer from an unrelated knowledge base.

**Retrieval** is TF-IDF-based (scikit-learn), chunked at the markdown
**section (`##`) level**, so every chunk can be cited as `file.md#Section`.
This requires no model downloads and runs fully offline.

**Generation** is pluggable (`src/llm_client.py`):
- `anthropic` — Claude API (`claude-sonnet-4-6` by default)
- `openai` — OpenAI Chat Completions
- `extractive` — **no API key needed.** Deterministic, clearly-cited answers
  built directly from retrieved passages. This is the default when no key is
  configured, so the whole prototype runs end-to-end with zero setup.

---

## 2. Project structure

```
.
├── app.py                       # Streamlit chatbot UI (memory + feedback + backend selectors)
├── pages/
│   └── 1_Admin_Dashboard.py     # Admin/Analytics dashboard (FAQs, activity, feedback)
├── cli_chat.py                   # Terminal chat, with conversation memory
├── evaluate.py                    # Runs evaluation_set.csv through the pipeline
├── generate_report.py              # Builds reports/EVALUATION_REPORT.md
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py               # Paths, routing thresholds, backend selection
│   ├── document_loader.py      # Markdown -> section-level Chunk objects
│   ├── retriever.py             # TF-IDF retrieval + per-route similarity scoring
│   ├── vector_retriever.py      # FAISS dense-vector retrieval (advanced backend)
│   ├── router.py                 # Rule-based + LLM-based query classification
│   ├── llm_client.py             # Pluggable LLM backend (anthropic/openai/extractive)
│   ├── prompts.py                 # System + user prompt templates (memory-aware)
│   ├── memory.py                   # Conversation memory (per-session, bounded window)
│   ├── analytics.py                # SQLite query/feedback logging for the dashboard
│   └── assistant.py                 # Orchestrator: route -> retrieve -> generate -> log
├── data/
│   ├── corpus/
│   │   ├── company/           # General company knowledge base
│   │   ├── roles/              # Role & team documents
│   │   ├── policies/           # Expense, leave, conduct, security
│   │   ├── admin/               # HR processes, IT access, timesheets, travel
│   │   └── faq/                  # Onboarding FAQ
│   ├── evaluation_set.csv       # 20 labeled Q/A pairs with expected routes
│   ├── dataset_manifest.json
│   └── analytics/                # SQLite DB of query/feedback logs (created at runtime)
├── tests/
│   ├── test_pipeline.py         # Routing + retrieval sanity tests (pytest)
│   └── test_advanced_features.py # FAISS, LLM router, memory, analytics tests
└── reports/                       # evaluate.py + generate_report.py write here
```

All corpus documents and the evaluation set are **synthetic / fictional**,
as required by the assignment.

---

## 2b. Feature checklist (capstone requirements)

| Requirement | Where it lives |
|---|---|
| Advanced vector database integration (FAISS) | `src/vector_retriever.py`, toggle with `RETRIEVAL_BACKEND=faiss` |
| LLM-based query classification | `src/router.py:classify_llm`, toggle with `ROUTER_BACKEND=llm` |
| Evaluation & Testing Report | `generate_report.py` → `reports/EVALUATION_REPORT.md` (accuracy metrics, response quality analysis, test cases, user feedback) |
| Source attribution / citations | Every RAG answer cites `file.md#Section`; shown inline and in a "📎 Sources" expander in the UI |
| Conversation memory | `src/memory.py`, wired into `assistant.ask(question, session_id=...)` |
| Admin/Analytics dashboard | `pages/1_Admin_Dashboard.py` (FAQs, user activity, query category stats, feedback) |

---

## 3. Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) configure an LLM backend
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY if you want
# full LLM-generated answers instead of the offline extractive mode.

# 4. (Optional) enable the live web search fallback
# edit .env and set SERPER_API_KEY (free key at https://serper.dev)
# so out-of-scope questions get answered from the live web instead of
# just being declined.
```

> **No API key? No problem.** The assistant automatically falls back to the
> offline `extractive` backend and is fully functional out of the box.
> Web search is the same: without `SERPER_API_KEY`, out-of-scope questions
> just get the plain "not in our docs, ask HR/your manager" response.

### Web search fallback (Groq or Serper.dev)

When a question doesn't match anything in the local onboarding knowledge
base, the assistant no longer just gives up. If `GROQ_API_KEY` is set,
Groq browser search is used first; otherwise, if `SERPER_API_KEY` is set, it:

1. Sends the question to Groq's server-side browser search, or to
   [Serper](https://serper.dev) (a Google Search API).
2. Feeds the top results back to the LLM (or, offline, formats them directly)
   to produce an answer clearly labeled as coming from the web, with
   numbered `[1] [2]` citations and clickable source links in a "🌐 Web
   sources" panel — never mixed in with internal document citations.
3. Falls back to the normal `direct_llm` response if the search itself
   fails or returns nothing useful, so a bad/rate-limited key never breaks
   the chat.

This route shows up in the Admin Dashboard's query category chart as
`web_search`, separate from the three document-backed routes.

### Voice input/output

The sidebar has a **🎙️ Ask by voice** control (uses your mic + free Google
speech recognition, no API key needed) — click it, speak your question, and
it's submitted just like typing it. Every assistant answer also gets a
**🔊 Listen** button (pure browser text-to-speech, works fully offline, no
API key), plus an optional "auto-read new answers aloud" toggle in the
sidebar. Voice input needs the `streamlit-mic-recorder` package (already in
`requirements.txt`); if it's missing, that section just shows a one-line
note instead of breaking the app. Voice output has no dependency at all —
it's built into every modern browser.

---

## 4. Running the assistant

### Chatbot UI (Streamlit)
```bash
streamlit run app.py
```
Opens a browser chat interface with:
- Routing info, source citations (📎 Sources expander), and a routing-scores debug panel
- **Conversation memory** — ask natural follow-ups ("what about for a PM instead?")
  and the assistant keeps context; "🔄 New conversation" resets it
- **👍 / 👎 feedback buttons** under every answer, logged for the Admin Dashboard
- Sidebar selectors to switch the **retrieval backend** (TF-IDF / FAISS) and
  **query router** (rule-based / LLM) live
- A link to the **Admin / Analytics Dashboard** page

### Admin / Analytics Dashboard
```bash
streamlit run app.py   # then open the "Admin Dashboard" page from the sidebar
```
Shows frequently asked questions, user activity over time, query category
(route) statistics, and user feedback results — all read live from the
same SQLite store (`data/analytics/analytics.db`) that the chat UI writes to.

### Terminal chat
```bash
python cli_chat.py
python cli_chat.py --retrieval faiss --router llm
```
Also has conversation memory; type `new` to reset it, `exit`/`quit` to leave.

### Evaluation
```bash
python evaluate.py
python evaluate.py --retrieval faiss     # evaluate the FAISS vector backend
python evaluate.py --router llm          # evaluate LLM-based routing
```
Runs all 20 labeled questions in `data/evaluation_set.csv` through the full
pipeline and reports:
- Routing accuracy (predicted vs. expected route)
- Citation match rate (did retrieval surface the gold source document?)
- Key-phrase presence rate in the generated answer

Results are written to `reports/eval_results.csv`.

**Current benchmark (offline extractive backend, TF-IDF retrieval, rule-based router):**
| Metric | Score |
|---|---|
| Routing accuracy | 19/20 (95%) |
| Citation match rate | 16/17 (94.1%) |
| Key-phrase presence rate | 13/17 (76.5%) |

**With the FAISS vector-DB retrieval backend** (`--retrieval faiss`): 18/20
(90%) routing accuracy, same 94.1% citation match — a useful accuracy/latency
trade-off to show in a report, and it upgrades automatically to true semantic
sentence embeddings if `sentence-transformers` + a locally cached model are
available (see `requirements.txt`).

### Evaluation & Testing Report
```bash
python generate_report.py
python generate_report.py --retrieval faiss --router llm
```
Writes `reports/EVALUATION_REPORT.md` with four sections: **Accuracy
Metrics**, **Response Quality Analysis** (answer length, citation coverage,
per-route breakdown), **Test Cases** (all 20 labeled questions with
pass/fail per check), and **User Feedback Results** (pulled live from the
👍/👎 buttons in the chat UI).

### Tests
```bash
pytest -q
```
23 tests covering routing, TF-IDF + FAISS retrieval, the LLM-router offline
fallback, conversation memory, and the analytics store.

---

## 5. Example interactions

| Question | Route | Behavior |
|---|---|---|
| "What are the company's core values?" | `general_company` | Retrieves `company_overview.md#Values`, answers with citation |
| "What are a Data Analyst's first 30 days expectations?" | `role_specific` | Retrieves `data_analyst_role.md#First 30 Days Expectations` |
| "How do I submit an expense claim?" | `admin_policy` | Retrieves `expense_policy.md#Submission Process` |
| "What is my exact salary breakup?" | `direct_llm` | Declines and redirects to HR payroll process (personal data) |
| "Can you approve my leave request right now?" | `direct_llm` | Declines and redirects to manager/HR portal approval |

---

## 6. Extending this project

- **True semantic embeddings for FAISS:** `src/vector_retriever.py` uses
  TF-IDF+SVD (LSA) embeddings by default so it needs no model download.
  Install `sentence-transformers`, point `SENTENCE_TRANSFORMERS_LOCAL_PATH`
  at a locally cached model, and it upgrades automatically — same
  `top_k()` / `score_routes()` interface, nothing else changes.
- **Swap in Pinecone/Chroma:** implement the same interface as
  `VectorRetriever` (`top_k`, `score_routes`, `route_indices`) against a
  hosted vector DB and select it in `assistant.py`.
- **Add a new route:** add a folder under `data/corpus/`, register it in
  `src/config.py:ROUTE_SOURCES`, and add keyword hints in `src/router.py`.
- **Add a new LLM backend:** implement `_generate_<name>()` in
  `src/llm_client.py` and add it to `_resolve_backend()`.
- **Persist conversation memory across restarts:** `src/memory.py` is
  in-process only; swap `SessionMemoryStore`'s dict for a Redis/DB-backed
  store to survive app restarts or scale across workers.

---

## 7. Limitations

- TF-IDF retrieval is fast and dependency-light but purely lexical — it can
  miss paraphrased questions that share no vocabulary with the source
  document. The FAISS backend's default LSA embeddings help somewhat but
  are still corpus-derived, not a pretrained semantic model — true
  semantic recall needs `sentence-transformers` (see above).
- The keyword-hint list in `router.py` is a small, hand-curated seed set and
  should grow with real usage data. LLM-based routing (`ROUTER_BACKEND=llm`)
  is more flexible but costs a model call per question and needs an API key.
- The `extractive` fallback produces readable, cited answers but does not
  synthesize/summarize across passages the way a full LLM would.
- Conversation memory is in-process and per-session; it does not persist
  across app restarts and isn't shared across multiple server workers.
- The analytics store (SQLite) is fine for a prototype/single-instance
  deployment but would need a proper database for multi-instance production use.
- No authentication, audit logging, or PII redaction is implemented — needed
  before any production/internal deployment with real employee data.
