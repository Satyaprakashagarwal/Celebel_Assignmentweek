# Chat With Your PDFs — RAG + Streamlit + MongoDB

A local Retrieval-Augmented Generation (RAG) app. Drop some PDFs in, and ask
questions about them in a chat UI — answers are grounded only in what's
actually in your documents, with sources shown for every reply. It also
remembers things about you (and your past questions) across sessions using
MongoDB, so it gets a bit more personalized the more you use it.

Started from the classic "RAG Tutorial v2" pattern (PDFs → Chroma → local
LLM), then extended with a hybrid retriever, a free hosted LLM, and
long-term memory.

## What it does

- **Ask questions about your PDFs** and get answers built only from the
  content it retrieves — if the docs don't cover it, it says so instead of
  guessing.
- **Understands typos and rephrased questions.** A single vector search is
  surprisingly brittle — this app layers a few things on top so sloppy
  questions still work:
  - **Better embeddings** — local Hugging Face `BAAI/bge-small-en-v1.5`
    instead of a weaker model, for stronger semantic matching.
  - **Hybrid retrieval** — keyword search (BM25) combined with vector
    search via LangChain's `EnsembleRetriever`, so both exact wording and
    paraphrasing work.
  - **Query rewriting** — `MultiQueryRetriever` asks the LLM to generate a
    few alternative phrasings of your question before searching.
  - **Spell correction** — a lightweight `pyspellchecker` pass cleans up
    obviously misspelled words before keyword search runs.
- **Fast, free LLM** — answers come from [Groq](https://console.groq.com/keys)'s
  free API running Llama 3.3 70B.
- **Remembers you across sessions** — MongoDB stores durable preferences
  ("I'm new to board games", "keep answers short") pulled automatically
  from your messages, plus a running history of past Q&A. Both get folded
  back into future prompts, so the app can stay personalized even after
  you close and reopen it. If MongoDB isn't reachable, this feature just
  turns itself off — the rest of the app keeps working.
- **Manage documents from the sidebar** — upload PDFs, build/update the
  vector database, delete a single document (and only its chunks), or wipe
  everything and start fresh.

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) — this is what
  actually answers your questions. No local model download needed.
- MongoDB, if you want the "remembers you" feature. Optional — the app
  runs fine without it, it just won't have long-term memory. Easiest way
  to get one running locally is `docker run -d -p 27017:27017 mongo`, or
  use a free [MongoDB Atlas](https://www.mongodb.com/atlas) cluster.
- No API key needed for embeddings — that model runs 100% locally via
  `sentence-transformers` (downloaded once, then cached).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Groq key (and MongoDB
settings, if you're using that):

```bash
cp .env.example .env
```

```
GROQ_API_KEY=gsk_...
MONGODB_URI=mongodb://localhost:27017   # optional, this is the default
MONGODB_DB=rag_chat_app                 # optional, this is the default
```

If you skip the `.env` file, you can also paste your Groq key straight
into the Streamlit sidebar each time.

## Option A: Streamlit UI (recommended)

```bash
streamlit run app.py
```

Then in the browser:

1. Confirm your Groq key is picked up (sidebar shows an error if it's
   missing).
2. Upload PDF(s), or just use the two sample PDFs already in `data/`
   (Monopoly and Ticket to Ride rulebooks).
3. Click **Build Database**.
4. Ask questions in the chat box — typos and paraphrased questions are
   fine. Each answer shows its source chunks under "Sources".
5. Use **Delete** next to a file to remove just that document, or
   **Delete All** to wipe the whole vector store and start over.
6. If MongoDB is connected, the sidebar also shows what the app has
   learned about you so far, with a **Forget Everything** button to clear
   it.

## Option B: Command line

```bash
# Build/update the vector database from PDFs in data/
python populate_database.py
python populate_database.py --reset   # wipe and rebuild from scratch

# Ask a question (uses GROQ_API_KEY from your environment/.env)
python query_data.py "How much money does a player start with in Monopoly?"
```

(The CLI path doesn't use the MongoDB memory feature — that's wired up in
the Streamlit app only.)

## Run tests

```bash
pytest test_rag.py
```

## Project structure

```
data/                       # Put your PDFs here
get_embedding_function.py   # Embedding model (local HuggingFace BGE by default)
llm_provider.py             # Sets up the Groq (free API) LLM
retriever.py                # Hybrid BM25 + vector retriever, query rewriting, spell-check
memory_store.py             # MongoDB-backed long-term memory (preferences + history)
populate_database.py        # CLI: load -> split -> embed -> store in Chroma
query_data.py                # CLI: retrieve -> prompt -> answer
app.py                        # Streamlit UI wrapping the same logic + memory
test_rag.py                   # Basic answer-quality tests
```

## Why this needed more than a basic vector search

A plain setup — one semantic vector search, five chunks, a small local
model — is very sensitive to exact wording:

| Problem | Fix |
|---|---|
| Weak embeddings | Switched to `BAAI/bge-small-en-v1.5` (HuggingFace, local) |
| Only semantic search | Added BM25 keyword search, merged via `EnsembleRetriever` |
| Typos / odd phrasing | `MultiQueryRetriever` rewrites the question a few ways using the LLM, plus a `pyspellchecker` pass |
| Weak LLM | Groq `llama-3.3-70b-versatile` (free API), strong at handling sloppy questions |
| Too few chunks | Raised `k` from 5 to 8 |
| No memory across sessions | Added MongoDB-backed preferences + interaction history, folded into every prompt |

A free Groq API key is required to run the LLM (embeddings still run
100% locally, no key needed for those). MongoDB is optional and only
powers the long-term memory feature in the Streamlit app.
