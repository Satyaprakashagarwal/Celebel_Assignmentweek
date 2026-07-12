# RAG Tutorial (2026 refresh, Streamlit UI, hybrid smart retrieval)

A local Retrieval-Augmented Generation app: index your PDFs into a Chroma
vector database, then ask questions and get answers grounded only in those
documents.

This version adds a "smarter brain" on top of the original tutorial logic:

- **Better embeddings** — local Hugging Face `BAAI/bge-small-en-v1.5`
  instead of `nomic-embed-text`, for stronger semantic matching.
- **Hybrid retrieval** — combines keyword search (BM25, great for exact
  words) with semantic vector search (great for paraphrases/synonyms) via
  LangChain's `EnsembleRetriever`.
- **Query rewriting** — `MultiQueryRetriever` uses the LLM to generate a
  few alternative phrasings of your question before searching, which fixes
  most typos and awkward wording automatically.
- **Spell correction** — a lightweight pass with `pyspellchecker` cleans up
  obviously misspelled words before keyword search.
- **A smart, free LLM** — [Groq](https://console.groq.com/keys)'s free API
  serving Llama 3.3 70B, which is great at understanding sloppy/paraphrased
  questions.

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (no local model
  download needed, very fast).
- No API key needed for embeddings — the default embedding model runs
  locally via `sentence-transformers` (downloaded once, then cached).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and paste in your Groq key so you
don't have to type it into the sidebar every time:

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

## Option A: Streamlit UI (recommended)

```bash
streamlit run app.py
```

In the browser:
1. In the sidebar, paste your free Groq API key.
2. Upload PDF(s) (or keep the two sample PDFs already in `data/`).
3. Click **Build / Update DB**.
4. Ask questions in the chat box — including paraphrased or slightly
   misspelled ones. Answers show their source chunks under "Sources".
5. Use **Reset DB** to wipe the vector store and start over.

## Option B: Command line

```bash
# Build/update the vector database from PDFs in data/
python populate_database.py
python populate_database.py --reset   # wipe and rebuild from scratch

# Ask a question (uses GROQ_API_KEY from your environment/.env)
python query_data.py "How much money does a player start with in Monopoly?"
```

## Run tests

```bash
pytest test_rag.py
```

## Project structure

```
data/                    # Put your PDFs here
get_embedding_function.py# Embedding model (local HuggingFace BGE by default)
llm_provider.py          # Sets up the Groq (free API) LLM
retriever.py             # Hybrid BM25 + vector retriever, query rewriting, spell-check
populate_database.py     # CLI: load -> split -> embed -> store in Chroma
query_data.py            # CLI: retrieve -> prompt -> answer
app.py                   # Streamlit UI wrapping the same logic
test_rag.py              # Basic answer-quality tests
```

## Why answers were too literal before, and what changed

The original setup only did a single semantic vector search (5 chunks) with
a weak local embedding model, then handed the result straight to a small 7B
model. That combination is very sensitive to exact wording:

- A typo shifts the embedding just enough to miss the right chunk.
- A paraphrased question ("what languages does he know" vs "technical
  skills") may not land close enough in vector space to the original PDF
  wording.
- A small local model, even with the right context, is weaker at reading
  between the lines than larger models.

This version addresses each layer:

| Problem | Fix |
|---|---|
| Weak embeddings | Switched to `BAAI/bge-small-en-v1.5` (HuggingFace, local) |
| Only semantic search | Added BM25 keyword search, merged via `EnsembleRetriever` |
| Typos / odd phrasing | `MultiQueryRetriever` rewrites the question a few ways using the LLM, plus a `pyspellchecker` pass |
| Weak LLM | Groq `llama-3.3-70b-versatile` (free API), strong at handling sloppy questions |
| Too few chunks | Raised `k` from 5 to 8 and switched to MMR search for more diverse context |

A free Groq API key is required to run the LLM (embeddings still run
100% locally, no key needed for those).
