"""
Builds a "smarter" retriever that is much more forgiving of typos, rephrased
questions, and questions that don't use the exact wording in the PDF.

It combines three techniques:

1. Hybrid search (EnsembleRetriever): a keyword-based BM25Retriever (great at
   exact/near-exact word matches) is combined with the semantic Chroma vector
   retriever (great at paraphrases/synonyms), so either style of match works.

2. Multi-query rewriting (MultiQueryRetriever): the LLM rewrites the user's
   question into a few alternative phrasings before retrieving, which fixes
   most typos and awkward phrasing automatically (a good LLM silently
   "understands" a misspelled question and rewrites it correctly).

3. Light spell correction: an extra, cheap pass with `pyspellchecker` that
   fixes obviously-misspelled individual words before they hit BM25 (which,
   unlike the vector search, has no fuzzy/semantic understanding at all).
"""

from spellchecker import SpellChecker

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# EnsembleRetriever / MultiQueryRetriever live in `langchain_classic` on
# LangChain 1.x (they were moved out of the core `langchain` package during
# the 1.0 restructuring). Falls back to the pre-1.0 location automatically.
try:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
except ImportError:  # pre-1.0 LangChain
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.multi_query import MultiQueryRetriever

_spell = SpellChecker()


def correct_spelling(text: str) -> str:
    """Fix individual misspelled words, leaving correctly-spelled and unknown
    (e.g. proper noun / technical) words untouched."""
    corrected_words = []
    for word in text.split():
        stripped = word.strip(".,?!:;\"'")
        if not stripped.isalpha() or len(stripped) < 4:
            corrected_words.append(word)
            continue
        if stripped.lower() in _spell:
            corrected_words.append(word)
            continue
        suggestion = _spell.correction(stripped.lower())
        if suggestion and suggestion != stripped.lower():
            corrected_words.append(word.replace(stripped, suggestion))
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


def build_hybrid_retriever(chroma_path: str, embedding_function, llm, k: int = 8):
    """Returns a retriever that hits the vector store (semantic) and a BM25
    index built from the same data (keyword), merges results, then wraps the
    whole thing in query-rewriting powered by `llm`."""
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Pull everything out of Chroma to build a matching BM25 keyword index.
    raw = db.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]

    if not all_docs:
        return None

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    vector_retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 4}
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    # Wrap with LLM-powered query rewriting for typo/phrasing tolerance.
    # include_original=True keeps searching with the exact original wording
    # too, so a perfectly-typed query never regresses versus the old behavior.
    smart_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever, llm=llm, include_original=True
    )

    return smart_retriever
