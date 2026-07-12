from __future__ import annotations

import argparse

from langchain_core.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function
from llm_provider import get_llm
from retriever import build_hybrid_retriever, correct_spelling

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context. If the context
does not contain the answer, say you don't know based on the provided
documents -- do not make up an answer.

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    response_text, sources = query_rag(args.query_text)
    print(f"Response: {response_text}\nSources: {sources}")


def query_rag(query_text: str):
    llm = get_llm()
    embedding_function = get_embedding_function()

    retriever = build_hybrid_retriever(CHROMA_PATH, embedding_function, llm)
    if retriever is None:
        return (
            "The database is empty. Run populate_database.py (or build it "
            "from the Streamlit sidebar) first.",
            [],
        )

    # Lightly spell-correct the query text to help the keyword (BM25) side of
    # the hybrid retriever; the LLM-based query rewriting handles the rest.
    cleaned_query = correct_spelling(query_text)

    results = retriever.invoke(cleaned_query)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response = llm.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("id", "unknown") for doc in results]
    return response_text, sources


if __name__ == "__main__":
    main()
