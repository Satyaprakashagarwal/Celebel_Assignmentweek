import os
import shutil
import gc
import time

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_embedding_function import get_embedding_function
from llm_provider import get_llm
from retriever import build_hybrid_retriever, correct_spelling
import memory_store

CHROMA_PATH = "chroma"
DATA_PATH = "data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context. If the context
does not contain the answer, say you don't know based on the provided
documents -- do not make up an answer.

{context}

---
{memory_context}
Answer the question based on the above context: {question}
"""

st.set_page_config(page_title="RAG PDF Chat", layout="wide")


# ----------------------------- Core RAG logic ----------------------------- #

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

    return len(new_chunks), len(existing_ids)


def delete_document(filename: str) -> int:
    """Delete one PDF from data/ and all of its chunks from the Chroma DB.

    Returns how many vector chunks were removed. Matches chunks by their
    stored `source` metadata (the file path used when the PDF was loaded),
    so only that file's vectors are removed -- everything else stays intact.
    """
    file_path = os.path.join(DATA_PATH, filename)
    removed_count = 0

    if os.path.exists(CHROMA_PATH):
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )
        existing = db.get(include=["metadatas"])
        ids_to_delete = [
            _id
            for _id, meta in zip(existing["ids"], existing["metadatas"])
            if meta.get("source") == file_path
        ]
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            removed_count = len(ids_to_delete)

    if os.path.exists(file_path):
        os.remove(file_path)

    return removed_count


def clear_database():
    """Delete the Chroma persistence folder.

    On Windows, chromadb keeps a cached SQLite connection open for each
    persist_directory for the lifetime of the process (Streamlit's server
    process stays alive across reruns), which makes shutil.rmtree fail with
    PermissionError/WinError32. Clearing chromadb's internal client cache
    first releases that lock; the retry loop below is a safety net for any
    straggling file handles (e.g. antivirus scanners briefly touching files).
    """
    if not os.path.exists(CHROMA_PATH):
        return

    try:
        import chromadb

        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    gc.collect()

    last_error = None
    for attempt in range(6):
        try:
            shutil.rmtree(CHROMA_PATH)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(0.5)
    raise last_error


@st.cache_resource(show_spinner=False)
def _cached_llm():
    return get_llm()


def query_rag(query_text: str):
    llm = _cached_llm()
    embedding_function = get_embedding_function()

    retriever = build_hybrid_retriever(CHROMA_PATH, embedding_function, llm)
    if retriever is None:
        return "The database is empty. Build it from the sidebar first.", []

    cleaned_query = correct_spelling(query_text)
    results = retriever.invoke(cleaned_query)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Long-term memory: fold in stored preferences + relevant past turns so
    # the answer can stay personalized/context-aware across sessions, not
    # just within the current chat window.
    memory = memory_store.load_memory()
    memory_block = memory_store.get_memory_context(memory)
    memory_context = f"\n{memory_block}\n" if memory_block else ""

    prompt = PROMPT_TEMPLATE.format(
        context=context_text, memory_context=memory_context, question=query_text
    )

    response = llm.invoke(prompt)
    sources = [doc.metadata.get("id", "unknown") for doc in results]

    # Update long-term memory: log this turn, and pull out any durable
    # preference/fact the user stated so future sessions can recall it too.
    memory_store.add_interaction(query_text, response.content)
    for pref in memory_store.extract_preferences(llm, query_text):
        memory_store.add_preference(pref)

    return response.content, sources


# --------------------------------- Sidebar --------------------------------- #

with st.sidebar:
    st.header("Model")
    st.caption("Powered by Groq (llama-3.3-70b - fast and free tier).")
    _groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
    if not _groq_key_set:
        st.error("GROQ_API_KEY is not set. Add it to your .env file.")

    st.divider()
    st.header("Your Documents")
    st.caption("Upload PDF files, then click Build Database to make them searchable.")

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        os.makedirs(DATA_PATH, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(DATA_PATH, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s).")

    existing_pdfs = (
        [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
        if os.path.isdir(DATA_PATH)
        else []
    )
    st.caption("Files currently uploaded:")
    if existing_pdfs:
        for pdf_name in existing_pdfs:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.write(pdf_name)
            with col_del:
                if st.button(
                    "Delete",
                    key=f"del_{pdf_name}",
                    help=f"Remove '{pdf_name}' and its saved data",
                    use_container_width=True,
                ):
                    removed = delete_document(pdf_name)
                    st.success(
                        f"Deleted '{pdf_name}' and {removed} related piece(s) "
                        "of saved data."
                    )
                    st.rerun()
    else:
        st.write("No files uploaded yet.")

    col1, col2 = st.columns(2)
    with col1:
        build_clicked = st.button("Build Database", use_container_width=True)
    with col2:
        reset_clicked = st.button("Delete All", use_container_width=True)

    if reset_clicked:
        clear_database()
        st.success("All saved data has been deleted.")

    if build_clicked:
        if not existing_pdfs:
            st.warning("No PDFs found. Upload some files first.")
        else:
            with st.spinner("Reading and saving your documents..."):
                docs = load_documents()
                chunks = split_documents(docs)
                added, existing = add_to_chroma(chunks)
            st.success(f"Done. Added {added} new piece(s). ({existing} already saved)")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Saved Preferences")
    st.caption(
        "This app remembers things you tell it, and your past questions, "
        "so it can give better answers over time -- even after you close "
        "and reopen the app."
    )
    if not memory_store.is_connected():
        st.error(
            f"Cannot connect to the database at `{memory_store.MONGODB_URI}`. "
            "This feature is turned off until it's reachable."
        )
    else:
        _memory = memory_store.load_memory()
        _prefs = _memory.get("preferences", [])
        _hist = _memory.get("interactions", [])
        if _prefs:
            st.caption(f"What it remembers about you ({len(_prefs)}):")
            for p in _prefs:
                st.write(f"- {p}")
        else:
            st.caption("Nothing saved yet.")
        st.caption(f"{len(_hist)} past question(s) saved.")
        if st.button("Forget Everything", use_container_width=True):
            memory_store.clear_memory()
            st.success("All saved preferences have been cleared.")
            st.rerun()


# --------------------------------- Main chat -------------------------------- #

st.title("Chat with your PDFs")
st.caption(
    "Ask a question about your uploaded documents. The app understands "
    "typos and differently worded questions, and only answers using what's "
    "in your files."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

query_text = st.chat_input("Ask a question about your documents...")

if query_text:
    st.session_state.messages.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    if not _groq_key_set:
        answer = "This app isn't set up yet. GROQ_API_KEY is missing from .env."
        sources = []
    elif not os.path.exists(CHROMA_PATH):
        answer = "No database found yet. Build the database from the sidebar first."
        sources = []
    else:
        with st.spinner("Thinking..."):
            try:
                answer, sources = query_rag(query_text)
            except Exception as e:
                answer = f"Something went wrong while answering: {e}"
                sources = []

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
