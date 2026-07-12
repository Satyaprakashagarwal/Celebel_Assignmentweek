"""
Long-term memory for the chat app, backed by MongoDB.

Unlike `st.session_state` (which is wiped whenever the Streamlit server
restarts or the browser session ends), this module persists to MongoDB, so
two things survive across sessions -- and across machines, if you point
MONGODB_URI at a shared/remote cluster instead of localhost:

1. **Preferences** -- short facts the user states about themselves or how
   they want to be helped (e.g. "I'm a beginner, keep explanations simple",
   "I'm comparing Monopoly and Ticket to Ride for game night"). These are
   pulled out of the user's messages automatically with a cheap LLM call.

2. **Interaction history** -- a running log of past question/answer pairs,
   independent of the current chat window.

Both are folded back into the RAG prompt on every new query (see
`get_memory_context`), so answers can stay personalized and context-aware
across restarts, not just within one chat session.

Connection: set MONGODB_URI in the environment / .env file; defaults to
mongodb://localhost:27017.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGODB_DB", "rag_chat_app")

MAX_INTERACTIONS = 200

_EXTRACTION_PROMPT = """You extract durable user preferences and facts from a
chat message, for a long-term memory system. Only extract things that would
still be true/useful in future, unrelated conversations -- e.g. stated
expertise level, goals, interests, tone/format preferences, or personal
context ("I'm new to board games", "I only care about the 2-player rules",
"keep answers short").

Do NOT extract: the question itself, one-off requests, or anything not
about the user's lasting preferences/context.

Message: "{message}"

Respond with ONLY a JSON array of short preference strings (max 2 items).
If there is nothing worth remembering, respond with exactly: []
"""

_client = None


def _get_db():
    """Lazily creates (and caches) the MongoDB connection. Returns None if
    MongoDB can't be reached, so the rest of the app can degrade gracefully
    instead of crashing the chat."""
    global _client
    if _client is None:
        from pymongo import MongoClient

        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    try:
        _client.admin.command("ping")
    except Exception:
        return None
    return _client[DB_NAME]


def _preferences_col():
    db = _get_db()
    return db["preferences"] if db is not None else None


def _interactions_col():
    db = _get_db()
    return db["interactions"] if db is not None else None


def load_memory() -> dict:
    """Returns {"preferences": [...], "interactions": [...]} read from
    MongoDB, in the same shape the rest of the app already expects."""
    prefs_col = _preferences_col()
    inter_col = _interactions_col()

    preferences, interactions = [], []
    try:
        if prefs_col is not None:
            preferences = [
                doc["text"] for doc in prefs_col.find({}, {"text": 1}).sort("_id", 1)
            ]
        if inter_col is not None:
            interactions = [
                {
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "timestamp": doc["timestamp"],
                }
                for doc in inter_col.find(
                    {}, {"question": 1, "answer": 1, "timestamp": 1}
                ).sort("_id", 1)
            ]
    except Exception:
        pass

    return {"preferences": preferences, "interactions": interactions}


def clear_memory():
    prefs_col = _preferences_col()
    inter_col = _interactions_col()
    try:
        if prefs_col is not None:
            prefs_col.delete_many({})
        if inter_col is not None:
            inter_col.delete_many({})
    except Exception:
        pass


def add_interaction(question: str, answer: str):
    inter_col = _interactions_col()
    if inter_col is None:
        return
    try:
        inter_col.insert_one(
            {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Keep the collection from growing unbounded -- only the most recent
        # MAX_INTERACTIONS turns are needed for recall purposes.
        count = inter_col.count_documents({})
        if count > MAX_INTERACTIONS:
            overflow = count - MAX_INTERACTIONS
            stale_ids = [
                doc["_id"]
                for doc in inter_col.find({}, {"_id": 1}).sort("_id", 1).limit(overflow)
            ]
            if stale_ids:
                inter_col.delete_many({"_id": {"$in": stale_ids}})
    except Exception:
        pass


def add_preference(text: str):
    prefs_col = _preferences_col()
    if prefs_col is None:
        return
    text = text.strip()
    if not text:
        return
    try:
        # Avoid storing exact duplicates.
        prefs_col.update_one({"text": text}, {"$setOnInsert": {"text": text}}, upsert=True)
    except Exception:
        pass


def extract_preferences(llm, user_message: str) -> list[str]:
    """Ask the LLM whether the message states any durable preference/fact.
    Fails safe: any error or unparsable response just means nothing new is
    remembered this turn (never blocks the actual answer)."""
    try:
        response = llm.invoke(_EXTRACTION_PROMPT.format(message=user_message))
        raw = response.content.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        items = json.loads(match.group(0))
        return [str(item).strip() for item in items if str(item).strip()]
    except Exception:
        return []


def get_memory_context(memory: dict, max_preferences: int = 10, max_interactions: int = 3) -> str:
    """Formats stored preferences + a few recent past interactions into a
    block that can be dropped into the prompt so the LLM can personalize its
    answer."""
    parts = []

    preferences = memory.get("preferences", [])[-max_preferences:]
    if preferences:
        pref_lines = "\n".join(f"- {p}" for p in preferences)
        parts.append(f"Known user preferences/context:\n{pref_lines}")

    interactions = memory.get("interactions", [])[-max_interactions:]
    if interactions:
        hist_lines = "\n".join(
            f'- Q: {i["question"]}\n  A: {i["answer"][:200]}' for i in interactions
        )
        parts.append(f"Relevant past conversation:\n{hist_lines}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def is_connected() -> bool:
    """Lets the UI show a clear warning if MongoDB isn't reachable, instead
    of silently losing memory."""
    return _get_db() is not None
