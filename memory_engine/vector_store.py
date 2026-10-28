"""
ChronoTrace / SynapseState Vector Memory Engine
Provides persistent vector storage and semantic similarity search over historical decisions
using ChromaDB with all-MiniLM-L6-v2 embeddings. Includes a resilient fallback store.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "chronotrace_decisions"


class VectorStoreManager:
    """
    Manages embedding generation, persistence, and semantic querying
    for cross-meeting decision evolution tracking.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = str(persist_directory or DEFAULT_CHROMA_DIR)
        os.makedirs(self.persist_directory, exist_ok=True)
        self._client = None
        self._collection = None
        self._fallback_store_file = Path(self.persist_directory) / "fallback_store.json"
        self._fallback_data: List[Dict[str, Any]] = self._load_fallback_data()

    def _init_chroma(self):
        """Initializes ChromaDB persistent client and collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.utils import embedding_functions

                self._client = chromadb.PersistentClient(path=self.persist_directory)
                try:
                    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                except Exception:
                    emb_fn = embedding_functions.DefaultEmbeddingFunction()

                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=emb_fn,
                    metadata={"description": "ChronoTrace Historical Decisions Semantic Index"}
                )
                print(f"[VectorStore] ChromaDB connected at: {self.persist_directory}")
            except Exception as e:
                print(f"[VectorStore] ChromaDB initialization note ({e}). Using embedded vector fallback.")
                self._collection = None

    def add_decision(
        self,
        decision_id: int,
        topic: str,
        decision_text: str,
        session_number: int,
        meeting_id: int,
        status: str = "Active",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stores a decision vector with rich metadata."""
        meta = {
            "decision_id": int(decision_id),
            "topic": topic,
            "session_number": int(session_number),
            "meeting_id": int(meeting_id),
            "status": status,
            **(metadata or {})
        }
        doc_text = f"Topic: {topic} | Decision: {decision_text}"

        self._init_chroma()
        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[f"dec_{decision_id}"],
                    documents=[doc_text],
                    metadatas=[meta]
                )
                return
            except Exception as e:
                print(f"[VectorStore] ChromaDB upsert warning: {e}. Writing to fallback store.")

        # Fallback in-memory / JSON store
        self._fallback_data = [item for item in self._fallback_data if item["decision_id"] != decision_id]
        self._fallback_data.append({
            "decision_id": decision_id,
            "topic": topic,
            "decision_text": decision_text,
            "document": doc_text,
            "session_number": session_number,
            "meeting_id": meeting_id,
            "status": status,
            "metadata": meta
        })
        self._save_fallback_data()

    def search_similar_decisions(
        self,
        query_text: str,
        n_results: int = 5,
        session_before: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top semantically similar historical decisions.
        Optionally filters for decisions made strictly before `session_before`.
        """
        self._init_chroma()
        if self._collection is not None:
            try:
                where_clause = None
                if session_before is not None:
                    where_clause = {"session_number": {"$lt": int(session_before)}}

                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where_clause
                )

                output = []
                if results and results.get("ids") and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        dist = results["distances"][0][i] if results.get("distances") else 0.5
                        sim = max(0.0, 1.0 - dist)
                        output.append({
                            "id": results["ids"][0][i],
                            "decision_id": results["metadatas"][0][i].get("decision_id"),
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity_score": round(sim, 3)
                        })
                return output
            except Exception as e:
                print(f"[VectorStore] ChromaDB query error: {e}. Using fallback similarity.")

        # Fallback Cosine / Token Similarity Search
        return self._search_fallback(query_text, n_results, session_before)

    def _search_fallback(
        self,
        query_text: str,
        n_results: int,
        session_before: Optional[int]
    ) -> List[Dict[str, Any]]:
        """TF-IDF / Lexical cosine similarity search over fallback records."""
        q_tokens = set(re.findall(r"\b\w+\b", query_text.lower()))
        scored = []

        for item in self._fallback_data:
            if session_before is not None and item.get("session_number", 0) >= session_before:
                continue

            doc_tokens = set(re.findall(r"\b\w+\b", item["document"].lower()))
            if not doc_tokens or not q_tokens:
                score = 0.0
            else:
                inter = len(q_tokens.intersection(doc_tokens))
                union = math.sqrt(len(q_tokens)) * math.sqrt(len(doc_tokens))
                score = round(inter / union if union > 0 else 0.0, 3)

            scored.append({
                "id": f"dec_{item['decision_id']}",
                "decision_id": item["decision_id"],
                "document": item["document"],
                "metadata": item.get("metadata", {}),
                "similarity_score": score
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:n_results]

    def _load_fallback_data(self) -> List[Dict[str, Any]]:
        if self._fallback_store_file.exists():
            try:
                with open(self._fallback_store_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_fallback_data(self) -> None:
        try:
            with open(self._fallback_store_file, "w", encoding="utf-8") as f:
                json.dump(self._fallback_data, f, indent=2)
        except Exception as e:
            print(f"[VectorStore] Error writing fallback store: {e}")

    def reset(self) -> None:
        """Clears all stored decision vectors."""
        self._init_chroma()
        if self._collection is not None and self._client is not None:
            try:
                self._client.delete_collection(COLLECTION_NAME)
                self._collection = None
            except Exception:
                pass
        self._fallback_data = []
        self._save_fallback_data()


if __name__ == "__main__":
    store = VectorStoreManager()
    store.add_decision(
        decision_id=1,
        topic="Primary Database Engine",
        decision_text="Adopt MongoDB Atlas as primary document store for rapid schema prototyping",
        session_number=101,
        meeting_id=101
    )
    store.add_decision(
        decision_id=2,
        topic="Authentication Protocol",
        decision_text="Use JWT bearer tokens with 15-minute expiry",
        session_number=101,
        meeting_id=101
    )

    query = "Switch from MongoDB to PostgreSQL for relational transactions"
    matches = store.search_similar_decisions(query, n_results=2)
    print(f"Query: '{query}'")
    print("Semantic Search Matches:")
    print(json.dumps(matches, indent=2))
