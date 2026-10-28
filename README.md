# ChronoTrace / SynapseState
### Multi-Meeting Intelligence & Conversation Debt Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2.5%2B-E92063.svg)](https://pydantic.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://trychroma.com)
[![SQLite](https://img.shields.io/badge/SQLite-ACID%20DBMS-003B57.svg)](https://sqlite.org)

---

## 📌 Project Architecture

ChronoTrace connects disjoint meeting streams into an evolving semantic graph. It identifies when past consensus is superseded, tracks cognitive conversation debt from unresolved questions, and grounds every extracted decision in verbatim speech timestamps with zero-shot DeBERTa NLI Claim Verification Rate (CVR).

```
ChronoTrace/
│
├── database/               # Relational DBMS Module (SQLite / PostgreSQL)
│   ├── schema.sql          # Full DDL with Foreign Keys, Check Constraints & B-Tree Indexes
│   ├── db_connection.py    # Connection manager with WAL mode, transactions & Row dicts
│   ├── queries.py          # 5 High-Impact Analytical SQL queries
│   └── seed_data.py        # 3-Meeting historical engineering sync dataset
│
├── audio_pipeline/          # Speech Processing & Diarization (Member 1)
│   ├── whisper_diarize.py  # Whisper STT + Pyannote 3.1 multi-speaker diarization
│   └── sample_output.json  # Output transcript format with speaker IDs & timestamps
│
├── nlp_engine/              # LLM Extraction & Entailment Grounding (Member 2)
│   ├── schemas.py          # Strict Pydantic v2 schemas for JSON outputs
│   ├── extractors.py       # LLM API prompts for decision & action extraction
│   └── verifier.py         # DeBERTa-v3 NLI Claim Verification Rate (CVR) engine
│
├── memory_engine/           # Vector Memory & State Transition Engine (Member 3)
│   ├── vector_store.py     # ChromaDB embedding store & semantic search
│   └── state_graph.py      # Decision evolution, contradiction detection & debt score
│
├── dashboard/               # Frontend & Visualization Interface (Member 4)
│   ├── app.py              # Main Streamlit web application dashboard
│   └── components/          # Decision timeline, Debt tracker, Audit popups
│       ├── timeline.py     # Decision evolution timeline component
│       ├── debt_tracker.py # Conversation debt and friction tracker
│       └── audit_popup.py  # 'Why?' forensic audit modal inspector
│
├── data/                    # Storage directory for chronotrace.db and ChromaDB
├── deploy_guide.md          # Step-by-step Streamlit Cloud deployment instructions
├── requirements.txt         # All global Python dependencies
└── main.py                  # Master orchestration execution file
```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Virtual Environment
```bash
cd ChronoTrace
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize and Seed the Database
```bash
python main.py --seed
```

### 3. Run the 5 Analytical DBMS Queries (CLI)
```bash
python main.py --run-queries
```

### 4. Run the Full End-to-End Master Pipeline
```bash
python main.py --process-sample
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Analytical DBMS Queries Specification

ChronoTrace implements 5 production-grade SQL queries directly fulfilling the DBMS mini-project requirements:

1. **Query 1 (Decision Evolution Lineage):** Self-joins historical decision rows to trace how choices evolve (e.g., Session 101 MongoDB superseded by Session 102 PostgreSQL).
2. **Query 2 (Conversation Debt Severity):** Evaluates open questions carried over 2+ sessions, calculating dynamic severity indices and risk tiers (`CRITICAL`, `HIGH`, `MODERATE`).
3. **Query 3 (Grounding & Ownership Risk Audit):** Flags action items with low CVR entailment (<0.60) or missing owners (`UNASSIGNED`).
4. **Query 4 (Member Task Accountability):** Aggregates explicit vs tentative commitments, completed tasks, and stalled velocity per engineer.
5. **Query 5 (Forensic 'Why?' Audit Trail):** Reconstructs complete evidence lineage, audio timestamps, CVR confidence, and transcript excerpts for modal inspection.
