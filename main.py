"""
ChronoTrace / SynapseState Master Orchestration Engine
Integrates:
  1. Audio Ingestion & Pyannote Diarization (Member 1)
  2. LLM Extraction & DeBERTa NLI Claim Verification (Member 2)
  3. ChromaDB Vector Memory & Decision State Graph (Member 3)
  4. Streamlit Multi-Meeting Dashboard UI (Member 4)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Safe encoding handling on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup module path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database.db_connection import init_db
from database.seed_data import seed_database
from database.queries import print_analytical_queries_demo
from audio_pipeline.whisper_diarize import AudioDiarizationPipeline
from nlp_engine.extractors import MeetingExtractor
from nlp_engine.verifier import NLIGroundingVerifier
from memory_engine.state_graph import StateGraphEngine


def run_full_pipeline(audio_file: str = "", session_number: int = 104, title: str = "Sprint 26 Architecture Review") -> None:
    """
    Executes the entire ChronoTrace pipeline sequentially:
    Audio -> Diarization -> Extraction -> Grounding -> Memory -> SQL
    """
    print("\n" + "=" * 80)
    print("CHRONOTRACE MASTER END-TO-END ORCHESTRATION PIPELINE")
    print("=" * 80)

    # Step 1: Audio Processing & Multi-Speaker Diarization
    print(f"\n[Step 1/5] Ingesting speech and diarizing speaker turns (Audio: '{audio_file or 'Reference Audio'}')...")
    audio_pipe = AudioDiarizationPipeline()
    if audio_file and Path(audio_file).exists():
        turns = audio_pipe.process_audio(audio_file)
    else:
        print("  -> Using ground-truth diarized turns from audio_pipeline/sample_output.json")
        turns = audio_pipe.get_reference_sample()
    print(f"  [OK] Diarized {len(turns)} multi-speaker turns with sub-second alignment timestamps.")

    # Step 2: NLP Extraction into Strict Pydantic v2 Schemas
    print("\n[Step 2/5] Extracting decisions, action items, and conversation debt triggers...")
    extractor = MeetingExtractor()
    summary = extractor.extract_meeting(
        turns=turns,
        session_number=session_number,
        meeting_title=title
    )
    print(f"  [OK] Extracted {len(summary.decisions)} decisions, {len(summary.action_items)} action items, {len(summary.open_questions)} open questions.")

    # Step 3: NLI Grounding & Claim Verification Rate (CVR)
    print("\n[Step 3/5] Computing Claim Verification Rate (CVR) with DeBERTa NLI verifier...")
    verifier = NLIGroundingVerifier(threshold=0.60)
    for dec in summary.decisions:
        verifier.verify_decision(dec)
        print(f"  -> Decision: '{dec.decision_text[:50]}...' | CVR: {dec.cvr_confidence_score * 100:.1f}%")
    print("  [OK] All decisions scored and grounded against verbatim transcript spans.")

    # Step 4: Long-Term Vector Memory & State Transition Engine
    print("\n[Step 4/5] Checking ChromaDB vector memory for decision evolution & superseding...")
    state_engine = StateGraphEngine()
    result = state_engine.ingest_summary(summary)

    if result.get("state_transitions"):
        print(f"  [*] STATE TRANSITIONS DETECTED: {len(result['state_transitions'])} decision(s) superseded!")
        for trans in result["state_transitions"]:
            print(f"     * Decision #{trans['original_decision_id']} superseded by Decision #{trans['revising_decision_id']} ({trans['session_delta']})")
    else:
        print("  [OK] Decisions ingested. No prior contradictions or superseded consensus detected.")

    # Step 5: Compound Conversation Debt
    print("\n[Step 5/5] Recalculating Conversation Debt scores across historical sessions...")
    debt_items = result.get("conversation_debt_audit", [])
    print(f"  [OK] Evaluated {len(debt_items)} carried-over questions:")
    for q in debt_items[:3]:
        print(f"     * Question #{q['question_id']}: Carried {q['carried_over_count']} sessions -> Compounded Debt: {q['compounded_debt_score']}")

    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY! Data committed to SQLite and ChromaDB.")
    print("=" * 80)


def launch_dashboard() -> None:
    """Launches the Streamlit web application."""
    app_path = BASE_DIR / "dashboard" / "app.py"
    print(f"\n[Dashboard] Launching Streamlit web application: {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)])


def main():
    parser = argparse.ArgumentParser(description="ChronoTrace / SynapseState Master Orchestration CLI")
    parser.add_argument("--seed", action="store_true", help="Seed database with 3-meeting historical scenario")
    parser.add_argument("--process-sample", action="store_true", help="Run end-to-end pipeline with sample transcript")
    parser.add_argument("--audio", type=str, default="", help="Path to input audio file for end-to-end processing")
    parser.add_argument("--session", type=int, default=104, help="Session number for newly ingested meeting")
    parser.add_argument("--title", type=str, default="Sprint 26 Architecture Review", help="Meeting title")
    parser.add_argument("--run-queries", action="store_true", help="Execute and print the 5 DBMS analytical queries")
    parser.add_argument("--run-dashboard", action="store_true", help="Launch Streamlit web dashboard")

    args = parser.parse_args()

    # If no flags passed, display overview and run queries
    if not (args.seed or args.process_sample or args.audio or args.run_queries or args.run_dashboard):
        banner = r"""
   ____ _                         _____                    
  / ___| |__  _ __ ___  _ __   ___|_   _| __ __ _  ___ ___ 
 | |   | '_ \| '__/ _ \| '_ \ / _ \ | || '__/ _` |/ __/ _ \
 | |___| | | | | | (_) | | | | (_) || || | | (_| | (_|  __/
  \____|_| |_|_|  \___/|_| |_|\___/ |_||_|  \__,_|\___\___|
        SynapseState Multi-Meeting Intelligence Engine
        """
        print(banner)
        seed_database(reset=False)
        print_analytical_queries_demo()
        print("\nCLI Commands:")
        print("  python main.py --process-sample   # Run end-to-end pipeline")
        print("  python main.py --run-queries      # Execute 5 analytical DBMS queries")
        print("  python main.py --run-dashboard    # Launch Streamlit dashboard UI")
        print("  python main.py --seed             # Re-seed test database")
        return

    if args.seed:
        seed_database(reset=True)

    if args.process_sample or args.audio:
        run_full_pipeline(audio_file=args.audio, session_number=args.session, title=args.title)

    if args.run_queries:
        print_analytical_queries_demo()

    if args.run_dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
