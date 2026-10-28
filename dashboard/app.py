"""
ChronoTrace / SynapseState Main Streamlit Web Application Dashboard
Enterprise-grade multi-meeting intelligence platform featuring live microphone recording,
real-time speech recognition (Google Web Speech / Gemini), decision evolution timelines,
conversation debt metrics, forensic 'Why?' audits, and an interactive DBMS playground.
"""

import json
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Setup path resolution for local modules
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database.db_connection import execute_query, init_db
from database.seed_data import seed_database
from database.queries import (
    get_superseded_decisions_lineage,
    get_conversation_debt_report,
    get_low_confidence_or_unassigned_actions,
    get_member_accountability_summary,
    get_decision_audit_trail,
    get_all_meetings,
    get_all_decisions_detailed
)
from dashboard.components.timeline import render_decision_timeline
from dashboard.components.debt_tracker import render_debt_tracker
from dashboard.components.audit_popup import render_audit_view
import importlib
import audio_pipeline.whisper_diarize
importlib.reload(audio_pipeline.whisper_diarize)
from audio_pipeline.whisper_diarize import AudioDiarizationPipeline
from nlp_engine.extractors import MeetingExtractor
from nlp_engine.verifier import NLIGroundingVerifier
from memory_engine.state_graph import StateGraphEngine

# Streamlit Page Config
st.set_page_config(
    page_title="ChronoTrace | Multi-Meeting Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    h1, h2, h3, h4 { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stMetric {
        background-color: #111827;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    div[data-testid="stSidebarNav"] { display: none; }
    .recording-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .step-pill {
        display: inline-block;
        background: #38bdf822;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    .transcript-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        padding: 14px;
        border-radius: 6px;
        color: #f1f5f9;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Ensures database is initialized and manages navigation state."""
    if "db_initialized" not in st.session_state:
        try:
            meetings = get_all_meetings()
            if not meetings:
                seed_database(reset=False)
            st.session_state["db_initialized"] = True
        except Exception:
            seed_database(reset=True)
            st.session_state["db_initialized"] = True

    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "🎙️ Record & Ingest Meeting"


init_session_state()

# Sidebar Navigation & Management
st.sidebar.markdown(
    """
    <div style="padding: 10px 0;">
        <h2 style="color: #38bdf8; margin: 0; font-size: 1.6rem;">⚡ ChronoTrace</h2>
        <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 500;">SynapseState Multi-Meeting Engine</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
nav_options = [
    "🎙️ Record & Ingest Meeting",
    "📊 Executive Overview",
    "⏳ Decision Timeline",
    "⚠️ Conversation Debt Tracker",
    "🔍 'Why?' Forensic Audit",
    "👥 Team Accountability Matrix",
    "🛠️ DBMS SQL Playground"
]

current_index = nav_options.index(st.session_state["current_view"]) if st.session_state["current_view"] in nav_options else 0
selected_view = st.sidebar.radio("NAVIGATION", options=nav_options, index=current_index)
st.session_state["current_view"] = selected_view

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Optional API Settings")
with st.sidebar.expander("Cloud LLM Keys (Optional)", expanded=False):
    ui_gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    ui_openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if ui_gemini_key:
        os.environ["GEMINI_API_KEY"] = ui_gemini_key
    if ui_openai_key:
        os.environ["OPENAI_API_KEY"] = ui_openai_key
    st.caption("Free Google SpeechRecognition + Local NLP are enabled by default with zero keys required.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Quick Actions")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

if st.sidebar.button("🌱 Re-Seed Sample Database", use_container_width=True):
    with st.spinner("Resetting to 3-session historical architecture baseline..."):
        seed_database(reset=True)
    st.sidebar.success("Database re-seeded successfully!")
    st.session_state["latest_ingestion_result"] = None
    st.rerun()

st.sidebar.markdown(
    """
    <div style="margin-top: 40px; font-size: 0.75rem; color: #64748b;">
        ChronoTrace v1.0.0 Enterprise Edition<br/>
        Multi-Meeting Intelligence & Conversation Debt
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# VIEW 1: RECORD & INGEST MEETING (Live Audio Ingestion Studio)
# ==============================================================================
if st.session_state["current_view"] == "🎙️ Record & Ingest Meeting":
    st.title("🎙️ Live Meeting Recording & Intelligence Studio")
    st.markdown(
        "Record through your microphone or upload any meeting audio. ChronoTrace **transcribes your actual speech**, "
        "extracts your consensus decisions and action items, checks for superseded choices against historical meetings, "
        "and updates conversation debt in real time."
    )

    # Calculate default session number
    max_sess_row = execute_query("SELECT COALESCE(MAX(session_number), 100) AS max_sess FROM Meetings;")
    next_session_default = (max_sess_row[0]["max_sess"] if max_sess_row else 100) + 1

    # Configuration Banner
    with st.container():
        st.markdown('<div class="recording-card">', unsafe_allow_html=True)
        st.markdown('<span class="step-pill">STEP 1: MEETING DETAILS & AUDIO CAPTURE</span>', unsafe_allow_html=True)

        col_cfg1, col_cfg2 = st.columns([3, 1])
        with col_cfg1:
            meeting_title = st.text_input(
                "Meeting Session Title",
                value="Sprint 26 Architecture Sync: Caching & Partitioning Strategy"
            )
        with col_cfg2:
            session_num = st.number_input(
                "Session Number",
                min_value=101,
                max_value=999,
                value=int(next_session_default)
            )

        # Audio Input Mode Selection
        input_method = st.radio(
            "Select Audio Capture Mode:",
            options=[
                "🎙️ Live Microphone Recording (In-Browser)",
                "📁 Upload Audio File (.wav, .mp3, .m4a)",
                "✏️ Direct Transcript Input (Text Mode)",
                "⚡ Architectural Simulation Scenario"
            ],
            horizontal=True
        )

        audio_bytes = None
        user_spoken_turns = None
        user_spoken_text = ""

        if input_method == "🎙️ Live Microphone Recording (In-Browser)":
            st.info("💡 **Speak into your microphone:** State your team's decisions (e.g. *'We decided to switch from Redis to Memcached for session storage'*), action tasks (*'Bob will write the adapter by Friday'*), or questions (*'What is our p99 latency SLA?'*). Click **Stop** when finished.")
            recorded_audio = st.audio_input("Record Meeting Audio:")
            if recorded_audio is not None:
                audio_bytes = recorded_audio.read()
                st.audio(audio_bytes, format="audio/wav")

        elif input_method == "📁 Upload Audio File (.wav, .mp3, .m4a)":
            uploaded_file = st.file_uploader("Upload meeting audio recording:", type=["wav", "mp3", "m4a", "ogg"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                st.audio(audio_bytes)

        elif input_method == "✏️ Direct Transcript Input (Text Mode)":
            st.caption("Type or paste what was spoken in your meeting:")
            direct_text = st.text_area(
                "Spoken Meeting Discussion:",
                value="Alice Chen: We officially decide to switch from Redis to Memcached for session token storage.\nBob Martinez: I will write the Memcached connection adapter by next Wednesday.\nCharlie Patel: What is our p99 latency SLA for cache misses?",
                height=120
            )
            if direct_text:
                # Convert into turns
                lines = [l.strip() for l in direct_text.split("\n") if l.strip()]
                user_spoken_turns = []
                cur_t = 0.0
                for line in lines:
                    if ":" in line:
                        spk, txt = line.split(":", 1)
                        spk = spk.strip()
                        txt = txt.strip()
                    else:
                        spk = "Speaker 1"
                        txt = line
                    dur = max(3.0, len(txt.split()) * 0.45)
                    user_spoken_turns.append({
                        "speaker": "SPEAKER_00",
                        "speaker_name": spk,
                        "start": round(cur_t, 1),
                        "end": round(cur_t + dur, 1),
                        "text": txt
                    })
                    cur_t += dur
                user_spoken_text = direct_text

        elif input_method == "⚡ Architectural Simulation Scenario":
            st.caption("Select a pre-configured architecture scenario:")
            scenario = st.selectbox(
                "Choose Scenario:",
                options=[
                    "Scenario A: Switch from Redis to Memcached (Supersedes Session 101 Auth Decision #2)",
                    "Scenario B: Resolve Disaster Recovery SLA (Resolves Session 101 Question #2, lowering debt)",
                    "Scenario C: Standard Feature Sync (Kafka Partitioning & TypeScript SDK)"
                ]
            )
            if "Scenario A" in scenario:
                user_spoken_turns = [
                    {"speaker": "SPEAKER_00", "speaker_name": "Alice Chen", "start": 0.0, "end": 7.5, "text": "Welcome to Session 104. We need to revisit our caching layer from Session 101."},
                    {"speaker": "SPEAKER_01", "speaker_name": "Bob Martinez", "start": 7.8, "end": 19.4, "text": "Redis cluster memory overhead is too high for simple key-value session tokens. I propose we switch from Redis to Memcached for lightweight session storage."},
                    {"speaker": "SPEAKER_00", "speaker_name": "Alice Chen", "start": 19.8, "end": 28.5, "text": "Agreed. We officially pivot from Redis to Memcached for session token storage."},
                    {"speaker": "SPEAKER_01", "speaker_name": "Bob Martinez", "start": 29.0, "end": 38.0, "text": "I will write the Memcached adapter and connection pooling by next Wednesday."},
                    {"speaker": "SPEAKER_02", "speaker_name": "Charlie Patel", "start": 38.5, "end": 49.0, "text": "What is our p99 latency SLA for cache misses before falling back to PostgreSQL?"}
                ]
            elif "Scenario B" in scenario:
                user_spoken_turns = [
                    {"speaker": "SPEAKER_00", "speaker_name": "Alice Chen", "start": 0.0, "end": 8.0, "text": "We are finally addressing Charlie's carried-over question on disaster recovery."},
                    {"speaker": "SPEAKER_02", "speaker_name": "Charlie Patel", "start": 8.5, "end": 22.0, "text": "Based on our compliance audit, we officially establish our disaster recovery RTO as 15 minutes and RPO as 1 minute across AWS regions."},
                    {"speaker": "SPEAKER_00", "speaker_name": "Alice Chen", "start": 22.5, "end": 31.0, "text": "Approved. The disaster recovery RTO and RPO question from Session 101 is now officially resolved."},
                    {"speaker": "SPEAKER_02", "speaker_name": "Charlie Patel", "start": 31.5, "end": 42.0, "text": "I will configure the cross-region backup replication automation in Terraform."}
                ]
            else:
                user_spoken_turns = AudioDiarizationPipeline.get_reference_sample()
            user_spoken_text = " ".join(t["text"] for t in user_spoken_turns)

        # Transcribe audio if real audio was supplied
        if audio_bytes and not user_spoken_turns:
            with st.spinner("🎙️ Transcribing your speech with Google Speech Recognition..."):
                pipe = AudioDiarizationPipeline()
                raw_result = pipe.process_audio(audio_bytes)
                if isinstance(raw_result, tuple) and len(raw_result) == 2:
                    user_spoken_turns = raw_result[0]
                    user_spoken_text = raw_result[1]
                elif isinstance(raw_result, list):
                    user_spoken_turns = raw_result
                    user_spoken_text = " ".join(t.get("text", "") for t in user_spoken_turns if isinstance(t, dict))
                else:
                    user_spoken_turns = []
                    user_spoken_text = ""

            if user_spoken_turns:
                st.success(f"✓ Transcribed {len(user_spoken_turns)} spoken sentence(s) from your audio!")
                with st.expander("📝 Review / Edit Transcribed Speech from Your Recording", expanded=True):
                    edited_transcript = st.text_area(
                        "Your Spoken Words (feel free to edit or add speaker tags if desired):",
                        value=user_spoken_text,
                        height=100
                    )
                    # Re-segment if user edited the text
                    if edited_transcript != user_spoken_text:
                        import re
                        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", edited_transcript) if s.strip()]
                        user_spoken_turns = [
                            {"speaker": f"SPEAKER_{i%3:02d}", "speaker_name": f"Speaker {(i%3)+1}", "start": float(i*4), "end": float((i+1)*4), "text": s}
                            for i, s in enumerate(sents or [edited_transcript])
                        ]
                        user_spoken_text = edited_transcript
            else:
                st.warning("⚠️ Could not detect clear spoken words in this audio. Please speak clearly into your microphone or enter what was said in 'Direct Transcript Input'.")

        st.markdown('</div>', unsafe_allow_html=True)

    # STEP 2: PIPELINE EXECUTION TRIGGER
    st.markdown('<div class="recording-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">STEP 2: RUN CHRONOTRACE AI PIPELINE ON YOUR AUDIO</span>', unsafe_allow_html=True)

    can_process = bool(user_spoken_turns and len(user_spoken_turns) > 0)

    if not can_process:
        st.info("👉 Speak into the microphone above or provide audio to begin analysis.")

    process_btn = st.button(
        "🚀 Process Recording & Extract Intelligence",
        type="primary",
        disabled=not can_process,
        use_container_width=True
    )

    if process_btn and user_spoken_turns:
        with st.status("Executing Multi-Meeting Intelligence Pipeline on YOUR Speech...", expanded=True) as status_box:
            # 1. Audio Processing & Diarization
            st.write(f"🎙️ **[1/5] Speech Turns Indexed**: {len(user_spoken_turns)} turns analyzed.")

            # 2. NLP Extraction into Pydantic v2
            st.write("🧠 **[2/5] Extracting Decisions, Action Items, and Open Questions from YOUR Audio**...")
            extractor = MeetingExtractor()
            summary = extractor.extract_meeting(
                turns=user_spoken_turns,
                session_number=int(session_num),
                meeting_title=meeting_title
            )
            st.write(f"✓ Extracted {len(summary.decisions)} decisions, {len(summary.action_items)} action items, {len(summary.open_questions)} open questions.")

            # 3. DeBERTa NLI Claim Verification (CVR)
            st.write("🎯 **[3/5] Calculating Claim Verification Rate (CVR) against Verbatim Speech Spans**...")
            verifier = NLIGroundingVerifier(threshold=0.60)
            for dec in summary.decisions:
                verifier.verify_decision(dec)
            st.write(f"✓ Grounded all extracted decisions against verbatim speech timestamps.")

            # 4. Long-Term Vector Memory & State Graph
            st.write("🔄 **[4/5] Cross-Meeting Semantic Search (ChromaDB) & Superseded Detection**...")
            state_engine = StateGraphEngine()
            result = state_engine.ingest_summary(summary)

            # 5. Conversation Debt Compounding
            st.write("⚠️ **[5/5] Compounding Conversation Debt Across Historical Sessions**...")
            st.write(f"✓ Saved to SQLite database.")

            status_box.update(label="🎉 YOUR Audio Successfully Processed and Grounded!", state="complete", expanded=False)

        st.session_state["latest_ingestion_result"] = {
            "summary": summary,
            "result": result,
            "turns": user_spoken_turns
        }
        st.success(f"Meeting '{meeting_title}' (Session {session_num}) has been successfully integrated from your voice recording!")

    st.markdown('</div>', unsafe_allow_html=True)

    # STEP 3: INSTANT INTERACTIVE RESULTS REVEAL
    if st.session_state.get("latest_ingestion_result"):
        res_data = st.session_state["latest_ingestion_result"]
        summary = res_data["summary"]
        pipeline_res = res_data["result"]
        turns = res_data["turns"]

        st.markdown('<div class="recording-card">', unsafe_allow_html=True)
        st.markdown('<span class="step-pill">STEP 3: REVEALED MEETING INTELLIGENCE (FROM YOUR AUDIO)</span>', unsafe_allow_html=True)
        st.markdown(f"### 📋 Results Extracted from Your Voice for Session {summary.session_number}: *{summary.meeting_title}*")

        # Superseded Revisions Alert
        transitions = pipeline_res.get("state_transitions", [])
        if transitions:
            for trans in transitions:
                st.markdown(
                    f"""
                    <div style="background-color: #451a03; border: 1px solid #f59e0b; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;">
                        <h4 style="color: #fbbf24; margin: 0 0 6px 0;">⚡ DECISION REVISION DETECTED:</h4>
                        <div style="color: #fef3c7;">
                            Historical <b>Decision #{trans['original_decision_id']}</b> on topic <code>{trans['topic']}</code> has been marked as <b style="color: #f59e0b;">SUPERSEDED</b> by new <b>Decision #{trans['revising_decision_id']}</b>!
                        </div>
                        <div style="font-size: 0.85rem; color: #fde68a; margin-top: 4px;">
                            Lineage Delta: {trans['session_delta']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Tabbed Results View
        tab_dec, tab_act, tab_debt, tab_turns = st.tabs([
            "⚖️ Consensus Decisions",
            "📌 Action Items",
            "⚠️ Conversation Debt Impact",
            "🎙️ Diarized Transcript"
        ])

        with tab_dec:
            if summary.decisions:
                for dec in summary.decisions:
                    cvr_pct = dec.cvr_confidence_score * 100
                    cvr_badge_color = "#10b981" if cvr_pct >= 60 else "#ef4444"
                    st.markdown(
                        f"""
                        <div style="background: #0f172a; border-left: 5px solid #10b981; border: 1px solid #334155; padding: 14px; border-radius: 8px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <b style="color: #38bdf8;">Topic: {dec.topic}</b>
                                <span style="background: {cvr_badge_color}22; color: {cvr_badge_color}; border: 1px solid {cvr_badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">
                                    CVR {cvr_pct:.0f}%
                                </span>
                            </div>
                            <div style="font-size: 1.05rem; color: #f8fafc; margin: 6px 0;">{dec.decision_text}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">
                                ⏱ <b>Timestamp:</b> {dec.evidence_timestamp} | 🎙️ <i>"{dec.evidence_quote}"</i>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No consensus decisions extracted in this session.")

        with tab_act:
            if summary.action_items:
                df_act = pd.DataFrame([a.model_dump() for a in summary.action_items])
                st.dataframe(df_act, use_container_width=True, hide_index=True)
            else:
                st.caption("No action items extracted.")

        with tab_debt:
            debt_items = pipeline_res.get("conversation_debt_audit", [])
            st.markdown(f"**Total Tracked Open Questions**: {len(debt_items)}")
            if debt_items:
                df_q = pd.DataFrame(debt_items)
                st.dataframe(df_q, use_container_width=True, hide_index=True)

        with tab_turns:
            for t in turns:
                spk = t.get("speaker_name") or t.get("speaker", "SPEAKER")
                st.markdown(f"**{spk}** `[{t.get('start', 0.0):.1f}s - {t.get('end', 0.0):.1f}s]`: {t.get('text', '')}")

        st.markdown("---")
        st.markdown("#### 🔗 Navigate to Updated Engine Views:")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("👉 View in Decision Timeline", use_container_width=True):
                st.session_state["current_view"] = "⏳ Decision Timeline"
                st.rerun()
        with btn_col2:
            if st.button("👉 View in Conversation Debt Tracker", use_container_width=True):
                st.session_state["current_view"] = "⚠️ Conversation Debt Tracker"
                st.rerun()
        with btn_col3:
            if st.button("👉 View in 'Why?' Forensic Audit", use_container_width=True):
                st.session_state["current_view"] = "🔍 'Why?' Forensic Audit"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: EXECUTIVE OVERVIEW
# ==============================================================================
elif st.session_state["current_view"] == "📊 Executive Overview":
    st.title("📊 Multi-Meeting Executive Intelligence Overview")
    st.markdown(
        "ChronoTrace links historical meeting sessions into an evolving graph of decisions, "
        "detecting contradictions, flagging cognitive conversation debt, and enforcing verifiable transcript grounding."
    )

    meetings = get_all_meetings()
    decisions = get_all_decisions_detailed()
    debt_report = get_conversation_debt_report(min_carried_over=1)
    flagged_tasks = get_low_confidence_or_unassigned_actions(max_cvr=0.60)

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_meetings = len(meetings)
    active_decisions = len([d for d in decisions if d["status"] == "Active"])
    superseded_decisions = len([d for d in decisions if d["status"] == "Superseded"])
    total_debt = sum(q.get("debt_score", 0.0) for q in debt_report)

    kpi1.metric("Indexed Meetings", total_meetings, delta="Historical Stream")
    kpi2.metric("Active Consensus Decisions", active_decisions, delta=f"{superseded_decisions} superseded", delta_color="normal")
    kpi3.metric("Conversation Debt Index", f"{total_debt:.1f}", delta="Unresolved friction", delta_color="inverse")
    kpi4.metric("Flagged Task Risks", len(flagged_tasks), delta="Low CVR / Unassigned", delta_color="inverse")

    st.markdown("---")

    # Recent Meetings Grid
    st.markdown("### 📅 Historical Meeting Sessions")
    cols = st.columns(len(meetings) if meetings else 1)
    for idx, m in enumerate(meetings):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1f2937; border-top: 4px solid #38bdf8; padding: 14px; border-radius: 8px;">
                    <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">SESSION {m['session_number']}</span>
                    <h4 style="margin: 6px 0; font-size: 1.05rem; color: #f8fafc;">{m['title']}</h4>
                    <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">Date: {m['created_at'][:10]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Quick Highlights Summary
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### 🔄 Recent Decision Revisions")
        lineage = get_superseded_decisions_lineage()
        if lineage:
            for item in lineage[:3]:
                st.markdown(f"- **{item['topic']}**: Pivoted from *'{item['original_decision'][:40]}...'* in Session {item['original_session']} to *'{item['revised_decision'][:45]}...'* in Session {item['revising_session']}")
        else:
            st.caption("No superseded decisions recorded.")

    with col_r:
        st.markdown("#### ⚠️ High Conversation Debt Alerts")
        if debt_report:
            for item in debt_report[:3]:
                st.markdown(f"- 🔴 **{item['risk_tier']} Risk**: '{item['question_text']}' *(Carried over {item['carried_over_count']} sessions | Debt: {item['debt_score']})*")
        else:
            st.caption("No open questions carried over.")


# ==============================================================================
# VIEW 3: DECISION TIMELINE
# ==============================================================================
elif st.session_state["current_view"] == "⏳ Decision Timeline":
    render_decision_timeline()


# ==============================================================================
# VIEW 4: CONVERSATION DEBT TRACKER
# ==============================================================================
elif st.session_state["current_view"] == "⚠️ Conversation Debt Tracker":
    render_debt_tracker()


# ==============================================================================
# VIEW 5: 'WHY?' FORENSIC AUDIT
# ==============================================================================
elif st.session_state["current_view"] == "🔍 'Why?' Forensic Audit":
    render_audit_view()


# ==============================================================================
# VIEW 6: TEAM ACCOUNTABILITY MATRIX
# ==============================================================================
elif st.session_state["current_view"] == "👥 Team Accountability Matrix":
    st.subheader("👥 Member Task Accountability & Commitment Matrix")
    st.markdown(
        "Evaluates execution velocity and ownership across engineering members by comparing "
        "explicit commitments, tentative tasks, completion rates, and stalled project friction."
    )

    acc_data = get_member_accountability_summary()
    if acc_data:
        df_acc = pd.DataFrame(acc_data)
        display_df = df_acc[[
            "member_name", "member_role", "total_tasks", "explicit_tasks",
            "tentative_tasks", "completed_tasks", "in_progress_tasks", "stalled_tasks",
            "completion_percentage", "stalled_risk_percentage"
        ]].copy()
        display_df.columns = [
            "Team Member", "Role", "Total Tasks", "Explicit Commitments",
            "Tentative", "Completed", "In Progress", "Stalled",
            "Completion %", "Stalled Risk %"
        ]
        display_df["Completion %"] = display_df["Completion %"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "N/A")
        display_df["Stalled Risk %"] = display_df["Stalled Risk %"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "0.0%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Task Execution Distribution")
        chart_data = df_acc[["member_name", "completed_tasks", "in_progress_tasks", "stalled_tasks"]].set_index("member_name")
        st.bar_chart(chart_data)
    else:
        st.info("No accountability metrics available.")


# ==============================================================================
# VIEW 7: DBMS SQL PLAYGROUND (Mini-Project Grading Module)
# ==============================================================================
elif st.session_state["current_view"] == "🛠️ DBMS SQL Playground":
    st.subheader("🛠️ DBMS Analytical Query Playground")
    st.markdown(
        "Execute and inspect the 5 core relational DBMS analytical queries directly against `chronotrace.db`. "
        "Each query demonstrates complex joins, subqueries, aggregation, and relational integrity constraints."
    )

    queries = {
        "Query 1: Superseded Decision Revisions Across Historical Streams": {
            "fn": get_superseded_decisions_lineage,
            "sql": """
SELECT 
    d_orig.decision_id AS original_decision_id,
    m_orig.session_number AS original_session,
    m_orig.title AS original_meeting_title,
    d_orig.topic,
    d_orig.decision_text AS original_decision,
    d_orig.status AS original_status,
    d_rev.decision_id AS superseded_by_id,
    m_rev.session_number AS revising_session,
    m_rev.title AS revising_meeting_title,
    d_rev.decision_text AS revised_decision,
    d_rev.cvr_confidence_score AS revised_cvr_score
FROM Decisions d_orig
JOIN Meetings m_orig ON d_orig.meeting_id = m_orig.meeting_id
JOIN Decisions d_rev ON d_orig.superseded_by_id = d_rev.decision_id
JOIN Meetings m_rev ON d_rev.meeting_id = m_rev.meeting_id
ORDER BY m_orig.session_number ASC, m_rev.session_number ASC;
            """
        },
        "Query 2: Conversation Debt for Carried-Over Questions (>= 2 Sessions)": {
            "fn": lambda: get_conversation_debt_report(min_carried_over=2),
            "sql": """
SELECT 
    q.question_id,
    m.session_number AS origin_session,
    m.title AS origin_meeting_title,
    q.question_text,
    COALESCE(u.name, 'Unspecified') AS asked_by_name,
    q.carried_over_count,
    q.debt_score,
    ROUND(q.debt_score * (1.0 + (q.carried_over_count * 0.5)), 2) AS dynamic_severity_index,
    CASE 
        WHEN q.carried_over_count >= 3 THEN 'CRITICAL'
        WHEN q.carried_over_count = 2 THEN 'HIGH'
        ELSE 'MODERATE'
    END AS risk_tier
FROM OpenQuestions q
JOIN Meetings m ON q.meeting_id = m.meeting_id
LEFT JOIN Users u ON q.asked_by_id = u.user_id
WHERE q.carried_over_count >= 2 AND q.status = 'Open'
ORDER BY q.debt_score DESC;
            """
        },
        "Query 3: Low-Confidence (<0.60 CVR) or Unassigned Action Items": {
            "fn": lambda: get_low_confidence_or_unassigned_actions(max_cvr=0.60),
            "sql": """
SELECT 
    a.task_id,
    m.session_number,
    a.task_description,
    COALESCE(u.name, 'UNASSIGNED') AS owner_name,
    a.commitment_level,
    a.status AS task_status,
    d.topic AS decision_topic,
    COALESCE(d.cvr_confidence_score, 0.0) AS cvr_confidence_score,
    CASE 
        WHEN (a.owner_id IS NULL OR a.commitment_level = 'Unassigned') 
             AND (d.cvr_confidence_score < 0.60) THEN 'Compound Risk (Unassigned & Low Grounding)'
        WHEN a.owner_id IS NULL OR a.commitment_level = 'Unassigned' THEN 'Unassigned Ownership Risk'
        WHEN d.cvr_confidence_score < 0.60 THEN 'Low Entailment Confidence (<0.60 CVR)'
        ELSE 'Standard Flag'
    END AS risk_category
FROM ActionItems a
JOIN Meetings m ON a.meeting_id = m.meeting_id
LEFT JOIN Users u ON a.owner_id = u.user_id
LEFT JOIN Decisions d ON a.decision_id = d.decision_id
WHERE (d.cvr_confidence_score < 0.60 OR a.owner_id IS NULL OR a.commitment_level = 'Unassigned')
ORDER BY cvr_confidence_score ASC;
            """
        },
        "Query 4: Member Task Accountability Summary": {
            "fn": get_member_accountability_summary,
            "sql": """
SELECT 
    u.user_id,
    u.name AS member_name,
    u.role AS member_role,
    COUNT(a.task_id) AS total_tasks,
    SUM(CASE WHEN a.commitment_level = 'Explicit' THEN 1 ELSE 0 END) AS explicit_tasks,
    SUM(CASE WHEN a.status = 'Completed' THEN 1 ELSE 0 END) AS completed_tasks,
    SUM(CASE WHEN a.status = 'Stalled' THEN 1 ELSE 0 END) AS stalled_tasks,
    ROUND(
        (CAST(SUM(CASE WHEN a.status = 'Completed' THEN 1 ELSE 0 END) AS REAL) / 
         NULLIF(COUNT(a.task_id), 0)) * 100.0, 1
    ) AS completion_percentage
FROM Users u
LEFT JOIN ActionItems a ON u.user_id = a.owner_id
GROUP BY u.user_id, u.name, u.role
ORDER BY total_tasks DESC;
            """
        },
        "Query 5: Full Decision Context Forensic Audit Trail": {
            "fn": lambda: get_decision_audit_trail(),
            "sql": """
SELECT 
    d.decision_id,
    m.session_number,
    m.title AS meeting_title,
    d.topic,
    d.decision_text,
    d.evidence_timestamp,
    d.cvr_confidence_score,
    d.status AS decision_status,
    d.superseded_by_id,
    COUNT(DISTINCT a.task_id) AS linked_action_items_count
FROM Decisions d
JOIN Meetings m ON d.meeting_id = m.meeting_id
LEFT JOIN ActionItems a ON d.decision_id = a.decision_id
GROUP BY d.decision_id
ORDER BY m.session_number DESC;
            """
        }
    }

    selected_query_title = st.selectbox("Select Analytical SQL Query to Execute:", list(queries.keys()))
    q_entry = queries[selected_query_title]

    with st.expander("📝 View Raw SQL DDL / Query Definition", expanded=True):
        st.code(q_entry["sql"].strip(), language="sql")

    if st.button("▶️ Execute Query Live"):
        results = q_entry["fn"]()
        if results:
            st.success(f"Query returned {len(results)} rows.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("Query returned 0 rows.")
