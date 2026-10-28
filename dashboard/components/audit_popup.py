"""
ChronoTrace 'Why?' Forensic Audit View & Grounding Inspector Component
Provides deep-dive decision verification, inspecting speaker turns, verbatim audio evidence,
timestamp intervals, CVR confidence scores, and revision pedigree.
"""

from typing import Optional
import streamlit as st
import pandas as pd
from database.queries import get_all_decisions_detailed, get_decision_audit_trail


def render_audit_view():
    """Renders the interactive forensic audit view with 'Why?' drilldown."""
    st.subheader("🔍 Forensic 'Why?' Grounding Audit Inspector")
    st.markdown(
        "Click on any architectural decision to audit **why** it was made, examine the verbatim transcript evidence quote, "
        "verify the DeBERTa-v3 NLI Claim Verification Rate (CVR), and inspect linked action items."
    )

    decisions = get_all_decisions_detailed()
    if not decisions:
        st.info("No decisions recorded yet. Seed the database to view audit trails.")
        return

    # Selection widget
    decision_options = {
        f"#{d['decision_id']} [Session {d['session_number']}] {d['topic']}: {d['decision_text'][:65]}...": d["decision_id"]
        for d in decisions
    }

    selected_label = st.selectbox(
        "Select a Decision to Audit:",
        options=list(decision_options.keys()),
        index=0
    )
    selected_id = decision_options[selected_label]

    # Fetch complete audit trail for this decision (Query 5)
    audit_data = get_decision_audit_trail(decision_id=selected_id)
    if not audit_data:
        st.warning("Audit details not found for this decision.")
        return

    record = audit_data[0]

    # Top summary card
    st.markdown("---")
    status = record["decision_status"]
    status_color = "#10b981" if status == "Active" else ("#f59e0b" if status == "Superseded" else "#ef4444")
    cvr_score = record["cvr_confidence_score"] or 0.0
    cvr_color = "#10b981" if cvr_score >= 0.60 else "#ef4444"

    col_meta1, col_meta2 = st.columns([3, 1])
    with col_meta1:
        st.markdown(f"### #{record['decision_id']} - {record['topic']}")
        st.markdown(f"**Session {record['session_number']}**: {record['meeting_title']}")
        st.caption(f"Recorded Date: {record['meeting_date']}")
    with col_meta2:
        st.markdown(
            f"""
            <div style="text-align: right; margin-top: 10px;">
                <span style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; padding: 6px 14px; border-radius: 16px; font-weight: 700; font-size: 0.9rem;">
                    {status.upper()}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Core Decision Statement
    st.markdown(
        f"""
        <div style="background-color: #0f172a; border-left: 5px solid #38bdf8; padding: 16px 20px; border-radius: 6px; margin: 15px 0;">
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Consensus Decision Statement</div>
            <div style="font-size: 1.15rem; color: #f8fafc; font-weight: 600; margin-top: 4px;">{record['decision_text']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Forensic Verification Metrics (CVR & Grounding)
    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        st.markdown("#### 🎯 Claim Verification Rate (CVR)")
        st.progress(cvr_score)
        if cvr_score >= 0.60:
            st.success(f"**Verified Grounding Score**: {cvr_score * 100:.1f}% (High Entailment)")
        else:
            st.error(f"**Low Grounding Warning**: {cvr_score * 100:.1f}% (Weak Entailment / Missing Evidence)")

    with col_v2:
        st.markdown("#### ⏱ Audio Alignment Interval")
        st.info(f"**Transcript Timestamp**: `{record['evidence_timestamp'] or '00:00:00 - 00:00:00'}`")

    # Verbatim Evidence Excerpt
    st.markdown("#### 🎙️ Verbatim Transcript Evidence Span")
    st.markdown(
        f"""
        <div style="background-color: #1e293b; border: 1px solid #475569; padding: 14px 18px; border-radius: 8px; font-style: italic; color: #e2e8f0; margin-bottom: 16px;">
            "{record['decision_text']}"
        </div>
        """,
        unsafe_allow_html=True
    )

    # Superseded Lineage Info
    if record["superseded_by_id"] or record["superseded_by_text"]:
        st.markdown("#### 🔄 Decision Modification Lineage")
        st.warning(
            f"This decision was **Superseded by Decision #{record['superseded_by_id']}**:\n\n"
            f"> *\"{record['superseded_by_text']}\"*"
        )

    # Linked Action Items
    st.markdown("#### 📌 Linked Action Items")
    if record["linked_tasks"]:
        tasks = record["linked_tasks"].split(",")
        for t in tasks:
            st.markdown(f"- 🔧 `{t.strip()}`")
    else:
        st.caption("No linked action items assigned directly to this decision.")

    # Full Meeting Transcript Drawer
    with st.expander("📄 View Full Session Transcript Text", expanded=False):
        st.text_area(
            "Session Transcript",
            value=record["transcript_text"] or "No transcript stored for this session.",
            height=200,
            disabled=True
        )
