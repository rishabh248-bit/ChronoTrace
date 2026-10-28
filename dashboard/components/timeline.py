"""
ChronoTrace Decision Evolution Timeline Component
Renders a multi-meeting sequential timeline graph tracing active, superseded,
and contradicted architectural decisions across sessions.
"""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from database.queries import get_all_decisions_detailed, get_superseded_decisions_lineage


def render_decision_timeline():
    """Renders the interactive decision lineage and evolution timeline."""
    st.subheader("⏳ Multi-Meeting Decision Evolution Timeline")
    st.markdown(
        "Trace how architectural choices evolve, get superseded, or undergo consensus reversals across historical meeting streams."
    )

    decisions = get_all_decisions_detailed()
    if not decisions:
        st.info("No decisions recorded yet. Seed the database to view historical decision streams.")
        return

    df = pd.DataFrame(decisions)

    # Top Filter Bar
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        status_filter = st.selectbox(
            "Filter by Lifecycle Status",
            options=["All", "Active", "Superseded", "Contradicted"],
            index=0
        )
    with col2:
        topics = ["All"] + sorted(list(df["topic"].unique()))
        topic_filter = st.selectbox("Filter by Topic Area", options=topics, index=0)
    with col3:
        search_kw = st.text_input("Search Decision Text", placeholder="e.g. MongoDB, Postgres, Kafka")

    # Apply Filters
    filtered_df = df.copy()
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["status"] == status_filter]
    if topic_filter != "All":
        filtered_df = filtered_df[filtered_df["topic"] == topic_filter]
    if search_kw:
        filtered_df = filtered_df[filtered_df["decision_text"].str.contains(search_kw, case=False, na=False)]

    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_decisions = len(df)
    active_count = len(df[df["status"] == "Active"])
    superseded_count = len(df[df["status"] == "Superseded"])
    avg_cvr = df["cvr_confidence_score"].mean() if not df.empty else 0.0

    m1.metric("Total Decisions", total_decisions)
    m2.metric("Active Consensus", active_count, delta=f"{int(active_count/total_decisions*100)}% of total" if total_decisions else None)
    m3.metric("Superseded / Revised", superseded_count, delta=f"-{superseded_count} revisions" if superseded_count else None, delta_color="inverse")
    m4.metric("Avg Claim Verification (CVR)", f"{avg_cvr * 100:.1f}%")

    st.markdown("---")

    # Superseded Lineage Callout
    lineage = get_superseded_decisions_lineage()
    if lineage:
        with st.expander("🔄 View Detected Decision Revisions & Lineage Chains", expanded=True):
            for item in lineage:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 12px 18px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #f59e0b;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">TOPIC: <b>{item['topic']}</b></span><br/>
                        <span style="color: #ef4444; text-decoration: line-through;">
                            [Session {item['original_session']}]: {item['original_decision']}
                        </span><br/>
                        <span style="color: #10b981; font-weight: 600;">
                            ↳ Revised in [Session {item['revising_session']}]: {item['revised_decision']}
                        </span>
                        <div style="margin-top: 6px; font-size: 0.8rem; color: #38bdf8;">
                            Grounding CVR: {item['revised_cvr_score'] * 100:.0f}% | Revised Meeting: {item['revising_meeting_title']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Group by Session Timeline Cards
    sessions = sorted(filtered_df["session_number"].unique())
    for session in sessions:
        session_rows = filtered_df[filtered_df["session_number"] == session]
        meeting_title = session_rows.iloc[0]["meeting_title"]

        st.markdown(f"#### 📅 Session {session}: {meeting_title}")

        for _, row in session_rows.iterrows():
            status = row["status"]
            status_color = "#10b981" if status == "Active" else ("#f59e0b" if status == "Superseded" else "#ef4444")
            cvr_score = row["cvr_confidence_score"] or 0.0
            cvr_badge_color = "#10b981" if cvr_score >= 0.60 else "#ef4444"

            superseded_notice = ""
            if row["superseded_by_id"]:
                superseded_notice = f"""
                <div style="margin-top: 8px; font-size: 0.85rem; color: #f59e0b;">
                    ⚠️ <b>Superseded Notice:</b> This consensus was modified/overturned by Decision #{row['superseded_by_id']}.
                </div>
                """

            st.markdown(
                f"""
                <div style="background: #0f172a; border: 1px solid #334155; border-left: 5px solid {status_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #38bdf8; font-size: 0.95rem;">#{row['decision_id']} | {row['topic']}</span>
                        <div>
                            <span style="background: {cvr_badge_color}22; color: {cvr_badge_color}; border: 1px solid {cvr_badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-right: 6px;">
                                CVR {cvr_score*100:.0f}%
                            </span>
                            <span style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                                {status.upper()}
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 1.05rem; color: #f8fafc; font-weight: 500; margin-bottom: 6px;">
                        {row['decision_text']}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        ⏱ Audio Timestamp: <b>{row['evidence_timestamp'] or 'Not provided'}</b>
                    </div>
                    {superseded_notice}
                </div>
                """,
                unsafe_allow_html=True
            )
