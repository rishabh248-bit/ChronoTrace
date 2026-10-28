"""
ChronoTrace Conversation Debt & Operational Friction Tracker
Monitors unresolved cognitive debt from carried-over questions, unassigned tasks,
and low-grounding decisions compounding across meeting streams.
"""

import streamlit as st
import pandas as pd
from database.queries import get_conversation_debt_report, get_low_confidence_or_unassigned_actions
from database.db_connection import execute_query


def render_debt_tracker():
    """Renders the Conversation Debt tracker view."""
    st.subheader("⚠️ Conversation Debt & Friction Engine")
    st.markdown(
        "Conversation Debt quantifies unresolved blockers, questions carried across multiple meeting sessions, "
        "and unassigned commitments compounding cognitive overhead on the engineering team."
    )

    # 1. Fetch Debt Metrics
    debt_rows = get_conversation_debt_report(min_carried_over=1)
    all_questions = execute_query(
        "SELECT question_id, question_text, carried_over_count, debt_score, status FROM OpenQuestions;"
    )
    flagged_actions = get_low_confidence_or_unassigned_actions(max_cvr=0.60)

    total_debt_score = sum(q.get("debt_score", 0.0) for q in all_questions if q.get("status") == "Open")
    critical_questions = [q for q in debt_rows if q.get("risk_tier") == "CRITICAL"]
    high_questions = [q for q in debt_rows if q.get("risk_tier") == "HIGH"]
    stalled_actions = [a for a in flagged_actions if a.get("task_status") == "Stalled"]

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Cumulative Debt Index",
        f"{total_debt_score:.1f}",
        delta=f"+{len(debt_rows)} carried questions",
        delta_color="inverse"
    )
    c2.metric("Critical Blockers (>=3 Sessions)", len(critical_questions), delta="High Escalation", delta_color="inverse")
    c3.metric("High Debt (>=2 Sessions)", len(high_questions), delta="Needs Resolution", delta_color="inverse")
    c4.metric("Stalled / Unassigned Tasks", len(flagged_actions), delta="Action Items at Risk", delta_color="inverse")

    st.markdown("---")

    # Critical Warning Alert Banner
    if critical_questions:
        st.error(
            f"🚨 **CRITICAL DEBT WARNING**: {len(critical_questions)} question(s) have been carried over 3 or more consecutive meeting sessions without resolution. Architectural velocity is at risk."
        )

    # Section 1: Carried-Over Questions Table
    st.markdown("### 📋 Unresolved Questions Compounding Conversation Debt")
    if debt_rows:
        df_debt = pd.DataFrame(debt_rows)

        # Style table columns
        display_df = df_debt[[
            "question_id", "risk_tier", "origin_session", "asked_by_name",
            "question_text", "carried_over_count", "debt_score", "dynamic_severity_index"
        ]].copy()
        display_df.columns = [
            "ID", "Risk Tier", "Origin Session", "Asked By",
            "Unresolved Question", "Sessions Carried Over", "Base Debt", "Dynamic Severity"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No questions have been carried over multiple sessions. Conversation debt is currently minimized!")

    st.markdown("---")

    # Section 2: Action Items with Low Grounding or Missing Owners
    st.markdown("### 🚩 Low Grounding & Unassigned Action Items Risk")
    st.markdown(
        "Action items lacking explicit commitment, missing owners, or grounded in low-confidence (<0.60 CVR) consensus."
    )

    if flagged_actions:
        df_actions = pd.DataFrame(flagged_actions)
        display_actions = df_actions[[
            "task_id", "session_number", "risk_category", "task_description",
            "owner_name", "commitment_level", "task_status", "cvr_confidence_score"
        ]].copy()
        display_actions.columns = [
            "Task ID", "Session", "Risk Factor", "Task Description",
            "Owner", "Commitment Level", "Status", "Decision CVR Score"
        ]

        # Format CVR column as percentage
        display_actions["Decision CVR Score"] = display_actions["Decision CVR Score"].apply(lambda v: f"{v*100:.0f}%")

        st.dataframe(
            display_actions,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No flagged action items detected.")
