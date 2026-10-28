"""
ChronoTrace / SynapseState Analytical Query Execution Engine
Provides 5 high-impact production SQL queries for multi-meeting intelligence,
conversation debt calculation, grounding risk, accountability, and forensic audit trails.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from database.db_connection import execute_query


def get_superseded_decisions_lineage() -> List[Dict[str, Any]]:
    """
    Query 1: Detect superseded decision revisions across historical meeting streams.
    Uses self-joins on Decisions and Meetings to trace original choices and their
    revisions across sequential sessions.
    """
    sql = """
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
    return execute_query(sql)


def get_conversation_debt_report(min_carried_over: int = 2) -> List[Dict[str, Any]]:
    """
    Query 2: Calculate conversation debt score for questions carried over 2+ sessions.
    Evaluates unresolved cognitive debt, multi-session drag, and risk tiering.
    """
    sql = """
    SELECT 
        q.question_id,
        m.session_number AS origin_session,
        m.title AS origin_meeting_title,
        q.question_text,
        COALESCE(u.name, 'Anonymous / Unspecified') AS asked_by_name,
        COALESCE(u.role, 'Participant') AS asked_by_role,
        q.carried_over_count,
        q.status,
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
    WHERE q.carried_over_count >= ? AND q.status = 'Open'
    ORDER BY q.debt_score DESC, q.carried_over_count DESC;
    """
    return execute_query(sql, (min_carried_over,))


def get_low_confidence_or_unassigned_actions(max_cvr: float = 0.60) -> List[Dict[str, Any]]:
    """
    Query 3: Retrieve low-confidence (<0.60 CVR) or unassigned action items.
    Exposes high-risk operational tasks lacking verifiable grounding or explicit owners.
    """
    sql = """
    SELECT 
        a.task_id,
        m.session_number,
        m.title AS meeting_title,
        a.task_description,
        COALESCE(u.name, 'UNASSIGNED') AS owner_name,
        COALESCE(u.role, 'None') AS owner_role,
        a.commitment_level,
        a.status AS task_status,
        d.decision_id,
        d.topic AS decision_topic,
        COALESCE(d.cvr_confidence_score, 0.0) AS cvr_confidence_score,
        CASE 
            WHEN (a.owner_id IS NULL OR a.commitment_level = 'Unassigned') 
                 AND (d.cvr_confidence_score < ?) THEN 'Compound Risk (Unassigned & Low Grounding)'
            WHEN a.owner_id IS NULL OR a.commitment_level = 'Unassigned' THEN 'Unassigned Ownership Risk'
            WHEN d.cvr_confidence_score < ? THEN 'Low Entailment Confidence (<0.60 CVR)'
            ELSE 'Standard Flag'
        END AS risk_category
    FROM ActionItems a
    JOIN Meetings m ON a.meeting_id = m.meeting_id
    LEFT JOIN Users u ON a.owner_id = u.user_id
    LEFT JOIN Decisions d ON a.decision_id = d.decision_id
    WHERE (d.cvr_confidence_score < ? OR a.owner_id IS NULL OR a.commitment_level = 'Unassigned')
    ORDER BY cvr_confidence_score ASC, m.session_number DESC;
    """
    return execute_query(sql, (max_cvr, max_cvr, max_cvr))


def get_member_accountability_summary() -> List[Dict[str, Any]]:
    """
    Query 4: Generate member task accountability summaries.
    Aggregates explicit vs tentative commitments, completion rate, and stalled velocity.
    """
    sql = """
    SELECT 
        u.user_id,
        u.name AS member_name,
        u.role AS member_role,
        COUNT(a.task_id) AS total_tasks,
        SUM(CASE WHEN a.commitment_level = 'Explicit' THEN 1 ELSE 0 END) AS explicit_tasks,
        SUM(CASE WHEN a.commitment_level = 'Tentative' THEN 1 ELSE 0 END) AS tentative_tasks,
        SUM(CASE WHEN a.status = 'Completed' THEN 1 ELSE 0 END) AS completed_tasks,
        SUM(CASE WHEN a.status = 'In Progress' THEN 1 ELSE 0 END) AS in_progress_tasks,
        SUM(CASE WHEN a.status = 'Stalled' THEN 1 ELSE 0 END) AS stalled_tasks,
        ROUND(
            (CAST(SUM(CASE WHEN a.status = 'Completed' THEN 1 ELSE 0 END) AS REAL) / 
             NULLIF(COUNT(a.task_id), 0)) * 100.0, 1
        ) AS completion_percentage,
        ROUND(
            (CAST(SUM(CASE WHEN a.status = 'Stalled' THEN 1 ELSE 0 END) AS REAL) / 
             NULLIF(COUNT(a.task_id), 0)) * 100.0, 1
        ) AS stalled_risk_percentage
    FROM Users u
    LEFT JOIN ActionItems a ON u.user_id = a.owner_id
    GROUP BY u.user_id, u.name, u.role
    ORDER BY total_tasks DESC, stalled_risk_percentage DESC;
    """
    return execute_query(sql)


def get_decision_audit_trail(decision_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Query 5: Full decision context audit trail for 'Why?' audit popups.
    Returns complete forensic pedigree: timestamp, CVR score, transcript excerpt,
    linked action tasks, and historical lineage.
    """
    sql = """
    SELECT 
        d.decision_id,
        m.session_number,
        m.title AS meeting_title,
        m.created_at AS meeting_date,
        d.topic,
        d.decision_text,
        d.evidence_timestamp,
        d.cvr_confidence_score,
        d.status AS decision_status,
        d.superseded_by_id,
        rev.decision_text AS superseded_by_text,
        COUNT(DISTINCT a.task_id) AS linked_action_items_count,
        GROUP_CONCAT(DISTINCT a.task_description) AS linked_tasks,
        m.transcript_text
    FROM Decisions d
    JOIN Meetings m ON d.meeting_id = m.meeting_id
    LEFT JOIN Decisions rev ON d.superseded_by_id = rev.decision_id
    LEFT JOIN ActionItems a ON d.decision_id = a.decision_id
    WHERE (? IS NULL OR d.decision_id = ?)
    GROUP BY d.decision_id
    ORDER BY m.session_number DESC, d.decision_id DESC;
    """
    return execute_query(sql, (decision_id, decision_id))


def get_all_meetings() -> List[Dict[str, Any]]:
    """Helper query returning all recorded meeting sessions."""
    sql = "SELECT * FROM Meetings ORDER BY session_number ASC;"
    return execute_query(sql)


def get_all_decisions_detailed() -> List[Dict[str, Any]]:
    """Helper query returning decisions with meeting title and session."""
    sql = """
    SELECT 
        d.decision_id,
        d.meeting_id,
        m.session_number,
        m.title AS meeting_title,
        d.topic,
        d.decision_text,
        d.evidence_timestamp,
        d.cvr_confidence_score,
        d.status,
        d.superseded_by_id
    FROM Decisions d
    JOIN Meetings m ON d.meeting_id = m.meeting_id
    ORDER BY m.session_number ASC, d.decision_id ASC;
    """
    return execute_query(sql)


def print_analytical_queries_demo() -> None:
    """Pretty prints the results of all 5 queries for DBMS testing and grading."""
    print("=" * 80)
    print("CHRONOTRACE DBMS ANALYTICAL QUERIES EXECUTION ENGINE")
    print("=" * 80)

    print("\n--- QUERY 1: Superseded Decision Revisions Across Sessions ---")
    q1 = get_superseded_decisions_lineage()
    print(pd.DataFrame(q1).to_string(index=False) if q1 else "No superseded decisions found.")

    print("\n--- QUERY 2: Conversation Debt for Carried-Over Questions (>= 2 Sessions) ---")
    q2 = get_conversation_debt_report(min_carried_over=2)
    print(pd.DataFrame(q2).to_string(index=False) if q2 else "No high-debt questions found.")

    print("\n--- QUERY 3: Low-Confidence (<0.60 CVR) or Unassigned Action Items ---")
    q3 = get_low_confidence_or_unassigned_actions(max_cvr=0.60)
    print(pd.DataFrame(q3).to_string(index=False) if q3 else "No flagged action items.")

    print("\n--- QUERY 4: Member Task Accountability Summary ---")
    q4 = get_member_accountability_summary()
    print(pd.DataFrame(q4).to_string(index=False) if q4 else "No member task data.")

    print("\n--- QUERY 5: Full Decision Forensic Audit Trail ('Why?' Audit Popups) ---")
    q5 = get_decision_audit_trail()
    if q5:
        df5 = pd.DataFrame(q5)[['decision_id', 'session_number', 'topic', 'decision_text', 'evidence_timestamp', 'cvr_confidence_score', 'decision_status']]
        print(df5.to_string(index=False))
    else:
        print("No decision audit trails found.")
    print("=" * 80)


if __name__ == "__main__":
    print_analytical_queries_demo()
