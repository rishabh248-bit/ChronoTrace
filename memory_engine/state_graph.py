"""
ChronoTrace / SynapseState Longitudinal Decision Evolution & Conversation Debt Engine
Tracks cross-meeting decision modifications, automatically detects contradictions / superseded choices,
and computes cumulative conversation debt across historical meeting streams.
"""

from typing import Any, Dict, List, Optional, Tuple
from database.db_connection import execute_insert, execute_query, execute_update
from memory_engine.vector_store import VectorStoreManager
from nlp_engine.schemas import DecisionSchema, MeetingSummarySchema, OpenQuestionSchema


class StateGraphEngine:
    """
    Longitudinal state transition and conversation debt calculation engine.
    Connects ChromaDB semantic search with relational SQLite state transitions.
    """

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def process_decision_evolution(
        self,
        new_decision: DecisionSchema,
        meeting_id: int,
        session_number: int
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Inserts new decision into SQLite, searches historical sessions in vector memory,
        and marks prior conflicting/superseded decisions if a reversal or evolution is detected.
        """
        # 1. Search prior historical decisions (session < current session)
        query = f"Topic: {new_decision.topic} | Decision: {new_decision.decision_text}"
        prior_matches = self.vector_store.search_similar_decisions(
            query_text=query,
            n_results=3,
            session_before=session_number
        )

        # 2. Insert new decision into SQLite as 'Active'
        insert_sql = """
        INSERT INTO Decisions (
            meeting_id, topic, decision_text, evidence_timestamp, 
            cvr_confidence_score, status, superseded_by_id
        ) VALUES (?, ?, ?, ?, ?, 'Active', NULL);
        """
        new_decision_id = execute_insert(
            insert_sql,
            (
                meeting_id,
                new_decision.topic,
                new_decision.decision_text,
                new_decision.evidence_timestamp,
                new_decision.cvr_confidence_score
            )
        )

        # 3. Add to vector store
        self.vector_store.add_decision(
            decision_id=new_decision_id,
            topic=new_decision.topic,
            decision_text=new_decision.decision_text,
            session_number=session_number,
            meeting_id=meeting_id,
            status="Active"
        )

        # 4. Check for superseded candidates
        transition_event = None
        for match in prior_matches:
            prior_id = match.get("decision_id")
            sim_score = match.get("similarity_score", 0.0)
            prior_topic = match.get("metadata", {}).get("topic", "")

            # Contradiction / Evolution heuristic:
            # Same topic or high semantic similarity, but contains change/pivot keywords or distinct technological entities
            is_same_topic = prior_topic.lower() == new_decision.topic.lower()
            is_pivot_language = any(kw in new_decision.decision_text.lower() for kw in ["pivot", "switch", "replace", "supersede", "instead of"])

            if prior_id and (is_same_topic or (sim_score >= 0.45 and is_pivot_language)):
                # Mark previous decision as Superseded in SQLite
                update_sql = """
                UPDATE Decisions 
                SET status = 'Superseded', superseded_by_id = ? 
                WHERE decision_id = ? AND status = 'Active';
                """
                affected = execute_update(update_sql, (new_decision_id, prior_id))
                if affected > 0:
                    transition_event = {
                        "original_decision_id": prior_id,
                        "revising_decision_id": new_decision_id,
                        "topic": new_decision.topic,
                        "superseded_by": new_decision.decision_text,
                        "session_delta": f"Session {match.get('metadata', {}).get('session_number')} -> Session {session_number}",
                        "similarity_score": sim_score
                    }
                    print(f"[StateGraph] Decision #{prior_id} SUPERSEDED by Decision #{new_decision_id} ({transition_event['session_delta']})")
                    break

        return new_decision_id, transition_event

    def update_conversation_debt(self, current_session: int) -> List[Dict[str, Any]]:
        """
        Calculates and compounds Conversation Debt across all unresolved questions.
        Formula:
            Debt = base_debt * (1.5 ** carried_over_count) + (stalled_tasks_penalty)
        """
        # Fetch open questions
        fetch_questions_sql = """
        SELECT q.question_id, q.meeting_id, m.session_number, q.question_text, 
               q.carried_over_count, q.debt_score
        FROM OpenQuestions q
        JOIN Meetings m ON q.meeting_id = m.meeting_id
        WHERE q.status = 'Open';
        """
        open_questions = execute_query(fetch_questions_sql)

        # Count stalled tasks to apply system-wide friction penalty
        stalled_count_row = execute_query("SELECT COUNT(*) AS count FROM ActionItems WHERE status = 'Stalled';")
        stalled_penalty = (stalled_count_row[0]["count"] if stalled_count_row else 0) * 2.0

        updated_records = []
        for q in open_questions:
            q_id = q["question_id"]
            origin_session = q["session_number"]
            sessions_elapsed = max(0, current_session - origin_session)

            # Carried over count represents how many sessions it remained unresolved
            carried_over = max(q["carried_over_count"], sessions_elapsed)
            base_score = 5.0
            compounded_debt = round((base_score * (1.5 ** carried_over)) + stalled_penalty, 2)

            execute_update(
                "UPDATE OpenQuestions SET carried_over_count = ?, debt_score = ? WHERE question_id = ?;",
                (carried_over, compounded_debt, q_id)
            )

            updated_records.append({
                "question_id": q_id,
                "question_text": q["question_text"],
                "origin_session": origin_session,
                "carried_over_count": carried_over,
                "compounded_debt_score": compounded_debt
            })

        return updated_records

    def ingest_summary(self, summary: MeetingSummarySchema) -> Dict[str, Any]:
        """
        Ingests an entire MeetingSummarySchema:
        Creates meeting record if needed, writes decisions, checks evolution,
        saves action items, inserts open questions, and recalculates conversation debt.
        """
        # 1. Insert Meeting
        meeting_sql = """
        INSERT INTO Meetings (title, session_number, transcript_text) 
        VALUES (?, ?, ?);
        """
        meeting_id = execute_insert(
            meeting_sql,
            (summary.meeting_title, summary.session_number, "\n".join(summary.key_takeaways))
        )

        transitions = []
        decision_id_map = {}

        # 2. Ingest decisions with evolution tracking
        for dec in summary.decisions:
            dec_id, trans = self.process_decision_evolution(
                new_decision=dec,
                meeting_id=meeting_id,
                session_number=summary.session_number
            )
            decision_id_map[dec.topic] = dec_id
            if trans:
                transitions.append(trans)

        # 3. Ingest Action Items
        users_rows = execute_query("SELECT user_id, name FROM Users;")
        user_name_to_id = {u["name"].lower(): u["user_id"] for u in users_rows}

        for action in summary.action_items:
            owner_id = None
            if action.owner_name:
                for uname, uid in user_name_to_id.items():
                    if uname in action.owner_name.lower() or action.owner_name.lower() in uname:
                        owner_id = uid
                        break

            # Associate with first available decision or None
            linked_dec_id = next(iter(decision_id_map.values()), None)

            action_sql = """
            INSERT INTO ActionItems (
                decision_id, meeting_id, task_description, owner_id, deadline, commitment_level, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            execute_insert(
                action_sql,
                (
                    linked_dec_id,
                    meeting_id,
                    action.task_description,
                    owner_id,
                    action.deadline or "TBD",
                    action.commitment_level,
                    action.status
                )
            )

        # 4. Ingest Open Questions
        for q in summary.open_questions:
            asked_by_id = None
            if q.asked_by:
                for uname, uid in user_name_to_id.items():
                    if uname in q.asked_by.lower() or q.asked_by.lower() in uname:
                        asked_by_id = uid
                        break

            q_sql = """
            INSERT INTO OpenQuestions (
                meeting_id, question_text, asked_by_id, carried_over_count, status, debt_score
            ) VALUES (?, ?, ?, 0, 'Open', ?);
            """
            execute_insert(q_sql, (meeting_id, q.question_text, asked_by_id, q.estimated_debt_impact))

        # 5. Compound Conversation Debt across history
        debt_audit = self.update_conversation_debt(current_session=summary.session_number)

        return {
            "meeting_id": meeting_id,
            "session_number": summary.session_number,
            "decisions_count": len(summary.decisions),
            "action_items_count": len(summary.action_items),
            "open_questions_count": len(summary.open_questions),
            "state_transitions": transitions,
            "conversation_debt_audit": debt_audit
        }


if __name__ == "__main__":
    from database.seed_data import seed_database
    seed_database(reset=True)

    engine = StateGraphEngine()
    debt = engine.update_conversation_debt(current_session=104)
    print("Conversation Debt compounding after Session 104:")
    for item in debt:
        print(f" - Question #{item['question_id']}: Carried over {item['carried_over_count']} sessions -> Debt Score: {item['compounded_debt_score']}")
