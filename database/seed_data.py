"""
ChronoTrace / SynapseState Seed Data Generator
Populates realistic, multi-session meeting engineering discussions, users,
historical decisions with revision lineage, action items, and conversation debt.
"""

from database.db_connection import get_db, init_db


def seed_database(reset: bool = True) -> None:
    """
    Initializes the schema and populates rich test data simulating 3 sequential
    engineering architecture meetings (Sessions 101, 102, 103).
    """
    if reset:
        init_db()

    with get_db() as conn:
        cursor = conn.cursor()

        # Clean existing test data (in reverse dependency order)
        cursor.execute("DELETE FROM OpenQuestions;")
        cursor.execute("DELETE FROM ActionItems;")
        cursor.execute("DELETE FROM Decisions;")
        cursor.execute("DELETE FROM Meetings;")
        cursor.execute("DELETE FROM Users;")

        # 1. Insert Team Users
        users = [
            (1, "Alice Chen", "alice@chronotrace.ai", "Lead Architect & EM"),
            (2, "Bob Martinez", "bob@chronotrace.ai", "Principal Backend Engineer"),
            (3, "Charlie Patel", "charlie@chronotrace.ai", "Senior DevOps / SRE"),
            (4, "Diana Vance", "diana@chronotrace.ai", "Lead Product Manager"),
        ]
        cursor.executemany(
            "INSERT INTO Users (user_id, name, email, role) VALUES (?, ?, ?, ?);",
            users
        )

        # 2. Insert Meetings (Sessions 101, 102, 103)
        meetings = [
            (
                101,
                "Sprint 24 Planning: Core Storage & Auth Architecture",
                101,
                "2026-10-10 10:00:00",
                "Alice (00:00): Welcome everyone. We need to finalize our storage and auth stack. "
                "Bob (04:15): I propose we adopt MongoDB Atlas as primary document store for rapid schema prototyping. "
                "Alice (06:00): Agreed on MongoDB for now. Let's move to auth. "
                "Charlie (14:10): Let's use JWT bearer tokens with 15-minute expiry and Redis token revocation list. "
                "Alice (16:30): Sounds solid. Alice (18:00): But wait, how will we guarantee strict multi-entity ACID consistency across financial ledger entries if using MongoDB? "
                "Charlie (22:15): What is our disaster recovery RTO/RPO target for cross-region failover? Let's discuss that next time."
            ),
            (
                102,
                "Sprint 24 Architecture Sync: ACID & Relational Pivot",
                102,
                "2026-10-17 10:00:00",
                "Alice (00:00): Reviewing last week's decisions. Our ledger requirements demand strict transactional guarantees. "
                "Bob (08:30): After running concurrency tests, MongoDB's distributed transactions are hitting deadlocks. "
                "Alice (11:20): Therefore, we pivot from MongoDB to PostgreSQL 16 with Row-Level Security for ACID ledger compliance. "
                "Charlie (22:00): For edge traffic, let's deploy Kong API Gateway with OpenID Connect plugin for centralized edge auth. "
                "Bob (27:10): Should we run Postgres on AWS Aurora Serverless v2 or self-managed on EKS with CloudNativePG?"
            ),
            (
                103,
                "Sprint 25 Retrospective: Production Readiness & Scale",
                103,
                "2026-10-24 10:00:00",
                "Alice (00:00): Welcome to Sprint 25 Retro. Let's review messaging and SDK support. "
                "Charlie (05:40): For event streaming, we select Apache Kafka with Strimzi operator for inter-service async event bus. "
                "Diana (35:10): Can we guarantee Python SDK at launch? Alice (36:20): Maybe, let's tentatively support only TypeScript SDK initially, deferring Python SDK. "
                "Diana (40:00): We still haven't answered Charlie's question from Session 101 on our disaster recovery RTO/RPO target!"
            )
        ]
        cursor.executemany(
            "INSERT INTO Meetings (meeting_id, title, session_number, created_at, transcript_text) VALUES (?, ?, ?, ?, ?);",
            meetings
        )

        # 3. Insert Decisions
        # Initial insertion with superseded_by_id as NULL to respect foreign key ordering
        decisions = [
            (
                1,
                101,
                "Primary Database Engine",
                "Adopt MongoDB Atlas as primary document store for rapid schema prototyping",
                "00:04:15 - 00:06:20",
                0.88,
                "Superseded",
                None
            ),
            (
                2,
                101,
                "Authentication Protocol",
                "Use JWT bearer tokens with 15-minute expiry and Redis token revocation list",
                "00:14:10 - 00:16:45",
                0.94,
                "Active",
                None
            ),
            (
                3,
                102,
                "Primary Database Engine",
                "Pivot from MongoDB to PostgreSQL 16 with Row-Level Security for ACID ledger compliance",
                "00:08:30 - 00:12:15",
                0.92,
                "Active",
                None
            ),
            (
                4,
                102,
                "API Gateway Architecture",
                "Deploy Kong API Gateway with OpenID Connect plugin for centralized edge auth",
                "00:22:00 - 00:24:10",
                0.85,
                "Active",
                None
            ),
            (
                5,
                103,
                "Event Streaming Platform",
                "Select Apache Kafka with Strimzi operator for inter-service async event bus",
                "00:05:40 - 00:09:12",
                0.91,
                "Active",
                None
            ),
            (
                6,
                103,
                "Client SDK Language Support",
                "Tentatively support only TypeScript SDK initially, deferring Python SDK",
                "00:35:10 - 00:36:50",
                0.54,  # Intentional low CVR score for Query 3 demonstration
                "Active",
                None
            )
        ]
        cursor.executemany(
            """INSERT INTO Decisions (
                decision_id, meeting_id, topic, decision_text, evidence_timestamp, 
                cvr_confidence_score, status, superseded_by_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);""",
            decisions
        )

        # Now update Decision 1 to point to its superseding revision (Decision 3)
        cursor.execute(
            "UPDATE Decisions SET superseded_by_id = 3 WHERE decision_id = 1;"
        )

        # 4. Insert Action Items
        action_items = [
            # Meeting 101 Tasks
            (1, 1, 101, "Set up MongoDB staging cluster and document collection schemas", 2, "2026-10-17", "Explicit", "Completed"),
            (2, 2, 101, "Configure Redis caching cluster with VPC peering", 3, "2026-10-18", "Explicit", "In Progress"),
            # Meeting 102 Tasks
            (3, 3, 102, "Write schema migration scripts from MongoDB prototypes to PostgreSQL DDL", 2, "2026-10-24", "Explicit", "In Progress"),
            (4, 4, 102, "Evaluate Kong vs Envoy gateway latency benchmarks on AWS EKS", None, "2026-10-22", "Unassigned", "Stalled"),  # High risk unassigned task
            # Meeting 103 Tasks
            (5, 5, 103, "Deploy Strimzi Kafka operator to staging Kubernetes cluster", 3, "2026-11-01", "Explicit", "In Progress"),
            (6, 6, 103, "Draft TypeScript SDK core client and authentication interceptors", 4, "2026-11-03", "Tentative", "Pending"),
            (7, 3, 103, "Audit PostgreSQL read-replica failover under simulated 10k QPS load", 2, "2026-11-05", "Explicit", "Completed"),
            (8, None, 103, "Define disaster recovery multi-region backup schedule and RTO/RPO SLAs", None, "2026-10-28", "Unassigned", "Stalled")  # High risk
        ]
        cursor.executemany(
            """INSERT INTO ActionItems (
                task_id, decision_id, meeting_id, task_description, owner_id, deadline, commitment_level, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);""",
            action_items
        )

        # 5. Insert Open Questions (Conversation Debt Tracker)
        open_questions = [
            (
                1,
                101,
                "How will we guarantee strict multi-entity ACID consistency across financial ledger entries if using MongoDB?",
                1,  # Alice
                2,  # Carried over 2 sessions
                "Open",
                24.5  # Elevated debt score
            ),
            (
                2,
                101,
                "What is our disaster recovery RTO/RPO target for cross-region database failover?",
                3,  # Charlie
                3,  # Carried over 3 sessions
                "Open",
                32.0  # High debt score
            ),
            (
                3,
                102,
                "Should we run Postgres on AWS Aurora Serverless v2 or self-managed on EKS with CloudNativePG?",
                2,  # Bob
                1,  # Carried over 1 session
                "Open",
                8.5
            ),
            (
                4,
                103,
                "Will Kafka topic partitioning scheme match user tenant IDs or event type clusters?",
                3,  # Charlie
                0,
                "Open",
                4.0
            )
        ]
        cursor.executemany(
            """INSERT INTO OpenQuestions (
                question_id, meeting_id, question_text, asked_by_id, carried_over_count, status, debt_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?);""",
            open_questions
        )

    print("Successfully seeded ChronoTrace database with 3 meetings, 6 decisions, 8 action items, and 4 questions.")


if __name__ == "__main__":
    seed_database(reset=True)
