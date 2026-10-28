-- ChronoTrace / SynapseState Relational Database Schema
-- Compatible with SQLite (default) and standard ANSI/PostgreSQL/MySQL
PRAGMA foreign_keys = ON;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL
);

-- 2. Meetings Table
CREATE TABLE IF NOT EXISTS Meetings (
    meeting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    session_number INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    transcript_text TEXT
);

-- 3. Decisions Table
CREATE TABLE IF NOT EXISTS Decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    decision_text TEXT NOT NULL,
    evidence_timestamp TEXT,
    cvr_confidence_score REAL DEFAULT 0.0,
    status TEXT CHECK(status IN ('Active', 'Superseded', 'Contradicted')) DEFAULT 'Active',
    superseded_by_id INTEGER,
    FOREIGN KEY (meeting_id) REFERENCES Meetings(meeting_id) ON DELETE CASCADE,
    FOREIGN KEY (superseded_by_id) REFERENCES Decisions(decision_id) ON DELETE SET NULL
);

-- 4. Action Items Table
CREATE TABLE IF NOT EXISTS ActionItems (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER,
    meeting_id INTEGER NOT NULL,
    task_description TEXT NOT NULL,
    owner_id INTEGER,
    deadline TEXT,
    commitment_level TEXT CHECK(commitment_level IN ('Explicit', 'Tentative', 'Unassigned')) DEFAULT 'Unassigned',
    status TEXT CHECK(status IN ('Pending', 'In Progress', 'Completed', 'Stalled')) DEFAULT 'Pending',
    FOREIGN KEY (decision_id) REFERENCES Decisions(decision_id) ON DELETE SET NULL,
    FOREIGN KEY (meeting_id) REFERENCES Meetings(meeting_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id) REFERENCES Users(user_id) ON DELETE SET NULL
);

-- 5. Open Questions Table (Conversation Debt Tracker)
CREATE TABLE IF NOT EXISTS OpenQuestions (
    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    asked_by_id INTEGER,
    carried_over_count INTEGER DEFAULT 0,
    status TEXT CHECK(status IN ('Open', 'Resolved')) DEFAULT 'Open',
    debt_score REAL DEFAULT 0.0,
    FOREIGN KEY (meeting_id) REFERENCES Meetings(meeting_id) ON DELETE CASCADE,
    FOREIGN KEY (asked_by_id) REFERENCES Users(user_id) ON DELETE SET NULL
);

-- Analytical & Performance Indexes
CREATE INDEX IF NOT EXISTS idx_decisions_meeting ON Decisions(meeting_id);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON Decisions(status);
CREATE INDEX IF NOT EXISTS idx_decisions_superseded ON Decisions(superseded_by_id);
CREATE INDEX IF NOT EXISTS idx_actions_meeting ON ActionItems(meeting_id);
CREATE INDEX IF NOT EXISTS idx_actions_owner ON ActionItems(owner_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON ActionItems(status);
CREATE INDEX IF NOT EXISTS idx_questions_meeting ON OpenQuestions(meeting_id);
CREATE INDEX IF NOT EXISTS idx_questions_status ON OpenQuestions(status);
CREATE INDEX IF NOT EXISTS idx_questions_debt ON OpenQuestions(debt_score);
