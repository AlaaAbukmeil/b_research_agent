import sqlite3
import json
import os
from datetime import datetime, timezone


class MemoryStore:
    """
    SQLite-backed persistent storage for the research agent's memory hierarchy.

    Tables:
        sessions          — One row per research session (query, cost, status)
        sub_questions      — Decomposed sub-questions per session
        episodic_buffer    — Level 1 memory: raw findings (bounded capacity)
        compressed_memory  — Level 2 memory: summarized findings (unbounded)
    """

    def __init__(self, db_path: str = "data/memory.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id          TEXT PRIMARY KEY,
                query               TEXT NOT NULL,
                started_at          TEXT NOT NULL,
                completed_at        TEXT,
                total_input_tokens  INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                estimated_cost_usd  REAL DEFAULT 0.0,
                status              TEXT DEFAULT 'active',
                final_answer        TEXT
            );

            CREATE TABLE IF NOT EXISTS sub_questions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                question    TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                status      TEXT DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS episodic_buffer (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                sub_question TEXT,
                finding      TEXT NOT NULL,
                token_count  INTEGER NOT NULL,
                keywords     TEXT,
                created_at   TEXT NOT NULL,
                compressed   INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS compressed_memory (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL,
                source_episodic_id  INTEGER,
                summary             TEXT NOT NULL,
                token_count         INTEGER NOT NULL,
                keywords            TEXT,
                topic               TEXT,
                relevance_score     REAL DEFAULT 0.0,
                created_at          TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_episodic_session
                ON episodic_buffer(session_id, compressed);
            CREATE INDEX IF NOT EXISTS idx_compressed_session
                ON compressed_memory(session_id);
            CREATE INDEX IF NOT EXISTS idx_subq_session
                ON sub_questions(session_id);
        """)
        self.conn.commit()

    # ── Session management ──────────────────────────────────────────

    def create_session(self, session_id: str, query: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO sessions (session_id, query, started_at) VALUES (?, ?, ?)",
            (session_id, query, now),
        )
        self.conn.commit()

    def update_session_cost(self, session_id: str, input_tokens: int,
                            output_tokens: int, cost: float):
        self.conn.execute(
            """UPDATE sessions
               SET total_input_tokens  = total_input_tokens  + ?,
                   total_output_tokens = total_output_tokens + ?,
                   estimated_cost_usd  = estimated_cost_usd  + ?
             WHERE session_id = ?""",
            (input_tokens, output_tokens, cost, session_id),
        )
        self.conn.commit()

    def get_session_cost(self, session_id: str) -> float:
        row = self.conn.execute(
            "SELECT estimated_cost_usd FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["estimated_cost_usd"] if row else 0.0

    def complete_session(self, session_id: str, final_answer: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE sessions
               SET status = 'completed', completed_at = ?, final_answer = ?
             WHERE session_id = ?""",
            (now, final_answer, session_id),
        )
        self.conn.commit()

    def get_session_summary(self, session_id: str) -> dict:
        session = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not session:
            return {}

        episodic_count = self.conn.execute(
            "SELECT COUNT(*) AS c FROM episodic_buffer WHERE session_id = ?",
            (session_id,),
        ).fetchone()["c"]

        compressed_count = self.conn.execute(
            "SELECT COUNT(*) AS c FROM compressed_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()["c"]

        subq_count = self.conn.execute(
            "SELECT COUNT(*) AS c FROM sub_questions WHERE session_id = ?",
            (session_id,),
        ).fetchone()["c"]

        return {
            **dict(session),
            "episodic_entries": episodic_count,
            "compressed_entries": compressed_count,
            "sub_questions_count": subq_count,
        }

    def get_all_sessions(self) -> list:
        rows = self.conn.execute(
            """SELECT session_id, query, started_at, status, estimated_cost_usd
                 FROM sessions ORDER BY started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Sub-questions ───────────────────────────────────────────────

    def store_sub_questions(self, session_id: str, questions: list):
        now = datetime.now(timezone.utc).isoformat()
        for i, q in enumerate(questions):
            self.conn.execute(
                """INSERT INTO sub_questions
                   (session_id, question, order_index, created_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, q, i, now),
            )
        self.conn.commit()

    def update_sub_question_status(self, session_id: str, order_index: int,
                                    status: str):
        self.conn.execute(
            """UPDATE sub_questions SET status = ?
             WHERE session_id = ? AND order_index = ?""",
            (status, session_id, order_index),
        )
        self.conn.commit()

    # ── Episodic buffer ─────────────────────────────────────────────

    def add_episodic_entry(self, session_id: str, sub_question: str,
                           finding: str, token_count: int,
                           keywords: list) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            """INSERT INTO episodic_buffer
               (session_id, sub_question, finding, token_count, keywords, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, sub_question, finding, token_count,
             json.dumps(keywords), now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_active_episodic_entries(self, session_id: str) -> list:
        rows = self.conn.execute(
            """SELECT * FROM episodic_buffer
             WHERE session_id = ? AND compressed = 0
             ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_episodic_compressed(self, entry_id: int):
        self.conn.execute(
            "UPDATE episodic_buffer SET compressed = 1 WHERE id = ?",
            (entry_id,),
        )
        self.conn.commit()

    # ── Compressed memory ───────────────────────────────────────────

    def add_compressed_memory(self, session_id: str, summary: str,
                              token_count: int, keywords: list,
                              topic: str,
                              source_episodic_id: int = None):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO compressed_memory
               (session_id, source_episodic_id, summary, token_count,
                keywords, topic, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, source_episodic_id, summary, token_count,
             json.dumps(keywords), topic, now),
        )
        self.conn.commit()

    def get_all_compressed_memories(self, session_id: str) -> list:
        rows = self.conn.execute(
            """SELECT * FROM compressed_memory
             WHERE session_id = ?
             ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()