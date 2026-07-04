"""SQLite storage for sessions, calls, turns, and events.

Layout:
  sessions -> calls -> turns -> events
A session groups calls. A call is one conversation with call_start as wallclock
zero. A turn (a "shot") is one Enter-to-answer cycle inside a call. Events are
the atomic timed records; per-turn rolled-up metrics live on the turns row so
the analysis space can query them without replaying every event.
"""
import json
import sqlite3
import threading
from typing import Any, Optional

from . import config

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    label             TEXT,
    created_iso       TEXT,
    created_epoch_ms  REAL
);

CREATE TABLE IF NOT EXISTS calls (
    id                TEXT PRIMARY KEY,
    session_id        TEXT REFERENCES sessions(id),
    started_iso       TEXT,
    started_epoch_ms  REAL,
    model             TEXT,
    voice             TEXT,
    language          TEXT,
    params_json       TEXT,
    ended_t_ms        REAL
);

CREATE TABLE IF NOT EXISTS turns (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id                   TEXT REFERENCES calls(id),
    turn_index                INTEGER,
    user_text                 TEXT,
    chars                     INTEGER,
    enter_t_ms                REAL,
    stt_start_t_ms            REAL,
    stt_done_t_ms             REAL,
    stt_ms                    REAL,
    llm_start_t_ms            REAL,
    llm_ttft_ms               REAL,
    llm_done_t_ms             REAL,
    llm_tokens                INTEGER,
    tts_start_t_ms            REAL,
    tts_ttfb_ms               REAL,
    first_audio_channel_t_ms  REAL,
    shot_latency_ms           REAL,
    audio_generated_ms        REAL,
    audio_channel_ms          REAL,
    underrun                  INTEGER,
    metrics_json              TEXT,
    timing_json               TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    seq          INTEGER,
    call_id      TEXT REFERENCES calls(id),
    turn_id      INTEGER REFERENCES turns(id),
    type         TEXT,
    t_ms         REAL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS session_state (
    session_id        TEXT PRIMARY KEY,
    history_json      TEXT,
    message_count     INTEGER,
    last_call_id      TEXT,
    updated_iso       TEXT,
    updated_epoch_ms  REAL
);

CREATE INDEX IF NOT EXISTS idx_events_call ON events(call_id, seq);
CREATE INDEX IF NOT EXISTS idx_turns_call ON turns(call_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session_id);
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.executescript(_SCHEMA)
        # Migrate older databases that predate later columns.
        for col, coltype in (
            ("stt_start_t_ms", "REAL"), ("stt_done_t_ms", "REAL"), ("stt_ms", "REAL"),
            ("timing_json", "TEXT"),
        ):
            try:
                _conn.execute(f"ALTER TABLE turns ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass  # column already exists
        _conn.commit()
    return _conn


def init_db() -> None:
    get_conn()


def create_session(session_id: str, label: str = "") -> None:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    conn = get_conn()
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(id, label, created_iso, created_epoch_ms) "
            "VALUES (?,?,?,?)",
            (session_id, label, now.isoformat(), now.timestamp() * 1000.0),
        )
        conn.commit()


def create_call(
    call_id: str,
    session_id: str,
    started_iso: str,
    started_epoch_ms: float,
    params: dict[str, Any],
) -> None:
    conn = get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO calls(id, session_id, started_iso, started_epoch_ms, "
            "model, voice, language, params_json, ended_t_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                call_id,
                session_id,
                started_iso,
                started_epoch_ms,
                params.get("model"),
                params.get("voice"),
                params.get("language"),
                json.dumps(params),
                None,
            ),
        )
        conn.commit()


def end_call(call_id: str, ended_t_ms: float) -> None:
    conn = get_conn()
    with _lock:
        conn.execute("UPDATE calls SET ended_t_ms=? WHERE id=?", (ended_t_ms, call_id))
        conn.commit()


def create_turn(
    call_id: str, turn_index: int, user_text: str, chars: int, enter_t_ms: float
) -> int:
    conn = get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO turns(call_id, turn_index, user_text, chars, enter_t_ms) "
            "VALUES (?,?,?,?,?)",
            (call_id, turn_index, user_text, chars, enter_t_ms),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_turn(turn_id: int, **fields: Any) -> None:
    if not fields:
        return
    if "metrics" in fields:
        fields["metrics_json"] = json.dumps(fields.pop("metrics"))
    cols = ", ".join(f"{k}=?" for k in fields)
    conn = get_conn()
    with _lock:
        conn.execute(f"UPDATE turns SET {cols} WHERE id=?", (*fields.values(), turn_id))
        conn.commit()


def save_session_state(
    session_id: str, history: list[dict[str, Any]], last_call_id: str
) -> None:
    """Upsert a session's conversation history so it survives eviction/restart."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    conn = get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO session_state"
            "(session_id, history_json, message_count, last_call_id, updated_iso, updated_epoch_ms) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "history_json=excluded.history_json, message_count=excluded.message_count, "
            "last_call_id=excluded.last_call_id, updated_iso=excluded.updated_iso, "
            "updated_epoch_ms=excluded.updated_epoch_ms",
            (session_id, json.dumps(history), len(history), last_call_id,
             now.isoformat(), now.timestamp() * 1000.0),
        )
        conn.commit()


def load_session_state(session_id: str) -> Optional[list[dict[str, Any]]]:
    """Return the stored conversation history for a session, or None."""
    rows = _rows("SELECT history_json FROM session_state WHERE session_id=?", (session_id,))
    if not rows:
        return None
    return json.loads(rows[0]["history_json"] or "[]")


def insert_event(event: dict[str, Any]) -> None:
    conn = get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO events(seq, call_id, turn_id, type, t_ms, payload_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                event["seq"],
                event["call_id"],
                event["turn_id"],
                event["type"],
                event["t_ms"],
                json.dumps(event["payload"]),
            ),
        )
        conn.commit()


# ---- read helpers for the analysis space and JSON export ----

def _rows(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    conn = get_conn()
    with _lock:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def list_calls() -> list[dict[str, Any]]:
    return _rows("SELECT * FROM calls ORDER BY started_epoch_ms")


def get_call(call_id: str) -> Optional[dict[str, Any]]:
    rows = _rows("SELECT * FROM calls WHERE id=?", (call_id,))
    return rows[0] if rows else None


def get_turns(call_id: str) -> list[dict[str, Any]]:
    return _rows("SELECT * FROM turns WHERE call_id=? ORDER BY turn_index", (call_id,))


def get_turn_by_index(call_id: str, turn_index: int) -> Optional[dict[str, Any]]:
    rows = _rows(
        "SELECT * FROM turns WHERE call_id=? AND turn_index=?", (call_id, turn_index)
    )
    return rows[0] if rows else None


def all_turns() -> list[dict[str, Any]]:
    return _rows("SELECT * FROM turns ORDER BY call_id, turn_index")


def get_events(call_id: str) -> list[dict[str, Any]]:
    rows = _rows("SELECT * FROM events WHERE call_id=? ORDER BY seq", (call_id,))
    for r in rows:
        r["payload"] = json.loads(r.pop("payload_json") or "{}")
    return rows
