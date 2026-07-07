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

-- Node stages: one row per node invocation on a turn
CREATE TABLE IF NOT EXISTS node_stages (
    stage_id            TEXT PRIMARY KEY,
    turn_id             TEXT NOT NULL,
    session_id          TEXT,
    conversation_id     TEXT,
    node_name           TEXT NOT NULL,
    node_class          TEXT NOT NULL,
    config_label        TEXT NOT NULL,
    start_wall_ms       REAL NOT NULL,
    end_wall_ms         REAL,
    timing_json         TEXT,
    summary_json        TEXT,
    FOREIGN KEY(turn_id) REFERENCES turns(id)
);

-- Pipe events (extended)
CREATE TABLE IF NOT EXISTS pipe_events (
    event_id                  TEXT PRIMARY KEY,
    type                      TEXT,
    node_name                 TEXT,
    turn_id                   TEXT,
    timestamp_wallclock_ms    REAL,
    timestamp_local_offset    REAL,
    payload_json              TEXT,
    parent_event_id           TEXT,
    stage_id                  TEXT,
    seq                       INTEGER
);

-- Inter-stage events
CREATE TABLE IF NOT EXISTS inter_stage_events (
    event_id            TEXT PRIMARY KEY,
    turn_id             TEXT NOT NULL,
    session_id          TEXT,
    conversation_id     TEXT,
    event_type          TEXT NOT NULL,
    wallclock_ms        REAL NOT NULL,
    local_offset_ms     REAL,
    seq                 INTEGER,
    from_stage_id       TEXT,
    to_stage_id         TEXT,
    payload_json        TEXT NOT NULL,
    UNIQUE(turn_id, seq)
);

-- Keep-alive pings
CREATE TABLE IF NOT EXISTS keep_warm_pings (
    ping_id             TEXT PRIMARY KEY,
    node_name           TEXT NOT NULL,
    wallclock_ms        REAL NOT NULL,
    rtt_ms              REAL NOT NULL,
    parent_turn_id      TEXT,
    FOREIGN KEY(parent_turn_id) REFERENCES turns(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pipe_events_stage ON pipe_events(stage_id);
CREATE INDEX IF NOT EXISTS idx_inter_stage_turn ON inter_stage_events(turn_id, seq);
CREATE INDEX IF NOT EXISTS idx_node_stages_turn ON node_stages(turn_id);
CREATE INDEX IF NOT EXISTS idx_node_stages_config ON node_stages(config_label);
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
        # Migrate pipe_events for stage_id/seq support
        for col, coltype in (
            ("stage_id", "TEXT"),
            ("seq", "INTEGER"),
        ):
            try:
                _conn.execute(f"ALTER TABLE pipe_events ADD COLUMN {col} {coltype}")
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


def log_pipe_event(event: dict[str, Any]) -> None:
    conn = get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO pipe_events "
            "(event_id, type, node_name, turn_id, timestamp_wallclock_ms, "
            "timestamp_local_offset, payload_json, parent_event_id, stage_id, seq) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                event["event_id"],
                event["type"],
                event["node_name"],
                event["turn_id"],
                event["timestamp_wallclock_ms"],
                event["timestamp_local_offset"],
                json.dumps(event["payload"]),
                event.get("parent_event_id"),
                event.get("stage_id"),
                event.get("seq"),
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


def log_node_event(event: dict[str, Any]) -> None:
    """Save a NodeEvent to the pipe_events table with stage_id and seq."""
    conn = get_conn()
    with _lock:
        conn.execute(
            """
            INSERT INTO pipe_events(event_id, type, node_name, turn_id, timestamp_wallclock_ms,
                                    timestamp_local_offset, payload_json, parent_event_id, stage_id, seq)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("event_id"),
                event.get("event_type") or event.get("type"),
                event.get("node_name"),
                event.get("turn_id"),
                event.get("wallclock_ms") or event.get("timestamp_wallclock_ms"),
                event.get("local_offset_ms") or event.get("timestamp_local_offset"),
                json.dumps(event.get("payload", {})),
                event.get("parent_event_id"),
                event.get("stage_id"),
                event.get("seq"),
            ),
        )
        conn.commit()


def insert_inter_stage_event(event: dict[str, Any]) -> None:
    """Save an InterStageEvent to the inter_stage_events table."""
    conn = get_conn()
    with _lock:
        conn.execute(
            """
            INSERT INTO inter_stage_events(event_id, turn_id, session_id, conversation_id,
                                          event_type, wallclock_ms, local_offset_ms, seq,
                                          from_stage_id, to_stage_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("event_id"),
                event.get("turn_id"),
                event.get("session_id"),
                event.get("conversation_id"),
                event.get("event_type"),
                event.get("wallclock_ms"),
                event.get("local_offset_ms"),
                event.get("seq"),
                event.get("from_stage_id"),
                event.get("to_stage_id"),
                json.dumps(event.get("payload", {})),
            ),
        )
        conn.commit()


def log_keep_warm_ping(
    node_name: str, wallclock_ms: float, rtt_ms: float, parent_turn_id: str = None
) -> None:
    """Save a keepalive ping to the keep_warm_pings table."""
    import uuid
    ping_id = str(uuid.uuid4())
    conn = get_conn()
    with _lock:
        conn.execute(
            "INSERT INTO keep_warm_pings(ping_id, node_name, wallclock_ms, rtt_ms, parent_turn_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (ping_id, node_name, wallclock_ms, rtt_ms, parent_turn_id),
        )
        conn.commit()


def log_node_stage(record: dict[str, Any]) -> None:
    """Save a NodeStageRecord to the node_stages table."""
    conn = get_conn()
    with _lock:
        conn.execute(
            """
            INSERT INTO node_stages(stage_id, turn_id, session_id, conversation_id, node_name,
                                   node_class, config_label, start_wall_ms, end_wall_ms,
                                   timing_json, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stage_id) DO UPDATE SET
                end_wall_ms=excluded.end_wall_ms,
                timing_json=excluded.timing_json,
                summary_json=excluded.summary_json
            """,
            (
                record.get("stage_id"),
                record.get("turn_id"),
                record.get("session_id"),
                record.get("conversation_id"),
                record.get("node_name"),
                record.get("node_class"),
                record.get("config_label"),
                record.get("start_wall_ms"),
                record.get("end_wall_ms"),
                json.dumps(record.get("timing", {})),
                json.dumps(record.get("summary", {})),
            ),
        )
        conn.commit()


def get_node_stages(turn_id: str) -> list[dict[str, Any]]:
    """Return all stage records for a turn."""
    return _rows(
        "SELECT * FROM node_stages WHERE turn_id=? ORDER BY start_wall_ms",
        (turn_id,),
    )


def get_inter_stage_events(turn_id: str) -> list[dict[str, Any]]:
    """Return all inter-stage events for a turn."""
    return _rows(
        "SELECT * FROM inter_stage_events WHERE turn_id=? ORDER BY seq",
        (turn_id,),
    )


def get_keep_warm_pings(parent_turn_id: str = None) -> list[dict[str, Any]]:
    """Return keep-alive pings, optionally filtered by parent turn."""
    if parent_turn_id:
        return _rows(
            "SELECT * FROM keep_warm_pings WHERE parent_turn_id=? ORDER BY wallclock_ms",
            (parent_turn_id,),
        )
    return _rows(
        "SELECT * FROM keep_warm_pings ORDER BY wallclock_ms",
    )
