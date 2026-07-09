"""LogManager: event queue with background flusher.

Two modes:
  - DB mode:  queue → batch INSERT into SQLite (background thread)
  - Console mode: queue → formatted print to stdout

Usage:
    # DB mode
    log = LogManager(db_path="data/sonication.db")
    log.start()
    log.enqueue({"type": "turn_start", ...})
    log.shutdown()

    # Console mode
    log = LogManager(mode="console")
    log.start()
    log.enqueue({"type": "turn_start", ...})
    log.shutdown()

    # No logging at all
    pipe = HotPipe(..., log_manager=None)  # drops all logging
"""
import asyncio
import json
import queue
import sqlite3
import threading
import time
from typing import Any, Optional


class LogManager:
    """Event queue with background flusher.

    Args:
        db_path: SQLite database path. Mutually exclusive with mode="console".
        mode: "db" (default) or "console" for stdout logging.
        max_queue: Maximum queue length. Raises RuntimeError if exceeded.
        flush_interval_s: How often the flusher thread runs (default 0.01s = 10ms).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        mode: str = "db",
        max_queue: int = 100_000,
        flush_interval_s: float = 0.01,
    ):
        if mode == "console":
            self._mode = "console"
            self._db_path = None
        elif mode == "db":
            if not db_path:
                raise ValueError("db_path required when mode='db'")
            self._mode = "db"
            self._db_path = db_path
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'db' or 'console'.")

        self._max_queue = max_queue
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._conn: Optional[sqlite3.Connection] = None
        self._flusher_task: Optional[threading.Thread] = None
        self._running = False

        # Global auto instance for backward compat
        self._auto_instance: Optional["LogManager"] = None

    # ------------------------------------------------------------------ init
    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        import os
        from sonication import config

        path = self._db_path or config.DB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start(self) -> None:
        """Start the background flusher thread and create tables."""
        if self._running:
            return
        if self._mode == "db":
            self._init_db()
        self._running = True
        self._flusher_task = threading.Thread(
            target=self._flush_loop, daemon=True, name="sonication-log-flusher"
        )
        self._flusher_task.start()

    def shutdown(self) -> None:
        """Drain queue, stop flusher, close DB connection."""
        if not self._running:
            return
        self._running = False
        if self._flusher_task:
            self._flusher_task.join(timeout=5.0)
            self._flusher_task = None
        # Flush remaining events
        self._flush_batch()
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------ queue
    def enqueue(self, event: dict[str, Any]) -> None:
        """Add an event to the queue. Non-blocking.

        Raises RuntimeError if queue is at max capacity.
        """
        if not self._running:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            raise RuntimeError(
                f"LogManager queue full ({self._max_queue} events). "
                "Events are being lost — check flush performance."
            )

    # ------------------------------------------------------------------ flush
    def _flush_loop(self) -> None:
        """Background thread: wake every flush_interval, grab batch, flush."""
        while self._running:
            self._flush_batch()
            # Use wait with timeout so we can exit quickly on shutdown
            self._queue.join()
            time.sleep(self._flush_interval_s)

    def _flush_batch(self) -> None:
        """Grab all queued events and write them."""
        batch: list[dict[str, Any]] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not batch:
            return

        if self._mode == "console":
            self._flush_console(batch)
        else:
            self._flush_db(batch)

        for _ in batch:
            self._queue.task_done()

    def _flush_console(self, batch: list[dict[str, Any]]) -> None:
        """Print events to stdout with formatting."""
        for event in batch:
            ts = time.strftime("%H:%M:%S", time.localtime(event.get("timestamp", time.time())))
            etype = event.get("type", event.get("event_type", "unknown"))
            node = event.get("node_name", "")
            turn = event.get("turn_id", "")
            payload = event.get("payload", {})

            # Truncate payload for readability
            if isinstance(payload, dict):
                parts = []
                for k, v in payload.items():
                    if k in ("pcm",):
                        continue
                    s = f"{k}={v!r}" if not isinstance(v, str) or len(v) < 100 else f"{k}={v[:80]}..."
                    parts.append(s)
                payload_str = ", ".join(parts)
            else:
                payload_str = str(payload)

            print(f"[{ts}] {etype} | node={node} turn={turn} | {payload_str}")

    def _flush_db(self, batch: list[dict[str, Any]]) -> None:
        """Batch INSERT events into SQLite using thread-local connection."""
        if not self._conn:
            return

        def _make_payload_json_safe(payload: dict) -> dict:
            """Convert bytes to base64 in payload for JSON serialization."""
            import base64
            safe = {}
            for k, v in payload.items():
                if isinstance(v, bytes):
                    safe[k] = base64.b64encode(v).decode("ascii")
                elif isinstance(v, dict):
                    safe[k] = _make_payload_json_safe(v)
                else:
                    safe[k] = v
            return safe

        # Separate events by table
        pipe_events = []
        node_stages = []
        inter_stage_events = []
        keep_warm_pings = []
        node_events = []

        for event in batch:
            kind = event.get("_log_kind", "pipe_event")
            if kind == "pipe_event":
                pipe_events.append(event)
            elif kind == "node_stage":
                node_stages.append(event)
            elif kind == "inter_stage":
                inter_stage_events.append(event)
            elif kind == "keep_warm_ping":
                keep_warm_pings.append(event)
            elif kind == "node_event":
                node_events.append(event)

        # Bulk insert pipe_events
        if pipe_events:
            self._conn.executemany(
                """INSERT INTO pipe_events
                   (event_id, type, node_name, turn_id, timestamp_wallclock_ms,
                    timestamp_local_offset, payload_json, parent_event_id, stage_id, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        e.get("event_id"),
                        e.get("type") or e.get("event_type"),
                        e.get("node_name"),
                        e.get("turn_id"),
                        e.get("timestamp_wallclock_ms") or e.get("wallclock_ms"),
                        e.get("timestamp_local_offset") or e.get("local_offset_ms"),
                        json.dumps(_make_payload_json_safe(e.get("payload", {}))),
                        e.get("parent_event_id"),
                        e.get("stage_id"),
                        e.get("seq"),
                    )
                    for e in pipe_events
                ],
            )

        # Bulk insert node_stages
        if node_stages:
            self._conn.executemany(
                """INSERT OR REPLACE INTO node_stages
                   (stage_id, turn_id, session_id, conversation_id, node_name,
                    node_class, config_label, start_wall_ms, end_wall_ms,
                    timing_json, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        e.get("stage_id"),
                        e.get("turn_id"),
                        e.get("session_id"),
                        e.get("conversation_id"),
                        e.get("node_name"),
                        e.get("node_class"),
                        e.get("config_label"),
                        e.get("start_wall_ms"),
                        e.get("end_wall_ms"),
                        json.dumps(e.get("timing", {})),
                        json.dumps(e.get("summary", {})),
                    )
                    for e in node_stages
                ],
            )

        # Bulk insert inter_stage_events
        if inter_stage_events:
            self._conn.executemany(
                """INSERT OR REPLACE INTO inter_stage_events
                   (event_id, turn_id, session_id, conversation_id, event_type,
                    wallclock_ms, local_offset_ms, seq, from_stage_id, to_stage_id,
                    payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        e.get("event_id"),
                        e.get("turn_id"),
                        e.get("session_id"),
                        e.get("conversation_id"),
                        e.get("event_type"),
                        e.get("wallclock_ms"),
                        e.get("local_offset_ms"),
                        e.get("seq"),
                        e.get("from_stage_id"),
                        e.get("to_stage_id"),
                        json.dumps(_make_payload_json_safe(e.get("payload", {}))),
                    )
                    for e in inter_stage_events
                ],
            )

        # Bulk insert keep_warm_pings
        if keep_warm_pings:
            self._conn.executemany(
                """INSERT INTO keep_warm_pings
                   (ping_id, node_name, wallclock_ms, rtt_ms, parent_turn_id)
                VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        e.get("ping_id"),
                        e.get("node_name"),
                        e.get("wallclock_ms"),
                        e.get("rtt_ms"),
                        e.get("parent_turn_id"),
                    )
                    for e in keep_warm_pings
                ],
            )

        # Bulk insert node_events
        if node_events:
            self._conn.executemany(
                """INSERT INTO pipe_events
                   (event_id, type, node_name, turn_id, timestamp_wallclock_ms,
                    timestamp_local_offset, payload_json, parent_event_id, stage_id, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        e.get("event_id"),
                        e.get("event_type") or e.get("type"),
                        e.get("node_name"),
                        e.get("turn_id"),
                        e.get("wallclock_ms") or e.get("timestamp_wallclock_ms"),
                        e.get("local_offset_ms") or e.get("timestamp_local_offset"),
                        json.dumps(_make_payload_json_safe(e.get("payload", {}))),
                        e.get("parent_event_id"),
                        e.get("stage_id"),
                        e.get("seq"),
                    )
                    for e in node_events
                ],
            )

        self._conn.commit()

    # ------------------------------------------------------------------ queries
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a SELECT query and return results as dicts."""
        if self._mode == "console":
            raise RuntimeError("Console mode does not support queries.")
        if not self._conn:
            return []
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    @property
    def tables(self) -> list[str]:
        """Return list of table names in the database."""
        if self._mode == "console":
            return []
        if not self._conn:
            return []
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------ auto
    @classmethod
    def auto_start(cls, db_path: Optional[str] = None, **kwargs) -> "LogManager":
        """Start a global LogManager instance for backward compat.

        Convenience for existing code that uses log_pipe_event() without
        explicit LogManager.
        """
        instance = cls(db_path=db_path, **kwargs)
        instance.start()
        cls._auto_instance = instance
        return instance

    @classmethod
    def auto_shutdown(cls) -> None:
        """Stop the global LogManager instance."""
        if cls._auto_instance:
            cls._auto_instance.shutdown()
            cls._auto_instance = None

    @classmethod
    def auto_instance(cls) -> Optional["LogManager"]:
        """Return the global auto instance, if any."""
        return cls._auto_instance


# ------------------------------------------------------------------ schema
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

CREATE TABLE IF NOT EXISTS keep_warm_pings (
    ping_id             TEXT PRIMARY KEY,
    node_name           TEXT NOT NULL,
    wallclock_ms        REAL NOT NULL,
    rtt_ms              REAL NOT NULL,
    parent_turn_id      TEXT,
    FOREIGN KEY(parent_turn_id) REFERENCES turns(id)
);

CREATE INDEX IF NOT EXISTS idx_pipe_events_stage ON pipe_events(stage_id);
CREATE INDEX IF NOT EXISTS idx_inter_stage_turn ON inter_stage_events(turn_id, seq);
CREATE INDEX IF NOT EXISTS idx_node_stages_turn ON node_stages(turn_id);
CREATE INDEX IF NOT EXISTS idx_node_stages_config ON node_stages(config_label);
"""