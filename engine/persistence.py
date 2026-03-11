"""
VØX Persistence Layer
---------------------

SQLite-based persistence for voice registries, consent records, and audit logs.
Redis-compatible rate limiting with fallback to SQLite.

This replaces the in-memory dictionaries with proper persistent storage.
"""

from __future__ import annotations

import sqlite3
import json
import time
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

# Try Redis, fall back to SQLite-based rate limiting
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class VoxDatabase:
    """
    SQLite database for VØX registries.

    Tables:
        - voices: Voice registry (id, category, owner, consent, allowed_uses)
        - consents: Consent records with proof
        - blocked_voices: Permanently blocked voices
        - audit_log: All governance decisions
    """

    def __init__(self, db_path: str = "~/.axiom_vox/vox.db"):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self.transaction() as conn:
            conn.executescript("""
                -- Voice Registry
                CREATE TABLE IF NOT EXISTS voices (
                    voice_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    owner_id TEXT,
                    consent_verified INTEGER DEFAULT 0,
                    allowed_uses TEXT,  -- JSON array
                    metadata TEXT,      -- JSON object
                    created_at REAL,
                    updated_at REAL
                );

                -- Consent Records
                CREATE TABLE IF NOT EXISTS consents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    voice_id TEXT NOT NULL,
                    consent_type TEXT NOT NULL,
                    proof_hash TEXT,
                    granted_by TEXT,
                    granted_at REAL,
                    expires_at REAL,
                    revoked INTEGER DEFAULT 0,
                    revoked_at REAL,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_consents_voice ON consents(voice_id);

                -- Blocked Voices
                CREATE TABLE IF NOT EXISTS blocked_voices (
                    voice_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    blocked_by TEXT,
                    blocked_at REAL,
                    permanent INTEGER DEFAULT 1
                );

                -- Audit Log
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    request_id TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    voice_id TEXT,
                    action TEXT NOT NULL,
                    passed INTEGER,
                    api_key_hash TEXT,
                    request_data TEXT,   -- JSON
                    result_data TEXT,    -- JSON
                    violations TEXT      -- JSON array
                );
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_voice ON audit_log(voice_id);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

                -- Rate Limiting (fallback when Redis unavailable)
                CREATE TABLE IF NOT EXISTS rate_limits (
                    key TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (key, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_rate_key ON rate_limits(key);

                -- Fine-Tuning Jobs
                CREATE TABLE IF NOT EXISTS finetune_jobs (
                    job_id TEXT PRIMARY KEY,
                    voice_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    sample_count INTEGER,
                    total_duration_seconds REAL,
                    consent_verified INTEGER DEFAULT 0,
                    requestor_id TEXT,
                    adapter_path TEXT,
                    run_id TEXT,
                    final_loss REAL,
                    epochs_completed INTEGER,
                    verification_passed INTEGER,
                    verification_score REAL,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_voice ON finetune_jobs(voice_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON finetune_jobs(status);

                -- Fine-Tuning Samples
                CREATE TABLE IF NOT EXISTS finetune_samples (
                    sample_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    file_path TEXT,
                    file_hash TEXT,
                    duration_seconds REAL,
                    sample_rate INTEGER,
                    processed INTEGER DEFAULT 0,
                    quality_score REAL,
                    created_at REAL,
                    FOREIGN KEY (job_id) REFERENCES finetune_jobs(job_id)
                );
                CREATE INDEX IF NOT EXISTS idx_samples_job ON finetune_samples(job_id);

                -- LoRA Adapters
                CREATE TABLE IF NOT EXISTS lora_adapters (
                    adapter_id TEXT PRIMARY KEY,
                    voice_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL,
                    adapter_path TEXT NOT NULL,
                    lora_rank INTEGER,
                    lora_alpha REAL,
                    parameter_count INTEGER,
                    inference_latency_ms REAL,
                    quality_score REAL,
                    is_active INTEGER DEFAULT 1,
                    created_at REAL,
                    last_used_at REAL,
                    FOREIGN KEY (job_id) REFERENCES finetune_jobs(job_id)
                );
                CREATE INDEX IF NOT EXISTS idx_adapters_voice ON lora_adapters(voice_id);

                -- Biometric Templates (v0.10.0)
                CREATE TABLE IF NOT EXISTS biometric_templates (
                    template_id TEXT PRIMARY KEY,
                    voice_id TEXT NOT NULL UNIQUE,
                    embedding BLOB NOT NULL,
                    embedding_version TEXT NOT NULL,
                    embedding_backend TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    confidence REAL,
                    consent_id INTEGER,
                    owner_id TEXT,
                    enrolled_at REAL NOT NULL,
                    updated_at REAL,
                    is_active INTEGER DEFAULT 1,
                    revoked INTEGER DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (voice_id) REFERENCES voices(voice_id)
                );
                CREATE INDEX IF NOT EXISTS idx_biometric_voice ON biometric_templates(voice_id);
                CREATE INDEX IF NOT EXISTS idx_biometric_active ON biometric_templates(is_active);

                -- Embedding History (for drift analysis)
                CREATE TABLE IF NOT EXISTS embedding_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    voice_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    timestamp REAL NOT NULL,
                    context TEXT,
                    similarity_to_template REAL,
                    quality_score REAL
                );
                CREATE INDEX IF NOT EXISTS idx_emb_history_voice ON embedding_history(voice_id);
                CREATE INDEX IF NOT EXISTS idx_emb_history_time ON embedding_history(timestamp);

                -- Biometric Audit Log
                CREATE TABLE IF NOT EXISTS biometric_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    voice_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    similarity_score REAL,
                    liveness_score REAL,
                    liveness_passed INTEGER,
                    drift_detected INTEGER DEFAULT 0,
                    error_message TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_bio_audit_voice ON biometric_audit(voice_id);
                CREATE INDEX IF NOT EXISTS idx_bio_audit_time ON biometric_audit(timestamp);
                CREATE INDEX IF NOT EXISTS idx_bio_audit_action ON biometric_audit(action);
            """)

    # ========================================================================
    # VOICE REGISTRY
    # ========================================================================

    def register_voice(
        self,
        voice_id: str,
        category: str,
        owner_id: Optional[str] = None,
        consent_verified: bool = False,
        allowed_uses: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a voice in the database."""
        now = time.time()
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO voices
                (voice_id, category, owner_id, consent_verified, allowed_uses, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM voices WHERE voice_id = ?), ?), ?)
            """, (
                voice_id,
                category,
                owner_id,
                1 if consent_verified else 0,
                json.dumps(allowed_uses or ["general"]),
                json.dumps(metadata or {}),
                voice_id,
                now,
                now,
            ))

    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get voice registration details."""
        row = self.conn.execute(
            "SELECT * FROM voices WHERE voice_id = ?", (voice_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "voice_id": row["voice_id"],
            "category": row["category"],
            "owner_id": row["owner_id"],
            "consent_verified": bool(row["consent_verified"]),
            "allowed_uses": json.loads(row["allowed_uses"] or "[]"),
            "metadata": json.loads(row["metadata"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_voices(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered voices."""
        if category:
            rows = self.conn.execute(
                "SELECT * FROM voices WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM voices").fetchall()

        return [
            {
                "voice_id": row["voice_id"],
                "category": row["category"],
                "consent_verified": bool(row["consent_verified"]),
            }
            for row in rows
        ]

    # ========================================================================
    # CONSENT REGISTRY
    # ========================================================================

    def grant_consent(
        self,
        voice_id: str,
        consent_type: str = "usage",
        proof: Optional[str] = None,
        granted_by: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Grant consent for a voice."""
        now = time.time()
        expires_at = None
        if expires_in_days:
            expires_at = now + (expires_in_days * 86400)

        proof_hash = None
        if proof:
            proof_hash = hashlib.sha256(proof.encode()).hexdigest()

        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO consents
                (voice_id, consent_type, proof_hash, granted_by, granted_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                voice_id,
                consent_type,
                proof_hash,
                granted_by,
                now,
                expires_at,
                json.dumps(metadata or {}),
            ))

            # Update voice registry
            conn.execute("""
                UPDATE voices SET consent_verified = 1, updated_at = ?
                WHERE voice_id = ?
            """, (now, voice_id))

            return cursor.lastrowid

    def check_consent(self, voice_id: str) -> bool:
        """Check if voice has valid (non-expired, non-revoked) consent."""
        now = time.time()
        row = self.conn.execute("""
            SELECT COUNT(*) as cnt FROM consents
            WHERE voice_id = ?
              AND revoked = 0
              AND (expires_at IS NULL OR expires_at > ?)
        """, (voice_id, now)).fetchone()

        return row["cnt"] > 0

    def revoke_consent(self, voice_id: str, consent_id: Optional[int] = None) -> None:
        """Revoke consent for a voice."""
        now = time.time()
        with self.transaction() as conn:
            if consent_id:
                conn.execute("""
                    UPDATE consents SET revoked = 1, revoked_at = ?
                    WHERE id = ?
                """, (now, consent_id))
            else:
                conn.execute("""
                    UPDATE consents SET revoked = 1, revoked_at = ?
                    WHERE voice_id = ?
                """, (now, voice_id))

            # Update voice registry if all consents revoked
            remaining = conn.execute("""
                SELECT COUNT(*) as cnt FROM consents
                WHERE voice_id = ? AND revoked = 0
            """, (voice_id,)).fetchone()

            if remaining["cnt"] == 0:
                conn.execute("""
                    UPDATE voices SET consent_verified = 0, updated_at = ?
                    WHERE voice_id = ?
                """, (now, voice_id))

    # ========================================================================
    # BLOCKED VOICES
    # ========================================================================

    def block_voice(
        self,
        voice_id: str,
        reason: str,
        blocked_by: Optional[str] = None,
        permanent: bool = True,
    ) -> None:
        """Block a voice."""
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO blocked_voices
                (voice_id, reason, blocked_by, blocked_at, permanent)
                VALUES (?, ?, ?, ?, ?)
            """, (voice_id, reason, blocked_by, time.time(), 1 if permanent else 0))

    def unblock_voice(self, voice_id: str) -> None:
        """Unblock a voice."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM blocked_voices WHERE voice_id = ?", (voice_id,))

    def is_blocked(self, voice_id: str) -> tuple[bool, Optional[str]]:
        """Check if voice is blocked."""
        row = self.conn.execute(
            "SELECT reason FROM blocked_voices WHERE voice_id = ?", (voice_id,)
        ).fetchone()

        if row:
            return True, row["reason"]
        return False, None

    # ========================================================================
    # AUDIT LOG
    # ========================================================================

    def log_audit(
        self,
        request_id: str,
        request_type: str,
        action: str,
        passed: bool,
        voice_id: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        request_data: Optional[Dict] = None,
        result_data: Optional[Dict] = None,
        violations: Optional[List[str]] = None,
    ) -> int:
        """Log an audit entry."""
        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO audit_log
                (timestamp, request_id, request_type, voice_id, action, passed,
                 api_key_hash, request_data, result_data, violations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                request_id,
                request_type,
                voice_id,
                action,
                1 if passed else 0,
                api_key_hash,
                json.dumps(request_data) if request_data else None,
                json.dumps(result_data) if result_data else None,
                json.dumps(violations) if violations else None,
            ))
            return cursor.lastrowid

    def get_audit_log(
        self,
        voice_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query audit log."""
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)
        if action:
            query += " AND action = ?"
            params.append(action)
        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "request_id": row["request_id"],
                "request_type": row["request_type"],
                "voice_id": row["voice_id"],
                "action": row["action"],
                "passed": bool(row["passed"]),
            }
            for row in rows
        ]

    # ========================================================================
    # FINE-TUNING JOBS
    # ========================================================================

    def create_finetune_job(
        self,
        voice_id: str,
        sample_count: int,
        total_duration: float,
        consent_verified: bool,
        requestor_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Create a new fine-tuning job."""
        import uuid
        if not job_id:
            job_id = f"ft_{uuid.uuid4().hex[:12]}"

        now = time.time()
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO finetune_jobs
                (job_id, voice_id, status, created_at, sample_count,
                 total_duration_seconds, consent_verified, requestor_id)
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)
            """, (
                job_id, voice_id, now, sample_count,
                total_duration, 1 if consent_verified else 0, requestor_id
            ))

        return job_id

    def update_job_status(
        self,
        job_id: str,
        status: str,
        final_loss: Optional[float] = None,
        epochs_completed: Optional[int] = None,
        verification_passed: Optional[bool] = None,
        adapter_path: Optional[str] = None,
        run_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status and optional fields."""
        now = time.time()

        updates = ["status = ?", "completed_at = ?"]
        params = [status, now]

        if final_loss is not None:
            updates.append("final_loss = ?")
            params.append(final_loss)
        if epochs_completed is not None:
            updates.append("epochs_completed = ?")
            params.append(epochs_completed)
        if verification_passed is not None:
            updates.append("verification_passed = ?")
            params.append(1 if verification_passed else 0)
        if adapter_path is not None:
            updates.append("adapter_path = ?")
            params.append(adapter_path)
        if run_id is not None:
            updates.append("run_id = ?")
            params.append(run_id)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(job_id)

        with self.transaction() as conn:
            conn.execute(f"""
                UPDATE finetune_jobs
                SET {', '.join(updates)}
                WHERE job_id = ?
            """, params)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details."""
        row = self.conn.execute(
            "SELECT * FROM finetune_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "job_id": row["job_id"],
            "voice_id": row["voice_id"],
            "status": row["status"],
            "created_at": datetime.fromtimestamp(row["created_at"]).isoformat() if row["created_at"] else None,
            "started_at": datetime.fromtimestamp(row["started_at"]).isoformat() if row["started_at"] else None,
            "completed_at": datetime.fromtimestamp(row["completed_at"]).isoformat() if row["completed_at"] else None,
            "sample_count": row["sample_count"],
            "total_duration_seconds": row["total_duration_seconds"],
            "consent_verified": bool(row["consent_verified"]),
            "adapter_path": row["adapter_path"],
            "run_id": row["run_id"],
            "final_loss": row["final_loss"],
            "epochs_completed": row["epochs_completed"],
            "verification_passed": bool(row["verification_passed"]) if row["verification_passed"] is not None else None,
            "error_message": row["error_message"],
        }

    def list_jobs(
        self,
        voice_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List fine-tuning jobs."""
        query = "SELECT * FROM finetune_jobs WHERE 1=1"
        params = []

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        return [
            {
                "job_id": row["job_id"],
                "voice_id": row["voice_id"],
                "status": row["status"],
                "created_at": datetime.fromtimestamp(row["created_at"]).isoformat() if row["created_at"] else None,
                "final_loss": row["final_loss"],
                "verification_passed": bool(row["verification_passed"]) if row["verification_passed"] is not None else None,
            }
            for row in rows
        ]

    # ========================================================================
    # LORA ADAPTERS
    # ========================================================================

    def register_adapter(
        self,
        voice_id: str,
        job_id: str,
        adapter_path: str,
        lora_rank: int,
        lora_alpha: float,
        parameter_count: int,
    ) -> str:
        """Register a trained LoRA adapter."""
        adapter_id = f"lora_{voice_id}"
        now = time.time()

        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lora_adapters
                (adapter_id, voice_id, job_id, adapter_path, lora_rank, lora_alpha,
                 parameter_count, is_active, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                adapter_id, voice_id, job_id, adapter_path,
                lora_rank, lora_alpha, parameter_count, now, now
            ))

        return adapter_id

    def get_adapter(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get adapter for a voice."""
        row = self.conn.execute(
            "SELECT * FROM lora_adapters WHERE voice_id = ? AND is_active = 1",
            (voice_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "adapter_id": row["adapter_id"],
            "voice_id": row["voice_id"],
            "job_id": row["job_id"],
            "adapter_path": row["adapter_path"],
            "lora_rank": row["lora_rank"],
            "lora_alpha": row["lora_alpha"],
            "parameter_count": row["parameter_count"],
            "quality_score": row["quality_score"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
        }

    def update_adapter_last_used(self, voice_id: str) -> None:
        """Update last used timestamp for an adapter."""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE lora_adapters SET last_used_at = ?
                WHERE voice_id = ?
            """, (time.time(), voice_id))

    def list_adapters(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all registered adapters."""
        query = "SELECT * FROM lora_adapters"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY created_at DESC"

        rows = self.conn.execute(query).fetchall()

        return [
            {
                "adapter_id": row["adapter_id"],
                "voice_id": row["voice_id"],
                "adapter_path": row["adapter_path"],
                "lora_rank": row["lora_rank"],
                "parameter_count": row["parameter_count"],
                "quality_score": row["quality_score"],
            }
            for row in rows
        ]

    def deactivate_adapter(self, voice_id: str) -> None:
        """Deactivate an adapter (soft delete)."""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE lora_adapters SET is_active = 0
                WHERE voice_id = ?
            """, (voice_id,))


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Rate limiter with Redis backend (falls back to SQLite).

    Tracks requests per key within a sliding window.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        redis_url: Optional[str] = None,
        db: Optional[VoxDatabase] = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.redis_client = None
        self.db = db

        # Try Redis first
        if redis_url and HAS_REDIS:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None

    def check(self, key: str) -> bool:
        """
        Check if request is allowed.

        Returns True if under rate limit, False if exceeded.
        """
        if self.redis_client:
            return self._check_redis(key)
        elif self.db:
            return self._check_sqlite(key)
        else:
            return True  # No persistence, allow all

    def _check_redis(self, key: str) -> bool:
        """Rate limit check using Redis."""
        now = time.time()
        window_start = now - self.window_seconds
        redis_key = f"vox:rate:{key}"

        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zadd(redis_key, {str(now): now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, self.window_seconds + 1)
        results = pipe.execute()

        count = results[2]
        return count <= self.requests_per_minute

    def _check_sqlite(self, key: str) -> bool:
        """Rate limit check using SQLite."""
        now = time.time()
        window_start = now - self.window_seconds

        with self.db.transaction() as conn:
            # Clean old entries
            conn.execute(
                "DELETE FROM rate_limits WHERE key = ? AND timestamp < ?",
                (key, window_start)
            )

            # Count current window
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM rate_limits WHERE key = ?",
                (key,)
            ).fetchone()

            if row["cnt"] >= self.requests_per_minute:
                return False

            # Add new entry
            conn.execute(
                "INSERT INTO rate_limits (key, timestamp) VALUES (?, ?)",
                (key, now)
            )

            return True


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_db_instance: Optional[VoxDatabase] = None
_rate_limiter: Optional[RateLimiter] = None


def get_database(db_path: str = "~/.axiom_vox/vox.db") -> VoxDatabase:
    """Get or create the database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = VoxDatabase(db_path)
    return _db_instance


def get_rate_limiter(
    requests_per_minute: int = 60,
    redis_url: Optional[str] = None,
) -> RateLimiter:
    """Get or create the rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            redis_url=redis_url or os.environ.get("REDIS_URL"),
            db=get_database(),
        )
    return _rate_limiter
