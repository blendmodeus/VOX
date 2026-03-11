"""
Biometric Storage
-----------------

Database operations for voice biometric templates, embeddings, and audit logs.

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .models import (
    BiometricTemplate,
    BiometricAuditEntry,
    BiometricAction,
    EmbeddingBackend,
    DriftSeverity,
)
from .embeddings import deserialize_embedding, serialize_embedding

logger = logging.getLogger(__name__)


# SQL for biometric tables (to be added to VoxDatabase._init_db)
BIOMETRIC_SCHEMA = """
-- Biometric Templates
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
"""


class BiometricStorage:
    """
    Storage operations for voice biometric data.

    Requires VoxDatabase instance with biometric tables initialized.
    """

    def __init__(self, db):
        """
        Initialize biometric storage.

        Args:
            db: VoxDatabase instance
        """
        self.db = db
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure biometric tables exist."""
        with self.db.transaction() as conn:
            conn.executescript(BIOMETRIC_SCHEMA)

    # ========================================================================
    # TEMPLATE OPERATIONS
    # ========================================================================

    def save_template(self, template: BiometricTemplate) -> str:
        """
        Save a biometric template.

        Args:
            template: BiometricTemplate to save

        Returns:
            template_id
        """
        if not template.template_id:
            template.template_id = f"bio_{uuid.uuid4().hex[:12]}"

        with self.db.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO biometric_templates
                (template_id, voice_id, embedding, embedding_version, embedding_backend,
                 embedding_dim, sample_count, confidence, consent_id, owner_id,
                 enrolled_at, updated_at, is_active, revoked, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template.template_id,
                template.voice_id,
                template.embedding,
                template.embedding_version,
                template.embedding_backend.value if isinstance(template.embedding_backend, EmbeddingBackend) else template.embedding_backend,
                template.embedding_dim,
                template.sample_count,
                template.confidence,
                template.consent_id,
                template.owner_id,
                template.enrolled_at,
                template.updated_at or time.time(),
                1 if template.is_active else 0,
                1 if template.revoked else 0,
                json.dumps(template.metadata),
            ))

        logger.info(f"Saved biometric template: {template.template_id} for voice {template.voice_id}")
        return template.template_id

    def get_template(self, voice_id: str) -> Optional[BiometricTemplate]:
        """
        Get biometric template for a voice.

        Args:
            voice_id: Voice ID to look up

        Returns:
            BiometricTemplate or None
        """
        row = self.db.conn.execute(
            "SELECT * FROM biometric_templates WHERE voice_id = ? AND is_active = 1 AND revoked = 0",
            (voice_id,)
        ).fetchone()

        if not row:
            return None

        return BiometricTemplate(
            template_id=row["template_id"],
            voice_id=row["voice_id"],
            embedding=row["embedding"],
            embedding_version=row["embedding_version"],
            embedding_backend=EmbeddingBackend(row["embedding_backend"]),
            embedding_dim=row["embedding_dim"],
            sample_count=row["sample_count"],
            confidence=row["confidence"],
            consent_id=row["consent_id"],
            owner_id=row["owner_id"],
            enrolled_at=row["enrolled_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
            revoked=bool(row["revoked"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_template_by_id(self, template_id: str) -> Optional[BiometricTemplate]:
        """Get template by template ID."""
        row = self.db.conn.execute(
            "SELECT * FROM biometric_templates WHERE template_id = ?",
            (template_id,)
        ).fetchone()

        if not row:
            return None

        return BiometricTemplate(
            template_id=row["template_id"],
            voice_id=row["voice_id"],
            embedding=row["embedding"],
            embedding_version=row["embedding_version"],
            embedding_backend=EmbeddingBackend(row["embedding_backend"]),
            embedding_dim=row["embedding_dim"],
            sample_count=row["sample_count"],
            confidence=row["confidence"],
            consent_id=row["consent_id"],
            owner_id=row["owner_id"],
            enrolled_at=row["enrolled_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
            revoked=bool(row["revoked"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def update_template(
        self,
        voice_id: str,
        embedding: bytes,
        sample_count: int,
        confidence: float,
    ) -> bool:
        """
        Update template with new embedding.

        Args:
            voice_id: Voice ID
            embedding: New averaged embedding
            sample_count: Updated sample count
            confidence: Updated confidence score

        Returns:
            True if updated
        """
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                UPDATE biometric_templates
                SET embedding = ?, sample_count = ?, confidence = ?, updated_at = ?
                WHERE voice_id = ? AND is_active = 1 AND revoked = 0
            """, (embedding, sample_count, confidence, time.time(), voice_id))

            return cursor.rowcount > 0

    def revoke_template(self, voice_id: str, reason: str = "") -> bool:
        """
        Revoke a biometric template.

        Args:
            voice_id: Voice ID to revoke
            reason: Reason for revocation

        Returns:
            True if revoked
        """
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                UPDATE biometric_templates
                SET revoked = 1, is_active = 0, updated_at = ?,
                    metadata = json_set(COALESCE(metadata, '{}'), '$.revoke_reason', ?)
                WHERE voice_id = ? AND revoked = 0
            """, (time.time(), reason, voice_id))

            if cursor.rowcount > 0:
                logger.info(f"Revoked biometric template for voice: {voice_id}")
                return True
            return False

    def delete_template(self, voice_id: str) -> bool:
        """
        Permanently delete a biometric template.

        Args:
            voice_id: Voice ID to delete

        Returns:
            True if deleted
        """
        with self.db.transaction() as conn:
            # Delete embedding history first
            conn.execute("DELETE FROM embedding_history WHERE voice_id = ?", (voice_id,))

            # Delete template
            cursor = conn.execute(
                "DELETE FROM biometric_templates WHERE voice_id = ?",
                (voice_id,)
            )

            if cursor.rowcount > 0:
                logger.info(f"Deleted biometric template for voice: {voice_id}")
                return True
            return False

    def is_enrolled(self, voice_id: str) -> bool:
        """Check if voice has an active biometric enrollment."""
        row = self.db.conn.execute(
            "SELECT 1 FROM biometric_templates WHERE voice_id = ? AND is_active = 1 AND revoked = 0",
            (voice_id,)
        ).fetchone()
        return row is not None

    def list_enrolled_voices(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all enrolled voices."""
        rows = self.db.conn.execute("""
            SELECT voice_id, template_id, sample_count, confidence, enrolled_at, updated_at
            FROM biometric_templates
            WHERE is_active = 1 AND revoked = 0
            ORDER BY enrolled_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [
            {
                "voice_id": row["voice_id"],
                "template_id": row["template_id"],
                "sample_count": row["sample_count"],
                "confidence": row["confidence"],
                "enrolled_at": row["enrolled_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # ========================================================================
    # EMBEDDING HISTORY
    # ========================================================================

    def save_embedding_history(
        self,
        voice_id: str,
        embedding: bytes,
        context: str = "verification",
        similarity_to_template: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> int:
        """
        Save embedding to history for drift analysis.

        Args:
            voice_id: Voice ID
            embedding: Serialized embedding
            context: Context ("enrollment", "verification", "update")
            similarity_to_template: Similarity score to enrolled template
            quality_score: Audio quality score

        Returns:
            History entry ID
        """
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO embedding_history
                (voice_id, embedding, timestamp, context, similarity_to_template, quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                voice_id,
                embedding,
                time.time(),
                context,
                similarity_to_template,
                quality_score,
            ))
            return cursor.lastrowid

    def get_embedding_history(
        self,
        voice_id: str,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get embedding history for drift analysis.

        Args:
            voice_id: Voice ID
            limit: Max entries to return
            since: Only entries after this timestamp

        Returns:
            List of embedding history entries
        """
        if since:
            rows = self.db.conn.execute("""
                SELECT * FROM embedding_history
                WHERE voice_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (voice_id, since, limit)).fetchall()
        else:
            rows = self.db.conn.execute("""
                SELECT * FROM embedding_history
                WHERE voice_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (voice_id, limit)).fetchall()

        return [
            {
                "id": row["id"],
                "voice_id": row["voice_id"],
                "embedding": row["embedding"],
                "timestamp": row["timestamp"],
                "context": row["context"],
                "similarity_to_template": row["similarity_to_template"],
                "quality_score": row["quality_score"],
            }
            for row in rows
        ]

    def get_drift_statistics(self, voice_id: str) -> Dict[str, Any]:
        """
        Get drift statistics for a voice.

        Returns:
            Statistics about embedding drift over time
        """
        history = self.get_embedding_history(voice_id, limit=1000)

        if len(history) < 2:
            return {
                "sample_count": len(history),
                "has_sufficient_data": False,
            }

        similarities = [
            h["similarity_to_template"]
            for h in history
            if h["similarity_to_template"] is not None
        ]

        if not similarities:
            return {
                "sample_count": len(history),
                "has_sufficient_data": False,
            }

        import numpy as np
        sim_array = np.array(similarities)

        # Time-based analysis
        timestamps = [h["timestamp"] for h in history if h["similarity_to_template"] is not None]

        return {
            "sample_count": len(history),
            "has_sufficient_data": True,
            "mean_similarity": float(np.mean(sim_array)),
            "std_similarity": float(np.std(sim_array)),
            "min_similarity": float(np.min(sim_array)),
            "max_similarity": float(np.max(sim_array)),
            "recent_mean": float(np.mean(sim_array[:10])) if len(sim_array) >= 10 else float(np.mean(sim_array)),
            "first_sample": min(timestamps) if timestamps else None,
            "last_sample": max(timestamps) if timestamps else None,
            "time_span_days": (max(timestamps) - min(timestamps)) / 86400 if len(timestamps) > 1 else 0,
        }

    # ========================================================================
    # AUDIT LOG
    # ========================================================================

    def log_biometric_action(
        self,
        entry: BiometricAuditEntry,
    ) -> int:
        """
        Log a biometric action for audit.

        Args:
            entry: Audit entry to log

        Returns:
            Audit log entry ID
        """
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO biometric_audit
                (timestamp, voice_id, action, result, similarity_score, liveness_score,
                 liveness_passed, drift_detected, error_message, ip_address, user_agent, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp,
                entry.voice_id,
                entry.action.value if isinstance(entry.action, BiometricAction) else entry.action,
                entry.result,
                entry.similarity_score,
                entry.liveness_score,
                1 if entry.liveness_passed else 0 if entry.liveness_passed is not None else None,
                1 if entry.drift_detected else 0,
                entry.error_message,
                entry.ip_address,
                entry.user_agent,
                json.dumps(entry.metadata),
            ))
            return cursor.lastrowid

    def get_audit_log(
        self,
        voice_id: Optional[str] = None,
        action: Optional[BiometricAction] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[BiometricAuditEntry]:
        """
        Get biometric audit log entries.

        Args:
            voice_id: Filter by voice ID
            action: Filter by action type
            since: Only entries after this timestamp
            limit: Max entries to return

        Returns:
            List of audit entries
        """
        query = "SELECT * FROM biometric_audit WHERE 1=1"
        params = []

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)

        if action:
            query += " AND action = ?"
            params.append(action.value if isinstance(action, BiometricAction) else action)

        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.db.conn.execute(query, params).fetchall()

        return [
            BiometricAuditEntry(
                timestamp=row["timestamp"],
                voice_id=row["voice_id"],
                action=BiometricAction(row["action"]),
                result=row["result"],
                similarity_score=row["similarity_score"],
                liveness_score=row["liveness_score"],
                liveness_passed=bool(row["liveness_passed"]) if row["liveness_passed"] is not None else None,
                drift_detected=bool(row["drift_detected"]),
                error_message=row["error_message"],
                ip_address=row["ip_address"],
                user_agent=row["user_agent"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def get_verification_stats(
        self,
        voice_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get verification statistics for a voice.

        Args:
            voice_id: Voice ID
            days: Number of days to analyze

        Returns:
            Verification statistics
        """
        since = time.time() - (days * 86400)

        rows = self.db.conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN result = 'failure' THEN 1 ELSE 0 END) as failure,
                SUM(CASE WHEN liveness_passed = 0 THEN 1 ELSE 0 END) as liveness_failures,
                AVG(similarity_score) as avg_similarity,
                AVG(liveness_score) as avg_liveness
            FROM biometric_audit
            WHERE voice_id = ? AND action = 'verify' AND timestamp > ?
        """, (voice_id, since)).fetchone()

        return {
            "voice_id": voice_id,
            "period_days": days,
            "total_attempts": rows["total"] or 0,
            "successful": rows["success"] or 0,
            "failed": rows["failure"] or 0,
            "liveness_failures": rows["liveness_failures"] or 0,
            "success_rate": (rows["success"] or 0) / (rows["total"] or 1),
            "avg_similarity": rows["avg_similarity"],
            "avg_liveness": rows["avg_liveness"],
        }


# Singleton instance
_storage_instance: Optional[BiometricStorage] = None


def get_biometric_storage(db=None) -> BiometricStorage:
    """Get or create biometric storage singleton."""
    global _storage_instance
    if _storage_instance is None:
        if db is None:
            from ..persistence import get_database
            db = get_database()
        _storage_instance = BiometricStorage(db)
    return _storage_instance


def set_biometric_storage(storage: BiometricStorage) -> None:
    """Set the biometric storage singleton."""
    global _storage_instance
    _storage_instance = storage
