"""Database models for the core application.

This module defines models for:
- QueryHistory: Stores query history for debugging and analytics
"""

import uuid

from django.db import models


class QueryHistory(models.Model):
    """Model for storing query history.

    Attributes:
        id: Unique identifier for the query.
        query_text: The original query text.
        intent: Detected query intent (rag, database, hybrid, smart).
        answer: Generated answer text.
        sources_json: JSON field containing source documents.
        metadata_json: JSON field containing query metadata.
        execution_time: Time taken to execute the query in seconds.
        success: Whether the query executed successfully.
        created_at: Timestamp when the query was created.
        session_id: Optional session identifier for grouping queries.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query_text = models.TextField(help_text="The original query text")
    intent = models.CharField(
        max_length=50,
        default="pending",
        help_text="Detected query intent (rag, database, hybrid, smart)",
    )
    answer = models.TextField(null=True, blank=True, help_text="Generated answer text")
    sources_json = models.JSONField(default=list, help_text="Source documents as JSON")
    metadata_json = models.JSONField(default=dict, help_text="Query metadata as JSON")
    execution_time = models.FloatField(null=True, blank=True, help_text="Execution time in seconds")
    success = models.BooleanField(default=False, help_text="Whether query succeeded")
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        db_index=True,
        help_text="Session identifier for grouping queries",
    )

    class Meta:
        """Meta options for QueryHistory model."""

        ordering = ["-created_at"]
        verbose_name = "Query History"
        verbose_name_plural = "Query Histories"
        indexes = [
            models.Index(fields=["intent"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["success"]),
        ]

    def __str__(self) -> str:
        """Return string representation of the query.

        Returns:
            str: Truncated query text with intent.
        """
        truncated = self.query_text[:50] + "..." if len(self.query_text) > 50 else self.query_text
        return f"[{self.intent}] {truncated}"
