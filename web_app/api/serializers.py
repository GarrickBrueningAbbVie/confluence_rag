"""Serializers for the Confluence RAG REST API.

This module provides DRF serializers for request/response validation.
"""

from typing import Any, Dict, List, Optional

from rest_framework import serializers


class QueryOptionsSerializer(serializers.Serializer):
    """Serializer for query options.

    Attributes:
        top_k: Number of documents to retrieve.
        enable_two_stage: Whether to use two-stage RAG.
        enable_reranking: Whether to enable reranking.
    """

    top_k = serializers.IntegerField(min_value=1, max_value=50, default=10)
    enable_two_stage = serializers.BooleanField(default=True)
    enable_reranking = serializers.BooleanField(default=True)


class QueryRequestSerializer(serializers.Serializer):
    """Serializer for query API requests.

    Attributes:
        query: The query text.
        mode: Routing mode (auto, rag, database, hybrid, smart).
        async_mode: Whether to execute asynchronously.
        query_id: Optional pre-generated query ID.
        options: Optional query options.
    """

    query = serializers.CharField(max_length=2000, required=True)
    mode = serializers.ChoiceField(
        choices=["auto", "rag", "database", "hybrid", "smart"],
        default="auto",
    )
    async_mode = serializers.BooleanField(default=False, source="async")
    query_id = serializers.UUIDField(required=False, allow_null=True)
    options = QueryOptionsSerializer(required=False)


class SourceSerializer(serializers.Serializer):
    """Serializer for source documents.

    Attributes:
        title: Document title.
        url: Document URL.
        type: Document type.
        relevance: Relevance score.
        document_index: Index in the response context.
        excerpt: Content excerpt.
    """

    title = serializers.CharField()
    url = serializers.URLField(allow_blank=True)
    type = serializers.CharField(required=False, default="Document")
    relevance = serializers.FloatField(required=False)
    document_index = serializers.IntegerField(required=False)
    excerpt = serializers.CharField(required=False, allow_blank=True)


class FigureSerializer(serializers.Serializer):
    """Serializer for chart figures.

    Attributes:
        figure: Plotly figure data (JSON).
        html: HTML representation.
        chart_type: Type of chart.
        caption: Chart caption.
    """

    figure = serializers.JSONField(required=False)
    html = serializers.CharField(required=False, allow_blank=True)
    chart_type = serializers.CharField(required=False)
    caption = serializers.CharField(required=False, allow_blank=True)


class TableSerializer(serializers.Serializer):
    """Serializer for table data.

    Attributes:
        html: HTML table representation.
        markdown: Markdown table representation.
        row_count: Number of rows.
        columns: List of column names.
        data: Raw table data.
    """

    html = serializers.CharField(required=False, allow_blank=True)
    markdown = serializers.CharField(required=False, allow_blank=True)
    row_count = serializers.IntegerField(required=False)
    columns = serializers.ListField(child=serializers.CharField(), required=False)
    data = serializers.ListField(required=False)


class MetadataSerializer(serializers.Serializer):
    """Serializer for query metadata.

    Attributes:
        intent: Detected intent.
        confidence: Confidence score.
        routing_mode: Routing mode used.
        extracted_entities: Extracted entities.
        sub_queries: List of sub-queries for smart routing.
    """

    intent = serializers.CharField(required=False)
    confidence = serializers.FloatField(required=False)
    routing_mode = serializers.CharField(required=False)
    extracted_entities = serializers.DictField(required=False)
    sub_queries = serializers.ListField(required=False)


class QueryResponseSerializer(serializers.Serializer):
    """Serializer for query API responses.

    Attributes:
        query_id: Unique query identifier.
        success: Whether the query succeeded.
        answer: Generated answer.
        sources: List of source documents.
        intent: Detected query intent.
        query: Generated database query (if applicable).
        execution_time: Execution time in seconds.
        metadata: Additional query metadata.
        figures: List of chart figures.
        tables: List of table data.
        error: Error message if query failed.
    """

    query_id = serializers.UUIDField()
    success = serializers.BooleanField()
    answer = serializers.CharField(allow_blank=True)
    sources = SourceSerializer(many=True, required=False)
    intent = serializers.CharField()
    query = serializers.CharField(allow_null=True, required=False)
    execution_time = serializers.FloatField()
    metadata = MetadataSerializer(required=False)
    figures = FigureSerializer(many=True, required=False)
    tables = TableSerializer(many=True, required=False)
    error = serializers.CharField(allow_null=True, required=False)


class QueryStatusResponseSerializer(serializers.Serializer):
    """Serializer for query status responses.

    Attributes:
        query_id: Unique query identifier.
        status: Current status (pending, processing, complete, failed).
        progress: Progress percentage (0-100).
        current_step: Description of current step.
        result: Query result (only if complete).
    """

    query_id = serializers.UUIDField()
    status = serializers.ChoiceField(
        choices=["pending", "processing", "complete", "failed"]
    )
    progress = serializers.IntegerField(min_value=0, max_value=100)
    current_step = serializers.CharField()
    result = QueryResponseSerializer(required=False)


class HealthResponseSerializer(serializers.Serializer):
    """Serializer for health check responses.

    Attributes:
        status: Overall health status.
        message: Status message.
        pipelines: Status of each pipeline component.
    """

    status = serializers.ChoiceField(choices=["healthy", "degraded", "unhealthy"])
    message = serializers.CharField()
    pipelines = serializers.DictField()
