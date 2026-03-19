"""API views for the Confluence RAG REST API.

This module provides REST API endpoints for:
- QueryAPIView: Execute queries programmatically
- QueryStatusView: Check query execution status
- HealthCheckView: Service health check
"""

import uuid
import threading
from typing import Any, Dict

from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from loguru import logger

from core.services import get_pipeline_service


# Cache key prefix for query status
QUERY_STATUS_PREFIX = "query_status_"
QUERY_RESULT_PREFIX = "query_result_"

# Status constants
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


def get_query_status_key(query_id: str) -> str:
    """Generate cache key for query status.

    Args:
        query_id: The query UUID.

    Returns:
        Cache key string.
    """
    return f"{QUERY_STATUS_PREFIX}{query_id}"


def get_query_result_key(query_id: str) -> str:
    """Generate cache key for query result.

    Args:
        query_id: The query UUID.

    Returns:
        Cache key string.
    """
    return f"{QUERY_RESULT_PREFIX}{query_id}"


@method_decorator(csrf_exempt, name="dispatch")
class QueryAPIView(APIView):
    """API endpoint for executing RAG queries.

    POST /api/v1/query/

    Request body:
        {
            "query": str,       # Required: The query text
            "mode": str,        # Optional: auto, rag, database, hybrid, smart
            "async": bool,      # Optional: Whether to execute asynchronously
            "query_id": str,    # Optional: Pre-generated query ID
            "top_k": int        # Optional: Number of documents to retrieve
        }

    Synchronous Response (async=false):
        {
            "query_id": str,
            "success": bool,
            "answer": str,
            "sources": list,
            "intent": str,
            "metadata": dict,
            "figures": list,
            "tables": list,
            "execution_time": float
        }

    Asynchronous Response (async=true):
        {
            "query_id": str,
            "status": "processing",
            "websocket_url": str
        }
    """

    def post(self, request: Request) -> Response:
        """Execute a query through the RAG pipeline.

        Args:
            request: The DRF request object.

        Returns:
            Response: JSON response with query results or status.
        """
        # Log the incoming request for debugging
        logger.debug(f"API request data: {request.data}")
        logger.debug(f"API request content type: {request.content_type}")

        # Validate query text
        query_text = request.data.get("query", "").strip()
        logger.info(f"Query text received: '{query_text[:100] if query_text else 'EMPTY'}'")

        if not query_text:
            return Response(
                {"error": "Query text is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Parse options
        mode = request.data.get("mode", "auto")
        is_async = request.data.get("async", False)
        query_id = request.data.get("query_id") or str(uuid.uuid4())
        top_k = request.data.get("top_k", 10)

        # Validate mode
        valid_modes = ["auto", "rag", "database", "hybrid", "smart"]
        if mode not in valid_modes:
            return Response(
                {"error": f"Invalid mode. Must be one of: {valid_modes}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if is_async:
            # Asynchronous execution
            self._start_async_query(query_id, query_text, mode, top_k)
            return Response(
                {
                    "query_id": query_id,
                    "status": STATUS_PROCESSING,
                    "websocket_url": f"/ws/query/{query_id}/",
                },
                status=status.HTTP_202_ACCEPTED,
            )
        else:
            # Synchronous execution
            result = self._execute_query_sync(query_id, query_text, mode, top_k)
            return Response(result, status=status.HTTP_200_OK)

    def _execute_query_sync(
        self,
        query_id: str,
        query_text: str,
        mode: str,
        top_k: int,
    ) -> Dict[str, Any]:
        """Execute query synchronously.

        Args:
            query_id: The query UUID.
            query_text: The query text.
            mode: The routing mode.
            top_k: Number of documents to retrieve.

        Returns:
            Query result dictionary.
        """
        service = get_pipeline_service()
        result = service.execute_query(query_text, mode=mode, top_k=top_k)

        response = result.to_dict()
        response["query_id"] = query_id
        return response

    def _start_async_query(
        self,
        query_id: str,
        query_text: str,
        mode: str,
        top_k: int,
    ) -> None:
        """Start asynchronous query execution in a background thread.

        Args:
            query_id: The query UUID.
            query_text: The query text.
            mode: The routing mode.
            top_k: Number of documents to retrieve.
        """
        # Set initial status
        cache.set(
            get_query_status_key(query_id),
            {
                "status": STATUS_PROCESSING,
                "progress": 0,
                "current_step": "Starting...",
            },
            timeout=300,  # 5 minute timeout
        )

        # Start background thread
        thread = threading.Thread(
            target=self._execute_query_async,
            args=(query_id, query_text, mode, top_k),
            daemon=True,
        )
        thread.start()

    def _execute_query_async(
        self,
        query_id: str,
        query_text: str,
        mode: str,
        top_k: int,
    ) -> None:
        """Execute query in background thread with progress updates.

        Args:
            query_id: The query UUID.
            query_text: The query text.
            mode: The routing mode.
            top_k: Number of documents to retrieve.
        """
        channel_layer = get_channel_layer()
        group_name = f"query_{query_id}"

        def progress_callback(percent: int, step: str) -> None:
            """Send progress update via WebSocket and cache."""
            # Update cache for polling fallback
            cache.set(
                get_query_status_key(query_id),
                {
                    "status": STATUS_PROCESSING,
                    "progress": percent,
                    "current_step": step,
                },
                timeout=300,
            )

            # Send via WebSocket
            try:
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        "type": "progress_update",
                        "step": step,
                        "description": step,
                        "percent": percent,
                    },
                )
            except Exception:
                pass  # WebSocket may not be connected

        try:
            service = get_pipeline_service()
            result = service.execute_query(
                query_text,
                mode=mode,
                top_k=top_k,
                progress_callback=progress_callback,
            )

            result_dict = result.to_dict()
            result_dict["query_id"] = query_id

            # Update cache with final result
            cache.set(
                get_query_status_key(query_id),
                {
                    "status": STATUS_COMPLETE,
                    "progress": 100,
                    "current_step": "Complete",
                },
                timeout=300,
            )
            cache.set(get_query_result_key(query_id), result_dict, timeout=300)

            # Send completion via WebSocket
            try:
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        "type": "query_complete",
                        "result": result_dict,
                    },
                )
            except Exception:
                pass

        except Exception as e:
            # Update cache with error
            cache.set(
                get_query_status_key(query_id),
                {
                    "status": STATUS_FAILED,
                    "progress": 0,
                    "current_step": f"Error: {str(e)}",
                },
                timeout=300,
            )

            # Send error via WebSocket
            try:
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        "type": "query_error",
                        "error": str(e),
                    },
                )
            except Exception:
                pass


class QueryStatusView(APIView):
    """API endpoint for checking query status.

    GET /api/v1/query/{query_id}/status/

    Response:
        {
            "query_id": str,
            "status": str,      # pending, processing, complete, failed
            "progress": int,    # 0-100
            "current_step": str,
            "result": dict      # Only if status is complete
        }
    """

    def get(self, request: Request, query_id: uuid.UUID) -> Response:
        """Get the status of a query execution.

        Args:
            request: The DRF request object.
            query_id: UUID of the query to check.

        Returns:
            Response: JSON response with query status.
        """
        query_id_str = str(query_id)

        # Get status from cache
        status_data = cache.get(get_query_status_key(query_id_str))

        if not status_data:
            return Response(
                {
                    "query_id": query_id_str,
                    "status": STATUS_PENDING,
                    "progress": 0,
                    "current_step": "Query not found or expired",
                },
                status=status.HTTP_200_OK,
            )

        response = {
            "query_id": query_id_str,
            "status": status_data.get("status", STATUS_PENDING),
            "progress": status_data.get("progress", 0),
            "current_step": status_data.get("current_step", ""),
        }

        # Include result if complete
        if status_data.get("status") == STATUS_COMPLETE:
            result = cache.get(get_query_result_key(query_id_str))
            if result:
                response["result"] = result

        return Response(response, status=status.HTTP_200_OK)


class HealthCheckView(APIView):
    """API endpoint for service health check.

    GET /api/v1/health/

    Response:
        {
            "status": str,      # healthy, degraded, unhealthy
            "pipelines": dict,  # Status of each pipeline component
            "message": str      # Additional status message
        }
    """

    def get(self, request: Request) -> Response:
        """Check the health of the service.

        Args:
            request: The DRF request object.

        Returns:
            Response: JSON response with health status.
        """
        service = get_pipeline_service()
        pipeline_status = service.get_status()

        # Determine overall health
        if not pipeline_status["initialized"]:
            health_status = "unhealthy"
            message = pipeline_status.get("init_error", "Not initialized")
        elif not pipeline_status["components"]["rag_pipeline"]["ready"]:
            health_status = "unhealthy"
            message = "RAG pipeline not ready"
        elif not pipeline_status["components"]["query_router"]["ready"]:
            health_status = "degraded"
            message = "Query router not available, using RAG-only mode"
        else:
            health_status = "healthy"
            message = "All systems operational"

        return Response(
            {
                "status": health_status,
                "message": message,
                "pipelines": pipeline_status["components"],
            },
            status=status.HTTP_200_OK,
        )
