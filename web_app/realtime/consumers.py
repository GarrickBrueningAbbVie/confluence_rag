"""WebSocket consumers for real-time query progress updates.

This module provides WebSocket consumers for:
- QueryProgressConsumer: Sends real-time progress updates during query execution
"""

import json
from typing import Any, Dict

from channels.generic.websocket import AsyncJsonWebsocketConsumer


class QueryProgressConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer for query progress updates.

    Handles WebSocket connections for real-time progress updates
    during query execution. Clients connect using:

        ws://host/ws/query/{query_id}/

    Progress messages are sent in the format:
        {
            "type": "progress",
            "step": str,
            "description": str,
            "percent": int,
            "sub_queries": list
        }

    Completion messages are sent in the format:
        {
            "type": "complete",
            "result": dict
        }
    """

    async def connect(self) -> None:
        """Handle WebSocket connection.

        Extracts query_id from URL and joins the corresponding group.
        """
        self.query_id = self.scope["url_route"]["kwargs"]["query_id"]
        self.room_group_name = f"query_{self.query_id}"

        # Join query-specific group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

        # Send initial connection confirmation
        await self.send_json(
            {
                "type": "connected",
                "query_id": self.query_id,
                "message": "Connected to query progress stream",
            }
        )

    async def disconnect(self, close_code: int) -> None:
        """Handle WebSocket disconnection.

        Args:
            close_code: The WebSocket close code.
        """
        # Leave query-specific group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive_json(self, content: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages.

        Args:
            content: The JSON content received from the client.
        """
        message_type = content.get("type", "")

        if message_type == "ping":
            await self.send_json({"type": "pong"})

    async def progress_update(self, event: Dict[str, Any]) -> None:
        """Send progress update to WebSocket client.

        Called when a progress update is sent to the group.

        Args:
            event: The progress event data containing step, description, and percent.
        """
        await self.send_json(
            {
                "type": "progress",
                "step": event.get("step", ""),
                "description": event.get("description", ""),
                "percent": event.get("percent", 0),
                "sub_queries": event.get("sub_queries", []),
            }
        )

    async def query_complete(self, event: Dict[str, Any]) -> None:
        """Send completion signal to WebSocket client.

        Called when query execution is complete.

        Args:
            event: The completion event data containing the result.
        """
        await self.send_json(
            {
                "type": "complete",
                "result": event.get("result", {}),
            }
        )

    async def query_error(self, event: Dict[str, Any]) -> None:
        """Send error signal to WebSocket client.

        Called when query execution fails.

        Args:
            event: The error event data containing error details.
        """
        await self.send_json(
            {
                "type": "error",
                "error": event.get("error", "Unknown error"),
                "details": event.get("details", ""),
            }
        )
