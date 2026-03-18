"""WebSocket URL routing for real-time updates."""

from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/query/(?P<query_id>[0-9a-f-]+)/$", consumers.QueryProgressConsumer.as_asgi()),
]
