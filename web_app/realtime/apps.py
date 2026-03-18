"""Django app configuration for realtime application."""

from django.apps import AppConfig


class RealtimeConfig(AppConfig):
    """Configuration for the realtime Django application.

    Attributes:
        default_auto_field: Default primary key field type.
        name: Python path to the application.
        verbose_name: Human-readable name for the application.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "realtime"
    verbose_name = "Confluence RAG Real-time"
