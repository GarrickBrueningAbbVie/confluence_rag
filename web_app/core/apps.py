"""Django app configuration for core application."""

from django.apps import AppConfig


class CoreConfig(AppConfig):
    """Configuration for the core Django application.

    Attributes:
        default_auto_field: Default primary key field type.
        name: Python path to the application.
        verbose_name: Human-readable name for the application.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "Confluence RAG Core"
