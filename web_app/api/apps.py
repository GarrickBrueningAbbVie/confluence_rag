"""Django app configuration for API application."""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    """Configuration for the API Django application.

    Attributes:
        default_auto_field: Default primary key field type.
        name: Python path to the application.
        verbose_name: Human-readable name for the application.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "Confluence RAG API"
