"""URL configuration for API application."""

from django.urls import path

from . import views

app_name = "api"

urlpatterns = [
    path("query/", views.QueryAPIView.as_view(), name="query"),
    path("query/<uuid:query_id>/status/", views.QueryStatusView.as_view(), name="query_status"),
    path("health/", views.HealthCheckView.as_view(), name="health"),
]
