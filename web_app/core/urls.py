"""URL configuration for core application."""

from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.LandingView.as_view(), name="landing"),
    path("search/", views.SearchView.as_view(), name="search"),
    path("results/<uuid:query_id>/", views.ResultsView.as_view(), name="results"),
]
