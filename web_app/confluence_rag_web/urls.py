"""
URL configuration for Confluence RAG web application.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    # Admin interface
    path("admin/", admin.site.urls),
    # Core application (landing page, search, results)
    path("", include("core.urls")),
    # REST API endpoints
    path("api/v1/", include("api.urls")),
]
