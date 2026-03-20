"""Views for the core application.

This module contains the main views for the Confluence RAG web interface:
- LandingView: Google-like landing page with search box
- SearchView: Handles query submission and initiates pipeline execution
- ResultsView: Displays query results with debug panel
"""

import uuid
from typing import Any, Dict

from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views import View


class LandingView(View):
    """Landing page view with centered search box.

    Renders a clean, Google-like landing page with the title
    "DSA Confluence RAG" and a search input.
    """

    template_name = "core/landing.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        """Handle GET request for landing page.

        Args:
            request: The HTTP request object.

        Returns:
            HttpResponse: Rendered landing page template.
        """
        return render(request, self.template_name)


class SearchView(View):
    """Search view that handles query submission.

    Accepts a query from the search form, creates a query record,
    and redirects to the results page.
    """

    def post(self, request: HttpRequest) -> HttpResponse:
        """Handle POST request for search submission.

        Args:
            request: The HTTP request object containing the query.

        Returns:
            HttpResponse: Redirect to results page with query_id.
        """
        query_text = request.POST.get("query", "").strip()

        if not query_text:
            return redirect("core:landing")

        # Generate a unique query ID
        query_id = uuid.uuid4()

        # Store query in session for results page
        request.session[f"query_{query_id}"] = {
            "query_text": query_text,
            "status": "pending",
        }

        return redirect("core:results", query_id=query_id)


class ResultsView(View):
    """Results view that displays query results.

    Shows the answer, debug panel, charts/tables, and source documents.
    """

    template_name = "core/results.html"

    def get(self, request: HttpRequest, query_id: uuid.UUID) -> HttpResponse:
        """Handle GET request for results page.

        Args:
            request: The HTTP request object.
            query_id: UUID of the query to display results for.

        Returns:
            HttpResponse: Rendered results page template.
        """
        # Get query from session
        query_data = request.session.get(f"query_{query_id}", {})
        query_text = query_data.get("query_text", "")

        context: Dict[str, Any] = {
            "query_id": str(query_id),
            "query_text": query_text,
        }

        return render(request, self.template_name, context)
