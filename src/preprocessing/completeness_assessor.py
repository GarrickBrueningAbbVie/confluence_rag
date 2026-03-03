"""
Project page completeness assessment.

This module evaluates project pages against a standard project charter
template and generates completeness scores and summaries.

The charter template includes sections like:
- Definition and Purpose
- Benefits
- Project Team
- Data Sources
- Stakeholders
- Timeline
- Meetings
- Tools/Technology
- Approach/Methodology
- Risks/Dependencies
- Expected Outcomes/Deliverables

Example:
    >>> from preprocessing.completeness_assessor import CompletenessAssessor
    >>> assessor = CompletenessAssessor()
    >>> score, summary = assessor.calculate_completeness(page_data)
    >>> print(f"Score: {score}, Summary: {summary}")

Usage as standalone script:
    python -m preprocessing.completeness_assessor
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient
    from iliad.analyze import DocumentAnalyzer
except ImportError:
    pass


@dataclass
class CharterSection:
    """A section of the project charter template.

    Attributes:
        name: Section name for reporting
        weight: Weight for scoring (0.0 to 1.0, should sum to 1.0)
        required_keywords: Keywords that should be present
        optional_keywords: Keywords that enhance the score
    """

    name: str
    weight: float
    required_keywords: List[str] = field(default_factory=list)
    optional_keywords: List[str] = field(default_factory=list)


# Standard project charter template based on the "Code Documentation Tool" project
PROJECT_CHARTER_TEMPLATE = [
    CharterSection(
        name="Definition and Purpose",
        weight=0.15,
        required_keywords=[
            "purpose", "objective", "goal", "problem", "need",
            "overview", "description", "about", "introduction",
        ],
        optional_keywords=[
            "vision", "mission", "scope", "background", "context",
        ],
    ),
    CharterSection(
        name="Benefits",
        weight=0.10,
        required_keywords=[
            "benefit", "value", "impact", "outcome", "advantage",
        ],
        optional_keywords=[
            "roi", "savings", "improvement", "efficiency", "productivity",
        ],
    ),
    CharterSection(
        name="Project Team",
        weight=0.10,
        required_keywords=[
            "team", "member", "role", "lead", "owner", "manager",
            "data scientist", "analyst", "developer", "engineer",
        ],
        optional_keywords=[
            "stakeholder", "sponsor", "contributor", "contact",
        ],
    ),
    CharterSection(
        name="Data Sources",
        weight=0.10,
        required_keywords=[
            "data", "source", "database", "input", "dataset",
        ],
        optional_keywords=[
            "api", "integration", "feed", "warehouse", "lake",
            "table", "schema", "etl",
        ],
    ),
    CharterSection(
        name="Stakeholders",
        weight=0.08,
        required_keywords=[
            "stakeholder", "user", "customer", "audience", "client",
        ],
        optional_keywords=[
            "consumer", "recipient", "business", "function", "department",
        ],
    ),
    CharterSection(
        name="Timeline",
        weight=0.10,
        required_keywords=[
            "timeline", "milestone", "date", "deadline", "phase",
            "schedule", "quarter", "q1", "q2", "q3", "q4",
        ],
        optional_keywords=[
            "roadmap", "plan", "sprint", "release", "delivery",
        ],
    ),
    CharterSection(
        name="Meetings",
        weight=0.05,
        required_keywords=[
            "meeting", "standup", "sync", "review", "demo",
        ],
        optional_keywords=[
            "agenda", "cadence", "weekly", "monthly", "bi-weekly",
        ],
    ),
    CharterSection(
        name="Tools/Technology",
        weight=0.10,
        required_keywords=[
            "tool", "technology", "platform", "framework", "software",
            "python", "r", "sql", "aws", "azure",
        ],
        optional_keywords=[
            "infrastructure", "stack", "library", "package", "environment",
        ],
    ),
    CharterSection(
        name="Approach/Methodology",
        weight=0.10,
        required_keywords=[
            "approach", "methodology", "method", "process", "workflow",
            "algorithm", "model", "analysis",
        ],
        optional_keywords=[
            "technique", "strategy", "framework", "pipeline",
        ],
    ),
    CharterSection(
        name="Risks/Dependencies",
        weight=0.07,
        required_keywords=[
            "risk", "dependency", "constraint", "blocker", "issue",
        ],
        optional_keywords=[
            "challenge", "limitation", "assumption", "mitigation",
        ],
    ),
    CharterSection(
        name="Expected Outcomes/Deliverables",
        weight=0.05,
        required_keywords=[
            "deliverable", "outcome", "output", "result", "product",
        ],
        optional_keywords=[
            "artifact", "report", "dashboard", "model", "documentation",
        ],
    ),
]


class CompletenessAssessor:
    """
    Assess project page completeness against a charter template.

    Generates:
    - completeness_score: 0-100 numeric score
    - completeness_summary: Text description of assessment

    Only main project pages are assessed; subpages receive NaN scores.

    Attributes:
        iliad_client: Optional Iliad client for enhanced LLM analysis
        template: List of CharterSection definitions
        min_depth: Minimum page depth to be considered a main project
        max_depth: Maximum page depth to be considered a main project

    Example:
        >>> assessor = CompletenessAssessor()
        >>> pages = assessor.process_pages(all_pages)
        >>> # Check scores
        >>> for p in pages:
        ...     if p['completeness_score'] is not None:
        ...         print(f"{p['title']}: {p['completeness_score']}")
    """

    def __init__(
        self,
        iliad_client: Optional["IliadClient"] = None,
        template: Optional[List[CharterSection]] = None,
        min_depth: int = 2,
        max_depth: int = 5,
    ) -> None:
        """Initialize completeness assessor.

        Args:
            iliad_client: Optional Iliad client for enhanced analysis
            template: Custom charter template (defaults to PROJECT_CHARTER_TEMPLATE)
            min_depth: Minimum depth for main project pages
            max_depth: Maximum depth for main project pages
        """
        self.iliad_client = iliad_client
        self.template = template or PROJECT_CHARTER_TEMPLATE
        self.min_depth = min_depth
        self.max_depth = max_depth

        if iliad_client:
            self.analyzer = DocumentAnalyzer(iliad_client)
        else:
            self.analyzer = None

        # Validate template weights sum to ~1.0
        total_weight = sum(s.weight for s in self.template)
        if not (0.95 <= total_weight <= 1.05):
            logger.warning(f"Template weights sum to {total_weight}, should be ~1.0")

        logger.info(f"Initialized CompletenessAssessor with {len(self.template)} sections")

    def _get_page_depth(self, page_data: Dict[str, Any]) -> int:
        """Calculate page depth from ancestors.

        Handles cases where depth field is None or missing.

        Args:
            page_data: Page dictionary

        Returns:
            Page depth (1 = top-level)
        """
        depth = page_data.get("depth")
        if depth is None:
            # Calculate from ancestors
            ancestors = page_data.get("ancestors", [])
            depth = len(ancestors) + 1
        return depth

    def is_main_project_page(
        self,
        page_data: Dict[str, Any],
    ) -> bool:
        """
        Determine if a page is a main project page vs subpage.

        Main project pages are typically:
        - At depth 2-5 in the hierarchy
        - Have child pages
        - Are directly under a project root category
        - Have parent_project matching their own title

        Args:
            page_data: Page dictionary with metadata

        Returns:
            True if page is a main project page

        Example:
            >>> if assessor.is_main_project_page(page):
            ...     score, summary = assessor.calculate_completeness(page)
        """
        depth = self._get_page_depth(page_data)
        has_children = len(page_data.get("children", [])) > 0
        parent_project = page_data.get("parent_project")
        title = page_data.get("title", "")

        # Check depth range
        if depth < self.min_depth or depth > self.max_depth:
            return False

        # Main project pages typically have children
        if not has_children:
            return False

        # If parent_project matches title, this is the project's main page
        if parent_project and parent_project.lower() == title.lower():
            return True

        # Also consider pages that are the parent_project for other pages
        # This is handled during batch processing

        return False

    def assess_section(
        self,
        content: str,
        section: CharterSection,
    ) -> Tuple[float, bool, int]:
        """
        Assess content against a single charter section.

        Args:
            content: Page content (lowercase for matching)
            section: Charter section to assess

        Returns:
            Tuple of (section_score, is_present, keyword_count)

        Example:
            >>> score, present, count = assessor.assess_section(content, section)
        """
        # Count keyword matches
        required_found = sum(
            1 for kw in section.required_keywords
            if kw.lower() in content
        )
        optional_found = sum(
            1 for kw in section.optional_keywords
            if kw.lower() in content
        )

        total_required = len(section.required_keywords)
        total_optional = len(section.optional_keywords)

        if total_required == 0:
            return 0.0, False, 0

        # Calculate score components
        # Required keywords: 70% weight
        # Optional keywords: 30% weight
        required_ratio = required_found / total_required
        optional_ratio = optional_found / max(total_optional, 1)

        section_score = (required_ratio * 0.7) + (optional_ratio * 0.3)

        # Section is "present" if at least 2 required keywords found
        is_present = required_found >= 2

        keyword_count = required_found + optional_found

        return section_score, is_present, keyword_count

    def calculate_completeness(
        self,
        page_data: Dict[str, Any],
    ) -> Tuple[Optional[float], str]:
        """
        Calculate completeness score and generate summary.

        Args:
            page_data: Page dictionary with content

        Returns:
            Tuple of (score 0-100 or None, summary text)
            Score is None (NaN) for subpages.

        Example:
            >>> score, summary = assessor.calculate_completeness(page_data)
            >>> if score is not None:
            ...     print(f"Completeness: {score}%")
        """
        # Check if this is a main project page
        if not self.is_main_project_page(page_data):
            return None, "N/A - Subpage or non-project page"

        # Combine all content for analysis
        content_text = page_data.get("content_text", "") or ""
        attachment_content = page_data.get("attachment_content", "") or ""
        title = page_data.get("title", "")

        # Combine and lowercase for matching
        full_content = f"{title}\n{content_text}\n{attachment_content}".lower()

        if not full_content.strip():
            return 0.0, "No content found"

        # Assess each section
        sections_present = []
        sections_missing = []
        section_scores = []
        total_score = 0.0

        for section in self.template:
            score, present, _ = self.assess_section(full_content, section)
            weighted_score = score * section.weight
            total_score += weighted_score
            section_scores.append((section.name, score, section.weight))

            if present:
                sections_present.append(section.name)
            else:
                sections_missing.append(section.name)

        # Convert to 0-100 scale
        completeness_score = round(total_score * 100, 1)

        # Generate summary
        summary_parts = [f"Score: {completeness_score}/100"]

        if sections_present:
            present_str = ", ".join(sections_present[:4])
            if len(sections_present) > 4:
                present_str += f" (+{len(sections_present) - 4} more)"
            summary_parts.append(f"Present: {present_str}")

        if sections_missing:
            missing_str = ", ".join(sections_missing[:3])
            if len(sections_missing) > 3:
                missing_str += f" (+{len(sections_missing) - 3} more)"
            summary_parts.append(f"Missing: {missing_str}")

        summary = ". ".join(summary_parts)

        return completeness_score, summary

    def calculate_completeness_llm(
        self,
        page_data: Dict[str, Any],
    ) -> Tuple[Optional[float], str]:
        """
        Calculate completeness using LLM for more accurate assessment.

        Requires iliad_client to be configured.

        Args:
            page_data: Page dictionary with content

        Returns:
            Tuple of (score 0-100 or None, summary text)
        """
        if not self.analyzer:
            logger.warning("LLM assessment requires iliad_client, falling back to keyword-based")
            return self.calculate_completeness(page_data)

        if not self.is_main_project_page(page_data):
            return None, "N/A - Subpage or non-project page"

        content_text = page_data.get("content_text", "") or ""
        title = page_data.get("title", "")

        section_names = [s.name for s in self.template]

        try:
            result = self.analyzer.assess_completeness(content_text, section_names)
            score = result.get("score", 0)
            summary = result.get("summary", "Assessment completed")
            return float(score), summary
        except Exception as e:
            logger.error(f"LLM completeness assessment failed: {e}")
            return self.calculate_completeness(page_data)

    def process_page(
        self,
        page_data: Dict[str, Any],
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single page and add completeness data.

        Args:
            page_data: Page dictionary to process
            use_llm: Whether to use LLM-based assessment

        Returns:
            Updated page dictionary with completeness fields

        Example:
            >>> page = assessor.process_page(page_data)
            >>> print(page["completeness_score"])
        """
        if use_llm and self.iliad_client:
            score, summary = self.calculate_completeness_llm(page_data)
        else:
            score, summary = self.calculate_completeness(page_data)

        # Handle NaN (None) score properly for JSON serialization
        page_data["completeness_score"] = score if score is not None else float("nan")
        page_data["completeness_summary"] = summary

        return page_data

    def process_pages(
        self,
        pages: List[Dict[str, Any]],
        use_llm: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process all pages and add completeness data.

        Args:
            pages: List of page dictionaries
            use_llm: Whether to use LLM-based assessment

        Returns:
            Updated pages with completeness_score and completeness_summary

        Example:
            >>> pages = assessor.process_pages(all_pages)
            >>> scored = [p for p in pages if p['completeness_score'] is not None]
        """
        logger.info(f"Assessing completeness for {len(pages)} pages")

        # First pass: identify main project pages
        main_pages = []
        sub_pages = []

        for page in pages:
            if self.is_main_project_page(page):
                main_pages.append(page)
            else:
                sub_pages.append(page)

        logger.info(f"Found {len(main_pages)} main project pages to assess")

        # Process main project pages
        for i, page in enumerate(main_pages):
            if (i + 1) % 10 == 0:
                logger.info(f"Assessing page {i + 1}/{len(main_pages)}")

            try:
                self.process_page(page, use_llm=use_llm)
            except Exception as e:
                title = page.get("title", "unknown")
                logger.error(f"Failed to assess '{title}': {e}")
                page["completeness_score"] = float("nan")
                page["completeness_summary"] = f"Assessment failed: {str(e)}"

        # Set subpages to NaN
        for page in sub_pages:
            page["completeness_score"] = float("nan")
            page["completeness_summary"] = "N/A - Subpage"

        # Log summary
        scored_pages = [p for p in pages if not math.isnan(p.get("completeness_score", float("nan")))]
        if scored_pages:
            avg_score = sum(p["completeness_score"] for p in scored_pages) / len(scored_pages)
            logger.info(f"Completeness assessment complete:")
            logger.info(f"  - {len(scored_pages)} main project pages assessed")
            logger.info(f"  - Average score: {avg_score:.1f}/100")
            logger.info(f"  - {len(sub_pages)} subpages marked N/A")
        else:
            logger.warning("No main project pages found for assessment")

        return pages

    def get_low_completeness_pages(
        self,
        pages: List[Dict[str, Any]],
        threshold: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """
        Get pages with completeness below a threshold.

        Useful for identifying projects needing documentation improvement.

        Args:
            pages: List of processed pages
            threshold: Score threshold (0-100)

        Returns:
            List of pages with score below threshold

        Example:
            >>> low_pages = assessor.get_low_completeness_pages(pages, threshold=60)
            >>> for p in low_pages:
            ...     print(f"{p['title']}: {p['completeness_score']}")
        """
        low_pages = []

        for page in pages:
            score = page.get("completeness_score")
            if score is not None and not math.isnan(score) and score < threshold:
                low_pages.append(page)

        # Sort by score ascending
        low_pages.sort(key=lambda p: p.get("completeness_score", 0))

        return low_pages


def main():
    """Run completeness assessment as standalone script."""
    import sys

    # Default path
    json_path = Path("Data_Storage/confluence_pages.json")

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        logger.error(f"File not found: {json_path}")
        sys.exit(1)

    # Load pages
    logger.info(f"Loading pages from {json_path}")
    with open(json_path, "r") as f:
        pages = json.load(f)

    logger.info(f"Loaded {len(pages)} pages")

    # Create assessor and process
    assessor = CompletenessAssessor()
    pages = assessor.process_pages(pages)

    # Report results
    scored = [p for p in pages if not math.isnan(p.get("completeness_score", float("nan")))]

    print("\n" + "=" * 60)
    print("COMPLETENESS ASSESSMENT RESULTS")
    print("=" * 60)

    if scored:
        print(f"\nAssessed {len(scored)} main project pages:\n")

        # Sort by score
        scored.sort(key=lambda p: p.get("completeness_score", 0), reverse=True)

        for p in scored[:20]:  # Show top 20
            score = p["completeness_score"]
            title = p["title"][:40]
            summary = p["completeness_summary"]
            print(f"  {score:5.1f}  {title:<40}  {summary}")

        if len(scored) > 20:
            print(f"\n  ... and {len(scored) - 20} more pages")
    else:
        print("\nNo main project pages found for assessment")

    # Optionally save updated JSON
    output_path = json_path.with_suffix(".assessed.json")
    logger.info(f"Saving results to {output_path}")

    with open(output_path, "w") as f:
        # Handle NaN for JSON
        def handle_nan(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            raise TypeError

        json.dump(pages, f, indent=2, default=handle_nan)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
