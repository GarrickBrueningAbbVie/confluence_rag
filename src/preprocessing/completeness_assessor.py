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

    Main project pages are assessed using aggregated content from ALL pages
    within the project (main page + all subpages). Subpages receive NaN scores.

    Attributes:
        iliad_client: Optional Iliad client for enhanced LLM analysis
        template: List of CharterSection definitions
        min_depth: Minimum page depth to be considered a main project
        max_depth: Maximum page depth to be considered a main project
        _pages_by_id: Internal index for quick page lookup
        _pages_by_parent_project: Internal index for project grouping

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
        self._pages_by_id: Dict[str, Dict[str, Any]] = {}
        self._pages_by_parent_project: Dict[str, List[Dict[str, Any]]] = {}

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
        """Calculate page depth from parents.

        Handles cases where depth field is None or missing.

        Args:
            page_data: Page dictionary

        Returns:
            Page depth (1 = top-level)
        """
        depth = page_data.get("depth")
        if depth is None:
            # Calculate from parents
            parents = page_data.get("parents", [])
            depth = len(parents) + 1
        return depth

    def _build_page_index(self, pages: List[Dict[str, Any]]) -> None:
        """Build internal indexes for efficient page lookup.

        Creates:
        - _pages_by_id: Quick lookup by page ID
        - _pages_by_parent_project: Group pages by their parent project

        Args:
            pages: List of all page dictionaries
        """
        self._pages_by_id = {}
        self._pages_by_parent_project = {}

        for page in pages:
            page_id = page.get("id")
            if page_id:
                self._pages_by_id[page_id] = page

            parent_project = page.get("parent_project")
            if parent_project:
                if parent_project not in self._pages_by_parent_project:
                    self._pages_by_parent_project[parent_project] = []
                self._pages_by_parent_project[parent_project].append(page)

        logger.debug(
            f"Built page index: {len(self._pages_by_id)} pages, "
            f"{len(self._pages_by_parent_project)} projects"
        )

    def get_project_pages(
        self,
        main_page: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Get all pages belonging to a project (main page + all subpages).

        Collects pages by:
        1. Finding all pages with matching parent_project
        2. Recursively collecting children of the main page

        Args:
            main_page: The main project page

        Returns:
            List of all pages in the project (including main page)
        """
        project_pages = []
        title = main_page.get("title", "")
        page_id = main_page.get("id")

        # Method 1: Get all pages with this as parent_project
        if title in self._pages_by_parent_project:
            project_pages.extend(self._pages_by_parent_project[title])

        # Method 2: Recursively collect children
        collected_ids = {p.get("id") for p in project_pages}

        def collect_children(page: Dict[str, Any]) -> None:
            children = page.get("children", [])
            for child_ref in children:
                child_id = child_ref.get("id")
                if child_id and child_id not in collected_ids:
                    child_page = self._pages_by_id.get(child_id)
                    if child_page:
                        project_pages.append(child_page)
                        collected_ids.add(child_id)
                        collect_children(child_page)

        collect_children(main_page)

        # Ensure main page is included
        if page_id not in collected_ids:
            project_pages.insert(0, main_page)

        return project_pages

    def aggregate_project_content(
        self,
        project_pages: List[Dict[str, Any]],
    ) -> Tuple[str, int]:
        """Aggregate content from all pages in a project.

        Combines title, content_text, and attachment_content from all pages.

        Args:
            project_pages: List of pages in the project

        Returns:
            Tuple of (aggregated_content, page_count)
        """
        content_parts = []

        for page in project_pages:
            title = page.get("title", "")
            content_text = page.get("content_text", "") or ""
            attachment_content = page.get("attachment_content", "") or ""

            # Add page content with title as header
            if title:
                content_parts.append(f"=== {title} ===")
            if content_text.strip():
                content_parts.append(content_text)
            if attachment_content.strip():
                content_parts.append(attachment_content)

        aggregated = "\n\n".join(content_parts)
        return aggregated.lower(), len(project_pages)

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
        use_aggregated: bool = True,
    ) -> Tuple[Optional[float], str]:
        """
        Calculate completeness score and generate summary.

        When use_aggregated=True (default), aggregates content from ALL pages
        within the project (main page + all subpages) for a more accurate
        completeness assessment.

        Args:
            page_data: Page dictionary with content
            use_aggregated: If True, aggregate content from all project pages

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

        # Get content - either aggregated from all project pages or just this page
        if use_aggregated and self._pages_by_id:
            project_pages = self.get_project_pages(page_data)
            full_content, page_count = self.aggregate_project_content(project_pages)
        else:
            # Fallback: use only this page's content
            content_text = page_data.get("content_text", "") or ""
            attachment_content = page_data.get("attachment_content", "") or ""
            title = page_data.get("title", "")
            full_content = f"{title}\n{content_text}\n{attachment_content}".lower()
            page_count = 1

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

        # Generate summary with page count
        summary_parts = [f"Score: {completeness_score}/100"]

        if page_count > 1:
            summary_parts.append(f"Based on {page_count} pages")

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
        use_aggregated: bool = True,
    ) -> Tuple[Optional[float], str, int]:
        """
        Calculate completeness using LLM for more accurate assessment.

        Uses a prompt that mirrors the keyword-based heuristic with weighted sections.
        Aggregates content from all pages within the project when use_aggregated=True.

        Requires iliad_client to be configured.

        Args:
            page_data: Page dictionary with content
            use_aggregated: If True, aggregate content from all project pages

        Returns:
            Tuple of (score 0-100 or None, summary text, page_count)
        """
        if not self.iliad_client:
            logger.warning("LLM assessment requires iliad_client, falling back to keyword-based")
            score, summary = self.calculate_completeness(page_data, use_aggregated)
            page_count = len(self.get_project_pages(page_data)) if use_aggregated else 1
            return score, summary, page_count

        if not self.is_main_project_page(page_data):
            return None, "N/A - Subpage or non-project page", 0

        # Get content - either aggregated or single page
        if use_aggregated and self._pages_by_id:
            project_pages = self.get_project_pages(page_data)
            full_content, page_count = self.aggregate_project_content(project_pages)
        else:
            content_text = page_data.get("content_text", "") or ""
            attachment_content = page_data.get("attachment_content", "") or ""
            title = page_data.get("title", "")
            full_content = f"{title}\n{content_text}\n{attachment_content}"
            page_count = 1

        if not full_content.strip():
            return 0.0, "No content found", page_count

        # Build the weighted sections description for the prompt
        sections_with_weights = []
        for section in self.template:
            weight_pct = int(section.weight * 100)
            required_kw = ", ".join(section.required_keywords[:5])
            optional_kw = ", ".join(section.optional_keywords[:3])
            sections_with_weights.append(
                f"- **{section.name}** (Weight: {weight_pct}%)\n"
                f"  Required concepts: {required_kw}\n"
                f"  Optional concepts: {optional_kw}"
            )

        sections_text = "\n".join(sections_with_weights)

        prompt = f"""You are assessing a project documentation page for completeness against a standard project charter template.

## Scoring Formula
For each section:
1. Calculate required_ratio = (required concepts found) / (total required concepts)
2. Calculate optional_ratio = (optional concepts found) / (total optional concepts)
3. section_score = (required_ratio × 0.70) + (optional_ratio × 0.30)
4. weighted_score = section_score × section_weight

Final Score = sum of all weighted_scores × 100 (scale 0-100)

## Sections to Evaluate
{sections_text}

## Instructions
1. Analyze the content for each section
2. Determine what percentage of required/optional concepts are adequately covered
3. Calculate the final weighted score using the formula above
4. A section is "present" if at least 2 required concepts are found

Return ONLY a JSON object with this exact format:
{{
  "score": <number 0-100>,
  "present": ["list of section names that are present"],
  "missing": ["list of section names that are missing or inadequate"],
  "summary": "<brief summary of assessment>"
}}

## Content to Assess (from {page_count} page(s)):
{full_content[:12000]}"""

        try:
            from iliad.client import IliadModel

            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages, model=IliadModel.GPT4O_MINI)
            response_text = self.iliad_client.extract_content(response)

            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            score = float(result.get("score", 0))
            present = result.get("present", [])
            missing = result.get("missing", [])
            llm_summary = result.get("summary", "")

            # Build summary with page count
            summary_parts = [f"Score: {score:.1f}/100"]
            if page_count > 1:
                summary_parts.append(f"Based on {page_count} pages")
            if present:
                present_str = ", ".join(present[:4])
                if len(present) > 4:
                    present_str += f" (+{len(present) - 4} more)"
                summary_parts.append(f"Present: {present_str}")
            if missing:
                missing_str = ", ".join(missing[:3])
                if len(missing) > 3:
                    missing_str += f" (+{len(missing) - 3} more)"
                summary_parts.append(f"Missing: {missing_str}")

            summary = ". ".join(summary_parts)

            return score, summary, page_count

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM completeness JSON: {e}, falling back to keyword-based")
            score, summary = self.calculate_completeness(page_data, use_aggregated)
            return score, summary, page_count
        except Exception as e:
            logger.error(f"LLM completeness assessment failed: {e}")
            score, summary = self.calculate_completeness(page_data, use_aggregated)
            return score, summary, page_count

    def process_page(
        self,
        page_data: Dict[str, Any],
        use_llm: bool = True,
        use_aggregated: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single page and add completeness data.

        Args:
            page_data: Page dictionary to process
            use_llm: Whether to use LLM-based assessment (default: True)
            use_aggregated: If True, aggregate content from all project pages

        Returns:
            Updated page dictionary with completeness fields

        Example:
            >>> page = assessor.process_page(page_data)
            >>> print(page["completeness_score"])
        """
        if use_llm and self.iliad_client:
            score, summary, pages_assessed = self.calculate_completeness_llm(
                page_data, use_aggregated=use_aggregated
            )
        else:
            score, summary = self.calculate_completeness(page_data, use_aggregated=use_aggregated)
            # Get page count if aggregated
            if use_aggregated and self._pages_by_id and self.is_main_project_page(page_data):
                pages_assessed = len(self.get_project_pages(page_data))
            else:
                pages_assessed = 1

        # Handle NaN (None) score properly for JSON serialization
        page_data["completeness_score"] = score if score is not None else float("nan")
        page_data["completeness_summary"] = summary
        page_data["completeness_pages_assessed"] = pages_assessed if score is not None else 0

        return page_data

    def process_pages(
        self,
        pages: List[Dict[str, Any]],
        use_llm: bool = True,
        use_aggregated: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process all pages and add completeness data.

        When use_aggregated=True (default), each main project page's score
        is calculated using content from ALL pages within that project
        (main page + all subpages).

        Args:
            pages: List of page dictionaries
            use_llm: Whether to use LLM-based assessment (default: True)
            use_aggregated: If True, aggregate content from all project pages

        Returns:
            Updated pages with completeness_score and completeness_summary

        Example:
            >>> pages = assessor.process_pages(all_pages)
            >>> scored = [p for p in pages if p['completeness_score'] is not None]
        """
        logger.info(f"Assessing completeness for {len(pages)} pages")

        # Build page index for efficient lookup and aggregation
        self._build_page_index(pages)
        logger.info(f"Built index: {len(self._pages_by_parent_project)} unique projects")

        # First pass: identify main project pages
        main_pages = []
        sub_pages = []

        for page in pages:
            if self.is_main_project_page(page):
                main_pages.append(page)
            else:
                sub_pages.append(page)

        logger.info(f"Found {len(main_pages)} main project pages to assess")
        if use_aggregated:
            logger.info("Using aggregated content from all project pages for scoring")
        if use_llm and self.iliad_client:
            logger.info("Using LLM-based assessment")
        else:
            logger.info("Using keyword-based assessment")

        # Process main project pages
        for i, page in enumerate(main_pages):
            if (i + 1) % 10 == 0:
                logger.info(f"Assessing page {i + 1}/{len(main_pages)}")

            try:
                if use_llm and self.iliad_client:
                    score, summary, pages_assessed = self.calculate_completeness_llm(
                        page, use_aggregated=use_aggregated
                    )
                else:
                    score, summary = self.calculate_completeness(page, use_aggregated=use_aggregated)
                    if use_aggregated and self._pages_by_id:
                        pages_assessed = len(self.get_project_pages(page))
                    else:
                        pages_assessed = 1

                page["completeness_score"] = score if score is not None else float("nan")
                page["completeness_summary"] = summary
                page["completeness_pages_assessed"] = pages_assessed if score is not None else 0
            except Exception as e:
                title = page.get("title", "unknown")
                logger.error(f"Failed to assess '{title}': {e}")
                page["completeness_score"] = float("nan")
                page["completeness_summary"] = f"Assessment failed: {str(e)}"
                page["completeness_pages_assessed"] = 0

        # Set subpages to NaN
        for page in sub_pages:
            page["completeness_score"] = float("nan")
            page["completeness_summary"] = "N/A - Subpage"
            page["completeness_pages_assessed"] = 0

        # Log summary
        scored_pages = [p for p in pages if not math.isnan(p.get("completeness_score", float("nan")))]
        if scored_pages:
            avg_score = sum(p["completeness_score"] for p in scored_pages) / len(scored_pages)
            logger.info(f"Completeness assessment complete:")
            logger.info(f"  - {len(scored_pages)} main project pages assessed")
            logger.info(f"  - Average score: {avg_score:.1f}/100")
            logger.info(f"  - {len(sub_pages)} subpages marked N/A")
            if use_aggregated:
                total_pages_in_projects = sum(
                    len(self.get_project_pages(p)) for p in scored_pages
                )
                logger.info(f"  - Total pages included in assessments: {total_pages_in_projects}")
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
