"""
Document analysis using Iliad API.

This module provides high-level document analysis capabilities including:
- Content summarization
- Technology extraction
- Project description generation
- Custom prompt-based analysis

Example:
    >>> from iliad.analyze import DocumentAnalyzer
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>> client = IliadClient(IliadClientConfig.from_env())
    >>> analyzer = DocumentAnalyzer(client)
    >>> technologies = analyzer.extract_technologies("We use Python and PostgreSQL...")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .client import IliadClient, IliadModel


class DocumentAnalyzer:
    """
    High-level document analysis using Iliad API.

    Provides specialized methods for:
    - Analyzing file attachments
    - Extracting technologies from text
    - Generating document descriptions
    - Custom prompt-based analysis

    Attributes:
        client: IliadClient instance for API calls
        default_model: Model to use for analysis (defaults to gpt-4o-mini-global)

    Example:
        >>> analyzer = DocumentAnalyzer(iliad_client)
        >>> desc = analyzer.generate_description("/path/to/document.pdf")
        >>> techs = analyzer.extract_technologies("We built this with React and Node.js")
    """

    def __init__(
        self,
        client: IliadClient,
        default_model: str = "gpt-4o-mini-global",
    ) -> None:
        """Initialize document analyzer.

        Args:
            client: Configured IliadClient instance
            default_model: Default model for analysis tasks
        """
        self.client = client
        self.default_model = default_model
        logger.info(f"Initialized DocumentAnalyzer with model: {default_model}")

    def analyze_file(
        self,
        file_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Analyze a file with a custom prompt.

        Args:
            file_path: Path to file to analyze
            prompt: Analysis instructions/questions
            model: Model to use (defaults to self.default_model)

        Returns:
            Analysis result text

        Example:
            >>> result = analyzer.analyze_file("doc.pdf", "What are the key points?")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Analyzing file: {path.name}")

        response = self.client.analyze(
            files=[str(path)],
            prompt=prompt,
            model=model or self.default_model,
        )

        # Extract analysis from pairs
        if "pairs" in response and response["pairs"]:
            return response["pairs"][0][0]

        return self.client.extract_content(response)

    def analyze_text(
        self,
        text: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Analyze text content with a custom prompt.

        Args:
            text: Text content to analyze
            prompt: Analysis instructions/questions
            model: Model to use

        Returns:
            Analysis result text

        Example:
            >>> result = analyzer.analyze_text(doc_content, "Summarize this in 3 bullets")
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": text},
        ]

        response = self.client.chat(messages)
        return self.client.extract_content(response)

    def extract_technologies(
        self,
        content: str,
        title: Optional[str] = None,
    ) -> List[str]:
        """Extract technologies, tools, and frameworks from content.

        Uses LLM to identify technical stack including:
        - Programming languages (Python, R, Java, etc.)
        - Frameworks (React, Django, TensorFlow, etc.)
        - Databases (PostgreSQL, MongoDB, etc.)
        - Cloud services (AWS, Azure, etc.)
        - Tools (Docker, Airflow, Kubernetes, etc.)

        Args:
            content: Text content to analyze
            title: Optional page/document title for context

        Returns:
            List of extracted technology names (deduplicated)

        Example:
            >>> techs = analyzer.extract_technologies("Built with Python and Airflow")
            >>> print(techs)
            ['Python', 'Airflow']
        """
        if not content.strip():
            return []

        context = f"Title: {title}\n\n" if title else ""
        full_content = f"{context}{content[:8000]}"  # Limit content length

        prompt = """Extract all technologies, tools, frameworks, platforms, and programming languages mentioned in this text.

Return ONLY a JSON array of technology names, nothing else.
Example output: ["Python", "PostgreSQL", "Airflow", "AWS"]

If no technologies are found, return an empty array: []

Do not include generic terms like "database" or "cloud" - only specific named technologies."""

        try:
            response = self.analyze_text(full_content, prompt, model=self.default_model)

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            technologies = json.loads(response)

            if isinstance(technologies, list):
                # Deduplicate while preserving order
                seen = set()
                unique_techs = []
                for tech in technologies:
                    tech_lower = str(tech).lower()
                    if tech_lower not in seen:
                        seen.add(tech_lower)
                        unique_techs.append(str(tech))
                return unique_techs

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse technologies JSON: {e}")
            # Fallback: try to extract from plain text response
            return self._extract_technologies_fallback(response)
        except Exception as e:
            logger.error(f"Technology extraction failed: {e}")

        return []

    def _extract_technologies_fallback(self, response: str) -> List[str]:
        """Fallback technology extraction from non-JSON response.

        Args:
            response: LLM response that wasn't valid JSON

        Returns:
            List of extracted technology names
        """
        # Common technology patterns
        tech_patterns = [
            "Python", "R", "Java", "JavaScript", "TypeScript", "SQL",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Airflow",
            "Spark", "Hadoop", "Kafka", "TensorFlow", "PyTorch", "Scikit-learn",
            "React", "Vue", "Angular", "Django", "Flask", "FastAPI",
            "Tableau", "Power BI", "Looker", "Snowflake", "Databricks",
            "Git", "Jenkins", "GitLab", "SAS", "SPSS", "Stata",
        ]

        found = []
        response_lower = response.lower()

        for tech in tech_patterns:
            if tech.lower() in response_lower:
                found.append(tech)

        return found

    def generate_description(
        self,
        file_path: Union[str, Path],
        max_length: int = 200,
    ) -> str:
        """Generate a brief description of a document.

        Args:
            file_path: Path to document
            max_length: Maximum description length in characters

        Returns:
            Brief description of document content

        Example:
            >>> desc = analyzer.generate_description("architecture.pdf")
            >>> print(desc)
            "System architecture diagram showing data flow between services..."
        """
        prompt = f"""Provide a brief, one-paragraph description of this document's content.
The description should be informative and capture the key purpose or content.
Keep it under {max_length} characters."""

        return self.analyze_file(file_path, prompt)

    def summarize_content(
        self,
        content: str,
        style: str = "bullet_points",
        max_points: int = 5,
    ) -> str:
        """Summarize text content in specified style.

        Args:
            content: Text to summarize
            style: Summary style - 'bullet_points', 'paragraph', 'executive'
            max_points: Maximum number of bullet points (if applicable)

        Returns:
            Summarized content

        Example:
            >>> summary = analyzer.summarize_content(long_text, style="bullet_points")
        """
        style_prompts = {
            "bullet_points": f"""Summarize this content as {max_points} concise bullet points.
Each bullet should capture a key point or finding.
Use clear, direct language.""",
            "paragraph": """Summarize this content in a single paragraph of 2-3 sentences.
Capture the main purpose, key findings, and conclusions.""",
            "executive": """Provide an executive summary of this content.
Include: Purpose, Key Findings, Recommendations (if applicable).
Keep it concise and action-oriented.""",
        }

        prompt = style_prompts.get(style, style_prompts["bullet_points"])

        return self.analyze_text(content, prompt)

    def assess_completeness(
        self,
        content: str,
        required_sections: List[str],
    ) -> Dict[str, Any]:
        """Assess content completeness against required sections.

        Args:
            content: Content to assess
            required_sections: List of section names that should be present

        Returns:
            Dict with 'score' (0-100), 'present', 'missing', 'summary'

        Example:
            >>> sections = ["Introduction", "Methods", "Results", "Conclusion"]
            >>> result = analyzer.assess_completeness(paper_text, sections)
            >>> print(f"Score: {result['score']}, Missing: {result['missing']}")
        """
        sections_list = "\n".join(f"- {s}" for s in required_sections)

        prompt = f"""Analyze this content and determine which of the following sections are present:

{sections_list}

Return a JSON object with:
{{
  "present": ["list of sections that are present"],
  "missing": ["list of sections that are missing or inadequate"],
  "score": <number from 0-100 representing completeness>
}}

A section is "present" if the content adequately covers that topic, even if not explicitly labeled.
Return ONLY the JSON object, no explanation."""

        try:
            response = self.analyze_text(content, prompt)

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            result = json.loads(response)

            # Ensure required fields
            result.setdefault("present", [])
            result.setdefault("missing", [])
            result.setdefault("score", 0)

            # Generate summary
            present_count = len(result["present"])
            total_count = len(required_sections)
            result["summary"] = (
                f"Score: {result['score']}/100. "
                f"{present_count}/{total_count} sections present. "
                f"Missing: {', '.join(result['missing'][:3]) if result['missing'] else 'None'}"
            )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse completeness JSON: {e}")
            return {
                "present": [],
                "missing": required_sections,
                "score": 0,
                "summary": "Assessment failed - could not parse response",
            }
        except Exception as e:
            logger.error(f"Completeness assessment failed: {e}")
            return {
                "present": [],
                "missing": required_sections,
                "score": 0,
                "summary": f"Assessment failed: {str(e)}",
            }
