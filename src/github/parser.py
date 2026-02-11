"""Parser for GitHub repository content."""

from typing import List, Dict, Any
import re
from loguru import logger


class GitHubParser:
    """
    Parser for processing GitHub repository content.

    This class extracts and structures information from GitHub repositories
    for use in the RAG pipeline.
    """

    def __init__(self) -> None:
        """Initialize the GitHub parser."""
        logger.debug("Initialized GitHub parser")

    def parse_readme(self, readme_content: str) -> Dict[str, Any]:
        """
        Parse README content to extract structured information.

        Args:
            readme_content: Raw README markdown content.

        Returns:
            Dictionary with parsed README sections and metadata.
        """
        if not readme_content:
            return {"sections": [], "links": [], "code_blocks": []}

        sections = self._extract_sections(readme_content)
        links = self._extract_markdown_links(readme_content)
        code_blocks = self._extract_code_blocks(readme_content)

        parsed = {
            "sections": sections,
            "links": links,
            "code_blocks": code_blocks,
            "raw_content": readme_content,
        }

        logger.debug(f"Parsed README with {len(sections)} sections")
        return parsed

    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extract markdown sections based on headers.

        Args:
            content: Markdown content.

        Returns:
            List of dictionaries with section headers and content.
        """
        sections = []
        lines = content.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_section:
                    sections.append(
                        {
                            "level": current_section["level"],
                            "title": current_section["title"],
                            "content": "\n".join(current_content).strip(),
                        }
                    )

                current_section = {
                    "level": len(header_match.group(1)),
                    "title": header_match.group(2),
                }
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections.append(
                {
                    "level": current_section["level"],
                    "title": current_section["title"],
                    "content": "\n".join(current_content).strip(),
                }
            )

        return sections

    def _extract_markdown_links(self, content: str) -> List[Dict[str, str]]:
        """
        Extract markdown links from content.

        Args:
            content: Markdown content.

        Returns:
            List of dictionaries with link text and URLs.
        """
        pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        matches = re.findall(pattern, content)
        links = [{"text": text, "url": url} for text, url in matches]
        return links

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from markdown content.

        Args:
            content: Markdown content.

        Returns:
            List of dictionaries with language and code content.
        """
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        code_blocks = [
            {"language": lang or "plain", "code": code.strip()} for lang, code in matches
        ]
        return code_blocks

    def parse_python_file(self, file_content: str, file_path: str) -> Dict[str, Any]:
        """
        Parse Python file to extract functions, classes, and docstrings.

        Args:
            file_content: Python source code content.
            file_path: Path to the Python file.

        Returns:
            Dictionary with parsed Python elements.
        """
        parsed = {
            "path": file_path,
            "imports": self._extract_imports(file_content),
            "functions": self._extract_functions(file_content),
            "classes": self._extract_classes(file_content),
            "docstring": self._extract_module_docstring(file_content),
            "content": file_content,
        }

        logger.debug(f"Parsed Python file: {file_path}")
        return parsed

    def _extract_imports(self, content: str) -> List[str]:
        """
        Extract import statements from Python code.

        Args:
            content: Python source code.

        Returns:
            List of import statements.
        """
        pattern = r"^(?:from\s+[\w.]+\s+)?import\s+.+$"
        imports = re.findall(pattern, content, re.MULTILINE)
        return imports

    def _extract_module_docstring(self, content: str) -> str:
        """
        Extract module-level docstring from Python code.

        Args:
            content: Python source code.

        Returns:
            Module docstring or empty string.
        """
        pattern = r'^"""(.*?)"""'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_functions(self, content: str) -> List[Dict[str, str]]:
        """
        Extract function definitions from Python code.

        Args:
            content: Python source code.

        Returns:
            List of dictionaries with function information.
        """
        pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*[\w\[\], ]+)?\s*:\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')?'
        matches = re.findall(pattern, content, re.DOTALL)

        functions = []
        for match in matches:
            name, params, docstring1, docstring2 = match
            docstring = (docstring1 or docstring2).strip()
            functions.append(
                {"name": name, "parameters": params.strip(), "docstring": docstring}
            )

        return functions

    def _extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from Python code.

        Args:
            content: Python source code.

        Returns:
            List of dictionaries with class information.
        """
        pattern = r'class\s+(\w+)(?:\((.*?)\))?\s*:\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')?'
        matches = re.findall(pattern, content, re.DOTALL)

        classes = []
        for match in matches:
            name, bases, docstring1, docstring2 = match
            docstring = (docstring1 or docstring2).strip()
            classes.append(
                {"name": name, "bases": bases.strip(), "docstring": docstring}
            )

        return classes

    def parse_repository_summary(self, repo_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse complete repository summary for RAG pipeline.

        Args:
            repo_summary: Repository summary from GitHubClient.

        Returns:
            Structured dictionary ready for vectorization.
        """
        info = repo_summary.get("info", {})
        readme = repo_summary.get("readme", "")

        parsed_readme = self.parse_readme(readme) if readme else {}

        structured_data = {
            "repository_name": info.get("full_name", ""),
            "description": info.get("description", ""),
            "url": repo_summary.get("url", ""),
            "language": info.get("language", ""),
            "topics": info.get("topics", []),
            "readme_sections": parsed_readme.get("sections", []),
            "readme_links": parsed_readme.get("links", []),
            "structure": repo_summary.get("root_structure", []),
            "metadata": {
                "stars": info.get("stars", 0),
                "forks": info.get("forks", 0),
                "created_at": info.get("created_at"),
                "updated_at": info.get("updated_at"),
            },
        }

        logger.debug(f"Parsed repository summary for: {info.get('full_name')}")
        return structured_data

    def create_text_summary(self, parsed_data: Dict[str, Any]) -> str:
        """
        Create a text summary from parsed repository data.

        Args:
            parsed_data: Parsed repository data.

        Returns:
            Human-readable text summary.
        """
        parts = []

        parts.append(f"Repository: {parsed_data['repository_name']}")
        if parsed_data.get("description"):
            parts.append(f"Description: {parsed_data['description']}")

        parts.append(f"Language: {parsed_data.get('language', 'Unknown')}")

        if parsed_data.get("topics"):
            parts.append(f"Topics: {', '.join(parsed_data['topics'])}")

        if parsed_data.get("readme_sections"):
            parts.append("\nREADME Contents:")
            for section in parsed_data["readme_sections"]:
                parts.append(f"\n{section['title']}")
                if section.get("content"):
                    parts.append(section["content"][:500])

        summary = "\n".join(parts)
        logger.debug("Created text summary for repository")
        return summary
