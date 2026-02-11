"""GitHub API client for retrieving repository information and code."""

from typing import List, Dict, Optional, Any
from github import Github, GithubException
from loguru import logger
import re


class GitHubClient:
    """
    Client for interacting with GitHub API.

    This class provides methods to connect to GitHub repositories,
    retrieve repository metadata, and extract code content.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token for authentication.
                   Optional for public repositories.
        """
        self.client = Github(token) if token else Github()
        self.token = token
        logger.info("Initialized GitHub client")

    def parse_github_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Parse GitHub URL to extract owner and repository name.

        Args:
            url: GitHub repository URL.

        Returns:
            Dictionary with 'owner' and 'repo' keys, or None if invalid.
        """
        pattern = r"https://github\.com/([\w\-]+)/([\w\-.]+?)(?:\.git|/|$)"
        match = re.match(pattern, url)

        if match:
            return {"owner": match.group(1), "repo": match.group(2)}

        logger.warning(f"Invalid GitHub URL format: {url}")
        return None

    def get_repository(self, owner: str, repo: str) -> Optional[Any]:
        """
        Get a GitHub repository object.

        Args:
            owner: Repository owner/organization name.
            repo: Repository name.

        Returns:
            GitHub repository object or None if not found.
        """
        try:
            repository = self.client.get_repo(f"{owner}/{repo}")
            logger.info(f"Successfully accessed repository: {owner}/{repo}")
            return repository
        except GithubException as e:
            logger.error(f"Error accessing repository {owner}/{repo}: {str(e)}")
            return None

    def get_repository_from_url(self, url: str) -> Optional[Any]:
        """
        Get repository object from GitHub URL.

        Args:
            url: Full GitHub repository URL.

        Returns:
            GitHub repository object or None if not found.
        """
        parsed = self.parse_github_url(url)
        if not parsed:
            return None

        return self.get_repository(parsed["owner"], parsed["repo"])

    def get_repository_info(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive repository information.

        Args:
            repo_url: GitHub repository URL.

        Returns:
            Dictionary containing repository metadata and statistics.
        """
        repo = self.get_repository_from_url(repo_url)
        if not repo:
            return None

        try:
            info = {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "language": repo.language,
                "languages": repo.get_languages(),
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "created_at": repo.created_at.isoformat() if repo.created_at else None,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                "topics": repo.get_topics(),
                "default_branch": repo.default_branch,
            }

            logger.debug(f"Retrieved info for repository: {repo.full_name}")
            return info

        except GithubException as e:
            logger.error(f"Error getting repository info: {str(e)}")
            return None

    def get_readme(self, repo_url: str) -> Optional[str]:
        """
        Get repository README content.

        Args:
            repo_url: GitHub repository URL.

        Returns:
            README content as string, or None if not found.
        """
        repo = self.get_repository_from_url(repo_url)
        if not repo:
            return None

        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode("utf-8")
            logger.debug(f"Retrieved README for: {repo.full_name}")
            return content
        except GithubException as e:
            logger.warning(f"No README found for {repo.full_name}: {str(e)}")
            return None

    def get_file_content(self, repo_url: str, file_path: str) -> Optional[str]:
        """
        Get content of a specific file from repository.

        Args:
            repo_url: GitHub repository URL.
            file_path: Path to file within repository.

        Returns:
            File content as string, or None if not found.
        """
        repo = self.get_repository_from_url(repo_url)
        if not repo:
            return None

        try:
            file_content = repo.get_contents(file_path)
            if isinstance(file_content, list):
                logger.warning(f"{file_path} is a directory, not a file")
                return None

            content = file_content.decoded_content.decode("utf-8")
            logger.debug(f"Retrieved file: {file_path} from {repo.full_name}")
            return content

        except GithubException as e:
            logger.warning(f"File {file_path} not found in {repo.full_name}: {str(e)}")
            return None

    def get_directory_contents(
        self, repo_url: str, path: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get contents of a directory in the repository.

        Args:
            repo_url: GitHub repository URL.
            path: Path to directory within repository. Defaults to root.

        Returns:
            List of dictionaries with file/directory information.
        """
        repo = self.get_repository_from_url(repo_url)
        if not repo:
            return []

        try:
            contents = repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]

            items = []
            for item in contents:
                items.append(
                    {
                        "name": item.name,
                        "path": item.path,
                        "type": item.type,
                        "size": item.size,
                        "url": item.html_url,
                    }
                )

            logger.debug(f"Retrieved {len(items)} items from {path or 'root'}")
            return items

        except GithubException as e:
            logger.error(f"Error getting directory contents: {str(e)}")
            return []

    def get_python_files(self, repo_url: str) -> List[Dict[str, str]]:
        """
        Recursively get all Python files from repository.

        Args:
            repo_url: GitHub repository URL.

        Returns:
            List of dictionaries with Python file paths and content.
        """
        repo = self.get_repository_from_url(repo_url)
        if not repo:
            return []

        python_files = []

        def traverse_directory(path: str = "") -> None:
            """Recursively traverse repository directories."""
            try:
                contents = repo.get_contents(path)
                if not isinstance(contents, list):
                    contents = [contents]

                for item in contents:
                    if item.type == "dir":
                        traverse_directory(item.path)
                    elif item.type == "file" and item.name.endswith(".py"):
                        try:
                            content = item.decoded_content.decode("utf-8")
                            python_files.append({"path": item.path, "content": content})
                            logger.debug(f"Retrieved Python file: {item.path}")
                        except Exception as e:
                            logger.warning(f"Could not decode {item.path}: {str(e)}")

            except GithubException as e:
                logger.warning(f"Error traversing {path}: {str(e)}")

        traverse_directory()
        logger.info(f"Retrieved {len(python_files)} Python files from repository")
        return python_files

    def get_repository_summary(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive repository summary including info, README, and code structure.

        Args:
            repo_url: GitHub repository URL.

        Returns:
            Dictionary with repository information, README, and file structure.
        """
        logger.info(f"Getting comprehensive summary for: {repo_url}")

        info = self.get_repository_info(repo_url)
        if not info:
            return None

        readme = self.get_readme(repo_url)
        root_contents = self.get_directory_contents(repo_url)

        summary = {
            "info": info,
            "readme": readme,
            "root_structure": root_contents,
            "url": repo_url,
        }

        logger.info(f"Successfully created summary for: {info['full_name']}")
        return summary
