"""
Unified Iliad API client with authentication and retry logic.

This module provides a centralized client for all Iliad API interactions,
supporting multiple endpoints and response formats.

Example:
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>> config = IliadClientConfig(api_key="your_key", base_url="https://iliad.abbvie.com")
    >>> client = IliadClient(config)
    >>> response = client.chat([{"role": "user", "content": "Hello"}])
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, BinaryIO, Dict, List, Optional, Union, Iterator

import requests
from loguru import logger


class IliadModel(Enum):
    """Available Iliad LLM models.

    Attributes:
        GPT_4O: OpenAI GPT-4o model
        GPT_4O_MINI: OpenAI GPT-4o-mini model (faster, cheaper)
        GPT_4O_MINI_GLOBAL: GPT-4o-mini with global routing
        GPT_5_MINI_GLOBAL: GPT-5-mini with global routing (recommended for metadata)
        CLAUDE_4_SONNET: Anthropic Claude 4 Sonnet
        CLAUDE_45_SONNET: Anthropic Claude 4.5 Sonnet
        CLAUDE_45_HAIKU: Anthropic Claude 4.5 Haiku (fastest Claude)
        CLAUDE_45_OPUS: Anthropic Claude 4.5 Opus (most capable)
    """

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O_MINI_GLOBAL = "gpt-4o-mini-global"
    GPT_41_GLOBAL = "gpt-41-global"
    GPT_41_MINI_GLOBAL = "gpt-41-mini-global"
    GPT_41_NANO_GLOBAL = "gpt-41-nano-global"
    GPT_5_MINI_GLOBAL = "gpt-5-mini-global"
    GPT_5_CHAT_GLOBAL = "gpt-5-chat-global"
    CLAUDE_4_SONNET = "claude-4-sonnet"
    CLAUDE_45_SONNET = "claude-4.5-sonnet"
    CLAUDE_45_HAIKU = "claude-4.5-haiku"
    CLAUDE_45_OPUS = "claude-4.5-opus"


@dataclass
class IliadClientConfig:
    """Configuration for Iliad API client.

    Attributes:
        api_key: Iliad API key for authentication
        base_url: Base URL for Iliad API endpoints
        default_model: Default model for chat completions
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
    """

    api_key: str
    base_url: str
    default_model: str = "gpt-4o-mini-global"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "IliadClientConfig":
        """Create config from environment variables.

        Returns:
            IliadClientConfig instance

        Raises:
            ValueError: If required environment variables are missing
        """
        api_key = os.getenv("ILIAD_API_KEY", "")
        base_url = os.getenv("ILIAD_API_URL", "")

        if not api_key:
            raise ValueError("ILIAD_API_KEY environment variable is required")
        if not base_url:
            raise ValueError("ILIAD_API_URL environment variable is required")

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            default_model=os.getenv("ILIAD_DEFAULT_MODEL", "gpt-4o-mini-global"),
        )


class IliadClient:
    """
    Unified client for Iliad API endpoints.

    Provides methods for:
    - /api/v1/analyze - Document analysis
    - /api/v1/recognize - Text extraction
    - /api/v1/recognize/ocr - OCR for images
    - /api/v1/router/chat - Chat with auto-routing (SSE)
    - Chat completions (default endpoint)

    Attributes:
        config: Client configuration
        session: Requests session for connection pooling

    Example:
        >>> client = IliadClient(IliadClientConfig.from_env())
        >>> response = client.chat([{"role": "user", "content": "Analyze this"}])
        >>> print(response["completion"]["content"])
    """

    def __init__(self, config: IliadClientConfig) -> None:
        """Initialize Iliad client.

        Args:
            config: Client configuration with API key and base URL
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": config.api_key,
                "Accept": "application/json",
            }
        )
        logger.info(f"Initialized IliadClient with base URL: {config.base_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: JSON body data
            data: Form data
            files: File uploads
            stream: Whether to stream response

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: If all retries fail
        """
        url = f"{self.config.base_url}{endpoint}"
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}/{self.config.max_retries}: {method} {url}")

                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    data=data,
                    files=files,
                    timeout=self.config.timeout,
                    stream=stream,
                )
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

        logger.error(f"All {self.config.max_retries} attempts failed for {url}")
        raise last_exception

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to config.default_model)

        Returns:
            Response dict with structure:
            {
                "response_id": "...",
                "completion": {
                    "role": "assistant",
                    "content": "response text"
                }
            }

        Example:
            >>> response = client.chat([{"role": "user", "content": "Hello"}])
            >>> print(response["completion"]["content"])
        """
        payload = {"messages": messages}
        if model:
            payload["model"] = model

        logger.debug(f"Chat request with {len(messages)} messages")
        response = self._make_request("POST", "", json_data=payload)

        result = response.json()
        logger.debug(f"Chat response received: {len(str(result))} chars")
        return result

    def analyze(
        self,
        files: List[Union[str, BinaryIO]],
        prompt: str,
        model: Optional[str] = None,
        source_language: str = "en-US",
    ) -> Dict[str, Any]:
        """Analyze documents using Iliad /analyze endpoint.

        Args:
            files: List of file paths or file objects to analyze
            prompt: Instructions/questions for the model
            model: Model to use (defaults to gpt-4o)
            source_language: Language of uploaded files

        Returns:
            Response dict with structure:
            {
                "response_id": "...",
                "pairs": [["analysis", "source_text"], ...],
                "cost": 2.75
            }

        Example:
            >>> result = client.analyze(["/path/to/doc.pdf"], "Summarize this document")
            >>> print(result["pairs"][0][0])  # Analysis text
        """
        file_handles = []
        files_param = []

        try:
            for f in files:
                if isinstance(f, str):
                    fh = open(f, "rb")
                    file_handles.append(fh)
                    files_param.append(("files", fh))
                else:
                    files_param.append(("files", f))

            data = {
                "prompt": prompt,
                "source_language": source_language,
            }
            if model:
                data["model"] = model

            logger.info(f"Analyzing {len(files)} file(s)")
            response = self._make_request(
                "POST",
                "/api/v1/analyze",
                data=data,
                files=files_param,
            )

            return response.json()

        finally:
            for fh in file_handles:
                fh.close()

    def recognize(
        self,
        file: Union[str, BinaryIO],
        source_language: str = "en-US",
    ) -> str:
        """Extract text from document using /recognize endpoint.

        Supported file types: .pdf, .docx, .pptx, .rtf, .jpg, .jpeg, .png,
        .html, .xhtml, .epub, .eml, .json, .md, .xml, .txt, .odt, .ods,
        .odp, .odg, .ppt, .doc, .xls, .xlsm, .xlsx, .csv, .tsv

        Args:
            file: File path or file object
            source_language: Language of the document

        Returns:
            Extracted text content

        Example:
            >>> text = client.recognize("/path/to/document.pdf")
            >>> print(text[:100])
        """
        file_handle = None

        try:
            if isinstance(file, str):
                file_handle = open(file, "rb")
                files_param = {"file": file_handle}
            else:
                files_param = {"file": file}

            data = {"source_language": source_language}

            logger.info("Recognizing text from document")
            response = self._make_request(
                "POST",
                "/api/v1/recognize",
                data=data,
                files=files_param,
            )

            result = response.json()
            return result.get("text", "")

        finally:
            if file_handle:
                file_handle.close()

    def recognize_ocr(
        self,
        image: Union[str, BinaryIO],
    ) -> str:
        """Extract text from image using OCR.

        Supported formats: jpeg, png, tiff

        Args:
            image: Image file path or file object

        Returns:
            Extracted text content

        Example:
            >>> text = client.recognize_ocr("/path/to/image.png")
            >>> print(text)
        """
        file_handle = None

        try:
            if isinstance(image, str):
                file_handle = open(image, "rb")
                files_param = {"file": file_handle}
            else:
                files_param = {"file": image}

            logger.info("Running OCR on image")
            response = self._make_request(
                "POST",
                "/api/v1/recognize/ocr",
                files=files_param,
            )

            result = response.json()
            return result.get("text", "")

        finally:
            if file_handle:
                file_handle.close()

    def recognize_markdown(
        self,
        file: Union[str, BinaryIO],
        max_workers: int = 16,
    ) -> str:
        """Convert document to markdown format.

        Supported file types: .pdf, .docx, .pptx, .rtf, .html, .xhtml,
        .eml, .xml, .txt, .odt, .ods, .odp, .odg, .ppt, .doc, .xls,
        .xlsm, .xlsx, .csv, .tsv, .svg

        Args:
            file: File path or file object
            max_workers: Number of concurrent LLM image captioning calls (1-32)

        Returns:
            Markdown content

        Example:
            >>> md = client.recognize_markdown("/path/to/presentation.pptx")
            >>> print(md)
        """
        file_handle = None

        try:
            if isinstance(file, str):
                file_handle = open(file, "rb")
                files_param = {"file": file_handle}
            else:
                files_param = {"file": file}

            data = {"max_workers": str(max_workers)}

            logger.info("Converting document to markdown")
            response = self._make_request(
                "POST",
                "/api/v1/recognize/markdown",
                data=data,
                files=files_param,
            )

            result = response.json()
            return result.get("text", result.get("markdown", ""))

        finally:
            if file_handle:
                file_handle.close()

    def router_chat(
        self,
        messages: List[Dict[str, str]],
        files: Optional[List[Dict[str, str]]] = None,
        model: str = "auto",
        system_prompt: str = "",
        max_iters: int = 10,
    ) -> Iterator[Dict[str, Any]]:
        """Send request to router/chat endpoint with SSE streaming.

        The router automatically determines which tools to use (web search,
        code execution, image generation, etc.).

        Args:
            messages: Conversation history
            files: Files to process, each with 'id' and 'text' preview
            model: Model to use (auto, gpt-4o, claude-4.5-sonnet, etc.)
            system_prompt: System instructions
            max_iters: Maximum tool-use iterations

        Yields:
            Event dictionaries from SSE stream with types:
            - 'start': Beginning of execution
            - 'delta': Incremental content updates
            - 'content': Complete non-streaming content
            - 'end': End of execution

        Example:
            >>> for event in client.router_chat(
            ...     messages=[{"role": "user", "content": "Create a bar chart"}],
            ...     files=[{"id": "data.csv", "text": "a,b\\n1,2"}]
            ... ):
            ...     if event.get("type") == "content":
            ...         print(event.get("content"))
        """
        payload = {
            "messages": messages,
            "model": model,
            "max_iters": max_iters,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if files:
            payload["files"] = files

        logger.info(f"Router chat request with {len(messages)} messages")
        response = self._make_request(
            "POST",
            "/api/v1/router/chat",
            json_data=payload,
            stream=True,
        )

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    import json

                    try:
                        data = json.loads(line_str[6:])
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE data: {line_str}")

    def extract_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from API response.

        Handles various response formats from different endpoints.

        Args:
            response: API response dictionary

        Returns:
            Extracted text content
        """
        if "completion" in response:
            completion = response["completion"]
            if isinstance(completion, dict):
                return completion.get("content", "")
            return str(completion)

        if "text" in response:
            return response["text"]

        if "pairs" in response and response["pairs"]:
            return response["pairs"][0][0]

        return str(response)
