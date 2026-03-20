"""Query preprocessing module for keyword extraction and query analysis.

This module provides functionality to extract structured information from user queries
using LLM-based analysis for improved document retrieval and re-ranking.

Uses the Iliad API for intelligent extraction of:
- Keywords and key phrases
- Project names and acronyms
- Person names
- Dates and time references
- Query intent and comparison detection
"""

import re
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Use shared JSON parsing utility
from utils.json_parser import parse_llm_json_response

# Runtime imports - IliadClient is optional
ILIAD_AVAILABLE = False
IliadClient: Any = None
IliadClientConfig: Any = None

try:
    from iliad.client import IliadClient, IliadClientConfig
    ILIAD_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ProcessedQuery:
    """Container for processed query information."""

    original_query: str
    cleaned_query: str
    keywords: List[str]
    lemmatized_keywords: List[str]
    potential_project_names: List[str]
    potential_person_names: List[str]
    is_comparative: bool = False
    comparative_entities: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    query_intent: str = "informational"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Initialize default values for optional fields."""
        if self.comparative_entities is None:
            self.comparative_entities = []
        if self.dates is None:
            self.dates = []
        if self.technologies is None:
            self.technologies = []


# Few-shot examples for query analysis
FEW_SHOT_EXAMPLES = """

Example 0:
Query: "What is passport?"
Output:
{
    "cleaned_query": "what is passport",
    "keywords": ["passport"],
    "project_names": ["passport"],
    "person_names": [],
    "dates": [],
    "technologies": [],
    "is_comparative": false,
    "comparative_entities": [],
    "query_intent": "informational"
}

Example 1:
Query: "What is the ALFA project and who is working on it?"
Output:
{
    "cleaned_query": "ALFA project who is working on it",
    "keywords": ["ALFA", "project", "working"],
    "project_names": ["ALFA"],
    "person_names": [],
    "dates": [],
    "technologies": [],
    "is_comparative": false,
    "comparative_entities": [],
    "query_intent": "informational"
}

Example 2:
Query: "Compare the GraphRAG approach to traditional RAG for document retrieval"
Output:
{
    "cleaned_query": "GraphRAG approach traditional RAG document retrieval",
    "keywords": ["GraphRAG", "traditional", "RAG", "document", "retrieval"],
    "project_names": ["GraphRAG"],
    "person_names": [],
    "dates": [],
    "technologies": ["RAG", "GraphRAG"],
    "is_comparative": true,
    "comparative_entities": ["GraphRAG", "traditional RAG"],
    "query_intent": "comparison"
}

Example 3:
Query: "What projects did John Smith work on in 2024 using Python and Airflow?"
Output:
{
    "cleaned_query": "projects John Smith work 2024 Python Airflow",
    "keywords": ["projects", "work", "Python", "Airflow"],
    "project_names": [],
    "person_names": ["John Smith"],
    "dates": ["2024"],
    "technologies": ["Python", "Airflow"],
    "is_comparative": false,
    "comparative_entities": [],
    "query_intent": "informational"
}

Example 4:
Query: "What are the data sources for the missing data KRI."
Output:
{
    "cleaned_query": "data source for missing data KRI",
    "keywords": ["data source"],
    "project_names": ["misisng data KRI"],
    "person_names": [""],
    "dates": [""],
    "technologies": ["data"],
    "is_comparative": false,
    "comparative_entities": [],
    "query_intent": ""
}


Example 5:
Query: "Compare the methods of clover to convoke"
Output:
{
    "cleaned_query": "compare method of clover and convoke",
    "keywords": ["compare", "method"],
    "project_names": ["clover", "convoke"],
    "person_names": [""],
    "dates": [""],
    "technologies": [""],
    "is_comparative": true,
    "comparative_entities": ["clover","convoke"],
    "query_intent": "comparitive"
}

"""

QUERY_ANALYSIS_PROMPT = """You are a query analysis assistant for a data science documentation system.
Your task is to extract structured information from user queries to improve search and retrieval.

Analyze the query and extract the following information:
- cleaned_query: The query with question words removed (what, how, etc.) but keeping important terms
- keywords: Important search terms (nouns, verbs, technical terms) - lowercase unless acronym
- project_names: Names of projects mentioned (often acronyms like ALFA, DSA, ATLAS)
- person_names: Full names of people mentioned (First Last format)
- dates: Any date references (years, quarters, months, date ranges)
- technologies: Programming languages, tools, frameworks, methodologies mentioned
- is_comparative: Boolean - true if query compares multiple things
- comparative_entities: If comparative, the things being compared
- query_intent: One of "informational", "comparison", "aggregation", "listing", "how-to"

Important guidelines:
- Project names can be acronyms, capatlized words, or lower case words
- Distinguish between project names and technology names (Python is technology, ALFA is project)
- Person names generally follow a First Last convenction, but can take other formats. (Smith, John == John Smith)
- Dates can be years (2024), quarters (Q1 2025), months (January 2024), or ranges
- Query intent helps route to appropriate search pipeline

{examples}

Now analyze this query:
Query: "{query}"

Respond with ONLY valid JSON matching this schema:
{{
    "cleaned_query": "string",
    "keywords": ["string"],
    "project_names": ["string"],
    "person_names": ["string"],
    "dates": ["string"],
    "technologies": ["string"],
    "is_comparative": boolean,
    "comparative_entities": ["string"],
    "query_intent": "string"
}}
"""


class QueryProcessor:
    """
    Processes user queries to extract structured information using LLM.

    This class provides methods to:
    - Extract keywords, entities, and metadata from queries
    - Identify projects, people, dates, and technologies
    - Detect comparative queries
    - Clean queries for improved search

    Uses the Iliad API for intelligent extraction with few-shot prompting.
    Falls back to regex-based extraction if LLM is unavailable.
    """

    def __init__(
        self,
        iliad_client: Optional[Any] = None,
        model: str = "gpt-5-mini-global",
        use_llm: bool = True,
        use_few_shot: bool = True,
    ) -> None:
        """
        Initialize the query processor.

        Args:
            iliad_client: Optional pre-configured Iliad client.
            model: Model to use for LLM extraction.
            use_llm: Whether to use LLM-based extraction (falls back to regex if False).
            use_few_shot: Whether to include few-shot examples in prompt.
        """
        self.iliad_client = iliad_client
        self.model = model
        self.use_llm = use_llm
        self.use_few_shot = use_few_shot

        # Try to initialize client from environment if not provided
        if self.iliad_client is None and use_llm and ILIAD_AVAILABLE:
            try:
                config = IliadClientConfig.from_env()
                self.iliad_client = IliadClient(config)
                logger.info("Initialized Iliad client from environment")
            except (ValueError, Exception) as e:
                logger.warning(f"Could not initialize Iliad client: {e}")
                self.use_llm = False

        logger.info(
            f"QueryProcessor initialized (LLM: {self.use_llm and self.iliad_client is not None}, "
            f"Model: {model}, Few-shot: {use_few_shot})"
        )

    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process a user query to extract structured information.

        Args:
            query: The original user query.

        Returns:
            ProcessedQuery object containing extracted information.
        """
        logger.info(f"Processing query: '{query}'")

        # Try LLM-based extraction first
        if self.use_llm and self.iliad_client is not None:
            try:
                result = self._extract_with_llm(query)
                if result is not None:
                    logger.info(
                        f"LLM extraction successful - "
                        f"Keywords: {len(result.keywords)}, "
                        f"Projects: {len(result.potential_project_names)}, "
                        f"People: {len(result.potential_person_names)}"
                    )
                    return result
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}, falling back to regex")

        # Fallback to regex-based extraction
        logger.info("Using regex-based extraction")
        return self._extract_with_regex(query)

    def _extract_with_llm(self, query: str) -> Optional[ProcessedQuery]:
        """
        Extract query information using LLM.

        Args:
            query: The user query.

        Returns:
            ProcessedQuery if successful, None otherwise.
        """
        # Build prompt with optional few-shot examples
        examples = FEW_SHOT_EXAMPLES if self.use_few_shot else ""
        prompt = QUERY_ANALYSIS_PROMPT.format(examples=examples, query=query)

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.iliad_client.chat(messages=messages, model=self.model)
            content = self.iliad_client.extract_content(response)

            # Parse JSON response
            parsed = self._parse_llm_response(content)
            if parsed is None:
                return None

            # Build ProcessedQuery from parsed response
            result = ProcessedQuery(
                original_query=query,
                cleaned_query=parsed.get("cleaned_query", query),
                keywords=parsed.get("keywords", []),
                lemmatized_keywords=parsed.get("keywords", []),  # LLM handles normalization
                potential_project_names=parsed.get("project_names", []),
                potential_person_names=parsed.get("person_names", []),
                is_comparative=parsed.get("is_comparative", False),
                comparative_entities=parsed.get("comparative_entities", []),
                dates=parsed.get("dates", []),
                technologies=parsed.get("technologies", []),
                query_intent=parsed.get("query_intent", "informational"),
                confidence=0.9,  # LLM extraction has high confidence
            )

            return result

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[dict]:
        """
        Parse JSON response from LLM.

        Uses the shared JSON parsing utility from utils.json_parser.

        Args:
            content: Raw LLM response string.

        Returns:
            Parsed dictionary or None if parsing fails.
        """
        return parse_llm_json_response(content)

    def _extract_with_regex(self, query: str) -> ProcessedQuery:
        """
        Fallback regex-based extraction.

        Args:
            query: The user query.

        Returns:
            ProcessedQuery with regex-extracted information.
        """
        # Clean query
        cleaned = self._remove_question_patterns(query)

        # Extract components using regex
        keywords = self._extract_keywords_regex(cleaned)
        project_names = self._extract_project_names_regex(query)
        person_names = self._extract_person_names_regex(query)
        dates = self._extract_dates_regex(query)
        technologies = self._extract_technologies_regex(query)
        is_comparative, comparative_entities = self._detect_comparative_regex(query)

        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            keywords=keywords,
            lemmatized_keywords=keywords,
            potential_project_names=project_names,
            potential_person_names=person_names,
            is_comparative=is_comparative,
            comparative_entities=comparative_entities,
            dates=dates,
            technologies=technologies,
            query_intent=self._infer_intent_regex(query),
            confidence=0.5,  # Regex has lower confidence
        )

    def _remove_question_patterns(self, text: str) -> str:
        """Remove common question patterns from text."""
        patterns = [
            r"^what\s+(is|are|was|were)\s+",
            r"^who\s+(is|are|was|were)\s+",
            r"^where\s+(is|are|was|were)\s+",
            r"^when\s+(is|are|was|were)\s+",
            r"^how\s+(do|does|did|is|are|can|could|would|should)\s+",
            r"^can\s+you\s+(tell|show|help|explain|describe)\s+",
            r"^could\s+you\s+(tell|show|help|explain|describe)\s+",
            r"^please\s+(tell|show|help|explain|describe)\s+",
            r"^tell\s+me\s+about\s+",
            r"^explain\s+",
            r"^describe\s+",
        ]

        result = text.strip()
        for pattern in patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        return result.rstrip("?").strip()

    def _extract_keywords_regex(self, text: str) -> List[str]:
        """Extract keywords using regex."""
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "tell", "please", "help", "find", "give", "get", "know", "let", "make",
        }

        # Tokenize
        tokens = re.findall(r"[\w'-]+", text.lower())

        # Filter
        keywords = []
        for token in tokens:
            clean = token.strip("-'")
            if len(clean) >= 2 and clean not in stop_words and not clean.isdigit():
                keywords.append(clean)

        # Add acronyms (preserve case)
        acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
        keywords.extend([a.lower() for a in acronyms])

        return list(dict.fromkeys(keywords))

    def _extract_project_names_regex(self, query: str) -> List[str]:
        """Extract project names using regex."""
        project_names = []

        # Acronyms (2-6 capital letters)
        acronyms = re.findall(r"\b([A-Z]{2,6})\b", query)
        project_names.extend([a.lower() for a in acronyms])

        # "X project" pattern
        project_pattern = re.findall(
            r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+project",
            query,
            re.IGNORECASE
        )
        for match in project_pattern:
            name = match.strip().lower()
            if len(name) >= 2:
                project_names.append(name)

        # Capitalized words mid-sentence
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper():
                clean = re.sub(r"[^\w]", "", word).lower()
                if len(clean) >= 2:
                    project_names.append(clean)

        return list(dict.fromkeys(project_names))

    def _extract_person_names_regex(self, query: str) -> List[str]:
        """Extract person names using regex."""
        # "First Last" pattern
        matches = re.findall(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", query)

        person_names = []
        false_positives = {"data science", "real world", "project name", "machine learning"}

        for first, last in matches:
            full_name = f"{first} {last}".lower()
            if full_name not in false_positives:
                person_names.append(full_name)

        return list(dict.fromkeys(person_names))

    def _extract_dates_regex(self, query: str) -> List[str]:
        """Extract date references using regex."""
        dates = []

        # Years (2020-2030)
        years = re.findall(r"\b(20[2-3]\d)\b", query)
        dates.extend(years)

        # Quarters (Q1 2024, Q2-2025)
        quarters = re.findall(r"\b(Q[1-4][\s-]?20[2-3]\d)\b", query, re.IGNORECASE)
        dates.extend(quarters)

        # Months with year
        months = re.findall(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20[2-3]\d)\b",
            query,
            re.IGNORECASE
        )
        dates.extend(months)

        # Date ranges
        ranges = re.findall(r"\b(20[2-3]\d\s*[-–to]+\s*20[2-3]\d)\b", query)
        dates.extend(ranges)

        return list(dict.fromkeys(dates))

    def _extract_technologies_regex(self, query: str) -> List[str]:
        """Extract technology names using regex."""
        # Common technologies to look for
        tech_patterns = [
            r"\b(Python|Java|JavaScript|TypeScript|R|SQL|Scala|Go|Rust)\b",
            r"\b(Airflow|Spark|Kafka|Docker|Kubernetes|AWS|Azure|GCP)\b",
            r"\b(TensorFlow|PyTorch|scikit-learn|Pandas|NumPy)\b",
            r"\b(RAG|GraphRAG|LLM|NLP|ML|AI|GPT|BERT|Transformer)\b",
            r"\b(PostgreSQL|MySQL|MongoDB|Redis|Snowflake|Databricks)\b",
            r"\b(React|Vue|Angular|FastAPI|Flask|Django)\b",
            r"\b(Git|GitHub|GitLab|Jenkins|CircleCI)\b",
            r"\b(machine learning|deep learning|natural language processing)\b",
        ]

        technologies = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            technologies.extend([m.lower() if isinstance(m, str) else m[0].lower() for m in matches])

        return list(dict.fromkeys(technologies))

    def _detect_comparative_regex(self, query: str) -> Tuple[bool, List[str]]:
        """Detect comparative queries using regex."""
        query_lower = query.lower()

        patterns = [
            r"compare\s+(\w+)\s+(?:to|with|and|vs\.?|versus)\s+(\w+)",
            r"(\w+)\s+vs\.?\s+(\w+)",
            r"difference(?:s)?\s+between\s+(\w+)\s+and\s+(\w+)",
            r"(\w+)\s+versus\s+(\w+)",
            r"similarities?\s+between\s+(\w+)\s+and\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities = [g for g in match.groups() if g and len(g) > 2]
                if len(entities) >= 2:
                    return (True, entities)

        # Check for multiple acronyms with comparison words
        acronyms = re.findall(r"\b([A-Z]{2,6})\b", query)
        if len(acronyms) >= 2:
            comparison_words = ["compare", "vs", "versus", "difference", "differ"]
            if any(word in query_lower for word in comparison_words):
                return (True, [a.lower() for a in acronyms])

        return (False, [])

    def _infer_intent_regex(self, query: str) -> str:
        """Infer query intent using regex patterns."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["compare", "vs", "versus", "difference"]):
            return "comparison"
        if any(w in query_lower for w in ["how many", "count", "total", "number of"]):
            return "aggregation"
        if any(w in query_lower for w in ["list", "show all", "get all", "what are"]):
            return "listing"
        if any(w in query_lower for w in ["how to", "how do", "how can"]):
            return "how-to"

        return "informational"

    def is_comparative_query(self, query: str) -> Tuple[bool, List[str]]:
        """
        Detect if a query is comparative.

        Args:
            query: The user's query string.

        Returns:
            Tuple of (is_comparative, list of entities to compare).
        """
        # Use the processed result if available, otherwise use regex
        if self.use_llm and self.iliad_client:
            result = self.process_query(query)
            return (result.is_comparative, result.comparative_entities)

        return self._detect_comparative_regex(query)

    def get_search_terms(self, processed_query: ProcessedQuery) -> List[str]:
        """
        Get combined search terms for document retrieval.

        Args:
            processed_query: Processed query object.

        Returns:
            List of all search terms (keywords + names + technologies).
        """
        terms = []
        terms.extend(processed_query.keywords)
        terms.extend(processed_query.potential_project_names)
        terms.extend(processed_query.potential_person_names)
        terms.extend(processed_query.technologies)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)

        return unique_terms


def extract_keywords(query: str) -> Tuple[str, List[str], List[str]]:
    """
    Convenience function to extract keywords from a query.

    Args:
        query: The user's query string.

    Returns:
        Tuple of (cleaned_query, keywords_list, lemmatized_keywords).
    """
    processor = QueryProcessor(use_llm=False)  # Use regex for quick extraction
    result = processor.process_query(query)
    return result.cleaned_query, result.keywords, result.lemmatized_keywords
