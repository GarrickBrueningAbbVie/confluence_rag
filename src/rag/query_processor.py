"""Query preprocessing module for keyword extraction and query cleaning.

This module provides functionality to extract keywords from user queries
for improved document retrieval and re-ranking.
"""

import re
from typing import List, Set, Tuple
from dataclasses import dataclass
from loguru import logger

# Use NLTK for stop words and lemmatization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data (only if not already present)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    NLTK_AVAILABLE = True
    ENGLISH_STOP_WORDS: Set[str] = set(stopwords.words("english"))
    logger.info("NLTK loaded successfully for query processing")
except ImportError:
    NLTK_AVAILABLE = False
    # Fallback stop words if NLTK is not available
    ENGLISH_STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "it", "its", "my", "your", "his", "her", "our", "their",
        "me", "him", "them", "us", "about", "into", "through", "during",
    }
    logger.warning("NLTK not available, using fallback stop words")

# Additional domain-specific stop words
ADDITIONAL_STOP_WORDS: Set[str] = {
    "tell", "please", "help", "find", "give", "get", "know", "let", "make",
    "see", "look", "show", "using", "used", "work", "working", "works",
    "question", "answer", "explain", "describe", "information", "details",
}

# Question patterns to remove
QUESTION_PATTERNS: List[str] = [
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
    r"^what\s+does\s+",
    r"^what\s+do\s+",
    r"^i\s+want\s+to\s+(know|learn|understand)\s+about\s+",
    r"^i\s+need\s+to\s+(know|learn|understand)\s+about\s+",
]


@dataclass
class ProcessedQuery:
    """Container for processed query information."""

    original_query: str
    cleaned_query: str
    keywords: List[str]
    lemmatized_keywords: List[str]
    potential_project_names: List[str]
    potential_person_names: List[str]


class QueryProcessor:
    """
    Processes user queries to extract keywords and clean text.

    This class provides methods to:
    - Remove stop words and question phrases
    - Lemmatize words to their base form
    - Extract key terms and potential entity names
    - Identify potential project names and person names
    """

    def __init__(
        self,
        additional_stop_words: Set[str] = None,
        min_keyword_length: int = 2,
        use_lemmatization: bool = True,
    ) -> None:
        """
        Initialize the query processor.

        Args:
            additional_stop_words: Additional words to filter out.
            min_keyword_length: Minimum character length for keywords.
            use_lemmatization: Whether to lemmatize keywords.
        """
        self.stop_words = ENGLISH_STOP_WORDS.copy()
        self.stop_words.update(ADDITIONAL_STOP_WORDS)
        if additional_stop_words:
            self.stop_words.update(additional_stop_words)

        self.min_keyword_length = min_keyword_length
        self.use_lemmatization = use_lemmatization and NLTK_AVAILABLE
        self.question_patterns = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]

        # Initialize lemmatizer if available
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

        logger.info(
            f"QueryProcessor initialized (NLTK: {NLTK_AVAILABLE}, "
            f"Lemmatization: {self.use_lemmatization}, "
            f"Stop words count: {len(self.stop_words)})"
        )

    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process a user query to extract keywords and clean text.

        Args:
            query: The original user query.

        Returns:
            ProcessedQuery object containing cleaned query and extracted info.
        """
        logger.info(f"Processing query: '{query}'")

        # Clean the query
        cleaned = self._remove_question_patterns(query)
        logger.debug(f"After removing question patterns: '{cleaned}'")

        # Extract keywords
        keywords = self._extract_keywords(cleaned)
        logger.info(f"Extracted keywords: {keywords}")

        # Lemmatize keywords if available
        if self.use_lemmatization:
            lemmatized = self._lemmatize_keywords(keywords)
            logger.info(f"Lemmatized keywords: {lemmatized}")
        else:
            lemmatized = keywords.copy()

        # Identify potential names
        project_names = self._identify_potential_project_names(query, keywords)
        person_names = self._identify_potential_person_names(query)

        if project_names:
            logger.info(f"Identified potential project names: {project_names}")
        if person_names:
            logger.info(f"Identified potential person names: {person_names}")

        result = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            keywords=keywords,
            lemmatized_keywords=lemmatized,
            potential_project_names=project_names,
            potential_person_names=person_names,
        )

        logger.info(
            f"Query processing complete - "
            f"Keywords: {len(keywords)}, "
            f"Projects: {len(project_names)}, "
            f"People: {len(person_names)}"
        )

        return result

    def _remove_question_patterns(self, text: str) -> str:
        """
        Remove common question patterns from text.

        Args:
            text: Input text.

        Returns:
            Text with question patterns removed.
        """
        result = text.strip()

        # Remove question patterns from beginning
        for pattern in self.question_patterns:
            result = pattern.sub("", result)

        # Remove trailing question mark
        result = result.rstrip("?").strip()

        return result

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}, using fallback")

        # Fallback tokenization
        return re.findall(r"[\w'-]+", text.lower())

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.

        Args:
            text: Input text.

        Returns:
            List of extracted keywords.
        """
        # Tokenize
        tokens = self._tokenize(text)

        # Filter tokens
        keywords = []
        for token in tokens:
            # Remove leading/trailing hyphens and apostrophes
            clean_token = token.strip("-'")

            # Skip if too short or is a stop word
            if len(clean_token) < self.min_keyword_length:
                continue
            if clean_token in self.stop_words:
                continue

            # Skip pure numbers unless they look like versions
            if clean_token.isdigit():
                continue

            keywords.append(clean_token)

        # Also extract multi-word phrases from original text
        phrases = self._extract_phrases(text)
        keywords.extend(phrases)

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def _lemmatize_keywords(self, keywords: List[str]) -> List[str]:
        """
        Lemmatize keywords to their base form.

        Args:
            keywords: List of keywords to lemmatize.

        Returns:
            List of lemmatized keywords.
        """
        if not self.lemmatizer:
            return keywords.copy()

        lemmatized = []
        for keyword in keywords:
            # Try noun first, then verb
            lemma = self.lemmatizer.lemmatize(keyword, pos="n")
            if lemma == keyword:
                lemma = self.lemmatizer.lemmatize(keyword, pos="v")
            lemmatized.append(lemma)

        # Remove duplicates while preserving order
        seen = set()
        unique_lemmas = []
        for lemma in lemmatized:
            if lemma not in seen:
                seen.add(lemma)
                unique_lemmas.append(lemma)

        return unique_lemmas

    def _extract_phrases(self, text: str) -> List[str]:
        """
        Extract potential multi-word phrases (project names, acronyms).

        Args:
            text: Input text.

        Returns:
            List of extracted phrases.
        """
        phrases = []

        # Find capitalized word sequences (potential proper nouns/names)
        cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for match in cap_pattern.findall(text):
            phrases.append(match.lower())

        # Find acronyms (2-6 capital letters)
        acronym_pattern = re.compile(r"\b([A-Z]{2,6})\b")
        for match in acronym_pattern.findall(text):
            phrases.append(match.lower())

        # Find quoted phrases
        quote_pattern = re.compile(r'"([^"]+)"|\'([^\']+)\'')
        for match in quote_pattern.findall(text):
            phrase = match[0] or match[1]
            if phrase:
                phrases.append(phrase.lower())

        return phrases

    def _identify_potential_project_names(
        self, query: str, keywords: List[str]
    ) -> List[str]:
        """
        Identify potential project names from query.

        Args:
            query: Original query text.
            keywords: Extracted keywords.

        Returns:
            List of potential project names.
        """
        project_names = []

        # Look for patterns like "the X project" or "X project"
        project_pattern = re.compile(
            r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+project",
            re.IGNORECASE
        )
        for match in project_pattern.findall(query):
            name = match.strip().lower()
            if name not in self.stop_words:
                project_names.append(name)

        # Look for acronyms which are often project names
        acronym_pattern = re.compile(r"\b([A-Z]{2,6})\b")
        for match in acronym_pattern.findall(query):
            project_names.append(match.lower())

        # Capitalized words that aren't at sentence start might be names
        words = query.split()
        for i, word in enumerate(words):
            if i == 0:
                continue
            if word and word[0].isupper() and word.lower() not in self.stop_words:
                clean = re.sub(r"[^\w]", "", word).lower()
                if len(clean) >= 2:
                    project_names.append(clean)

        return list(dict.fromkeys(project_names))

    def _identify_potential_person_names(self, query: str) -> List[str]:
        """
        Identify potential person names from query.

        Args:
            query: Original query text.

        Returns:
            List of potential person names.
        """
        person_names = []

        # Pattern for "FirstName LastName"
        name_pattern = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")
        for match in name_pattern.findall(query):
            full_name = f"{match[0]} {match[1]}".lower()
            # Filter out common false positives
            if not any(
                word in full_name
                for word in ["data science", "real world", "project name"]
            ):
                person_names.append(full_name)

        return list(dict.fromkeys(person_names))

    def get_search_terms(self, processed_query: ProcessedQuery) -> List[str]:
        """
        Get combined search terms for document retrieval.

        Args:
            processed_query: Processed query object.

        Returns:
            List of all search terms (keywords + names).
        """
        terms = []
        terms.extend(processed_query.lemmatized_keywords)
        terms.extend(processed_query.potential_project_names)
        terms.extend(processed_query.potential_person_names)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
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
    processor = QueryProcessor()
    result = processor.process_query(query)
    return result.cleaned_query, result.keywords, result.lemmatized_keywords
