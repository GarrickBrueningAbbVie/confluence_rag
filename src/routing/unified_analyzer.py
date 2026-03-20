"""
Unified Query Analyzer - Single LLM call for complete query analysis.

This module provides a unified analysis that combines:
- Entity extraction (projects, people, technologies, dates)
- Intent classification (rag, database, hybrid, chart, table)
- Query decomposition into atomic sub-queries
- Comparison detection

This replaces the previous multi-call approach:
- QueryProcessor (entity extraction) - LLM call #1
- IntentClassifier (intent) - optional LLM call #2
- LLMQueryAnalyzer (decomposition) - LLM call #3

Now everything is done in ONE LLM call.

Example:
    >>> from routing.unified_analyzer import UnifiedQueryAnalyzer
    >>> analyzer = UnifiedQueryAnalyzer(iliad_client)
    >>> result = analyzer.analyze("Compare ALFA and CloverX projects")
    >>> print(result.sub_queries)  # Returns 3 sub-queries: describe ALFA, describe CloverX, compare
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .types import QueryIntent, SubQuery
from .patterns import classify_intent_fallback, is_comparison_query, is_list_describe_query

# Import JSON parser
try:
    from utils.json_parser import parse_llm_json_response
except ImportError:
    from src.utils.json_parser import parse_llm_json_response

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


@dataclass
class EntityExtractionResult:
    """Extracted entities from query."""

    projects: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)


@dataclass
class UnifiedAnalysisResult:
    """Complete analysis result from single LLM call.

    Attributes:
        original_query: The original user query
        entities: Extracted entities (projects, people, technologies, dates)
        primary_intent: Overall query intent
        confidence: Confidence score (0.0 to 1.0)
        sub_queries: Decomposed sub-queries with individual intents
        is_complex: Whether query was decomposed into multiple sub-queries
        is_comparative: Whether query compares entities
        comparative_entities: Entities being compared (if comparative)
        reasoning: Explanation for the analysis
    """

    original_query: str
    entities: EntityExtractionResult
    primary_intent: QueryIntent
    confidence: float
    sub_queries: List[SubQuery]
    is_complex: bool = False
    is_comparative: bool = False
    comparative_entities: List[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return bool(
            self.entities.projects
            or self.entities.people
            or self.entities.technologies
            or self.entities.dates
        )


# =============================================================================
# Unified Analysis Prompt
# =============================================================================

UNIFIED_ANALYSIS_PROMPT = '''You are a query analyzer for a knowledge base system about 
data science projects within a clinical development group.

Analyze the user's query and extract ALL of the following in a single response:

1. **Entities**: Projects, people, technologies, and dates mentioned
2. **Intent**: The primary type of query (rag, database, hybrid, chart, table)
3. **Sub-queries**: Decompose complex queries into atomic sub-queries
4. **Comparison**: Detect if the query compares entities

## CRITICAL: Spelling and Grammar Correction

Users may have typos or misspellings in their queries. You MUST:
1. Always correct spelling errors in the sub-query "text" field
2. Correct common word typos (e.g., "wat" → "what", "prjects" → "projects")
3. Fix technology names if misspelled (e.g., "xgbost" → "XGBoost", "pythn" → "Python")
4. Keep entity extraction accurate even when spelled correctly in original

The sub-query text must be clean, properly spelled English.

## Intent Types

- **rag**: Semantic search for explanations, descriptions, concepts
  - "What is X?", "Explain Y", "How does Z work?"
- **database**: Structured queries for counts, lists, aggregations
  - "How many X?", "List all Y", "Count Z by category"
- **hybrid**: Needs both semantic AND structured data
  - "List projects and describe them", "Compare X and Y"
- **chart**: Visualization requests
  - "Show a chart of X", "Graph Y over time"
- **table**: Tabular data display
  - "Show a table of X", "Display Y in table format"

## Sub-Query Rules

1. ALWAYS decompose multi-part questions (connected by "and", "also", etc.)
2. Each sub-query should be atomic (one question, one intent)
3. Use `depends_on` when a query needs results from a previous query
4. Use `context_from` when a query should reference results from other queries
5. Order by priority (0 = highest priority, execute first)

## CRITICAL: Comparison Query Handling

For comparison queries (e.g., "Compare X and Y"), you MUST decompose into:
1. First sub-query: Describe entity X (intent: rag)
2. Second sub-query: Describe entity Y (intent: rag)
3. Third sub-query: Compare X and Y (intent: rag, context_from: [0, 1])

This ensures BOTH entities are properly described before comparison.

## Output Format

Return ONLY valid JSON:
```json
{
  "entities": {
    "projects": ["list of project names/acronyms"],
    "people": ["list of person names"],
    "technologies": ["list of technologies/tools"],
    "dates": ["list of dates/years"]
  },
  "primary_intent": "rag|database|hybrid|chart|table",
  "confidence": 0.0-1.0,
  "sub_queries": [
    {
      "text": "rewritten atomic query",
      "intent": "rag|database|hybrid|chart|table",
      "priority": 0,
      "depends_on": null,
      "context_from": []
    }
  ],
  "is_comparative": true|false,
  "comparative_entities": ["entity1", "entity2"],
  "reasoning": "brief explanation"
}
```

## Examples

### Simple RAG Query
Query: "What is the ALFA project?"
```json
{
  "entities": {"projects": ["ALFA"], "people": [], "technologies": [], "dates": []},
  "primary_intent": "rag",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "Describe the ALFA project, its purpose, and key details", "intent": "rag", "priority": 0, "depends_on": null, "context_from": []}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Single project description request"
}
```

### Simple Database Query
Query: "How many projects use Python?"
```json
{
  "entities": {"projects": [], "people": [], "technologies": ["Python"], "dates": []},
  "primary_intent": "database",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "Count all projects that use Python", "intent": "database", "priority": 0, "depends_on": null, "context_from": []}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Counting query for technology filter"
}
```

### Technology Listing Query
Query: "List projects that use XGBoost"
```json
{
  "entities": {"projects": [], "people": [], "technologies": ["XGBoost"], "dates": []},
  "primary_intent": "database",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "List all projects that use XGBoost technology", "intent": "database", "priority": 0, "depends_on": null, "context_from": []}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Listing query filtered by technology"
}
```

### Query with Typos (SPELLING CORRECTION)
Query: "Wat prjects use xgbost?"
```json
{
  "entities": {"projects": [], "people": [], "technologies": ["XGBoost"], "dates": []},
  "primary_intent": "database",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "What projects use XGBoost?", "intent": "database", "priority": 0, "depends_on": null, "context_from": []}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Corrected spelling: 'Wat'->'What', 'prjects'->'projects', 'xgbost'->'XGBoost'"
}
```

### COMPARISON Query (CRITICAL)
Query: "Compare ALFA and CloverX projects"
```json
{
  "entities": {"projects": ["ALFA", "CloverX"], "people": [], "technologies": [], "dates": []},
  "primary_intent": "hybrid",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "Describe the ALFA project in detail including its purpose, technologies, and key features", "intent": "rag", "priority": 0, "depends_on": null, "context_from": []},
    {"text": "Describe the CloverX project in detail including its purpose, technologies, and key features", "intent": "rag", "priority": 0, "depends_on": null, "context_from": []},
    {"text": "Compare and contrast ALFA and CloverX projects based on their purposes, technologies, approaches, and outcomes", "intent": "rag", "priority": 1, "depends_on": null, "context_from": [0, 1]}
  ],
  "is_comparative": true,
  "comparative_entities": ["ALFA", "CloverX"],
  "reasoning": "Comparison requires describing both entities first, then comparing"
}
```

### Multi-part Query
Query: "What is ALFA and how many pages mention it?"
```json
{
  "entities": {"projects": ["ALFA"], "people": [], "technologies": [], "dates": []},
  "primary_intent": "hybrid",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "Describe the ALFA project, its purpose, and key details", "intent": "rag", "priority": 0, "depends_on": null, "context_from": []},
    {"text": "Count how many pages mention or reference ALFA", "intent": "database", "priority": 1, "depends_on": null, "context_from": []}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Two-part query: description (RAG) and count (database)"
}
```

### List + Describe Query
Query: "List all projects using XGBoost and describe them"
```json
{
  "entities": {"projects": [], "people": [], "technologies": ["XGBoost"], "dates": []},
  "primary_intent": "hybrid",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "List all projects that use XGBoost technology", "intent": "database", "priority": 0, "depends_on": null, "context_from": []},
    {"text": "Describe each project that uses XGBoost, including their purposes and applications", "intent": "rag", "priority": 1, "depends_on": 0, "context_from": [0]}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "List from database, then describe each via RAG"
}
```

### Chart Query
Query: "Show a bar chart of pages by author"
```json
{
  "entities": {"projects": [], "people": [], "technologies": [], "dates": []},
  "primary_intent": "chart",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "Count pages grouped by author", "intent": "database", "priority": 0, "depends_on": null, "context_from": []},
    {"text": "Create a bar chart showing the number of pages by each author", "intent": "chart", "priority": 1, "depends_on": 0, "context_from": [0]}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Chart requires data from database query first"
}
```

### Table Query
Query: "Show a table of all projects and their technologies"
```json
{
  "entities": {"projects": [], "people": [], "technologies": [], "dates": []},
  "primary_intent": "table",
  "confidence": 0.95,
  "sub_queries": [
    {"text": "List all projects with their associated technologies in tabular format", "intent": "table", "priority": 0, "depends_on": null, "context_from": []}
  ],
  "is_comparative": false,
  "comparative_entities": [],
  "reasoning": "Direct table display request"
}
```

## User Query to Analyze

{query}
'''


class UnifiedQueryAnalyzer:
    """
    Single LLM call for complete query analysis.

    Combines entity extraction, intent classification, and query
    decomposition into one efficient LLM call.

    Attributes:
        iliad_client: Iliad API client
        model: Model to use for analysis
        max_sub_queries: Maximum sub-queries to generate

    Example:
        >>> analyzer = UnifiedQueryAnalyzer(iliad_client)
        >>> result = analyzer.analyze("Compare ALFA and CloverX")
        >>> print(len(result.sub_queries))  # 3
    """

    def __init__(
        self,
        iliad_client: "IliadClient",
        model: str = "gpt-5-mini-global",
        max_sub_queries: int = 5,
    ) -> None:
        """Initialize the unified analyzer.

        Args:
            iliad_client: Configured Iliad client
            model: Model to use for analysis
            max_sub_queries: Maximum sub-queries to generate
        """
        self.iliad_client = iliad_client
        self.model = model
        self.max_sub_queries = max_sub_queries

        logger.info(f"Initialized UnifiedQueryAnalyzer with model: {model}")

    def analyze(self, query: str) -> UnifiedAnalysisResult:
        """
        Analyze a query with a single LLM call.

        Extracts entities, classifies intent, and decomposes into
        sub-queries all in one request.

        Args:
            query: User's natural language query

        Returns:
            UnifiedAnalysisResult with complete analysis

        Example:
            >>> result = analyzer.analyze("What is ALFA and who created it?")
            >>> print(result.primary_intent)  # QueryIntent.HYBRID
            >>> print(len(result.sub_queries))  # 2
        """
        logger.info(f"Analyzing query: {query[:100]}...")

        try:
            # Build prompt
            logger.info(f"Building Prompt from query {query}")
            prompt = UNIFIED_ANALYSIS_PROMPT.replace("{query}", query)

            # Single LLM call
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            content = self.iliad_client.extract_content(response)

            # Parse response
            logger.info('Parsing Response')
            result = self._parse_response(content, query)

            logger.info(
                f"Analysis complete: intent={result.primary_intent.value}, "
                f"sub_queries={len(result.sub_queries)}, "
                f"comparative={result.is_comparative}"
            )

            return result

        except Exception as e:
            logger.error(f"Unified analysis failed: {e}")
            return self._fallback_analysis(query, str(e))

    def _parse_response(self, response: str, original_query: str) -> UnifiedAnalysisResult:
        """Parse LLM response into UnifiedAnalysisResult.

        Args:
            response: Raw LLM response
            original_query: Original user query

        Returns:
            Parsed UnifiedAnalysisResult
        """
        parsed = parse_llm_json_response(response)

        if parsed is None:
            logger.warning("Failed to parse LLM response, using fallback")
            return self._fallback_analysis(original_query, "JSON parse error")

        # Extract entities
        entities_data = parsed.get("entities", {})
        entities = EntityExtractionResult(
            projects=entities_data.get("projects", []),
            people=entities_data.get("people", []),
            technologies=entities_data.get("technologies", []),
            dates=entities_data.get("dates", []),
        )

        # Extract primary intent
        intent_str = parsed.get("primary_intent", "rag").lower()
        intent_map = {
            "rag": QueryIntent.RAG,
            "database": QueryIntent.DATABASE,
            "hybrid": QueryIntent.HYBRID,
            "chart": QueryIntent.CHART,
            "table": QueryIntent.TABLE,
        }
        primary_intent = intent_map.get(intent_str, QueryIntent.RAG)

        # Extract sub-queries
        sub_queries = []
        raw_sub_queries = parsed.get("sub_queries", [])

        for i, sq in enumerate(raw_sub_queries[: self.max_sub_queries]):
            sq_intent_str = sq.get("intent", "rag").lower()
            sq_intent = intent_map.get(sq_intent_str, QueryIntent.RAG)

            sub_queries.append(
                SubQuery(
                    text=sq.get("text", original_query),
                    intent=sq_intent,
                    depends_on=sq.get("depends_on"),
                    priority=sq.get("priority", i),
                    context_from=sq.get("context_from", []),
                )
            )

        # Ensure at least one sub-query
        if not sub_queries:
            sub_queries.append(
                SubQuery(text=original_query, intent=primary_intent)
            )

        return UnifiedAnalysisResult(
            original_query=original_query,
            entities=entities,
            primary_intent=primary_intent,
            confidence=parsed.get("confidence", 0.9),
            sub_queries=sub_queries,
            is_complex=len(sub_queries) > 1,
            is_comparative=parsed.get("is_comparative", False),
            comparative_entities=parsed.get("comparative_entities", []),
            reasoning=parsed.get("reasoning", ""),
        )

    def _fallback_analysis(self, query: str, error: str) -> UnifiedAnalysisResult:
        """Create fallback analysis using pattern matching.

        Args:
            query: Original query
            error: Error message that triggered fallback

        Returns:
            Pattern-based UnifiedAnalysisResult
        """
        logger.warning(f"Using fallback analysis due to: {error}")

        # Use pattern-based classification
        intent, confidence, reasoning = classify_intent_fallback(query)

        # Simple entity extraction via patterns
        entities = self._extract_entities_fallback(query)

        # Check for comparison
        is_comparative = is_comparison_query(query)
        comparative_entities = []

        # Build sub-queries
        sub_queries = []

        if is_comparative:
            # For comparisons, create describe + compare sub-queries
            # Extract potential entities to compare
            comparative_entities = entities.projects[:2] if entities.projects else []

            if len(comparative_entities) >= 2:
                sub_queries.append(
                    SubQuery(
                        text=f"Describe {comparative_entities[0]}",
                        intent=QueryIntent.RAG,
                        priority=0,
                    )
                )
                sub_queries.append(
                    SubQuery(
                        text=f"Describe {comparative_entities[1]}",
                        intent=QueryIntent.RAG,
                        priority=0,
                    )
                )
                sub_queries.append(
                    SubQuery(
                        text=f"Compare {comparative_entities[0]} and {comparative_entities[1]}",
                        intent=QueryIntent.RAG,
                        priority=1,
                        context_from=[0, 1],
                    )
                )
            else:
                sub_queries.append(SubQuery(text=query, intent=intent))
        elif is_list_describe_query(query):
            # List + describe pattern
            sub_queries.append(
                SubQuery(
                    text=query.split("and")[0].strip() if "and" in query else query,
                    intent=QueryIntent.DATABASE,
                    priority=0,
                )
            )
            sub_queries.append(
                SubQuery(
                    text="Describe the listed items",
                    intent=QueryIntent.RAG,
                    priority=1,
                    depends_on=0,
                )
            )
        else:
            sub_queries.append(SubQuery(text=query, intent=intent))

        return UnifiedAnalysisResult(
            original_query=query,
            entities=entities,
            primary_intent=intent,
            confidence=confidence * 0.8,  # Lower confidence for fallback
            sub_queries=sub_queries,
            is_complex=len(sub_queries) > 1,
            is_comparative=is_comparative,
            comparative_entities=comparative_entities,
            reasoning=f"Fallback analysis: {reasoning}",
        )

    def _extract_entities_fallback(self, query: str) -> EntityExtractionResult:
        """Extract entities using simple patterns.

        Args:
            query: User query

        Returns:
            EntityExtractionResult with extracted entities
        """
        import re

        projects = []
        people = []
        technologies = []
        dates = []

        # Extract acronyms (potential projects)
        acronyms = re.findall(r"\b([A-Z]{2,6})\b", query)
        projects.extend([a for a in acronyms if a not in ["RAG", "LLM", "API", "SQL"]])

        # Extract capitalized words mid-sentence (potential projects)
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper():
                clean = re.sub(r"[^\w]", "", word)
                if len(clean) >= 3 and clean not in ["The", "And", "For", "With"]:
                    projects.append(clean)

        # Extract person names (First Last pattern)
        name_matches = re.findall(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", query)
        for first, last in name_matches:
            full_name = f"{first} {last}"
            if full_name.lower() not in ["data science", "machine learning"]:
                people.append(full_name)

        # Extract common technologies
        tech_patterns = [
            r"\b(Python|Java|JavaScript|TypeScript|R|SQL|Scala|Go|Rust)\b",
            r"\b(Airflow|Spark|Kafka|Docker|Kubernetes|AWS|Azure|GCP)\b",
            r"\b(TensorFlow|PyTorch|XGBoost|scikit-learn|Pandas|NumPy)\b",
            r"\b(PostgreSQL|MySQL|MongoDB|Redis|Snowflake|Databricks)\b",
        ]
        for pattern in tech_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            technologies.extend(matches)

        # Extract years
        years = re.findall(r"\b(20[2-3]\d)\b", query)
        dates.extend(years)

        return EntityExtractionResult(
            projects=list(set(projects)),
            people=list(set(people)),
            technologies=list(set(technologies)),
            dates=list(set(dates)),
        )

    def analyze_batch(self, queries: List[str]) -> List[UnifiedAnalysisResult]:
        """
        Analyze multiple queries.

        Args:
            queries: List of queries to analyze

        Returns:
            List of UnifiedAnalysisResult objects
        """
        results = []
        for query in queries:
            results.append(self.analyze(query))
        return results
