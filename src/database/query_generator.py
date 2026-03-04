"""
Natural language to pandas query generator.

This module uses Iliad API to generate pandas queries from natural
language questions about Confluence page data.

Example:
    >>> from database.query_generator import QueryGenerator
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>>
    >>> iliad = IliadClient(IliadClientConfig.from_env())
    >>> generator = QueryGenerator(iliad, schema_info)
    >>> query = generator.generate("How many pages use Python?")
"""

from typing import Any, Dict, List, Optional

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


# Few-shot examples for query generation
FEW_SHOT_EXAMPLES = [
    {
        "question": "How many pages has John Smith created?",
        "query": "df[df['created_by'] == 'John Smith'].shape[0]",
        "explanation": "Filter by created_by column and count rows",
    },
    {
        "question": "What products use airflow?",
        "query": "df[df['technologies'].apply(lambda x: 'Airflow' in x if isinstance(x, list) else False)]['title'].tolist()",
        "explanation": "Filter where technologies list contains Airflow, return titles",
    },
    {
        "question": "List all projects with completeness score above 80",
        "query": "df[df['completeness_score'] > 80][['title', 'completeness_score']].to_dict('records')",
        "explanation": "Filter by completeness_score, select relevant columns",
    },
    {
        "question": "What is the average completeness score?",
        "query": "df['completeness_score'].mean()",
        "explanation": "Calculate mean of completeness_score column",
    },
    {
        "question": "How many pages are in each project?",
        "query": "df.groupby('parent_project').size().to_dict()",
        "explanation": "Group by parent_project and count",
    },
    {
        "question": "What technologies are most commonly used?",
        "query": "pd.Series([t for techs in df['technologies'].dropna() for t in techs if isinstance(techs, list)]).value_counts().head(10).to_dict()",
        "explanation": "Flatten technologies lists, count occurrences",
    },
    {
        "question": "Show pages created in the last 30 days",
        "query": "df[df['created_date'] > pd.Timestamp.now() - pd.Timedelta(days=30)][['title', 'created_date']].to_dict('records')",
        "explanation": "Filter by recent created_date",
    },
    {
        "question": "Which projects have the lowest completeness scores?",
        "query": "df[df['completeness_score'].notna()].nsmallest(10, 'completeness_score')[['title', 'parent_project', 'completeness_score']].to_dict('records')",
        "explanation": "Sort by completeness_score ascending, take top 10",
    },
    {
        "question": "How many pages exist at each depth level?",
        "query": "df['depth'].value_counts().sort_index().to_dict()",
        "explanation": "Count pages per depth level",
    },
    {
        "question": "List all unique technologies",
        "query": "sorted(set(t for techs in df['technologies'].dropna() for t in techs if isinstance(techs, list)))",
        "explanation": "Extract and deduplicate all technologies",
    },
]


class QueryGenerator:
    """
    Generate pandas queries from natural language.

    Uses Iliad API with few-shot prompting to generate safe,
    executable pandas queries.

    Attributes:
        iliad_client: Iliad API client
        schema_info: DataFrame schema description
        model: Model to use for generation

    Example:
        >>> generator = QueryGenerator(iliad_client, schema_info)
        >>> result = generator.generate("Count pages by author")
        >>> print(result.query)
    """

    def __init__(
        self,
        iliad_client: "IliadClient",
        schema_info: str,
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize query generator.

        Args:
            iliad_client: Configured Iliad client
            schema_info: Description of DataFrame columns
            model: Model to use for query generation
        """
        self.iliad_client = iliad_client
        self.schema_info = schema_info
        self.model = model

        logger.info(f"Initialized QueryGenerator with model: {model}")

    def _build_prompt(self, question: str) -> str:
        """Build the prompt for query generation.

        Args:
            question: Natural language question

        Returns:
            Complete prompt with schema and examples
        """
        examples_text = "\n\n".join(
            f"Question: {ex['question']}\n"
            f"Query: {ex['query']}\n"
            f"Explanation: {ex['explanation']}"
            for ex in FEW_SHOT_EXAMPLES
        )

        prompt = f"""You are a pandas query generator. Given a natural language question about Confluence page data, generate a pandas query to answer it.

## DataFrame Schema
The DataFrame is named `df` and has the following columns:

{self.schema_info}

## CRITICAL FORMAT RULES (MUST FOLLOW)
1. Output ONLY a single pandas expression that starts with `df` or `pd`
2. DO NOT use variable assignments (e.g., `result = df[...]` is WRONG)
3. DO NOT use semicolons or multiple statements
4. The query must be executable directly with Python's eval() function
5. Return results as dictionaries, lists, or scalar values

CORRECT format examples:
- df.shape[0]
- df[df['col'] == 'value']['title'].tolist()
- df.groupby('col').size().to_dict()

WRONG format examples (DO NOT DO THIS):
- result = df.shape[0]  <-- NO variable assignments!
- filtered_df = df[df['col'] == 'value']  <-- NO variable assignments!
- df.shape[0]; df.head()  <-- NO semicolons!

## Important Notes
- The 'technologies' column contains lists of strings (e.g., ['Python', 'Airflow'])
- Use .apply(lambda x: ...) to search within list columns
- Always handle None/NaN values appropriately
- Use .to_dict('records') to convert DataFrames to list of dicts

## Examples

{examples_text}

## Your Task

Question: {question}

Output ONLY the pandas expression (starting with df or pd). No variable names, no assignments, no explanations:"""

        return prompt

    def generate(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Generate a pandas query from natural language.

        Args:
            question: Natural language question

        Returns:
            Dict with:
            - query: Generated pandas query string
            - explanation: Brief explanation of the query
            - success: Whether generation succeeded
            - error: Error message if failed

        Example:
            >>> result = generator.generate("Count pages by author")
            >>> if result['success']:
            ...     print(result['query'])
        """
        result = {
            "query": "",
            "explanation": "",
            "success": False,
            "error": None,
        }

        try:
            prompt = self._build_prompt(question)

            # Format as messages for chat API
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)

            # Extract content from response
            content = self.iliad_client.extract_content(response)

            # Extract query from response
            query = self._extract_query(content)

            if query:
                result["query"] = query
                result["success"] = True
                logger.info(f"Generated query: {query[:100]}...")
            else:
                result["error"] = "Failed to extract valid query from response"
                logger.warning(f"Failed to extract query from: {response[:200]}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Query generation failed: {e}")

        return result

    def _extract_query(self, response: str) -> Optional[str]:
        """Extract pandas query from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Extracted query string or None
        """
        import re

        # Clean up response
        query = response.strip()

        # Remove markdown code blocks if present
        if query.startswith("```python"):
            query = query[9:]
        elif query.startswith("```"):
            query = query[3:]

        if query.endswith("```"):
            query = query[:-3]

        query = query.strip()

        # Fix common LLM issues: strip variable assignment from start
        # Pattern: variable_name = actual_expression
        assignment_match = re.match(r'^\s*\w+\s*=\s*(.+)$', query, re.DOTALL)
        if assignment_match:
            # Extract just the expression part (after the =)
            extracted = assignment_match.group(1).strip()
            # Only use it if it still references df
            if "df" in extracted:
                logger.info(f"Stripped variable assignment from query: {query[:50]}... -> {extracted[:50]}...")
                query = extracted

        # Remove any trailing semicolons or extra statements
        if ';' in query:
            # Take only the first statement
            query = query.split(';')[0].strip()
            logger.info(f"Stripped multi-statement query to first expression")

        # Basic validation
        if not query:
            return None

        # Must reference df
        if "df" not in query:
            return None

        # Check for dangerous operations
        dangerous_patterns = [
            "import ",
            "exec(",
            "eval(",
            "open(",
            "__",
            "os.",
            "sys.",
            "subprocess",
            "shutil",
            "pathlib",
        ]

        for pattern in dangerous_patterns:
            if pattern in query:
                logger.warning(f"Blocked dangerous pattern in query: {pattern}")
                return None

        return query

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a generated query for safety.

        Args:
            query: Query string to validate

        Returns:
            Dict with:
            - valid: Whether query is safe to execute
            - issues: List of issues found
        """
        issues = []

        # Check for dangerous operations
        dangerous = [
            ("import ", "Import statements not allowed"),
            ("exec(", "exec() not allowed"),
            ("eval(", "Nested eval() not allowed"),
            ("open(", "File operations not allowed"),
            ("__", "Dunder methods not allowed"),
            ("os.", "OS operations not allowed"),
            ("sys.", "sys module not allowed"),
            ("subprocess", "subprocess not allowed"),
            ("lambda x: x(", "Function calls in lambda not allowed"),
        ]

        for pattern, message in dangerous:
            if pattern in query:
                issues.append(message)

        # Check for basic structure
        if "df" not in query:
            issues.append("Query must reference 'df'")

        # Check query length
        if len(query) > 1000:
            issues.append("Query too long (max 1000 chars)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }
