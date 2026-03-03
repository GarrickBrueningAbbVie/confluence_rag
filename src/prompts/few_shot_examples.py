"""
Few-shot examples for various prompts.

This module provides curated examples for few-shot prompting
across different use cases.

Example:
    >>> from prompts.few_shot_examples import FewShotExamples
    >>> examples = FewShotExamples.query_generation()
"""

from typing import Dict, List


class FewShotExamples:
    """
    Few-shot examples for prompts.

    Provides curated examples for:
    - Query generation
    - Intent classification
    - Technology extraction
    - Completeness assessment

    Example:
        >>> examples = FewShotExamples.query_generation()
        >>> formatted = FewShotExamples.format_examples(examples)
    """

    @staticmethod
    def query_generation() -> List[Dict[str, str]]:
        """Get few-shot examples for pandas query generation.

        Returns:
            List of example dictionaries
        """
        return [
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
            {
                "question": "Who are the top 5 content creators?",
                "query": "df['created_by'].value_counts().head(5).to_dict()",
                "explanation": "Count pages per author, get top 5",
            },
            {
                "question": "What percentage of pages have technologies listed?",
                "query": "round(df['technologies'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).mean() * 100, 2)",
                "explanation": "Calculate percentage of non-empty technology lists",
            },
        ]

    @staticmethod
    def intent_classification() -> List[Dict[str, str]]:
        """Get few-shot examples for intent classification.

        Returns:
            List of example dictionaries
        """
        return [
            {
                "query": "How many pages use Python?",
                "intent": "DATABASE",
                "reasoning": "Counting/aggregation query",
            },
            {
                "query": "What is the RAG pipeline?",
                "intent": "RAG",
                "reasoning": "Conceptual/explanatory question",
            },
            {
                "query": "List all projects and explain their purpose",
                "intent": "HYBRID",
                "reasoning": "Needs both structured listing and semantic explanation",
            },
            {
                "query": "Show me a chart of pages by author",
                "intent": "CHART",
                "reasoning": "Explicit visualization request",
            },
            {
                "query": "Explain how authentication works",
                "intent": "RAG",
                "reasoning": "Technical explanation request",
            },
            {
                "query": "Which projects have completeness below 50?",
                "intent": "DATABASE",
                "reasoning": "Filtering by numeric threshold",
            },
            {
                "query": "What are the main features of Project X?",
                "intent": "RAG",
                "reasoning": "Feature description request",
            },
            {
                "query": "Compare the technologies used across projects",
                "intent": "HYBRID",
                "reasoning": "Needs aggregation and semantic comparison",
            },
        ]

    @staticmethod
    def technology_extraction() -> List[Dict[str, str]]:
        """Get few-shot examples for technology extraction.

        Returns:
            List of example dictionaries
        """
        return [
            {
                "content": "This project uses Python and pandas for data processing. We store data in PostgreSQL and deploy on AWS EC2.",
                "technologies": ["Python", "pandas", "PostgreSQL", "AWS", "EC2"],
            },
            {
                "content": "The frontend is built with React and TypeScript. We use Jest for testing and deploy to Vercel.",
                "technologies": ["React", "TypeScript", "Jest", "Vercel"],
            },
            {
                "content": "Our ETL pipeline runs on Airflow, processing data from S3 and loading into Snowflake.",
                "technologies": ["Airflow", "S3", "Snowflake"],
            },
            {
                "content": "This is a general project overview document.",
                "technologies": [],
            },
        ]

    @staticmethod
    def format_examples(
        examples: List[Dict[str, str]],
        input_key: str = "question",
        output_key: str = "query",
    ) -> str:
        """Format examples as a string for prompts.

        Args:
            examples: List of example dictionaries
            input_key: Key for input field
            output_key: Key for output field

        Returns:
            Formatted examples string
        """
        formatted = []

        for ex in examples:
            input_val = ex.get(input_key, "")
            output_val = ex.get(output_key, "")
            explanation = ex.get("explanation", "")

            text = f"{input_key.title()}: {input_val}\n{output_key.title()}: {output_val}"
            if explanation:
                text += f"\nExplanation: {explanation}"

            formatted.append(text)

        return "\n\n".join(formatted)

    @staticmethod
    def get_random_examples(
        examples: List[Dict[str, str]],
        n: int = 5,
    ) -> List[Dict[str, str]]:
        """Get random subset of examples.

        Args:
            examples: Full list of examples
            n: Number of examples to return

        Returns:
            Random subset of examples
        """
        import random
        return random.sample(examples, min(n, len(examples)))
