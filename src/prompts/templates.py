"""
Centralized prompt templates for the RAG system.

This module provides reusable prompt templates for various
components of the system.

Example:
    >>> from prompts.templates import PromptTemplates
    >>> prompt = PromptTemplates.rag_response(question, context)
"""


class PromptTemplates:
    """
    Centralized prompt templates.

    Provides consistent prompts across the system for:
    - RAG responses
    - Query generation
    - Intent classification
    - Technology extraction
    - Completeness assessment

    Example:
        >>> prompt = PromptTemplates.rag_response("What is X?", context)
    """

    @staticmethod
    def rag_response(
        question: str,
        context: str,
        instructions: str = "",
    ) -> str:
        """Generate RAG response prompt.

        Args:
            question: User question
            context: Retrieved context
            instructions: Optional formatting instructions

        Returns:
            Complete prompt for LLM
        """
        base = f"""You are a helpful assistant that answers questions based on the provided context.
            Use only the information from the context to answer. If the context doesn't contain
            relevant information, say so clearly.

            Context:
            {context}

            Question: {question}
        """

        if instructions:
            base += f"\n\nAdditional instructions: {instructions}"

        base += "\n\nAnswer:"

        return base

    @staticmethod
    def query_generation(
        question: str,
        schema_info: str,
        examples: str = "",
    ) -> str:
        """Generate pandas query prompt.

        Args:
            question: Natural language question
            schema_info: DataFrame schema description
            examples: Few-shot examples

        Returns:
            Complete prompt for query generation
        """
        prompt = f"""You are a pandas query generator. Given a natural language question about
            Confluence page data, generate a pandas query to answer it.

            ## DataFrame Schema
            The DataFrame is named `df` and has the following columns:

            {schema_info}

            ## Important Notes
            - The 'technologies' column contains lists of strings
            - Use .apply(lambda x: ...) to search within list columns
            - Always handle None/NaN values appropriately
            - Return results as dictionaries, lists, or scalar values
            - Use .to_dict('records') to convert DataFrames to list of dicts
        """

        if examples:
            prompt += f"\n\n## Examples\n{examples}"

        prompt += f"""

## Your Task

Question: {question}

Generate ONLY the pandas query code. No explanation, no markdown.

Query:"""

        return prompt

    @staticmethod
    def intent_classification(question: str) -> str:
        """Generate intent classification prompt.

        Args:
            question: User question

        Returns:
            Prompt for intent classification
        """
        return f"""Classify this query into one of these categories:
- RAG: Questions about concepts, explanations, documentation (e.g., "What is X?", "Explain Y")
- DATABASE: Questions about counts, lists, filters, aggregations (e.g., "How many?", "List all")
- HYBRID: Questions needing both semantic search and structured data
- CHART: Requests for visualizations

Query: "{question}"

Respond with ONLY one word: RAG, DATABASE, HYBRID, or CHART"""

    @staticmethod
    def technology_extraction(content: str, title: str = "") -> str:
        """Generate technology extraction prompt.

        Args:
            content: Page content to analyze
            title: Optional page title

        Returns:
            Prompt for technology extraction
        """
        prompt = """Extract technologies mentioned in this Confluence page content.

Technologies include:
- Programming languages (Python, R, Java, JavaScript, etc.)
- Frameworks (Django, React, TensorFlow, Flask, etc.)
- Databases (PostgreSQL, MongoDB, MySQL, Oracle, etc.)
- Cloud services (AWS, Azure, GCP, S3, EC2, etc.)
- Tools (Docker, Kubernetes, Airflow, Git, Jenkins, etc.)
- Libraries (pandas, numpy, scikit-learn, etc.)

"""
        if title:
            prompt += f"Page Title: {title}\n\n"

        prompt += f"""Content:
{content[:4000]}

List ONLY the technology names found, one per line.
Do not include generic terms like "API", "database", "cloud".
If no specific technologies are mentioned, respond with "NONE".

Technologies:"""

        return prompt

    @staticmethod
    def completeness_assessment(content: str, title: str) -> str:
        """Generate completeness assessment prompt.

        Args:
            content: Page content to assess
            title: Page title

        Returns:
            Prompt for completeness assessment
        """
        return f"""Assess the completeness of this project documentation page against a standard project charter template.

        Project Charter Template Sections:
        1. Definition/Purpose (15%) - Project goals and objectives
        2. Benefits (10%) - Expected benefits and value
        3. Project Team (10%) - Team members and roles
        4. Milestones/Timeline (10%) - Key dates and deliverables
        5. Risks/Dependencies (10%) - Known risks and dependencies
        6. Technical Approach (15%) - Architecture, technologies, design
        7. Data Sources (10%) - Data inputs and outputs
        8. Integration Points (5%) - External system connections
        9. Security/Compliance (5%) - Security considerations
        10. Success Metrics (5%) - KPIs and success criteria
        11. Maintenance Plan (5%) - Ongoing support plan

        Page Title: {title}

        Content:
        {content[:6000]}

        For each section, indicate if it's:
        - PRESENT: Content clearly addresses this section
        - PARTIAL: Some relevant content but incomplete
        - MISSING: No content for this section

        Then calculate an overall score (0-100) and provide a brief summary.

        Format your response as:
        SECTION_SCORES:
        1. Definition/Purpose: [PRESENT/PARTIAL/MISSING]
        2. Benefits: [PRESENT/PARTIAL/MISSING]
        ...

        SCORE: [0-100]
        SUMMARY: [Brief description of gaps
    """

    @staticmethod
    def chart_generation(
        question: str,
        data: str,
        chart_type: str = "auto",
    ) -> str:
        """Generate chart code generation prompt.

        Args:
            question: User's chart request
            data: Data to visualize (JSON)
            chart_type: Preferred chart type

        Returns:
            Prompt for Plotly code generation
        """
        return f"""Generate Python code using Plotly to create a visualization.

User Request: {question}

Data:
{data}

Requirements:
- Use Plotly Express (px) or Plotly Graph Objects (go)
- Create a clear, professional visualization
- Include appropriate title and labels
- Use a clean color scheme
- The code should be self-contained and executable
- Store the final figure in a variable called `fig`

{"Preferred chart type: " + chart_type if chart_type != "auto" else "Choose the most appropriate chart type."}

Generate only the Python code, no explanations:"""

    @staticmethod
    def response_synthesis(
        question: str,
        rag_response: str,
        db_response: str,
    ) -> str:
        """Generate response synthesis prompt.

        Args:
            question: Original user question
            rag_response: Response from RAG pipeline
            db_response: Response from database pipeline

        Returns:
            Prompt for synthesizing responses
        """
        return f"""Synthesize information from two sources to answer a user's question.

User Question: {question}

Documentation Search Result:
{rag_response}

Database Query Result:
{db_response}

Instructions:
- Integrate the factual data with the contextual information
- Don't repeat information unnecessarily
- If the results contradict, note the discrepancy
- Keep the response concise and focused on the question

Answer:"""

    @staticmethod
    def document_description(
        content: str,
        filename: str,
        max_length: int = 200,
    ) -> str:
        """Generate document description prompt.

        Args:
            content: Document content
            filename: Original filename
            max_length: Maximum description length

        Returns:
            Prompt for description generation
        """
        return f"""Provide a one-sentence description of this document.

Filename: {filename}

Content (excerpt):
{content[:3000]}

Description (keep under {max_length} characters):"""
