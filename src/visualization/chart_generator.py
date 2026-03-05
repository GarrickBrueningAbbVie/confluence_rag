"""
Chart generation using Iliad API.

This module generates visualizations by using Iliad API to create
Plotly code based on user requests and data.

Example:
    >>> from visualization.chart_generator import ChartGenerator
    >>> generator = ChartGenerator(iliad_client)
    >>> result = generator.generate("Show pages by author", data)
    >>> if result['success']:
    ...     result['figure'].show()
"""

import json
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .code_executor import CodeExecutor

# Import types for type hints
try:
    from iliad.client import IliadClient
    import plotly.graph_objects as go
except ImportError:
    pass


class ChartGenerator:
    """
    Generate charts using Iliad API.

    Uses LLM to generate Plotly code based on user requests,
    then safely executes the code to create visualizations.

    Attributes:
        iliad_client: Iliad API client
        executor: Safe code executor
        model: Model to use for code generation

    Example:
        >>> generator = ChartGenerator(iliad_client)
        >>> result = generator.generate("Bar chart of pages per author", data)
    """

    def __init__(
        self,
        iliad_client: "IliadClient",
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize chart generator.

        Args:
            iliad_client: Configured Iliad client
            model: Model to use for code generation
        """
        self.iliad_client = iliad_client
        self.model = model
        self.executor = CodeExecutor()

        logger.info(f"Initialized ChartGenerator with model: {model}")

    def generate(
        self,
        request: str,
        data: Union[Dict, List, str],
        chart_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Generate a chart from natural language request.

        Args:
            request: User's chart request
            data: Data to visualize (dict, list, or JSON string)
            chart_type: Preferred chart type (auto, bar, line, pie, scatter)

        Returns:
            Dict with:
            - success: Whether generation succeeded
            - figure: Plotly figure object (if successful)
            - code: Generated Python code
            - error: Error message (if failed)
            - html: HTML representation of figure

        Example:
            >>> result = generator.generate("Bar chart of top 10 authors", data)
            >>> if result['success']:
            ...     result['figure'].show()
        """
        result = {
            "success": False,
            "figure": None,
            "code": None,
            "error": None,
            "html": None,
        }

        # Convert data to string representation
        data_str = self._format_data_for_prompt(data)

        if not data_str:
            result["error"] = "No data provided for visualization"
            return result

        # Generate code
        prompt = self._build_prompt(request, data_str, chart_type)

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            code = self.iliad_client.extract_content(response)

            code = self._clean_code(code)
            result["code"] = code

            logger.debug(f"Generated code:\n{code}")

            # Execute code
            exec_result = self.executor.execute(code, data)

            if exec_result["success"]:
                result["success"] = True
                result["figure"] = exec_result["figure"]
                result["html"] = exec_result.get("html")
            else:
                result["error"] = exec_result.get("error")

        except Exception as e:
            result["error"] = f"Code generation failed: {e}"
            logger.error(f"Chart generation failed: {e}")

        return result

    def _format_data_for_prompt(
        self,
        data: Union[Dict, List, str],
    ) -> Optional[str]:
        """Format data for inclusion in prompt.

        Args:
            data: Data in various formats

        Returns:
            JSON string representation or None
        """
        if not data:
            return None

        if isinstance(data, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(data)
                return json.dumps(parsed, indent=2)[:3000]
            except json.JSONDecodeError:
                return data[:3000]

        try:
            return json.dumps(data, indent=2, default=str)[:3000]
        except (TypeError, ValueError):
            return str(data)[:3000]

    def _build_prompt(
        self,
        request: str,
        data_str: str,
        chart_type: str,
    ) -> str:
        """Build the code generation prompt.

        Args:
            request: User request
            data_str: Formatted data string
            chart_type: Preferred chart type

        Returns:
            Complete prompt
        """
        chart_guidance = ""
        if chart_type != "auto":
            chart_guidance = f"Create a {chart_type} chart."

        return f"""Generate Python code using Plotly to create a visualization.

User Request: {request}
{chart_guidance}

Data:
```json
{data_str}
```

Requirements:
1. Use Plotly Express (import plotly.express as px) or Plotly Graph Objects (import plotly.graph_objects as go)
2. The data is provided in a variable called `data`
3. Create a clear, professional visualization with:
   - Appropriate title
   - Labeled axes
   - Clean color scheme
4. Store the final figure in a variable called `fig`
5. Use appropriate chart type based on the data and request
6. Handle edge cases (empty data, missing values)

Common patterns:
- For counts/aggregations: bar charts
- For time series: line charts
- For distributions: histograms
- For proportions: pie charts
- For correlations: scatter plots

Generate only executable Python code. No markdown, no explanations.
Start with the imports and end with `fig` as the last line.

Code:"""

    def _clean_code(self, code: str) -> str:
        """Clean generated code.

        Args:
            code: Raw generated code

        Returns:
            Cleaned code
        """
        # Remove markdown code blocks
        code = code.strip()

        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        return code.strip()

    def generate_quick_chart(
        self,
        data: Union[Dict, List],
        chart_type: str = "bar",
        title: str = "",
        x_label: str = "",
        y_label: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a quick chart without LLM.

        For simple, common chart types that don't need LLM generation.

        Args:
            data: Data to visualize
            chart_type: Chart type (bar, pie, line)
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            Dict with figure and success status
        """
        try:
            import plotly.express as px
            import pandas as pd

            result = {
                "success": False,
                "figure": None,
                "error": None,
            }

            # Convert dict to appropriate format
            if isinstance(data, dict):
                df = pd.DataFrame(list(data.items()), columns=["x", "y"])
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame({"x": range(len(data)), "y": data})
            else:
                result["error"] = "Unsupported data format"
                return result

            # Create chart based on type
            if chart_type == "bar":
                fig = px.bar(
                    df,
                    x=df.columns[0],
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title=title,
                )
            elif chart_type == "pie":
                fig = px.pie(
                    df,
                    names=df.columns[0],
                    values=df.columns[1] if len(df.columns) > 1 else None,
                    title=title,
                )
            elif chart_type == "line":
                fig = px.line(
                    df,
                    x=df.columns[0],
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title=title,
                )
            else:
                result["error"] = f"Unsupported chart type: {chart_type}"
                return result

            # Update labels
            if x_label:
                fig.update_xaxes(title_text=x_label)
            if y_label:
                fig.update_yaxes(title_text=y_label)

            result["success"] = True
            result["figure"] = fig
            result["html"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

            return result

        except Exception as e:
            return {
                "success": False,
                "figure": None,
                "error": str(e),
            }

    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types.

        Returns:
            List of chart type names
        """
        return [
            "auto",
            "bar",
            "line",
            "pie",
            "scatter",
            "histogram",
            "heatmap",
            "treemap",
        ]
