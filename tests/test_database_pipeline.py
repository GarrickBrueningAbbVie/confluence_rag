"""
Tests for the database query pipeline.

Tests validation logic, query generation, and execution to identify issues.
"""

import pandas as pd
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.query_executor import QueryExecutor


def create_test_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": ["1", "2", "3"],
        "title": ["Project Alpha", "Project Beta", "Project Gamma"],
        "created_by": ["John Smith", "Jane Doe", "John Smith"],
        "technologies": [["Python", "Airflow"], ["Java", "Spark"], ["Python", "statistical knowledge"]],
        "completeness_score": [85.0, 72.5, 90.0],
        "parent_project": ["Alpha", "Beta", "Gamma"],
    })


class TestQueryValidation:
    """Test the query validation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.df = create_test_dataframe()
        self.executor = QueryExecutor(self.df)

    def test_valid_simple_query(self):
        """Test that valid simple queries pass validation."""
        queries = [
            "df.shape[0]",
            "df['title'].tolist()",
            "df[df['created_by'] == 'John Smith'].shape[0]",
            "df['completeness_score'].mean()",
        ]
        for query in queries:
            result = self.executor._validate_query(query)
            assert result["valid"], f"Query should be valid: {query}, issues: {result['issues']}"

    def test_variable_assignment_rejected(self):
        """Test that variable assignments are rejected."""
        queries = [
            "x = df.shape[0]",
            "result = df['title'].tolist()",
            "df_filtered = df[df['created_by'] == 'John Smith']",
            "df_with_statistical_knowledge = df[df['technologies'].apply(lambda x: 'test' in x)]",
            "count = df.shape[0]",
            "my_var = 5",
        ]
        for query in queries:
            result = self.executor._validate_query(query)
            print(f"Testing: {query[:60]}...")
            print(f"  Valid: {result['valid']}, Issues: {result['issues']}")
            assert not result["valid"], f"Variable assignment should be rejected: {query}"
            assert any("assignment" in issue.lower() for issue in result["issues"]), \
                f"Should mention assignment issue: {query}"

    def test_multi_statement_rejected(self):
        """Test that multi-statement queries are rejected."""
        queries = [
            "x = 1; df.shape[0]",
            "print('test'); df['title']",
        ]
        for query in queries:
            result = self.executor._validate_query(query)
            print(f"Testing: {query}")
            print(f"  Valid: {result['valid']}, Issues: {result['issues']}")
            assert not result["valid"], f"Multi-statement should be rejected: {query}"

    def test_dangerous_patterns_rejected(self):
        """Test that dangerous patterns are rejected."""
        queries = [
            "import os; df.shape[0]",
            "df.__class__.__bases__",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
            "os.system('ls')",
        ]
        for query in queries:
            result = self.executor._validate_query(query)
            print(f"Testing: {query}")
            print(f"  Valid: {result['valid']}, Issues: {result['issues']}")
            assert not result["valid"], f"Dangerous pattern should be rejected: {query}"

    def test_comparison_operators_allowed(self):
        """Test that comparison operators (==, !=, <=, >=) are allowed."""
        queries = [
            "df[df['completeness_score'] == 85.0].shape[0]",
            "df[df['completeness_score'] != 85.0].shape[0]",
            "df[df['completeness_score'] <= 85.0].shape[0]",
            "df[df['completeness_score'] >= 85.0].shape[0]",
            "df[df['created_by'] == 'John Smith']['title'].tolist()",
        ]
        for query in queries:
            result = self.executor._validate_query(query)
            print(f"Testing: {query}")
            print(f"  Valid: {result['valid']}, Issues: {result['issues']}")
            assert result["valid"], f"Comparison operator should be allowed: {query}, issues: {result['issues']}"

    def test_lambda_with_in_operator(self):
        """Test that lambda with 'in' operator works (the failing case)."""
        queries = [
            "df[df['technologies'].apply(lambda x: 'Python' in x if isinstance(x, list) else False)]['title'].tolist()",
            "df[df['technologies'].apply(lambda x: 'statistical knowledge' in x if isinstance(x, list) else False)]['title'].tolist()",
        ]
        for query in queries:
            result = self.executor._validate_query(query)
            print(f"Testing: {query[:80]}...")
            print(f"  Valid: {result['valid']}, Issues: {result['issues']}")
            assert result["valid"], f"Lambda with 'in' should be allowed: {query}, issues: {result['issues']}"


class TestQueryExecution:
    """Test query execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.df = create_test_dataframe()
        self.executor = QueryExecutor(self.df)

    def test_execute_valid_query(self):
        """Test executing valid queries."""
        result = self.executor.execute("df.shape[0]")
        print(f"Result: {result}")
        assert result["success"], f"Should succeed: {result['error']}"
        assert result["result"] == 3

    def test_execute_filter_query(self):
        """Test executing filter queries."""
        result = self.executor.execute("df[df['created_by'] == 'John Smith'].shape[0]")
        print(f"Result: {result}")
        assert result["success"], f"Should succeed: {result['error']}"
        assert result["result"] == 2

    def test_execute_lambda_query(self):
        """Test executing lambda queries (the failing case)."""
        query = "df[df['technologies'].apply(lambda x: 'Python' in x if isinstance(x, list) else False)]['title'].tolist()"
        result = self.executor.execute(query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        assert result["success"], f"Should succeed: {result['error']}"
        assert "Project Alpha" in result["result"]

    def test_reject_variable_assignment(self):
        """Test that variable assignments are rejected during execution."""
        query = "df_filtered = df[df['created_by'] == 'John Smith']"
        result = self.executor.execute(query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        assert not result["success"], "Variable assignment should fail"
        assert "validation failed" in result["error"].lower() or "assignment" in result["error"].lower()


class TestQueryExtraction:
    """Test the query extraction and cleaning logic in QueryGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        # We need to test _extract_query without actually calling the LLM
        # So we'll create a mock generator
        from unittest.mock import MagicMock
        from database.query_generator import QueryGenerator

        mock_client = MagicMock()
        self.generator = QueryGenerator(mock_client, "test schema")

    def test_strips_variable_assignment(self):
        """Test that variable assignments are stripped from LLM output."""
        test_cases = [
            # (input, expected_output)
            ("result = df.shape[0]", "df.shape[0]"),
            ("df_filtered = df[df['col'] == 'x']", "df[df['col'] == 'x']"),
            ("count = df.shape[0]", "df.shape[0]"),
            ("df_with_statistical_knowledge = df[df['technologies'].apply(lambda x: 'test' in x)]",
             "df[df['technologies'].apply(lambda x: 'test' in x)]"),
            # Normal queries should be unchanged
            ("df.shape[0]", "df.shape[0]"),
            ("df[df['col'] == 'x']['title'].tolist()", "df[df['col'] == 'x']['title'].tolist()"),
        ]

        for input_query, expected in test_cases:
            result = self.generator._extract_query(input_query)
            print(f"Input: {input_query[:50]}...")
            print(f"  Expected: {expected[:50]}...")
            print(f"  Got: {result[:50] if result else None}...")
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_strips_semicolons(self):
        """Test that semicolons and multi-statement code is handled."""
        test_cases = [
            ("df.shape[0]; df.head()", "df.shape[0]"),
            ("result = df.shape[0]; print(result)", "df.shape[0]"),
        ]

        for input_query, expected in test_cases:
            result = self.generator._extract_query(input_query)
            print(f"Input: {input_query}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_removes_markdown_code_blocks(self):
        """Test that markdown code blocks are removed."""
        test_cases = [
            ("```python\ndf.shape[0]\n```", "df.shape[0]"),
            ("```\ndf.shape[0]\n```", "df.shape[0]"),
        ]

        for input_query, expected in test_cases:
            result = self.generator._extract_query(input_query)
            print(f"Input: {repr(input_query)}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_rejects_dangerous_patterns(self):
        """Test that dangerous patterns return None."""
        dangerous_queries = [
            "import os; df.shape[0]",
            "df.__class__.__bases__",
            "exec('df.shape[0]')",
        ]

        for query in dangerous_queries:
            result = self.generator._extract_query(query)
            print(f"Input: {query}")
            print(f"  Result: {result}")
            assert result is None, f"Dangerous query should return None: {query}"


class TestRegexPatterns:
    """Test the regex patterns used for validation."""

    def test_assignment_detection_regex(self):
        """Test the regex pattern for detecting assignments."""
        # Pattern to detect '=' that's not part of ==, !=, <=, >=
        pattern1 = r'(?<![<>=!])=(?!=)'

        # Test cases that SHOULD match (have assignment-like =)
        should_match = [
            "x = 5",
            "df_filtered = df",
            "result = df.shape[0]",
            "df_with_statistical_knowledge = df[df['test']]",
        ]

        # Test cases that should NOT match (only comparison operators)
        should_not_match = [
            "df[df['x'] == 5]",
            "df[df['x'] != 5]",
            "df[df['x'] <= 5]",
            "df[df['x'] >= 5]",
        ]

        print("\n=== Testing assignment detection regex ===")
        print(f"Pattern: {pattern1}")

        print("\nShould match (has assignment =):")
        for s in should_match:
            match = re.search(pattern1, s)
            print(f"  '{s}' -> {bool(match)}")
            assert match, f"Should match: {s}"

        print("\nShould NOT match (only comparison operators):")
        for s in should_not_match:
            match = re.search(pattern1, s)
            print(f"  '{s}' -> {bool(match)}")
            # Note: These may still match if there are other = signs
            # The key is the second pattern check

    def test_start_of_line_assignment_regex(self):
        """Test the regex for assignment at start of query."""
        pattern2 = r'^\s*\w+\s*='

        should_match = [
            "x = 5",
            "df_filtered = df",
            "result = df.shape[0]",
            "df_with_statistical_knowledge = df[df['test']]",
            "  x = 5",  # with leading whitespace
        ]

        should_not_match = [
            "df[df['x'] == 5]",
            "df.assign(x=5)",
            "lambda x: x == 5",
        ]

        print("\n=== Testing start-of-line assignment regex ===")
        print(f"Pattern: {pattern2}")

        print("\nShould match (assignment at start):")
        for s in should_match:
            match = re.search(pattern2, s)
            print(f"  '{s[:50]}' -> {bool(match)}")
            assert match, f"Should match: {s}"

        print("\nShould NOT match:")
        for s in should_not_match:
            match = re.search(pattern2, s)
            print(f"  '{s[:50]}' -> {bool(match)}")
            assert not match, f"Should NOT match: {s}"


def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("DATABASE PIPELINE TESTS")
    print("=" * 60)

    # Test query extraction (new tests)
    print("\n" + "=" * 60)
    print("QUERY EXTRACTION TESTS")
    print("=" * 60)
    extraction_tests = TestQueryExtraction()
    extraction_tests.setup_method()

    try:
        extraction_tests.test_strips_variable_assignment()
        print("✓ Variable assignment stripping tests passed")
    except AssertionError as e:
        print(f"✗ Variable assignment stripping tests FAILED: {e}")

    try:
        extraction_tests.test_strips_semicolons()
        print("✓ Semicolon stripping tests passed")
    except AssertionError as e:
        print(f"✗ Semicolon stripping tests FAILED: {e}")

    try:
        extraction_tests.test_removes_markdown_code_blocks()
        print("✓ Markdown code block removal tests passed")
    except AssertionError as e:
        print(f"✗ Markdown code block removal tests FAILED: {e}")

    try:
        extraction_tests.test_rejects_dangerous_patterns()
        print("✓ Dangerous pattern rejection tests passed")
    except AssertionError as e:
        print(f"✗ Dangerous pattern rejection tests FAILED: {e}")

    # Test regex patterns (isolated)
    print("\n" + "=" * 60)
    print("REGEX PATTERN TESTS")
    print("=" * 60)
    regex_tests = TestRegexPatterns()
    try:
        regex_tests.test_assignment_detection_regex()
        print("✓ Assignment detection regex tests passed")
    except AssertionError as e:
        print(f"✗ Assignment detection regex tests FAILED: {e}")

    try:
        regex_tests.test_start_of_line_assignment_regex()
        print("✓ Start-of-line assignment regex tests passed")
    except AssertionError as e:
        print(f"✗ Start-of-line assignment regex tests FAILED: {e}")

    # Test validation
    print("\n" + "=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)
    validation_tests = TestQueryValidation()
    validation_tests.setup_method()

    try:
        validation_tests.test_valid_simple_query()
        print("✓ Valid simple query tests passed")
    except AssertionError as e:
        print(f"✗ Valid simple query tests FAILED: {e}")

    try:
        validation_tests.test_variable_assignment_rejected()
        print("✓ Variable assignment rejection tests passed")
    except AssertionError as e:
        print(f"✗ Variable assignment rejection tests FAILED: {e}")

    try:
        validation_tests.test_multi_statement_rejected()
        print("✓ Multi-statement rejection tests passed")
    except AssertionError as e:
        print(f"✗ Multi-statement rejection tests FAILED: {e}")

    try:
        validation_tests.test_dangerous_patterns_rejected()
        print("✓ Dangerous pattern rejection tests passed")
    except AssertionError as e:
        print(f"✗ Dangerous pattern rejection tests FAILED: {e}")

    try:
        validation_tests.test_comparison_operators_allowed()
        print("✓ Comparison operator tests passed")
    except AssertionError as e:
        print(f"✗ Comparison operator tests FAILED: {e}")

    try:
        validation_tests.test_lambda_with_in_operator()
        print("✓ Lambda with 'in' operator tests passed")
    except AssertionError as e:
        print(f"✗ Lambda with 'in' operator tests FAILED: {e}")

    # Test execution
    print("\n" + "=" * 60)
    print("EXECUTION TESTS")
    print("=" * 60)
    execution_tests = TestQueryExecution()
    execution_tests.setup_method()

    try:
        execution_tests.test_execute_valid_query()
        print("✓ Valid query execution tests passed")
    except AssertionError as e:
        print(f"✗ Valid query execution tests FAILED: {e}")

    try:
        execution_tests.test_execute_filter_query()
        print("✓ Filter query execution tests passed")
    except AssertionError as e:
        print(f"✗ Filter query execution tests FAILED: {e}")

    try:
        execution_tests.test_execute_lambda_query()
        print("✓ Lambda query execution tests passed")
    except AssertionError as e:
        print(f"✗ Lambda query execution tests FAILED: {e}")

    try:
        execution_tests.test_reject_variable_assignment()
        print("✓ Variable assignment rejection execution tests passed")
    except AssertionError as e:
        print(f"✗ Variable assignment rejection execution tests FAILED: {e}")

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
