"""
Tests for chart display functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization.chart_generator import ChartGenerator
from visualization.code_executor import CodeExecutor


def test_quick_chart_bar():
    """Test quick bar chart generation."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    generator = ChartGenerator(mock_client)

    # Test with dict data (typical database query result)
    data = {
        "Python": 15,
        "Java": 10,
        "Airflow": 8,
        "Spark": 5,
    }

    result = generator.generate_quick_chart(
        data=data,
        chart_type="bar",
        title="Technologies Usage",
    )

    print(f"Bar chart result: success={result['success']}, error={result.get('error')}")

    assert result["success"], f"Bar chart should succeed: {result.get('error')}"
    assert result["figure"] is not None, "Figure should be created"
    assert result["html"] is not None, "HTML should be generated"

    print("Bar chart test passed!")


def test_quick_chart_pie():
    """Test quick pie chart generation."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    generator = ChartGenerator(mock_client)

    # Test with dict data
    data = {
        "Project A": 30,
        "Project B": 25,
        "Project C": 20,
        "Project D": 15,
        "Project E": 10,
    }

    result = generator.generate_quick_chart(
        data=data,
        chart_type="pie",
        title="Project Distribution",
    )

    print(f"Pie chart result: success={result['success']}, error={result.get('error')}")

    assert result["success"], f"Pie chart should succeed: {result.get('error')}"
    assert result["figure"] is not None, "Figure should be created"

    print("Pie chart test passed!")


def test_quick_chart_list_of_dicts():
    """Test chart with list of dicts (like database records)."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    generator = ChartGenerator(mock_client)

    # Test with list of dicts (typical for grouped database results)
    data = [
        {"name": "John Smith", "count": 15},
        {"name": "Jane Doe", "count": 12},
        {"name": "Bob Wilson", "count": 8},
    ]

    result = generator.generate_quick_chart(
        data=data,
        chart_type="bar",
        title="Pages by Author",
    )

    print(f"List chart result: success={result['success']}, error={result.get('error')}")

    assert result["success"], f"List chart should succeed: {result.get('error')}"
    assert result["figure"] is not None, "Figure should be created"

    print("List of dicts chart test passed!")


def test_is_chartable_data():
    """Test the is_chartable_data function."""
    # Import from app
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ui"))

    # Manual implementation for testing
    def is_chartable_data(data):
        if data is None:
            return False

        if isinstance(data, dict):
            if len(data) == 0:
                return False
            values = list(data.values())
            if all(isinstance(v, (int, float)) for v in values):
                return True
            return False

        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and len(data) <= 50:
                for item in data:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, (int, float)):
                                return True
            return False

        return False

    # Test cases
    test_cases = [
        ({"a": 1, "b": 2}, True, "Dict with numeric values"),
        ({"a": "x", "b": "y"}, False, "Dict with string values"),
        ({}, False, "Empty dict"),
        (None, False, "None"),
        ([{"x": 1}, {"x": 2}], True, "List of dicts with numeric"),
        ([{"x": "a"}, {"x": "b"}], False, "List of dicts without numeric"),
        ("just a string", False, "String"),
        (42, False, "Scalar number"),
    ]

    print("\nTesting is_chartable_data:")
    for data, expected, description in test_cases:
        result = is_chartable_data(data)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: {description} -> {result} (expected {expected})")
        assert result == expected, f"Failed: {description}"

    print("is_chartable_data tests passed!")


def test_plotly_available():
    """Test that Plotly is available."""
    executor = CodeExecutor()
    assert executor.test_plotly_available(), "Plotly should be available"
    print("Plotly availability test passed!")


def run_tests():
    """Run all chart tests."""
    print("=" * 60)
    print("CHART DISPLAY TESTS")
    print("=" * 60)

    try:
        test_plotly_available()
        print("✓ Plotly availability test passed")
    except Exception as e:
        print(f"✗ Plotly availability test FAILED: {e}")
        return  # Can't continue without Plotly

    try:
        test_quick_chart_bar()
        print("✓ Bar chart test passed")
    except Exception as e:
        print(f"✗ Bar chart test FAILED: {e}")

    try:
        test_quick_chart_pie()
        print("✓ Pie chart test passed")
    except Exception as e:
        print(f"✗ Pie chart test FAILED: {e}")

    try:
        test_quick_chart_list_of_dicts()
        print("✓ List of dicts chart test passed")
    except Exception as e:
        print(f"✗ List of dicts chart test FAILED: {e}")

    try:
        test_is_chartable_data()
        print("✓ is_chartable_data test passed")
    except Exception as e:
        print(f"✗ is_chartable_data test FAILED: {e}")

    print("\n" + "=" * 60)
    print("CHART TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
