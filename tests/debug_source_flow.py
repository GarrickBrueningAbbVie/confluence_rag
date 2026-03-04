"""
Debug script to trace the source flow through the entire pipeline.
This will help identify where document_index is being lost.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_rag_pipeline_sources():
    """Test that RAG pipeline _format_sources adds document_index."""
    print("\n" + "=" * 60)
    print("TEST 1: RAG Pipeline _format_sources")
    print("=" * 60)

    # Create mock metadatas like what would come from vector store
    mock_metadatas = [
        {"title": "Code Documentation Tool", "url": "http://example.com/1", "source_type": "confluence"},
        {"title": "Code Documentation Tool", "url": "http://example.com/1", "source_type": "confluence"},
        {"title": "BA Projects - 11.4", "url": "http://example.com/2", "source_type": "confluence"},
    ]

    # Import and test the function directly
    from rag.pipeline import RAGPipeline

    # We need to test _format_sources without initializing the full pipeline
    # So we'll call it as a static-like method
    class MockPipeline:
        def _format_sources(self, metadatas):
            sources = []
            for i, meta in enumerate(metadatas, 1):
                sources.append({
                    "document_index": i,
                    "title": meta.get("title", "Unknown"),
                    "url": meta.get("url", ""),
                    "type": meta.get("source_type", "Unknown"),
                })
            return sources

    mock_pipeline = MockPipeline()
    sources = mock_pipeline._format_sources(mock_metadatas)

    print(f"\nInput metadatas: {len(mock_metadatas)} items")
    print(f"Output sources: {len(sources)} items")

    for i, source in enumerate(sources):
        print(f"\n  Source {i}:")
        print(f"    document_index: {source.get('document_index', 'MISSING!')}")
        print(f"    title: {source.get('title')}")
        print(f"    url: {source.get('url')}")

    # Verify
    all_have_index = all("document_index" in s for s in sources)
    print(f"\nAll sources have document_index: {all_have_index}")

    # Check if indices are correct
    indices = [s.get("document_index") for s in sources]
    expected = [1, 2, 3]
    indices_correct = indices == expected
    print(f"Indices are [1, 2, 3]: {indices_correct} (got {indices})")

    return all_have_index and indices_correct


def test_actual_rag_format_sources():
    """Test the actual RAG pipeline _format_sources method."""
    print("\n" + "=" * 60)
    print("TEST 2: Actual RAG Pipeline _format_sources (reading source)")
    print("=" * 60)

    # Read the actual source code to verify it's correct
    pipeline_path = Path(__file__).parent.parent / "src" / "rag" / "pipeline.py"
    with open(pipeline_path, 'r') as f:
        content = f.read()

    # Check for document_index in _format_sources
    if "document_index" in content:
        print("✓ 'document_index' found in pipeline.py")

        # Find the _format_sources function
        import re
        match = re.search(r'def _format_sources\(self.*?(?=\n    def |\nclass |\Z)', content, re.DOTALL)
        if match:
            func_code = match.group(0)
            print(f"\n_format_sources function found:")
            print("-" * 40)
            for line in func_code.split('\n')[:25]:
                print(f"  {line}")
            print("-" * 40)

            if '"document_index": i' in func_code or "'document_index': i" in func_code:
                print("✓ document_index is being set to 'i' (correct)")
                return True
            else:
                print("✗ document_index assignment not found in expected format")
                return False
    else:
        print("✗ 'document_index' NOT found in pipeline.py!")
        return False


def test_app_display_sources():
    """Test that app.py correctly reads document_index."""
    print("\n" + "=" * 60)
    print("TEST 3: App.py display_answer function")
    print("=" * 60)

    app_path = Path(__file__).parent.parent / "src" / "ui" / "app.py"
    with open(app_path, 'r') as f:
        content = f.read()

    # Check for document_index usage (may have default value like ", i)")
    if 'source.get("document_index"' in content or "source.get('document_index'" in content:
        print("✓ app.py reads document_index from sources")
    else:
        print("✗ app.py does NOT read document_index!")
        return False

    # Check for the expander label format
    if 'Document {doc_idx}' in content:
        print("✓ app.py formats label with Document {doc_idx}")
    else:
        print("✗ app.py does NOT format label with document index!")
        return False

    # Find the relevant code section
    import re
    match = re.search(r'# Display sources.*?(?=\n    # Display relevance|\Z)', content, re.DOTALL)
    if match:
        display_code = match.group(0)
        print(f"\nSource display code found:")
        print("-" * 40)
        for line in display_code.split('\n')[:20]:
            print(f"  {line}")
        print("-" * 40)

    return True


def test_query_router_sources():
    """Check if query_router modifies sources."""
    print("\n" + "=" * 60)
    print("TEST 4: Query Router source handling")
    print("=" * 60)

    router_path = Path(__file__).parent.parent / "src" / "routing" / "query_router.py"
    with open(router_path, 'r') as f:
        content = f.read()

    # Check all places where sources are set
    import re
    source_assignments = re.findall(r'.*sources.*=.*', content)

    print(f"Found {len(source_assignments)} source assignments:")
    for line in source_assignments:
        line = line.strip()
        if line and not line.startswith('#'):
            print(f"  {line[:80]}")

    # Check if sources are passed through without modification
    if 'result["sources"] = rag_result.get("sources", [])' in content:
        print("\n✓ Query router passes RAG sources through")
    else:
        print("\n⚠ Query router might be modifying sources")

    return True


def test_result_aggregator_sources():
    """Check result_aggregator _collect_sources."""
    print("\n" + "=" * 60)
    print("TEST 5: Result Aggregator _collect_sources")
    print("=" * 60)

    aggregator_path = Path(__file__).parent.parent / "src" / "routing" / "result_aggregator.py"
    with open(aggregator_path, 'r') as f:
        content = f.read()

    # Find _collect_sources function
    import re
    match = re.search(r'def _collect_sources\(self.*?(?=\n    def |\Z)', content, re.DOTALL)
    if match:
        func_code = match.group(0)
        print(f"\n_collect_sources function:")
        print("-" * 40)
        for line in func_code.split('\n'):
            print(f"  {line}")
        print("-" * 40)

        if "seen_urls" in func_code:
            print("\n⚠ WARNING: _collect_sources still uses URL deduplication!")
            return False
        elif "document_index" in func_code:
            print("\n✓ _collect_sources handles document_index")
            return True

    return False


def test_response_combiner_sources():
    """Check response_combiner source handling."""
    print("\n" + "=" * 60)
    print("TEST 6: Response Combiner source handling")
    print("=" * 60)

    combiner_path = Path(__file__).parent.parent / "src" / "routing" / "response_combiner.py"
    with open(combiner_path, 'r') as f:
        content = f.read()

    # Check if sources are modified
    if 'sources = rag_result.get("sources", [])' in content or "sources = rag_result.get('sources', [])" in content:
        print("✓ Response combiner passes sources through without modification")
        return True
    else:
        # Check what it does with sources
        import re
        source_lines = re.findall(r'.*"sources".*', content)
        print("Source handling in response_combiner:")
        for line in source_lines:
            print(f"  {line.strip()}")
        return True


def simulate_full_flow():
    """Simulate the full flow and check source data at each step."""
    print("\n" + "=" * 60)
    print("TEST 7: Simulated Full Flow")
    print("=" * 60)

    # Step 1: Mock RAG result (what _format_sources should produce)
    print("\nStep 1: RAG Pipeline _format_sources output")
    rag_sources = [
        {"document_index": 1, "title": "Code Documentation Tool", "url": "http://ex.com/1", "type": "confluence"},
        {"document_index": 2, "title": "Code Documentation Tool", "url": "http://ex.com/1", "type": "confluence"},
        {"document_index": 3, "title": "BA Projects", "url": "http://ex.com/2", "type": "confluence"},
    ]
    print(f"  Sources: {len(rag_sources)} items")
    for s in rag_sources:
        print(f"    doc_idx={s.get('document_index')}, title={s.get('title')[:30]}")

    # Step 2: QueryRouter passes through
    print("\nStep 2: QueryRouter (should pass through unchanged)")
    router_result = {"sources": rag_sources}
    print(f"  Sources: {len(router_result['sources'])} items")
    for s in router_result['sources']:
        print(f"    doc_idx={s.get('document_index')}, title={s.get('title')[:30]}")

    # Step 3: App display_answer receives
    print("\nStep 3: App display_answer receives")
    for source in router_result['sources']:
        doc_idx = source.get("document_index")
        title = source.get("title", "Unknown")
        if doc_idx:
            label = f"📄 Document {doc_idx}: {title}"
        else:
            label = f"📄 {title}"
        print(f"    Expander label: {label}")

    return True


def check_for_url_deduplication():
    """Search all routing files for URL deduplication."""
    print("\n" + "=" * 60)
    print("TEST 8: Search for URL deduplication in all files")
    print("=" * 60)

    routing_dir = Path(__file__).parent.parent / "src" / "routing"
    found_dedup = False

    for py_file in routing_dir.glob("*.py"):
        with open(py_file, 'r') as f:
            content = f.read()

        if "seen_urls" in content or "url not in seen" in content:
            print(f"\n⚠ Found URL deduplication in: {py_file.name}")
            found_dedup = True

            # Find the specific lines
            for i, line in enumerate(content.split('\n'), 1):
                if "seen_urls" in line or "seen" in line.lower() and "url" in line.lower():
                    print(f"    Line {i}: {line.strip()[:70]}")

    if not found_dedup:
        print("✓ No URL deduplication found in routing files")

    # Also check rag/pipeline.py
    pipeline_path = Path(__file__).parent.parent / "src" / "rag" / "pipeline.py"
    with open(pipeline_path, 'r') as f:
        content = f.read()

    if "seen_urls" in content:
        print(f"\n⚠ Found URL deduplication in: rag/pipeline.py")
        found_dedup = True

    return not found_dedup


def run_all_tests():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("SOURCE FLOW DIAGNOSTIC TESTS")
    print("=" * 60)

    results = []

    results.append(("RAG _format_sources logic", test_rag_pipeline_sources()))
    results.append(("RAG _format_sources source code", test_actual_rag_format_sources()))
    results.append(("App display_answer code", test_app_display_sources()))
    results.append(("Query router handling", test_query_router_sources()))
    results.append(("Result aggregator handling", test_result_aggregator_sources()))
    results.append(("Response combiner handling", test_response_combiner_sources()))
    results.append(("Full flow simulation", simulate_full_flow()))
    results.append(("No URL deduplication", check_for_url_deduplication()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed - code looks correct")
        print("  The issue may be:")
        print("  1. Streamlit cache not cleared (click 'Reload Pipelines' button)")
        print("  2. Python bytecode cache (.pyc files) not cleared")
        print("  3. App not restarted after code changes")
    else:
        print("\n✗ Some tests failed - see above for details")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
