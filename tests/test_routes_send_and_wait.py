def test_routes_no_send_and_wait():
    """Verify routes.py doesn't use deprecated send_and_wait in production code."""
    with open("api/routes.py") as f:
        content = f.read()
    # Allow test files but not production code
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'send_and_wait' in line and 'def test_' not in line:
            import pytest
            pytest.fail(f"routes.py line {i} still uses send_and_wait: {line.strip()}")
