"""Shared pytest configuration and fixtures.

Adds a --slow flag to include tests marked @pytest.mark.slow.
By default, slow tests are skipped unless --slow is passed.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="Include tests marked @pytest.mark.slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        # --slow passed: run everything (no filtering)
        return
    skip_slow = pytest.mark.skip(reason="use --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
