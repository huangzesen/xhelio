def test_pipeline_no_send_and_wait():
    """Verify pipeline.py doesn't use deprecated send_and_wait."""
    with open("agent/tool_handlers/pipeline.py") as f:
        content = f.read()
    assert "send_and_wait" not in content, "pipeline.py still uses send_and_wait"
