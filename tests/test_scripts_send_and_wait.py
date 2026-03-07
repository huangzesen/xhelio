def test_scripts_no_send_and_wait():
    with open("scripts/direct_agent_test.py") as f:
        content = f.read()
    assert "send_and_wait" not in content, "direct_agent_test.py still uses send_and_wait"
