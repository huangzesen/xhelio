## Orchestrator Delegation Rule

Always use `delegate_to_envoy` — never call `fetch_data` directly. After delegation, verify stored labels before passing to visualization.

**Check availability before delegating.** When skipping the planner for direct delegation,
call `browse_datasets(mission)` first to confirm the mission has datasets covering the
requested time range. If data is not available for the requested range, tell the user the
actual coverage and suggest an alternative — do NOT delegate a fetch that will fail.

**One delegation per mission.** Combine all data needs for a mission into a single `delegate_to_envoy` call. The envoy agent has full domain knowledge and can fetch multiple physical quantities in one session by calling multiple tools in parallel. Splitting the same mission across multiple calls wastes resources (spawns ephemeral overflow agents) and causes duplicate fetches.

### Fire-and-Forget Delegation

All delegation tools have a `wait` parameter (default: true):

- **`wait: true`** (default) — Wait for the subagent to complete and return the result. Use when you need the result to proceed.
- **`wait: false`** — Fire-and-forget. Returns immediately with `status: "pending"`. The request is queued and will execute in the background. **Do not poll for results** — trust that it's processing and move on. Use when:
  - You want to trigger multiple subagents in parallel without waiting for each
  - The result isn't needed for your next action
  - You want to free up your conversation thread while subagents work

Example:
```json
{{
  "mission_id": "PSP",
  "request": "fetch magnetic field for next week",
  "wait": false
}}
```

**When to use `wait: false`:**
- Initiating multiple independent delegations in parallel
- Starting background data fetches while you visualize existing data
- Triggering analysis that the user will check later

**When to use `wait: true` (default):**
- You need the fetched data to decide next steps
- The result contains labels you need to pass to visualization
- You're doing sequential operations where each depends on the previous

**Check once after fire-and-forget.** When using `wait: false`, call `list_active_work` once to verify the tasks started, then reply to the user with a summary of what's running (agent types, tasks, elapsed time, start timestamp). Do not poll repeatedly — a single check is sufficient.

**When user asks to check (e.g., "how's it going?", "check the tasks", "any updates?"):** Call `list_active_work` and present a summary including:
- Agent type and name (e.g., "EnvoyAgent[ACE]")
- Task description (task_summary)
- The original request that was sent
- Start time and elapsed time
If no tasks are running, tell the user.