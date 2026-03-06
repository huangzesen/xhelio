## Task Organization Rules

- **One task per mission**: if a user asks for 3 data types from PSP, create ONE PSP task
  that requests all 3. The envoy agent handles internal parallelization.
- **Different missions are separate tasks**: fetching PSP data and ACE data → two tasks.
- **Include computation tasks**: if derived quantities are needed (magnitude, ratio),
  add a task with mission="__data_ops__". Note: these depend on fetch results, so the
  orchestrator will execute them after fetches complete.
- **Include visualization tasks**: if the user asked to plot/show data, add a task with
  mission="__visualization__". The orchestrator executes this last.
- **NEVER create duplicate tasks**: one task per mission, combine all data needs.
