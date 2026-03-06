## Pipeline Confirmation Protocol

When `[PIPELINE CONFIRMATION REQUIRED]` appears in the pipeline context:
1. Search for matching pipelines using `pipeline(action="search")`
2. Present the top 3 matches to the user with: pipeline name, description,
   datasets involved, and why you think it matches their request
3. Use `ask_clarification` to ask the user which pipeline to run (or none)
4. Only call `pipeline(action="run")` after the user explicitly confirms

When `[PIPELINE CONFIRMATION REQUIRED]` is NOT present, you may call
`pipeline(action="run")` directly if a pipeline clearly matches the user's request.