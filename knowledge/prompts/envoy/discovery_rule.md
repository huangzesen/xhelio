## Dataset Discovery

Your system prompt contains the complete dataset catalog for this mission — every instrument,
dataset ID, description, and time coverage. Use this to identify the right dataset for the
user's request. Then call `browse_parameters(dataset_id)` to see available variables before
fetching.