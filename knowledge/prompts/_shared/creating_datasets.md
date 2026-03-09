## Creating Datasets from Search Results or Documents

Google Search results or document contents can be turned into plottable datasets:

1. Use `web_search` to find event data (solar flares, CME catalogs, ICME lists, etc.)
2. Route to the DataIO agent to create a DataFrame from the text data
   - Provide the data and desired label, e.g.: "Create a DataFrame from these X-class flares: [dates and values]. Label it 'xclass_flares_2024'."
   - For documents: "Extract the data table from report.pdf"
3. The DataIO agent uses `run_code` with `store_as` (and optionally `read_document`) to construct and store the DataFrame
4. Visualize the result via the visualization agent

This is useful for requests like "search for X-class flares and plot them", "find ICME events and make a timeline", or "extract data from this PDF".