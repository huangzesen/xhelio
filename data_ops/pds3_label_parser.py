"""
Minimal PDS3 ODL label parser for fixed-width ASCII tables.

Parses ``.lbl`` label files that accompany PDS3 data files (``.sts``,
``.TAB``, ``.tab``, ``.dat``).  Uses pure regex — no external dependency.

Returns the same dict shape as ``fetch_ppi_archive._parse_xml_label()``
so that ``_read_table()`` can handle both PDS3 and PDS4 data transparently.
"""

import re


def parse_pds3_label(label_text: str) -> dict:
    """Parse a PDS3 ODL label and extract table structure.

    Args:
        label_text: Full text content of a ``.lbl`` file.

    Returns:
        Dict with keys:
        - ``table_type``: always ``"fixed_width"``
        - ``fields``: list of column dicts (name, offset, length, type,
          unit, description, column_number)
        - ``records``: number of data rows
        - ``delimiter``: always ``None``
        - ``header_bytes``: number of bytes to skip before data
          (from ``^TABLE`` pointer)
        - ``row_bytes``: bytes per row (from ``ROW_BYTES``)

    Raises:
        ValueError: If no TABLE object is found.
    """
    # Find ^TABLE pointer (bytes offset where data starts)
    header_bytes = _parse_table_pointer(label_text)

    # Find the TABLE object block
    table_match = re.search(
        r"^\s*OBJECT\s*=\s*TABLE\b(.*?)^\s*END_OBJECT\s*=\s*TABLE\b",
        label_text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if table_match is None:
        raise ValueError("No OBJECT = TABLE found in PDS3 label")

    table_block = table_match.group(1)

    # Extract table-level keywords
    rows = _extract_int(table_block, "ROWS")
    row_bytes = _extract_int(table_block, "ROW_BYTES")

    # Parse COLUMN objects
    fields = _parse_columns(table_block)

    return {
        "table_type": "fixed_width",
        "fields": fields,
        "records": rows,
        "delimiter": None,
        "header_bytes": header_bytes,
        "row_bytes": row_bytes,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_table_pointer(label_text: str) -> int:
    """Extract the byte offset from the ^TABLE pointer.

    Formats:
    - ``^TABLE = ("filename.sts", 8758<BYTES>)``  → 8757 (0-based)
    - ``^TABLE = ("filename.sts", 120)``           → 119 * record_bytes
    - ``^TABLE = 1``                                → 0 (record 1 = start)

    We only need the bytes-offset form for PDS3 files with headers.
    """
    # Bytes form: ^TABLE = ("...", NNN<BYTES>)
    m = re.search(
        r"\^\s*TABLE\s*=\s*\(\s*\"[^\"]+\"\s*,\s*(\d+)\s*<BYTES>\s*\)",
        label_text, re.IGNORECASE,
    )
    if m:
        return int(m.group(1)) - 1  # Convert 1-based to 0-based

    # Record form: ^TABLE = ("...", NNN)  — NNN is record number (1-based)
    # We'd need ROW_BYTES to convert, but for simplicity return 0
    # and let the reader handle it
    m = re.search(
        r"\^\s*TABLE\s*=\s*\(\s*\"[^\"]+\"\s*,\s*(\d+)\s*\)",
        label_text, re.IGNORECASE,
    )
    if m:
        record_num = int(m.group(1))
        # Try to find ROW_BYTES to convert record number to byte offset
        row_bytes = _extract_int_from_text(label_text, "ROW_BYTES")
        if row_bytes:
            return (record_num - 1) * row_bytes
        return 0

    # Simple form: ^TABLE = NNN
    m = re.search(r"\^\s*TABLE\s*=\s*(\d+)", label_text, re.IGNORECASE)
    if m:
        record_num = int(m.group(1))
        if record_num <= 1:
            return 0
        row_bytes = _extract_int_from_text(label_text, "ROW_BYTES")
        if row_bytes:
            return (record_num - 1) * row_bytes
        return 0

    return 0


def _parse_columns(table_block: str) -> list[dict]:
    """Parse all COLUMN objects within a TABLE block."""
    columns = []

    # Find all COLUMN blocks (they may be nested in the TABLE)
    pattern = re.compile(
        r"OBJECT\s*=\s*COLUMN\b(.*?)END_OBJECT\s*=\s*COLUMN\b",
        re.DOTALL | re.IGNORECASE,
    )

    for match in pattern.finditer(table_block):
        col_block = match.group(1)
        col = _parse_single_column(col_block)
        columns.append(col)

    return columns


def _parse_single_column(col_block: str) -> dict:
    """Parse a single COLUMN object block into a field dict."""
    name = _extract_quoted_or_bare(col_block, "NAME") or "UNKNOWN"
    col_num = _extract_int(col_block, "COLUMN_NUMBER")
    start_byte = _extract_int(col_block, "START_BYTE") or 1
    byte_length = _extract_int(col_block, "BYTES") or 0
    data_type = _extract_quoted_or_bare(col_block, "DATA_TYPE") or "CHARACTER"
    unit = _extract_quoted_or_bare(col_block, "UNIT") or ""
    fmt = _extract_quoted_or_bare(col_block, "FORMAT") or ""
    description = _extract_multiline_string(col_block, "DESCRIPTION") or ""
    null_constant = _extract_quoted_or_bare(col_block, "NULL_CONSTANT")

    # Handle ITEMS / ITEM_BYTES for vector columns
    items = _extract_int(col_block, "ITEMS")
    item_bytes = _extract_int(col_block, "ITEM_BYTES")

    return {
        "name": name.strip(),
        "offset": start_byte,  # 1-based (matches PDS4 convention)
        "length": byte_length,
        "type": data_type.strip().strip('"'),
        "unit": unit.strip().strip('"'),
        "format": fmt.strip().strip('"'),
        "description": description.strip(),
        "column_number": col_num,
        "null_constant": null_constant,
        "items": items,
        "item_bytes": item_bytes,
    }


def _extract_int(text: str, keyword: str) -> int | None:
    """Extract an integer value for a keyword."""
    m = re.search(
        rf"^\s*{keyword}\s*=\s*(\d+)",
        text, re.MULTILINE | re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


def _extract_int_from_text(text: str, keyword: str) -> int | None:
    """Extract an integer value for a keyword from arbitrary text."""
    return _extract_int(text, keyword)


def _extract_quoted_or_bare(text: str, keyword: str) -> str | None:
    """Extract a value that may be quoted or bare.

    Handles: ``NAME = "SAMPLE UTC"`` and ``NAME = FGM``
    """
    # Quoted value
    m = re.search(
        rf'^\s*{keyword}\s*=\s*"([^"]*)"',
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return m.group(1)

    # Bare value (single line, no quote)
    m = re.search(
        rf"^\s*{keyword}\s*=\s*(\S+)",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return m.group(1)

    return None


def _extract_multiline_string(text: str, keyword: str) -> str | None:
    """Extract a multi-line quoted string value.

    Handles PDS3 multi-line descriptions like::

        DESCRIPTION = "
        This is a multi-line
        description that may span
        several lines.
        "
    """
    # Find the keyword followed by = and opening quote
    m = re.search(
        rf'^\s*{keyword}\s*=\s*"',
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m is None:
        return None

    # Find the closing quote (on a line by itself or at end of text)
    start = m.end()
    # Look for a standalone closing quote (possibly with trailing whitespace)
    close = re.search(r'^\s*"', text[start:], re.MULTILINE)
    if close:
        value = text[start:start + close.start()]
    else:
        # Try inline closing quote
        close = text.find('"', start)
        if close >= 0:
            value = text[start:close]
        else:
            value = text[start:]

    # Clean up: collapse internal whitespace, strip leading/trailing
    lines = value.split("\n")
    cleaned = " ".join(line.strip() for line in lines if line.strip())
    return cleaned
