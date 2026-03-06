"""Document tool handlers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core import OrchestratorAgent


def handle_read_document(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from pathlib import Path

    file_path = tool_args["file_path"]
    if not Path(file_path).is_file():
        return {"status": "error", "message": f"File not found: {file_path}"}

    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    ext = Path(file_path).suffix.lower()
    mime_type = mime_map.get(ext)
    if not mime_type:
        supported = ", ".join(sorted(mime_map.keys()))
        return {
            "status": "error",
            "message": f"Unsupported file format '{ext}'. Supported: {supported}",
        }

    try:
        import shutil

        file_bytes = Path(file_path).read_bytes()

        custom_prompt = tool_args.get("prompt", "")
        if custom_prompt:
            extraction_prompt = custom_prompt
        elif ext == ".pdf":
            extraction_prompt = (
                "Extract all text content from this document. "
                "Preserve the document structure (headings, paragraphs, lists). "
                "Render tables as markdown tables. "
                "Describe any figures or charts briefly."
            )
        else:
            extraction_prompt = (
                "Extract all text and data from this image. "
                "If it contains a table or chart, transcribe the data. "
                "If it contains text, transcribe it faithfully. "
                "Describe any visual elements briefly."
            )

        if hasattr(orch.adapter, "make_bytes_part") and hasattr(
            orch.adapter, "generate_multimodal"
        ):
            doc_part = orch.adapter.make_bytes_part(
                data=file_bytes, mime_type=mime_type
            )
            from agent.model_fallback import get_active_model

            response = orch.adapter.generate_multimodal(
                model=get_active_model(orch.model_name),
                contents=[doc_part, extraction_prompt],
            )
        else:
            from agent.model_fallback import get_active_model

            response = orch.adapter.generate(
                model=get_active_model(orch.model_name),
                contents=extraction_prompt,
            )
        orch._last_tool_context = "extract_document"
        orch._track_usage(response)

        full_text = response.text or ""

        from config import get_data_dir

        docs_dir = get_data_dir() / "documents"
        src = Path(file_path)
        stem = src.stem

        folder = docs_dir / stem
        counter = 1
        while folder.exists():
            folder = docs_dir / f"{stem}_{counter}"
            counter += 1
        folder.mkdir(parents=True, exist_ok=True)

        original_copy = folder / src.name
        shutil.copy2(str(src), str(original_copy))

        out_path = folder / f"{stem}.md"
        out_path.write_text(full_text, encoding="utf-8")
        from agent.event_bus import DEBUG
        orch._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Document] Saved to {folder} ({len(full_text)} chars)",
        )

        from agent.truncation import get_limit

        max_chars = get_limit("context.document")
        text = full_text
        truncated = max_chars > 0 and len(full_text) > max_chars
        if truncated:
            text = full_text[:max_chars]

        return {
            "status": "success",
            "file": Path(file_path).name,
            "original_saved_to": str(original_copy),
            "text_saved_to": str(out_path),
            "char_count": len(full_text),
            "truncated": truncated,
            "content": text,
        }
    except Exception as e:
        return {"status": "error", "message": f"Document reading failed: {e}"}


def handle_search_function_docs(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.function_catalog import search_functions

    query = tool_args["query"]
    package = tool_args.get("package")
    results = search_functions(query, package=package)
    return {
        "status": "success",
        "query": query,
        "count": len(results),
        "functions": results,
    }


def handle_get_function_docs(orch: "OrchestratorAgent", tool_args: dict) -> dict:
    from knowledge.function_catalog import get_function_docstring

    package = tool_args.get("package")
    function_name = tool_args.get("function_name")
    if not package or not function_name:
        return {
            "status": "error",
            "message": "Both 'package' (e.g. 'scipy.signal') and 'function_name' are required",
        }
    result = get_function_docstring(package, function_name)
    if "error" in result:
        return {"status": "error", "message": result["error"]}
    return {"status": "success", **result}
