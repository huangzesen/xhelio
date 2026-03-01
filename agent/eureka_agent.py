"""
EurekaAgent — Discovery agent that scans session assets and proposes interesting scientific findings.
"""

import json
import base64
import logging
import threading
import uuid
import io
from datetime import datetime
from typing import Any, List, Dict, Optional

import config
from .eureka_store import EurekaStore, EurekaEntry
from .event_bus import (
    get_event_bus,
    EUREKA_FINDING,
    EUREKA_EXTRACTION_START,
    EUREKA_EXTRACTION_DONE,
    EUREKA_EXTRACTION_ERROR,
    USER_MESSAGE,
    TOOL_CALL,
    TOOL_RESULT,
    RENDER_EXECUTED
)
from knowledge.prompt_builder import build_eureka_prompt
from data_ops.store import get_store
from .llm import LLMAdapter, FunctionSchema
from .token_counter import count_tokens

logger = logging.getLogger(__name__)

class EurekaAgent:
    """Async discovery agent — scans session assets, proposes interesting findings.

    Two-phase cycle:
    1. Think: Call tools to inspect data store, figures, session history, memories
    2. Eureka: Produce structured findings (max 3 per cycle)

    Runs on daemon thread, same cadence as MemoryAgent.
    """

    def __init__(self, adapter: LLMAdapter, model_name: str, event_bus, orchestrator_ref):
        self.adapter = adapter
        self.model_name = model_name
        self.bus = event_bus
        self.orchestrator = orchestrator_ref
        self.eureka_store = EurekaStore()

    def run(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the Eureka cycle."""
        # Step 2 instructions: Create a ChatSession with build_eureka_prompt() as system prompt and the 4 tool schemas
        prompt = build_eureka_prompt()
        
        tools = [
            FunctionSchema(
                name="list_session_assets",
                description="Returns summary of all fetched datasets from the DataStore singleton.",
                parameters={"type": "object", "properties": {}}
            ),
            FunctionSchema(
                name="get_session_figure",
                description="Exports current Plotly figure to PNG bytes, returns as base64-encoded image for multimodal analysis.",
                parameters={"type": "object", "properties": {}}
            ),
            FunctionSchema(
                name="read_session_history",
                description="Returns curated EventBus events (data operations, renders, user messages).",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of events to return.",
                            "default": 20
                        }
                    }
                }
            ),
            FunctionSchema(
                name="read_memories",
                description="Returns active memories relevant to current session.",
                parameters={"type": "object", "properties": {}}
            )
        ]

        chat = self.adapter.create_chat(
            model=self.model_name,
            system_prompt=prompt,
            tools=tools
        )

        initial_message = self._build_initial_message(context)
        
        # Phase 1: Think (tool-calling loop)
        try:
            response = chat.send(initial_message)
        except Exception as e:
            logger.warning(f"EurekaAgent failed initial call: {e}")
            return []

        # Max turns for tool calling to prevent infinite loops
        max_turns = 10
        turn = 0
        while response.tool_calls and turn < max_turns:
            turn += 1
            tool_results = []
            for tc in response.tool_calls:
                result = self._execute_tool(tc.name, tc.args)
                tool_results.append(
                    self.adapter.make_tool_result_message(tc.name, result, tool_call_id=tc.id)
                )
            try:
                response = chat.send(tool_results)
            except Exception as e:
                logger.warning(f"EurekaAgent failed tool-result call: {e}")
                break

        # Phase 2: Eureka (structured JSON parsing)
        eurekas_data = self._parse_eurekas(response.text)
        
        processed = []
        session_id = context.get("session_id", "unknown")
        timestamp = datetime.now().isoformat()
        
        for e in eurekas_data:
            try:
                # Basic validation of required fields
                if not e.get("title") or not e.get("observation"):
                    continue
                
                # Align with EurekaEntry in agent/eureka_store.py
                entry = EurekaEntry(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    timestamp=timestamp,
                    title=e["title"],
                    observation=e["observation"],
                    hypothesis=e.get("hypothesis", ""),
                    evidence=e.get("evidence", []),
                    confidence=float(e.get("confidence", 0.5)),
                    tags=e.get("tags", []),
                    status="proposed"
                )
                self.eureka_store.add(entry)
                
                # Emit finding event
                from dataclasses import asdict
                self.bus.emit(
                    EUREKA_FINDING,
                    agent="EurekaAgent",
                    level="info",
                    msg=f"[Eureka] {entry.title}",
                    data=asdict(entry)
                )
                processed.append(e)
                
                if len(processed) >= 3: # Hard limit per cycle (from prompt)
                    break
            except Exception as ex:
                logger.warning(f"Failed to process eureka entry: {ex}")

        return processed

    def _build_initial_message(self, context: Dict[str, Any]) -> str:
        session_id = context.get("session_id", "unknown")
        asset_keys = context.get("data_store_keys", [])
        has_figure = context.get("has_figure", False)
        recent_msgs = context.get("recent_messages", [])
        
        lines = [
            f"Session ID: {session_id}",
            f"Active Data Store Keys: {', '.join(asset_keys) if asset_keys else 'None'}",
            f"Current Figure Available: {'Yes' if has_figure else 'No'}"
        ]
        if recent_msgs:
            lines.append("\nRecent User Messages:")
            for m in recent_msgs:
                lines.append(f"- {m}")
        
        lines.append("\nInspect the session and report any scientific findings or interesting patterns.")
        return "\n".join(lines)

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool called by the LLM."""
        try:
            if name == "list_session_assets":
                return self._tool_list_session_assets()
            elif name == "get_session_figure":
                return self._tool_get_session_figure()
            elif name == "read_session_history":
                return self._tool_read_session_history(args.get("limit", 20))
            elif name == "read_memories":
                return self._tool_read_memories()
            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}

    def _tool_list_session_assets(self) -> Dict[str, Any]:
        """Returns summary of all fetched datasets from the DataStore singleton."""
        store = get_store()
        assets = {}
        for key in store.keys():
            df = store.get(key)
            if df is not None:
                # Find time column
                time_col = next((c for c in ["time", "Epoch", "Time"] if c in df.columns), None)
                info = {
                    "shape": df.shape,
                    "parameters": list(df.columns)
                }
                if time_col:
                    info["time_range"] = {
                        "min": str(df[time_col].min()),
                        "max": str(df[time_col].max())
                    }
                assets[key] = info
        return assets

    def _tool_get_session_figure(self) -> Dict[str, Any]:
        """Exports current Plotly figure to PNG bytes, returns as base64-encoded image."""
        renderer = getattr(self.orchestrator, "_renderer", None)
        if not renderer:
            return {"error": "No renderer available"}
        
        fig = renderer.get_figure()
        if not fig:
            return {"error": "No figure currently rendered"}
        
        try:
            # Requires kaleido
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
            # For multimodal analysis, we return a dict that the LLM can interpret.
            # Base64-encoded image data is standard for many vision models in tool responses.
            return {
                "mime_type": "image/png",
                "data": b64_data,
                "msg": "Figure exported successfully."
            }
        except Exception as e:
            return {"error": f"Figure PNG export failed: {e}"}

    def _tool_read_session_history(self, limit: int) -> List[str]:
        """Returns curated EventBus events."""
        relevant_types = {USER_MESSAGE, TOOL_CALL, TOOL_RESULT, RENDER_EXECUTED}
        # Data/viz tools specifically mentioned in scope
        target_tools = {
            "fetch_data", "custom_operation", "store_dataframe",
            "render_plotly_json", "manage_plot",
            "delegate_to_mission", "delegate_to_viz_plotly", 
            "delegate_to_viz_mpl", "delegate_to_data_ops"
        }
        
        all_events = self.bus.get_events()
        curated = []
        for ev in all_events:
            if ev.type not in relevant_types:
                continue
            
            if ev.type in {TOOL_CALL, TOOL_RESULT}:
                tool_name = ev.data.get("tool", "") if ev.data else ""
                if tool_name not in target_tools:
                    continue
            
            ts = ev.ts.strftime("%H:%M:%S") if ev.ts else ""
            if ev.type == USER_MESSAGE:
                curated.append(f"[{ts}] USER: {ev.msg}")
            elif ev.type == TOOL_CALL:
                tool = ev.data.get("tool", "")
                curated.append(f"[{ts}] CALL: {tool}")
            elif ev.type == TOOL_RESULT:
                tool = ev.data.get("tool", "")
                status = ev.data.get("status", "")
                curated.append(f"[{ts}] RESULT: {tool} ({status})")
            elif ev.type == RENDER_EXECUTED:
                curated.append(f"[{ts}] RENDER: Plotly figure produced")
        
        return curated[-limit:]

    def _tool_read_memories(self) -> List[Dict[str, Any]]:
        """Returns active memories relevant to current session."""
        memory_store = getattr(self.orchestrator, "_memory_store", None)
        if not memory_store:
            return []
        
        # Pattern from _build_memory_context() in core.py: read with session scopes.
        # detected from active actors.
        active_scopes = ["generic"]
        tracker = getattr(self.orchestrator, "_ctx_tracker", None)
        # simplified: get all enabled and let LLM filter or just take recent
        memories = memory_store.get_enabled()
        return [
            {"id": m.id, "type": m.type, "content": m.content, "scopes": m.scopes}
            for m in memories[:20]
        ]

    def _parse_eurekas(self, text: str) -> List[Dict[str, Any]]:
        """Parse structured JSON (eurekas array) from LLM text response."""
        if not text:
            return []
        try:
            # Look for JSON between markers or just the whole text
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            match = re.search(r'\{.*"eurekas"\s*:\s*\[.*\]\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("eurekas", [])
            
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("eurekas", [])
        except Exception:
            pass
        return []
