"""
DiscoveryAgent — LLM-based extraction of scientific knowledge from data exploration sessions.

Three operations:
1. extract_discoveries — Analyze ops log + conversation turns, produce discoveries with pipeline provenance
2. consolidate_session — Merge multiple raw entries within a session into coherent discoveries
3. consolidate_cross_session — LLM-based merge when total tokens exceed budget
"""

import json
import re
from datetime import datetime
from typing import Optional

import config
from .llm import LLMAdapter
from .discovery_store import Discovery, DiscoveryStore
from .memory import generate_tags, estimate_tokens
from .model_fallback import get_active_model
from .event_bus import get_event_bus, DEBUG


class DiscoveryAgent:
    """Extracts scientific discoveries from agent sessions."""

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        discovery_store: DiscoveryStore,
        session_id: str = "",
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.store = discovery_store
        self.session_id = session_id

    # ---- Extraction ----

    def extract_discoveries(
        self,
        operations_log: list[dict],
        conversation_turns: list[str],
    ) -> list[Discovery]:
        """Extract scientific discoveries from ops log and conversation.

        Args:
            operations_log: List of operation records from the session.
            conversation_turns: List of "User: ..." / "Agent: ..." strings.

        Returns:
            List of newly created Discovery objects (already saved to store).
        """
        if not operations_log:
            return []

        ops_summary = self._summarize_operations(operations_log)
        conv_text = "\n".join(conversation_turns[-20:]) if conversation_turns else "(no conversation)"

        # Get existing discoveries for dedup
        existing = self.store.get_session_discoveries(self.session_id)
        existing_summaries = [d.summary for d in existing if d.summary]

        prompt = self._build_extraction_prompt(ops_summary, conv_text, existing_summaries)

        try:
            response = self.adapter.generate(
                model=get_active_model(self.model_name),
                contents=prompt,
                temperature=0.2,
            )
        except Exception as e:
            get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="debug", msg=f"[DiscoveryAgent] LLM call failed: {e}")
            return []

        text = (response.text or "").strip()
        entries = self._parse_json_response(text)
        if entries is None:
            return []

        # Build op lookup for pipeline tracing
        op_by_id = {r["id"]: r for r in operations_log}

        discoveries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            summary = entry.get("summary", "").strip()
            content = entry.get("content", "").strip()
            if not summary or not content:
                continue

            # Dedup: skip if very similar to existing
            if self._is_duplicate(summary, existing_summaries):
                continue

            # Pipeline tracing: BFS backwards from relevant_ops
            relevant_ops = entry.get("relevant_ops", [])
            pipeline = self._trace_pipeline(relevant_ops, op_by_id, operations_log)

            # Extract missions and datasets from pipeline
            missions = set()
            datasets = set()
            for step in pipeline:
                args = step.get("args", {})
                ds = args.get("dataset_id", "")
                if ds:
                    datasets.add(ds)
                    # Infer mission from dataset prefix (e.g., "AC_H2_MFI" -> "ACE", "PSP_FLD_..." -> "PSP")
                    prefix = ds.split("_")[0] if "_" in ds else ds
                    missions.add(prefix.upper())

            # Also use LLM-provided missions if available
            llm_missions = entry.get("missions", [])
            if llm_missions:
                missions.update(m.upper() for m in llm_missions if isinstance(m, str))

            reasoning = entry.get("reasoning", "").strip()
            tags = generate_tags(f"{summary} {content}", ["generic"])

            discovery = Discovery(
                summary=summary,
                content=content,
                tags=tags,
                missions=sorted(missions),
                datasets=sorted(datasets),
                pipeline=pipeline,
                reasoning=reasoning,
                created_at=datetime.now().isoformat(),
                source_agent=entry.get("source_agent", "orchestrator"),
            )
            self.store.add(self.session_id, discovery)
            existing_summaries.append(summary)
            discoveries.append(discovery)

        if discoveries:
            get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="info", msg=f"[DiscoveryAgent] Extracted {len(discoveries)} discoveries")

        return discoveries

    # ---- Session consolidation ----

    def consolidate_session(self, session_id: str) -> int:
        """Merge multiple raw entries within a session into coherent discoveries.

        Returns the number of entries after consolidation (0 if nothing to consolidate).
        """
        discoveries = self.store.get_session_discoveries(session_id)
        if len(discoveries) <= 1:
            return len(discoveries)

        # Build prompt with all session discoveries
        entries_json = json.dumps(
            [{"summary": d.summary, "content": d.content, "reasoning": d.reasoning,
              "missions": d.missions, "datasets": d.datasets}
             for d in discoveries],
            indent=1,
        )

        prompt = f"""These discoveries were extracted from a single analysis session at different points.
Consolidate overlapping or related findings into coherent entries. Merge duplicates, combine related observations, and remove redundancy.

Current discoveries:
{entries_json}

Return a JSON array of consolidated discoveries. Each entry:
{{"summary": "...", "content": "...", "reasoning": "...", "missions": [...], "datasets": [...]}}

Rules:
- Merge entries that describe the same phenomenon
- Keep distinct observations separate
- Preserve mission/dataset references
- Content can be long — preserve important details
- Return JSON array only, no markdown fencing"""

        try:
            response = self.adapter.generate(
                model=get_active_model(self.model_name),
                contents=prompt,
                temperature=0.1,
            )
        except Exception as e:
            get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="debug", msg=f"[DiscoveryAgent] Session consolidation LLM call failed: {e}")
            return len(discoveries)

        text = (response.text or "").strip()
        entries = self._parse_json_response(text)
        if entries is None:
            return len(discoveries)

        # Build consolidated discoveries, preserving pipeline from originals
        consolidated = []
        all_pipelines = []
        for d in discoveries:
            all_pipelines.extend(d.pipeline)

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            summary = entry.get("summary", "").strip()
            content = entry.get("content", "").strip()
            if not summary:
                continue

            missions = [m for m in entry.get("missions", []) if isinstance(m, str)]
            datasets_list = [ds for ds in entry.get("datasets", []) if isinstance(ds, str)]

            # Include all pipelines from original discoveries that match these datasets
            relevant_pipeline = [
                p for p in all_pipelines
                if p.get("args", {}).get("dataset_id") in datasets_list
            ] if datasets_list else all_pipelines

            tags = generate_tags(f"{summary} {content}", ["generic"])

            consolidated.append(Discovery(
                summary=summary,
                content=content,
                tags=tags,
                missions=missions,
                datasets=datasets_list,
                pipeline=relevant_pipeline,
                reasoning=entry.get("reasoning", ""),
                created_at=datetime.now().isoformat(),
                source_agent="consolidation",
            ))

        if consolidated:
            self.store.replace_session(session_id, consolidated)
            get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="info", msg=f"[DiscoveryAgent] Session consolidation: {len(discoveries)} → {len(consolidated)}")
            return len(consolidated)

        return len(discoveries)

    # ---- Cross-session consolidation ----

    def consolidate_cross_session(self, token_budget: int = 100_000) -> bool:
        """Merge discoveries across sessions when total tokens exceed budget.

        Returns True if consolidation was performed.
        """
        current_tokens = self.store.total_tokens()
        if current_tokens <= token_budget:
            return False

        all_discoveries = self.store.get_all_flat()
        if not all_discoveries:
            return False

        entries_json = json.dumps(
            [{"summary": d.summary, "content": d.content[:500],
              "missions": d.missions, "datasets": d.datasets,
              "reasoning": d.reasoning}
             for d in all_discoveries],
            indent=1,
        )

        target_count = max(len(all_discoveries) // 2, 5)

        prompt = f"""These scientific discoveries were collected across multiple analysis sessions.
The total exceeds the token budget. Consolidate by merging overlapping findings.

Target: reduce to ~{target_count} entries while preserving unique knowledge.

Current discoveries ({len(all_discoveries)} entries):
{entries_json}

Return a JSON array of consolidated discoveries. Each entry:
{{"summary": "...", "content": "...", "reasoning": "...", "missions": [...], "datasets": [...]}}

Rules:
- Merge entries about the same phenomenon or dataset behavior
- Keep distinct scientific observations separate
- Preserve all mission/dataset references
- Content should be comprehensive but concise
- Return JSON array only, no markdown fencing"""

        try:
            response = self.adapter.generate(
                model=get_active_model(self.model_name),
                contents=prompt,
                temperature=0.1,
            )
        except Exception as e:
            get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="debug", msg=f"[DiscoveryAgent] Cross-session consolidation failed: {e}")
            return False

        text = (response.text or "").strip()
        entries = self._parse_json_response(text)
        if entries is None:
            return False

        # Group consolidated entries under a synthetic session key
        consolidated = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            summary = entry.get("summary", "").strip()
            content = entry.get("content", "").strip()
            if not summary:
                continue

            missions = [m for m in entry.get("missions", []) if isinstance(m, str)]
            datasets_list = [ds for ds in entry.get("datasets", []) if isinstance(ds, str)]
            tags = generate_tags(f"{summary} {content}", ["generic"])

            consolidated.append(Discovery(
                summary=summary,
                content=content,
                tags=tags,
                missions=missions,
                datasets=datasets_list,
                pipeline=[],  # Pipeline provenance lost in cross-session consolidation
                reasoning=entry.get("reasoning", ""),
                created_at=datetime.now().isoformat(),
                source_agent="cross_session_consolidation",
            ))

        if not consolidated:
            return False

        # Check we actually reduced tokens
        new_tokens = sum(
            estimate_tokens(d.summary) + estimate_tokens(d.content) + estimate_tokens(d.reasoning)
            for d in consolidated
        )
        if new_tokens > current_tokens * 0.9:
            get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="debug", msg="[DiscoveryAgent] Cross-session consolidation didn't reduce tokens enough")
            return False

        synthetic_session = f"consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.store.replace_all_sessions({synthetic_session: consolidated})
        get_event_bus().emit(DEBUG, agent="DiscoveryAgent", level="info", msg=f"[DiscoveryAgent] Cross-session consolidation: {len(all_discoveries)} → {len(consolidated)} entries, {current_tokens} → {new_tokens} tokens")
        return True

    # ---- Pipeline tracing ----

    def _trace_pipeline(
        self,
        relevant_op_ids: list[str],
        op_by_id: dict[str, dict],
        all_ops: list[dict],
    ) -> list[dict]:
        """BFS backwards from relevant_ops using input_producers to get minimal chain.

        Returns list of {tool, args, output_label} dicts.
        """
        if not relevant_op_ids:
            return []

        # Collect all ops in the chain via BFS
        visited: set[str] = set()
        queue = list(relevant_op_ids)
        chain_ids: set[str] = set()

        while queue:
            op_id = queue.pop(0)
            if op_id in visited:
                continue
            visited.add(op_id)
            rec = op_by_id.get(op_id)
            if rec is None:
                continue
            chain_ids.add(op_id)

            # Walk backwards using input_producers
            input_producers = rec.get("input_producers", {})
            for label, producer_id in input_producers.items():
                if producer_id not in visited:
                    queue.append(producer_id)

            # Also try direct input labels if no input_producers
            if not input_producers:
                for inp_label in rec.get("inputs", []):
                    for other_rec in all_ops:
                        if other_rec["status"] == "success" and inp_label in other_rec.get("outputs", []):
                            if other_rec["id"] not in visited:
                                queue.append(other_rec["id"])

        # Build stripped pipeline in chronological order
        pipeline = []
        for rec in all_ops:
            if rec["id"] in chain_ids and rec["status"] == "success":
                outputs = rec.get("outputs", [])
                pipeline.append({
                    "tool": rec["tool"],
                    "args": rec["args"],
                    "output_label": outputs[0] if outputs else "",
                })

        return pipeline

    # ---- Prompt building ----

    def _build_extraction_prompt(
        self,
        ops_summary: str,
        conversation_text: str,
        existing_summaries: list[str],
    ) -> str:
        existing_section = ""
        if existing_summaries:
            existing_section = "\n\nExisting discoveries for this session (do NOT duplicate):\n"
            existing_section += "\n".join(f"- {s}" for s in existing_summaries[:20])

        return f"""Analyze this data exploration session and extract scientific discoveries — observations about data behavior, correlations, anomalies, or instrument characteristics.

Only extract discoveries that represent genuine scientific knowledge learned from examining the data. Do NOT extract:
- Operational lessons (use pitfalls/reflections for those)
- User preferences
- Generic facts anyone would know

Operations performed:
{ops_summary}

Conversation context:
{conversation_text}
{existing_section}

Respond with a JSON array only (no markdown fencing). Each entry:
{{"summary": "1-2 sentence abstract", "content": "Full detailed observation (can be multiple paragraphs)", "reasoning": "Why this matters", "relevant_ops": ["op_001", "op_003"], "missions": ["PSP"], "source_agent": "orchestrator"}}

Rules:
- summary: concise abstract for search results
- content: detailed knowledge — include specific values, time ranges, comparisons
- relevant_ops: op IDs from the operations log that produced the data for this discovery
- missions: spacecraft mission IDs involved
- Return empty array [] if no scientific discoveries were made
- Do NOT duplicate existing discoveries listed above"""

    def _summarize_operations(self, operations_log: list[dict]) -> str:
        """Build a concise summary of the session's operations for the LLM prompt."""
        lines = []
        for op in operations_log:
            op_id = op.get("id", "?")
            tool = op.get("tool", "unknown")
            status = op.get("status", "unknown")
            args = op.get("args", {})
            outputs = op.get("outputs", [])
            error = op.get("error")

            if tool == "fetch_data":
                ds = args.get("dataset_id", "?")
                param = args.get("parameter_id", "?")
                ts = args.get("time_start", "")
                te = args.get("time_end", "")
                tr = f"{ts} to {te}" if ts and te else args.get("time_range", "?")
                line = f"  [{op_id}] fetch_data({ds}/{param}, {tr}) → {status}"
            elif tool == "custom_operation":
                desc = args.get("description", "?")
                label = args.get("output_label", "?")
                line = f"  [{op_id}] custom_operation({desc}) → {label} [{status}]"
            elif tool == "render_plotly_json":
                line = f"  [{op_id}] render_plotly_json → {status}"
            else:
                line = f"  [{op_id}] {tool}({', '.join(f'{k}={v}' for k, v in list(args.items())[:3])}) → {status}"

            if outputs:
                line += f" [outputs: {', '.join(outputs)}]"
            if error:
                line += f" ERROR: {error[:100]}"
            lines.append(line)

        return "\n".join(lines) if lines else "(no operations)"

    # ---- Helpers ----

    @staticmethod
    def _parse_json_response(text: str) -> list[dict] | None:
        """Parse JSON array from LLM response, stripping markdown fencing."""
        if not text:
            return None
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return None
        except (json.JSONDecodeError, ValueError):
            return None

    def _is_duplicate(
        self,
        summary: str,
        existing_summaries: list[str],
        threshold: float = 0.85,
    ) -> bool:
        """Check if summary is a duplicate of any existing discovery."""
        if not summary or not existing_summaries:
            return False
        sim = self.store.embeddings.pairwise_max_similarity(summary, existing_summaries)
        return sim > threshold
