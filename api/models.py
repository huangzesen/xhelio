"""Pydantic request/response schemas for the FastAPI backend."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---- Requests ----

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to send to the agent")


class FollowUpRequest(BaseModel):
    count: int = Field(default=3, ge=1, le=5, description="Number of suggestions to generate")


# ---- Responses ----

class SessionInfo(BaseModel):
    session_id: str
    model: str
    viz_backend: str = "plotly"
    created_at: datetime
    last_active: datetime
    busy: bool = False


class SessionDetail(SessionInfo):
    token_usage: dict[str, int] = Field(default_factory=dict)
    data_entries: int = 0
    plan_status: Optional[str] = None


class ServerStatus(BaseModel):
    status: str = "ok"
    active_sessions: int = 0
    max_sessions: int = 10
    uptime_seconds: float = 0.0
    api_key_configured: bool = False


class DataEntrySummary(BaseModel):
    label: str
    columns: Optional[list[str]] = None
    dims: Optional[dict[str, int]] = None
    num_points: int = 0
    shape: str = ""
    units: str = ""
    description: str = ""
    source: str = ""
    is_timeseries: bool = True
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    memory_bytes: int = 0


class SavedSessionInfo(BaseModel):
    id: str
    name: Optional[str] = None
    model: Optional[str] = None
    turn_count: int = 0
    round_count: int = 0
    last_message_preview: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    token_usage: dict[str, int] = Field(default_factory=dict)


class ResumeSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Saved session ID to resume")


class RenameSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class ErrorResponse(BaseModel):
    detail: str


# ---- Catalog ----

class MissionInfo(BaseModel):
    id: str
    name: str


class DatasetInfo(BaseModel):
    id: str
    name: str = ""
    description: str = ""


class ParameterInfo(BaseModel):
    name: str
    description: str = ""
    units: str = ""
    size: int = 1
    dataset_id: str = ""


class TimeRange(BaseModel):
    start: Optional[str] = None
    stop: Optional[str] = None


# ---- Data Fetch ----

class FetchDataRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    parameter_id: str = Field(..., min_length=1)
    time_min: str = Field(..., min_length=1)
    time_max: str = Field(..., min_length=1)


# ---- Config ----

class ConfigUpdate(BaseModel):
    config: dict[str, Any] = Field(..., description="Partial config to merge")


class ApiKeyUpdate(BaseModel):
    key: str = Field(..., min_length=1)
    provider: str = "gemini"


# ---- Memory ----

class MemoryInfo(BaseModel):
    id: str
    type: str
    scopes: list[str] = Field(default_factory=list)
    content: str
    created_at: str
    enabled: bool
    source: str = "extracted"
    tags: list[str] = Field(default_factory=list)
    access_count: int = 0
    last_accessed: str = ""
    version: int = 1
    supersedes: str = ""
    source_session: str = ""
    review_of: str = ""


class ToggleGlobalMemoryRequest(BaseModel):
    enabled: bool


# ---- Pipeline ----

class SavedSessionWithOps(BaseModel):
    id: str
    name: Optional[str] = None
    model: Optional[str] = None
    turn_count: int = 0
    last_message_preview: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    op_count: int = 0
    has_renders: bool = False


class ReplayRequest(BaseModel):
    use_cache: bool = True
    render_op_id: Optional[str] = None


# ---- Autocomplete ----

class CommandRequest(BaseModel):
    command: str = Field(..., min_length=1, description="Slash command name (without leading /)")


class CommandResponse(BaseModel):
    command: str
    content: str = Field(..., description="Markdown-formatted text for display")
    data: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    partial: str = Field(..., min_length=1)


class InputHistoryEntry(BaseModel):
    text: str = Field(..., min_length=1)


# ---- SSE Event Types ----

class SSEToolCallEvent(BaseModel):
    type: str = "tool_call"
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class SSEToolResultEvent(BaseModel):
    type: str = "tool_result"
    tool_name: str
    status: str  # "success" or "error"


class SSETextDeltaEvent(BaseModel):
    type: str = "text_delta"
    text: str


class SSEPlotEvent(BaseModel):
    type: str = "plot"
    available: bool = True


class SSEDoneEvent(BaseModel):
    type: str = "done"
    token_usage: dict[str, int] = Field(default_factory=dict)


class SSESessionTitleEvent(BaseModel):
    type: str = "session_title"
    name: str


class SSEErrorEvent(BaseModel):
    type: str = "error"
    message: str


# ---- Gallery ----

class GallerySaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    session_id: str
    render_op_id: str


class GalleryItemInfo(BaseModel):
    id: str
    name: str
    session_id: str
    render_op_id: str
    created_at: datetime
    thumbnail: str  # filename, e.g. "abc123.png"


# ---- Saved Pipelines ----

class SavePipelineRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    render_op_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class UpdatePipelineRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    steps: Optional[list[dict]] = None


class PipelineFeedbackRequest(BaseModel):
    comment: str = Field(..., min_length=1)


class ExecutePipelineRequest(BaseModel):
    time_start: str = Field(..., min_length=1)
    time_end: str = Field(..., min_length=1)


# ---- Asset Management ----

class DirStatsResponse(BaseModel):
    name: str
    path: str
    total_bytes: int
    file_count: int
    oldest_mtime: Optional[str] = None   # ISO 8601
    newest_mtime: Optional[str] = None
    turn_count: Optional[int] = None     # sessions only
    session_name: Optional[str] = None   # sessions only


class AssetCategoryResponse(BaseModel):
    name: str
    path: str
    total_bytes: int
    file_count: int
    subcategories: list[DirStatsResponse]


class AssetOverviewResponse(BaseModel):
    categories: list[AssetCategoryResponse]
    total_bytes: int
    scan_time_ms: int


class CleanupRequest(BaseModel):
    targets: list[str] = Field(default_factory=list)
    older_than_days: Optional[int] = None
    empty_only: bool = False
    dry_run: bool = False


class CleanupResponse(BaseModel):
    deleted_count: int
    freed_bytes: int
    freed_human: str
    dry_run: bool
