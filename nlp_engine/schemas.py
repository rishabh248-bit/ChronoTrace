"""
ChronoTrace / SynapseState Pydantic v2 Structured Extraction Schemas
Strict data contracts for LLM outputs ensuring zero-fail downstream ingestion.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class ActionItemSchema(BaseModel):
    """Schema representing an extracted task, owner assignment, and commitment tier."""
    model_config = ConfigDict(extra="ignore")

    task_description: str = Field(..., description="Actionable description of the work item")
    owner_name: Optional[str] = Field(None, description="Name or speaker ID of the assigned individual")
    deadline: Optional[str] = Field(None, description="Explicit or inferred delivery target / sprint")
    commitment_level: Literal["Explicit", "Tentative", "Unassigned"] = Field(
        default="Unassigned",
        description="Tier of commitment: Explicit (clear promise), Tentative ('maybe/could'), or Unassigned"
    )
    status: Literal["Pending", "In Progress", "Completed", "Stalled"] = Field(
        default="Pending",
        description="Operational lifecycle state of the task"
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence score of the extraction"
    )


class DecisionSchema(BaseModel):
    """Schema representing an architectural or procedural decision with transcript evidence."""
    model_config = ConfigDict(extra="ignore")

    topic: str = Field(..., description="Core subject area (e.g. 'Primary Database Engine')")
    decision_text: str = Field(..., description="Clear, declarative consensus statement of the decision")
    evidence_quote: str = Field(..., description="Verbatim transcript excerpt serving as ground truth")
    evidence_timestamp: str = Field(..., description="Timestamp interval e.g. '00:08:30 - 00:12:15'")
    rationale: Optional[str] = Field(None, description="Underlying architectural trade-off or reason cited")
    status: Literal["Active", "Superseded", "Contradicted"] = Field(
        default="Active",
        description="Lifecycle status in relation to historical decisions"
    )
    cvr_confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Claim Verification Rate computed by NLI entailment verifier"
    )


class OpenQuestionSchema(BaseModel):
    """Schema representing unresolved questions contributing to conversation debt."""
    model_config = ConfigDict(extra="ignore")

    question_text: str = Field(..., description="The unresolved inquiry or unanswered blocker")
    asked_by: Optional[str] = Field(None, description="Name or speaker ID who raised the question")
    context: Optional[str] = Field(None, description="Meeting context surrounding the question")
    status: Literal["Open", "Resolved"] = Field(
        default="Open",
        description="Status of the question"
    )
    estimated_debt_impact: float = Field(
        default=5.0,
        ge=0.0,
        description="Initial debt score weighting based on architectural importance"
    )


class MeetingSummarySchema(BaseModel):
    """Master structured envelope for an entire analyzed meeting session."""
    model_config = ConfigDict(extra="ignore")

    meeting_title: str = Field(..., description="Title of the meeting session")
    session_number: int = Field(..., description="Chronological session number (e.g. 101, 102)")
    key_takeaways: List[str] = Field(default_factory=list, description="High-level summary bullets")
    decisions: List[DecisionSchema] = Field(default_factory=list, description="All architectural decisions")
    action_items: List[ActionItemSchema] = Field(default_factory=list, description="All assigned and unassigned tasks")
    open_questions: List[OpenQuestionSchema] = Field(default_factory=list, description="Unresolved questions tracking debt")
